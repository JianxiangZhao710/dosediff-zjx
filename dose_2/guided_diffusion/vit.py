import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def triple(t): # 3D helper
    return t if isinstance(t, tuple) else (t, t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_q, x_k, x_v):
        q = self.to_q(x_q)
        k = self.to_k(x_k)
        v = self.to_v(x_v)
        
        # Split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])

        # Dot product attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class ViT_fusion_3D(nn.Module):
    def __init__(self, image_size, patch_size, dim, heads, mlp_dim, channels=1, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        
        # image_size: (D, H, W) of the feature map at bottleneck
        # patch_size: (pd, ph, pw)
        
        self.image_depth, self.image_height, self.image_width = triple(image_size)
        self.patch_depth, self.patch_height, self.patch_width = triple(patch_size)

        # Check divisibility
        assert self.image_depth % self.patch_depth == 0, f"Depth {self.image_depth} not divisible by {self.patch_depth}"
        assert self.image_height % self.patch_height == 0, f"Height {self.image_height} not divisible by {self.patch_height}"
        assert self.image_width % self.patch_width == 0, f"Width {self.image_width} not divisible by {self.patch_width}"

        num_patches = (self.image_depth // self.patch_depth) * \
                      (self.image_height // self.patch_height) * \
                      (self.image_width // self.patch_width)
                      
        patch_dim = channels * self.patch_depth * self.patch_height * self.patch_width

        # Patch Embedding Layers for Q, K, V
        # Input: (B, C, D, H, W) -> Output: (B, N, Dim)
        
        self.to_patch_embedding_q = nn.Sequential(
            Rearrange('b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)', 
                      p1=self.patch_depth, p2=self.patch_height, p3=self.patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_embedding_k = nn.Sequential(
            Rearrange('b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)', 
                      p1=self.patch_depth, p2=self.patch_height, p3=self.patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_embedding_v = nn.Sequential(
            Rearrange('b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)', 
                      p1=self.patch_depth, p2=self.patch_height, p3=self.patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer Block (Standard Self-Attention)
        # Note: In fusion context, we use Q, K, V from different sources
        # Q = CT, K = Dis, V = Main (or similar combination)
        self.transformer_block = nn.ModuleList([
            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        ])

        # Reconstruct back to 3D feature map
        self.return_linear = nn.Linear(dim, patch_dim)
        
        self.reshape_back = Rearrange('b (d h w) (p1 p2 p3 c) -> b c (d p1) (h p2) (w p3)', 
                                      d=self.image_depth // self.patch_depth,
                                      h=self.image_height // self.patch_height,
                                      w=self.image_width // self.patch_width,
                                      p1=self.patch_depth, p2=self.patch_height, p3=self.patch_width)

    def forward(self, x_q, x_k, x_v):
        # Legacy 3-input: x_q=CT, x_k=DIS, x_v=Main (for backwards compatibility)
        return self.forward_4(x_q, x_k, x_k, x_v)

    def forward_4(self, x_ct, x_syn, x_dis, x_main):
        # x_ct: CT features
        # x_syn: Synthetic dose features
        # x_dis: DIS features
        # x_main: Main features
        
        # Fuse conditions: CT + SYN + DIS -> merged condition
        x_k = x_ct + x_syn + x_dis
        
        # Patch Embed
        q = self.to_patch_embedding_q(x_ct)
        k = self.to_patch_embedding_k(x_k)
        v = self.to_patch_embedding_v(x_main)
        
        # Add Positional Embedding
        q += self.pos_embedding
        k += self.pos_embedding
        v += self.pos_embedding
        
        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)
        
        # Attention Fusion
        attn, ff = self.transformer_block
        # Cross Attention: Q queries K, applied to V
        # Here we follow DoseDiff logic: use CT/Dis as context to modulate Main
        # Or typical QKV attention: Attention(Q, K, V)
        x = attn(q, k, v) + v # Residual connection to V
        x = ff(x) + x
        
        # Project back
        x = self.return_linear(x)
        x = self.reshape_back(x)
        
        return x
