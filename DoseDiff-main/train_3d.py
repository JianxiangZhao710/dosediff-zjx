import os
import argparse
import torch
import numpy as np
import torch.distributed as dist
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR
import shutil

# 新的组件
from dataset_3d import Dataset_PSDM_3D_Train
from guided_diffusion.unet_3d import UNetModel_MS_Former_3D
from flow_matching import FlowMatching

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="0", help='which gpu is used')
parser.add_argument('--bs', type=int, default=1, help='batch size per gpu')
parser.add_argument('--epoch', type=int, default=2000, help='all_epochs')
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--steps", type=int, default=50, help='sampling steps')

args = parser.parse_args()

if args.local_rank == -1:
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend="nccl", init_method="env://")
device = torch.device("cuda", args.local_rank)

if dist.get_rank() == 0:
    print(
        f"[rank {dist.get_rank()}] LOCAL_RANK={args.local_rank} "
        f"device={device}",
        flush=True
    )

train_bs = args.bs
lr_max = 0.0001
# Patch Size: (Depth, Height, Width)
# 显存优化: 32x128x128. 如果显存不够，减小到 16x128x128 或 32x64x64
patch_size = (32, 128, 128) 

all_epochs = args.epoch
data_root_train = '/data0/zhaojianxiang/preprocessed_data/NPY/train'

L2 = 0.0001
save_name = 'Flow_3D_bs{}_epoch{}'.format(dist.get_world_size() * args.bs, args.epoch)

if dist.get_rank() == 0:
    if os.path.exists(os.path.join('trained_models', save_name)):
        pass
    os.makedirs(os.path.join('trained_models', save_name), exist_ok=True)
    # train_writer = SummaryWriter(os.path.join('trained_models', save_name, 'log/train'), flush_secs=2)

# Dataset
train_data = Dataset_PSDM_3D_Train(data_root=data_root_train, patch_size=patch_size)
train_samper = Data.distributed.DistributedSampler(train_data)
train_dataloader = DataLoader(dataset=train_data, batch_size=train_bs, sampler=train_samper, 
                              shuffle=False, num_workers=4, pin_memory=True)

if dist.get_rank() == 0:
    print(f"Dataset size: {len(train_data)}")

# Model
dis_channels = 20
model = UNetModel_MS_Former_3D(
    image_size=patch_size,
    in_channels=1, 
    ct_channels=1, 
    dis_channels=dis_channels,
    model_channels=64, # Reduced from 128 for 3D memory safety
    out_channels=1, 
    num_res_blocks=2, 
    attention_resolutions=(8, 16), # Adjust based on patch size
    channel_mult=(1, 2, 4, 4), # 4 levels
    dims=3
)

model = model.to(device)

# Flow Matching Wrapper
flow_model = FlowMatching(model)

# DDP Wrapper
ddp_model = DDP(
    model,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
    find_unused_parameters=False
)
# Assign ddp_model back to flow_model for training
flow_model.net = ddp_model

optimizer = optim.AdamW(ddp_model.parameters(), lr=lr_max, weight_decay=L2)
lr_scheduler = MultiStepLR(optimizer, milestones=[int((7 / 10) * args.epoch)], gamma=0.1, last_epoch=-1)

for epoch in range(all_epochs):
    train_samper.set_epoch(epoch)
    lr = optimizer.param_groups[0]['lr']
    ddp_model.train()
    
    train_epoch_loss = []
    
    for i, (ct, dis, rtdose) in enumerate(train_dataloader):
        ct = ct.to(device, non_blocking=True).float()
        dis = dis.to(device, non_blocking=True).float()
        rtdose = rtdose.to(device, non_blocking=True).float()
        
        optimizer.zero_grad()
        
        # Flow Matching Loss
        loss = flow_model.get_loss(rtdose, ct, dis)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
        optimizer.step()
        
        # Reduce loss for logging
        dist.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
        loss = loss / dist.get_world_size()
        
        train_epoch_loss.append(loss.item())
        
        if dist.get_rank() == 0 and i % 10 == 0:
            print('[%d/%d, %d/%d] train_loss: %.3f' %
                  (epoch + 1, all_epochs, i + 1, len(train_dataloader), loss.item()))
                  
    lr_scheduler.step()
    
    if dist.get_rank() == 0:
        mean_loss = np.mean(train_epoch_loss)
        # train_writer.add_scalar('lr', lr, epoch + 1)
        # train_writer.add_scalar('train_loss', mean_loss, epoch + 1)
        
        # Save Model
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(),
                       os.path.join('trained_models', save_name, 'model_epoch' + str(epoch + 1) + '.pth'))

if dist.get_rank() == 0:
    # train_writer.close()
    print('Training finished.')
