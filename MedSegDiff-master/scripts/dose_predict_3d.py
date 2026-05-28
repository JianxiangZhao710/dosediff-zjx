import sys
import gc
sys.path.append("../")
sys.path.append("./")

import os
import argparse
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入模型和数据加载器
from guided_diffusion.dose_loader_3d import Dataset_PSDM_3D_Train, OPENKBP_MASK_NAMES
from guided_diffusion.unet_3d import UNetModel_MS_Former_3D
from guided_diffusion.unet_3d_v1_1 import UNetModel_GatedXQueryViT_3D
from guided_diffusion.unet_3d_v1_2 import UNetModel_GatedXQueryViT_3D_v1_2
from guided_diffusion.unet_3d_v2 import UNetModel_ControlSwinFlow_3D
from flow_matching import FlowMatching


# Model registry: must match scripts/dose_train_3d.py
MODEL_REGISTRY = {
    'v1': UNetModel_MS_Former_3D,
    'v1_1_gated_xquery_vit': UNetModel_GatedXQueryViT_3D,
    'v1_2_gated_xquery_vit': UNetModel_GatedXQueryViT_3D_v1_2,
    'v2_control_swin_flow_unet': UNetModel_ControlSwinFlow_3D,
}

def denormalize_dose(dose, dose_max=80.0, dose_norm_factor=40.0):
    """反归一化剂量值：从归一化范围 (-1 到 1) 转回原始单位 (0 到 dose_max Gy)"""
    dose = (dose + 1.0) * dose_norm_factor
    dose = np.clip(dose, 0, dose_max)
    return dose

def normalize_ct(img):
    """归一化CT"""
    img = np.clip(img, 0, 2500)
    img = img / 1250.0 - 1.0
    return img

def predict_full_volume(model, ct_volume, syn_volume, cond_volume, patch_size=(64, 128, 128),
                        batch_size=1, steps=50, device='cuda'):
    """
    对整个3D体积进行预测（使用滑动窗口）

    Args:
        model: FlowMatching模型
        ct_volume: CT数据 (D, H, W)
        syn_volume: Synthetic dose数据 (D, H, W)
        cond_volume: 条件通道 (C, D, H, W)，OpenKBP 配置下 C=11（11 个 mask）
        patch_size: patch大小 (D, H, W)
        batch_size: batch大小
        steps: Flow Matching采样步数
        device: 设备

    Returns:
        预测的剂量体积 (D, H, W)，已反归一化到Gy单位
    """
    D, H, W = ct_volume.shape
    patch_d, patch_h, patch_w = patch_size
    
    # 初始化输出
    pred_dose = np.zeros((D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)

    def _offsets(dim_size, patch_size, overlap_ratio=0.25):
        """Generate starting offsets for sliding window along one dimension.

        - If patch_size >= dim_size, returns [0] (single patch, will be padded if needed).
        - Otherwise generates evenly stepped offsets with the given overlap and
          appends a final offset (dim_size - patch_size) to cover the tail.
        """
        if patch_size >= dim_size:
            return [0]
        step = max(1, int(patch_size * (1.0 - overlap_ratio)))
        offsets = list(range(0, dim_size - patch_size, step))
        last = dim_size - patch_size
        if not offsets or offsets[-1] != last:
            offsets.append(last)
        return sorted(set(offsets))

    d_offsets = _offsets(D, patch_d, 0.25)
    h_offsets = _offsets(H, patch_h, 0.25)
    w_offsets = _offsets(W, patch_w, 0.25)

    model.eval()
    with torch.no_grad():
        patches = []
        positions = []

        for d in d_offsets:
            for h in h_offsets:
                for w in w_offsets:
                    d_end = min(d + patch_d, D)
                    h_end = min(h + patch_h, H)
                    w_end = min(w + patch_w, W)
                    
                    ct_patch = ct_volume[d:d_end, h:h_end, w:w_end]
                    syn_patch = syn_volume[d:d_end, h:h_end, w:w_end]
                    cond_patch = cond_volume[:, d:d_end, h:h_end, w:w_end]

                    if ct_patch.shape != patch_size:
                        pad_d = patch_d - ct_patch.shape[0]
                        pad_h = patch_h - ct_patch.shape[1]
                        pad_w = patch_w - ct_patch.shape[2]

                        ct_patch = np.pad(
                            ct_patch,
                            ((0, pad_d), (0, pad_h), (0, pad_w)),
                            'constant', constant_values=-1.0,
                        )
                        syn_patch = np.pad(
                            syn_patch,
                            ((0, pad_d), (0, pad_h), (0, pad_w)),
                            'constant', constant_values=-1.0,
                        )
                        cond_patch = np.pad(
                            cond_patch,
                            ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                            'constant', constant_values=0.0,
                        )

                    patches.append((ct_patch, syn_patch, cond_patch))
                    positions.append((d, h, w, d_end, h_end, w_end))
        
        # 批量预测（逐个处理以减少内存使用）
        for i in tqdm(range(0, len(patches), batch_size), desc="Predicting patches"):
            batch_patches = patches[i:i+batch_size]
            batch_positions = positions[i:i+batch_size]
            
            for patch_idx, (ct_patch, syn_patch, cond_patch) in enumerate(batch_patches):
                d, h, w, d_end, h_end, w_end = batch_positions[patch_idx]

                ct_tensor = torch.from_numpy(ct_patch).unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, D, H, W)
                syn_tensor = torch.from_numpy(syn_patch).unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, D, H, W)
                cond_tensor = torch.from_numpy(cond_patch).unsqueeze(0).float().to(device)  # (1, C, D, H, W)

                with torch.cuda.device(device):
                    pred_patch = model.sample(ct_tensor, syn_tensor, cond_tensor, steps=steps)  # (1, 1, D, H, W)
                    pred_patch = pred_patch.squeeze(0).squeeze(0).cpu().numpy()  # (D, H, W)

                del ct_tensor, syn_tensor, cond_tensor
                torch.cuda.empty_cache()
                
                # 裁剪到实际大小
                actual_d = d_end - d
                actual_h = h_end - h
                actual_w = w_end - w
                
                pred_patch_crop = pred_patch[:actual_d, :actual_h, :actual_w]
                pred_patch_crop = denormalize_dose(pred_patch_crop)
                
                # 累加到输出
                pred_dose[d:d_end, h:h_end, w:w_end] += pred_patch_crop
                count_map[d:d_end, h:h_end, w:w_end] += 1.0
                
                # 清理
                del pred_patch
        
        # 平均重叠区域
        count_map[count_map == 0] = 1.0  # 避免除零
        pred_dose = pred_dose / count_map
    
    return pred_dose

def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型进行3D剂量预测')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='模型权重路径，例如: trained_models/MedSegDiff_Flow_3D_bs1_epoch600/model_epoch200.pth')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='测试数据目录，例如: /data0/zhaojianxiang/preprocessed_data/test-pats_preprocess')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='预测结果保存目录')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 128, 128],
                        help='Patch大小 (D, H, W)，默认: 64 128 128')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch大小，默认: 1（实际会逐个处理以减少内存）')
    parser.add_argument('--steps', type=int, default=30, help='Flow Matching采样步数，默认: 30（减少步数可降低内存使用）')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU编号，默认: 0')
    parser.add_argument('--model_channels', type=int, default=64,
                        help='UNet 基础通道数，必须与训练时一致（默认 64）')
    parser.add_argument('--model_name', type=str, default='v1',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='velocity-field network variant. v1=baseline, v1_1_gated_xquery_vit=v1.1 gated+X-query ViT')

    args = parser.parse_args()
    
    # 设置设备
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print(f"Loading model from: {args.model_path}")
    patch_size = tuple(args.patch_size)
    if any(s % 16 != 0 for s in patch_size):
        raise ValueError(f"patch_size 必须为 16 的倍数，当前: {patch_size}")
    # OpenKBP: 11 binary masks (PTV70/63/56 + 7 OARs + possible_dose_mask)
    dis_channels = len(OPENKBP_MASK_NAMES)
    
    if args.model_channels % 32 != 0:
        raise ValueError(f"--model_channels 必须为 32 的倍数 (GroupNorm32), 当前: {args.model_channels}")

    ModelClass = MODEL_REGISTRY[args.model_name]
    print(f"[Model] using model_name='{args.model_name}' -> {ModelClass.__name__}")

    model = ModelClass(
        image_size=patch_size,
        in_channels=1,
        ct_channels=1,
        syn_channels=1,
        dis_channels=dis_channels,
        model_channels=args.model_channels,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        channel_mult=(1, 2, 4, 4),
        dims=3
    )
    
    # 加载权重
    print("Loading checkpoint...")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    # 处理DDP权重（移除'module.'前缀）
    if any(k.startswith('module.') for k in checkpoint.keys()):
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    
    # 清理CPU内存
    del checkpoint
    import gc
    gc.collect()
    
    print("Moving model to device...")
    model = model.to(device)
    model.eval()
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    print("Model loaded successfully")
    
    # 创建Flow Matching包装器
    flow_model = FlowMatching(model)
    
    # 获取患者列表
    patient_list = sorted([p for p in os.listdir(args.data_dir) 
                           if os.path.isdir(os.path.join(args.data_dir, p))])
    print(f"Found {len(patient_list)} patients in {args.data_dir}")
    
    # OpenKBP: 11 个二值 mask 作为条件（顺序与训练加载器一致）
    mask_names = list(OPENKBP_MASK_NAMES)
    print(f"Condition channels ({len(mask_names)}): {mask_names}")

    for patient_id in tqdm(patient_list, desc="Processing patients"):
        patient_dir = os.path.join(args.data_dir, patient_id)
        output_patient_dir = os.path.join(args.output_dir, patient_id)
        if os.path.exists(os.path.join(output_patient_dir, 'dose.nii.gz')):
            print(f"Skipping {patient_id} (already exists)")
            continue

        os.makedirs(output_patient_dir, exist_ok=True)

        ct_path = os.path.join(patient_dir, 'ct.nii.gz')
        if not os.path.exists(ct_path):
            print(f"Warning: CT file not found for {patient_id}, skipping...")
            continue

        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata().astype(np.float32)
        ct_data = ct_data.transpose(2, 0, 1)  # (H, W, D) -> (D, H, W)

        ct_data = normalize_ct(ct_data)

        # Load synthetic dose
        syn_path = os.path.join(patient_dir, 'dose_synthetic.nii.gz')
        if os.path.exists(syn_path):
            syn_data = nib.load(syn_path).get_fdata().astype(np.float32).transpose(2, 0, 1)
            syn_data = np.clip(syn_data, 0, 80)
            syn_data = syn_data / 40.0 - 1.0  # normalize same as dose
        else:
            syn_data = np.zeros_like(ct_data)

        empty_shape = ct_data.shape
        mask_channels = []
        for name in mask_names:
            fpath = os.path.join(patient_dir, '{}.nii.gz'.format(name))
            if os.path.exists(fpath):
                m = nib.load(fpath).get_fdata().astype(np.float32).transpose(2, 0, 1)
                if m.shape != empty_shape:
                    m = np.zeros(empty_shape, dtype=np.float32)
            else:
                m = np.zeros(empty_shape, dtype=np.float32)
            mask_channels.append((m > 0).astype(np.float32))
        cond_volume = np.stack(mask_channels, axis=0)  # (11, D, H, W)

        print(f"Predicting for {patient_id}...")
        pred_dose = predict_full_volume(
            flow_model, ct_data, syn_data, cond_volume,
            patch_size=patch_size,
            batch_size=args.batch_size,
            steps=args.steps,
            device=device
        )
        
        # 转回原始方向 (D, H, W) -> (H, W, D)
        pred_dose = pred_dose.transpose(1, 2, 0)
        
        # 保存预测结果
        pred_nii = nib.Nifti1Image(pred_dose, ct_nii.affine, ct_nii.header)
        output_path = os.path.join(output_patient_dir, 'dose.nii.gz')
        nib.save(pred_nii, output_path)
        print(f"Saved prediction to: {output_path}")
    
    print(f"\nAll predictions saved to: {args.output_dir}")
    print(f"\nTo evaluate, run:")
    print(f"python evaluate_openKBP.py --prediction_dir {args.output_dir} --gt_dir <gt_dir> --denormalize 0")

if __name__ == '__main__':
    main()
