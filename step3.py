import os
import pydicom
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from rt_utils import RTStructBuilder

def process_patient(sample_dir, output_dir):
    sample_path = Path(sample_dir)
    out_path = Path(output_dir)
    
    # 确保输出目录存在
    out_path.mkdir(parents=True, exist_ok=True)
    masks_out_dir = out_path / "masks"
    masks_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"正在处理样本: {sample_dir}")
    
    # 查找输入文件夹
    ct_dir = sample_path / "CT"
    dose_dir = sample_path / "RTDOSE"
    rtstruct_dir = sample_path / "RTSTRUCT"
    
    if not ct_dir.exists() or not dose_dir.exists() or not rtstruct_dir.exists():
        print("错误：CT、RTDOSE 或 RTSTRUCT 目录不存在。")
        return

    # =========================================================================
    # 步骤 1：CT 序列张量化
    # =========================================================================
    print("步骤 1：读取 CT 序列...")
    # 查找实际包含 dicom 文件的子目录
    ct_dcm_files = list(ct_dir.rglob("*.dcm"))
    if not ct_dcm_files:
        print("错误：未在 CT 目录找到 DICOM 序列。")
        return
        
    actual_ct_dir = ct_dcm_files[0].parent
    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(actual_ct_dir))
    if not dicom_names:
        print("错误：无法在子目录解析 DICOM 序列。")
        return
        
    reader.SetFileNames(dicom_names)
    ct_image = reader.Execute()
    
    size = ct_image.GetSize()
    spacing = ct_image.GetSpacing()
    print(f"CT 尺寸: {size} (X, Y, Z)")
    print(f"CT 空间分辨率 (Spacing): {spacing}")
    
    # 保存 CT 结果 (由于只保留mask，这里跳过保存)
    # sitk.WriteImage(ct_image, str(out_path / "ct.nii.gz"))
    # print(f"CT 已保存至: {out_path / 'ct.nii.gz'}")

    # =========================================================================
    # 步骤 2：Dose 剂量图强行同化
    # =========================================================================
    print("步骤 2：读取 RTDOSE 并进行重采样...")
    dose_files = list(dose_dir.rglob("*.dcm"))
    if not dose_files:
        print("错误：未找到 Dose 文件。")
        return
    dose_file = dose_files[0]
    
    # 1. 提取物理剂量缩放因子
    try:
        dicom_hdr = pydicom.dcmread(str(dose_file), force=True)
        scaling_factor = float(dicom_hdr.DoseGridScaling)
        print(f"检测到 Dose Grid Scaling 因子: {scaling_factor}")
    except Exception as e:
        print(f"⚠️ 无法读取 DoseGridScaling: {e}，默认使用 1.0")
        scaling_factor = 1.0
        
    # 2. 读取图像并转换为真实物理剂量
    dose_image = sitk.ReadImage(str(dose_file))
    dose_image = sitk.Cast(dose_image, sitk.sitkFloat32)
    dose_image = dose_image * scaling_factor
    
    print("=== Original Image Info ===")
    print(f"CT Size: {ct_image.GetSize()}, Origin: {ct_image.GetOrigin()}")
    print(f"Dose Size: {dose_image.GetSize()}, Origin: {dose_image.GetOrigin()}")
    
    # 3. 修复 Z 轴反转 Bug (修正物理原点计算)
    if dose_image.GetDirection()[-1] == -1.0:
        print("🔄 检测到 Dose Z轴反转，正在执行物理空间修正...")
        dose_array = sitk.GetArrayFromImage(dose_image)
        # 翻转 numpy 矩阵
        dose_array = dose_array[::-1, :, :]
        
        # 【修正点】：计算 Dose 自己原来的 Z 轴终点，作为翻转后的新起点
        old_origin = dose_image.GetOrigin()
        spacing = dose_image.GetSpacing()
        size = dose_image.GetSize()
        # Dose 自己的真实物理 Z 轴尽头
        true_z_origin = old_origin[2] - (size[2] - 1) * spacing[2] 
        
        corrected_dose = sitk.GetImageFromArray(dose_array)
        corrected_dose.SetSpacing(spacing)
        # 只改变 Z 轴起点，X 和 Y 保持原样
        corrected_dose.SetOrigin((old_origin[0], old_origin[1], true_z_origin))
        corrected_dose.SetDirection(ct_image.GetDirection())
    else:
        corrected_dose = dose_image
        
    # 4. 空间重采样 (对齐到 CT)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(ct_image.GetSpacing())
    resampler.SetSize(ct_image.GetSize())
    resampler.SetOutputDirection(ct_image.GetDirection())
    resampler.SetOutputOrigin(ct_image.GetOrigin())
    resampler.SetTransform(sitk.Transform()) # 恒等变换
    resampler.SetDefaultPixelValue(0.0)      # 剂量越界填 0
    resampler.SetInterpolator(sitk.sitkLinear)# 必须用线性插值
    
    resampled_dose = resampler.Execute(corrected_dose)
    
    # 5. 验证与保存
    result_array = sitk.GetArrayFromImage(resampled_dose)
    print("\n=== Final Result Validation ===")
    print(f"Result shape: {result_array.shape}")
    print(f"Real physical Dose range (Gy): [{np.min(result_array):.2f}, {np.max(result_array):.2f}]")
    
    # 保存重采样后的 Dose (由于只保留mask，这里跳过保存)
    # sitk.WriteImage(resampled_dose, str(out_path / "dose_resampled.nii.gz"))
    # print(f"重采样 Dose 已保存至: {out_path / 'dose_resampled.nii.gz'}")
    
    # =========================================================================
    # 步骤 3：RTSTRUCT 几何坐标光栅化
    # =========================================================================
    print("步骤 3：加载 RTSTRUCT 并进行光栅化...")
    rtstruct_files = list(rtstruct_dir.rglob("*.dcm"))
    if not rtstruct_files:
        print("错误：未找到 RTSTRUCT 文件。")
        return
    rtstruct_file = rtstruct_files[0]
    
    try:
        # 使用 rt-utils 加载
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(actual_ct_dir),
            rt_struct_path=str(rtstruct_file)
        )
        
        roi_names = rtstruct.get_roi_names()
        print(f"共发现 {len(roi_names)} 个 ROI (勾画器官)。开始生成 0/1 Mask...")
        
        for roi in roi_names:
            try:
                # rtutils 默认返回的 bool 数组 shape 为 (512, 512, Z) (即 X, Y, Z)
                mask_3d = rtstruct.get_roi_mask_by_name(roi)
                
                # SimpleITK 的 numpy 格式需要是 (Z, Y, X)
                # 注：rt_utils 返回的掩码实际形状类似于 (Y, X, Z)，因此转置应该是 (2, 0, 1)
                mask_3d_zyx = np.transpose(mask_3d, (2, 0, 1))
                
                mask_uint8 = mask_3d_zyx.astype(np.uint8)
                
                sitk_mask = sitk.GetImageFromArray(mask_uint8)
                sitk_mask.CopyInformation(ct_image)
                
                safe_roi_name = "".join([c for c in roi if c.isalnum() or c in (' ', '_', '-')]).strip()
                mask_path = masks_out_dir / f"{safe_roi_name}.nii.gz"
                sitk.WriteImage(sitk_mask, str(mask_path))
                print(f"  [成功] 保存 ROI: {roi}")
            except Exception as e:
                print(f"  [失败] 无法处理 ROI '{roi}': {e}")
            
        print(f"所有 ROI 掩码已成功保存在: {masks_out_dir}")
        
    except Exception as e:
        print(f"RTSTRUCT 光栅化出现异常: {e}")
        
    print("全部操作完成！")

if __name__ == "__main__":
    BASE_INPUT_DIR = Path("/data1/home/zhaojianxiang/Dosedata/test_dose/")
    BASE_OUTPUT_DIR = Path("/data1/home/zhaojianxiang/Dosedata/dose_mask/")
    
    # 获取所有的样本文件夹
    patient_dirs = [d for d in BASE_INPUT_DIR.iterdir() if d.is_dir()]
    
    print(f"共发现 {len(patient_dirs)} 个样本需要处理。")
    
    for i, patient_dir in enumerate(patient_dirs, 1):
        print(f"\n=========================================================")
        print(f"[{i}/{len(patient_dirs)}] 正在处理样本: {patient_dir.name}")
        print(f"=========================================================")
        
        output_dir = BASE_OUTPUT_DIR / patient_dir.name
        try:
            process_patient(str(patient_dir), str(output_dir))
        except Exception as e:
            print(f"处理样本 {patient_dir.name} 时发生错误: {e}")
