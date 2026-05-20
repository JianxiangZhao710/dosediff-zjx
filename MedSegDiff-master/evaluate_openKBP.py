import numpy as np
import os
import SimpleITK as sitk
from tqdm import tqdm

"""
These codes are modified from https://github.com/ababier/open-kbp
"""


def denormalize_dose(dose, dose_max=80.0, dose_norm_factor=40.0):
    """
    反归一化剂量值：从归一化范围 (-1 到 1) 转回原始单位 (0 到 dose_max Gy)
    
    Args:
        dose: 归一化后的剂量值（范围 -1 到 1）
        dose_max: 最大剂量值（默认 80 Gy）
        dose_norm_factor: 归一化因子（默认 40，对应归一化公式 dose/40 - 1）
    
    Returns:
        反归一化后的剂量值（单位：Gy，范围 0 到 dose_max）
    """
    dose = (dose + 1.0) * dose_norm_factor
    dose = np.clip(dose, 0, dose_max)
    return dose


def get_3D_Dose_dif(pred, gt, possible_dose_mask=None):
    if possible_dose_mask is not None:
        pred = pred[possible_dose_mask > 0]
        gt = gt[possible_dose_mask > 0]

    dif = np.mean(np.abs(pred - gt))
    return dif


def get_DVH_metrics(_dose, _mask, mode, spacing=None):
    output = {}

    if mode == 'target':
        _roi_dose = _dose[_mask > 0]
        if len(_roi_dose) == 0:
            output['D1'] = 0.0
            output['D95'] = 0.0
            output['D99'] = 0.0
            return output

        # D1
        output['D1'] = np.percentile(_roi_dose, 99)
        # D95
        output['D95'] = np.percentile(_roi_dose, 5)
        # D99
        output['D99'] = np.percentile(_roi_dose, 1)

    elif mode == 'OAR':
        if spacing is None:
            raise Exception('calculate OAR metrics need spacing')

        _roi_dose = _dose[_mask > 0]
        _roi_size = len(_roi_dose)
        if _roi_size == 0:
            output['D_0.1_cc'] = 0.0
            output['mean'] = 0.0
            return output

        _voxel_size = np.prod(spacing)
        voxels_in_tenth_of_cc = np.maximum(1, np.round(100 / _voxel_size))
        # D_0.1_cc
        fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc / _roi_size * 100
        # 保护逻辑：如果体积太小，percentile 限制在 [0, 100]
        fractional_volume_to_evaluate = np.clip(fractional_volume_to_evaluate, 0, 100)
        output['D_0.1_cc'] = np.percentile(_roi_dose, fractional_volume_to_evaluate)
        # Dmean
        output['mean'] = np.mean(_roi_dose)
    else:
        raise Exception('Unknown mode!')

    return output


def get_Dose_score_and_DVH_score(
    prediction_dir,
    gt_dir,
    denormalize=True,
    dose_max=80.0,
    dose_norm_factor=40.0,
    return_detail=False,
):
    """
    计算3D剂量分布的平均绝对误差和各结构DVH指标的绝对误差
    
    Args:
        prediction_dir: 预测结果目录
        gt_dir: 真实数据目录
        denormalize: 是否对预测结果进行反归一化（默认 True，适用于 MedSegDiff-master 的输出）
        dose_max: 最大剂量值（默认 80 Gy）
        dose_norm_factor: 归一化因子（默认 40，对应归一化公式 dose/40 - 1）
    
    Returns:
        default(return_detail=False):
            dose_mae_mean, dose_mae_std, dvh_score_mean, dvh_score_std
        when return_detail=True:
            dose_mae_mean, dose_mae_std, dvh_score_mean, dvh_score_std, dvh_results
    """
    list_dose_dif = []
    list_DVH_dif = []  # Added for global DVH Score
    # 使用字典存储每个结构的DVH差异
    dvh_dif_dict = {}

    # Only iterate over real patient subdirectories (skip stray files like evaluation_results.txt)
    patient_ids = sorted([
        p for p in os.listdir(prediction_dir)
        if os.path.isdir(os.path.join(prediction_dir, p))
    ])
    list_patient_ids = tqdm(patient_ids)
    for patient_id in list_patient_ids:
        pred_nii = sitk.ReadImage(prediction_dir + '/' + patient_id + '/dose.nii.gz')
        pred = sitk.GetArrayFromImage(pred_nii)
        
        # 如果预测结果是归一化的，进行反归一化
        if denormalize:
            pred = denormalize_dose(pred, dose_max=dose_max, dose_norm_factor=dose_norm_factor)

        gt_nii = sitk.ReadImage(gt_dir + '/' + patient_id + '/dose.nii.gz')
        gt = sitk.GetArrayFromImage(gt_nii)

        # 计算3D剂量分布的平均绝对误差（MAE）
        possible_dose_mask_nii = sitk.ReadImage(gt_dir + '/' + patient_id + '/Mask_possible_dose_mask.nii.gz')
        possible_dose_mask = sitk.GetArrayFromImage(possible_dose_mask_nii)
        dose_mae = get_3D_Dose_dif(pred, gt, possible_dose_mask)
        list_dose_dif.append(dose_mae)

        # 计算各结构的DVH指标绝对误差
        for structure_name in ['Brainstem',
                               'SpinalCord',
                               'RightParotid',
                               'LeftParotid',
                               'Esophagus',
                               'Larynx',
                               'Mandible',
                               'PTV70',
                               'PTV63',
                               'PTV56']:
            structure_file = gt_dir + '/' + patient_id + '/Mask_' + structure_name + '.nii.gz'

            # If the structure has been delineated
            if os.path.exists(structure_file):
                structure_nii = sitk.ReadImage(structure_file, sitk.sitkUInt8)
                structure = sitk.GetArrayFromImage(structure_nii)

                spacing = structure_nii.GetSpacing()
                if structure_name.find('PTV') > -1:
                    mode = 'target'
                else:
                    mode = 'OAR'
                
                pred_DVH = get_DVH_metrics(pred, structure, mode=mode, spacing=spacing)
                gt_DVH = get_DVH_metrics(gt, structure, mode=mode, spacing=spacing)

                # 初始化该结构的字典（如果不存在）
                if structure_name not in dvh_dif_dict:
                    dvh_dif_dict[structure_name] = {}

                # 计算每个DVH指标的绝对误差
                for metric in gt_DVH.keys():
                    # 初始化该指标的列表（如果不存在）
                    if metric not in dvh_dif_dict[structure_name]:
                        dvh_dif_dict[structure_name][metric] = []
                    
                    dvh_abs_error = abs(gt_DVH[metric] - pred_DVH[metric])
                    dvh_dif_dict[structure_name][metric].append(dvh_abs_error)
                    list_DVH_dif.append(dvh_abs_error) # Collect for global DVH Score

    # 计算3D剂量MAE的统计值
    dose_mae_mean = np.mean(list_dose_dif)
    dose_mae_std = np.std(list_dose_dif)

    # 计算全局 DVH Score
    dvh_score_mean = np.mean(list_DVH_dif)
    dvh_score_std = np.std(list_DVH_dif)

    # 计算各结构DVH差异的统计值
    dvh_results = {}
    for structure_name, metrics_dict in dvh_dif_dict.items():
        dvh_results[structure_name] = {}
        for metric, error_list in metrics_dict.items():
            dvh_results[structure_name][metric] = {
                'mean': np.mean(error_list),
                'std': np.std(error_list)
            }

    if return_detail:
        return dose_mae_mean, dose_mae_std, dvh_score_mean, dvh_score_std, dvh_results
    return dose_mae_mean, dose_mae_std, dvh_score_mean, dvh_score_std


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='评估OpenKBP数据集上的剂量预测结果')
    parser.add_argument('--prediction_dir', type=str, required=True, help='预测结果目录路径')
    parser.add_argument('--gt_dir', type=str, required=True, help='真实数据目录路径')
    parser.add_argument('--denormalize', type=int, default=1, help='是否对预测结果进行反归一化（1=是，0=否，默认1）')
    parser.add_argument('--dose_max', type=float, default=80.0, help='最大剂量值（默认80.0 Gy）')
    parser.add_argument('--dose_norm_factor', type=float, default=40.0, help='归一化因子（默认40.0，对应归一化公式dose/40-1）')
    args = parser.parse_args()
    
    print("=" * 80)
    print("开始评估剂量预测结果...")
    print("=" * 80)
    if args.denormalize:
        print(f"将进行反归一化：dose = (dose + 1) * {args.dose_norm_factor}，范围 [0, {args.dose_max}] Gy")
    else:
        print("预测结果已为原始单位（Gy），不进行反归一化")
    print("=" * 80)
    
    # 计算评估指标
    dose_mae_mean, dose_mae_std, dvh_score_mean, dvh_score_std, dvh_results = get_Dose_score_and_DVH_score(
        args.prediction_dir, args.gt_dir, 
        denormalize=bool(args.denormalize),
        dose_max=args.dose_max,
        dose_norm_factor=args.dose_norm_factor,
        return_detail=True,
    )
    
    # 打印 Dose Score
    print("\n" + "=" * 80)
    print("Dose Score (3D 剂量分布 MAE):")
    print("=" * 80)
    print(f"Dose_score: {dose_mae_mean:.4f}")
    print(f"Dose_std  : {dose_mae_std:.4f}")

    # 打印 DVH Score
    print("\n" + "=" * 80)
    print("DVH Score (所有结构所有指标的 MAE):")
    print("=" * 80)
    print(f"DVH_score : {dvh_score_mean:.4f}")
    print(f"DVH_std   : {dvh_score_std:.4f}")
    
    # 打印各结构DVH差异结果
    print("\n" + "=" * 80)
    print("各结构 DVH 指标的绝对误差详情:")
    print("=" * 80)
    
    for structure_name, metrics_dict in sorted(dvh_results.items()):
        print(f"\n【{structure_name}】")
        for metric, stats in sorted(metrics_dict.items()):
            print(f"  {metric:15s}: 平均值 = {stats['mean']:8.4f} Gy, "
                  f"标准差 = {stats['std']:8.4f} Gy")
    
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)
