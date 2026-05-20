import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

def categorize_roi_name(file_name):
    """
    核心分拣引擎：根据文件名判断所属的放射生物学类别
    """
    # 1. 极速清洗：去后缀、转小写、去空格
    name = file_name.replace('.nii.gz', '').strip().lower()
    
    # ==========================================
    # 拦截网 1：纯数字、剂量线与非法字符 (最高优先级丢弃)
    # ==========================================
    # 拦截如 "5000", ">6000", "<4000", "dose 104[%]" 等
    if name.replace('.', '').isdigit() or any(c in name for c in ['>', '<', '%']):
        return "DROP"
        
    # ==========================================
    # 拦截网 2：临床垃圾词汇黑名单 (ring, nt, Body 等无用辅助线)
    # ==========================================
    garbage_keywords = [
        'ring', 'nt', 'mark', 'body', 'external', 'couch', 'sp', 'bs', 
        'iso', 'limit', 'fan', 'ball', 'skin', 'bolus', 'point',
        'block', 'copy', 'setup', 'temp', 'water', 'air', 'artifact',
        'dose', 'line', 'contour', 'plan', 'setup'
    ]
    # 如果名字里包含垃圾词汇，直接丢弃
    if any(kw in name for kw in garbage_keywords):
        return "DROP"
        
    # 拦截布尔运算产生的残缺结构 (如 "lung-ptv" 或 "ptv-bowel")
    if '-' in name and any(kw in name for kw in ['ptv', 'gtv', 'ctv']):
        # 注意：这里可能误伤正常的 "ptv-1"，您可以根据实际情况微调
        # 为了稳妥，含有减号且组合命名的通常是辅助结构
        if 'lung' in name or 'bowel' in name or 'heart' in name:
            return "DROP"

    # ==========================================
    # 目标网 1：进攻目标 Class 1 (PTV, GTV, CTV 等靶区 - 最高优先级)
    # ==========================================
    class1_keywords = ['ptv', 'gtv', 'ctv', 'ptv', 'gtv', 'ctv']
    if any(kw in name for kw in class1_keywords):
        return 1

    # ==========================================
    # 目标网 2：致命串行 Class 2 (绝对禁区) - Spinal_Cord 等
    # ==========================================
    class2_keywords = ['cord', 'esophagus', 'eso', 'trachea', 'brainstem', 'brain_stem', 'optic', 'chiasm', 'medulla', 'spinal']
    if any(kw in name for kw in class2_keywords):
        return 2

    # ==========================================
    # 目标网 3：关键并行 Class 3 (容积约束器官)
    # ==========================================
    class3_keywords = ['lung', 'liver', 'heart', 'kidney', 'stomach', 'bowel', 'rectum', 'bladder', 'intestine', 'lens', 'eye', 'thyroid', 'parotid', 'mandible', 'breast']
    if any(kw in name for kw in class3_keywords):
        return 3

    # ==========================================
    # 兜底：未能识别的结构，交由 AI 底图处理，此处直接丢弃
    # ==========================================
    return "DROP"


def step5_clinical_override(ai_base_map_path, mask_folder, output_path):
    """
    执行步骤 5：读取散装 Mask，进行优先级分拣并覆写到 AI 底图上
    """
    # 1. 铺开步骤 4 生成的 AI 底图作为基础画布 (内部已有 0, 2, 3, 4)
    base_img = sitk.ReadImage(ai_base_map_path)
    canvas = sitk.GetArrayFromImage(base_img)
    
    # 2. 准备分类队列 (确保覆写的层级顺序)
    masks_class_3 = [] # 最底层覆写 (并联器官)
    masks_class_2 = [] # 中间层覆写 (串联禁区)
    masks_class_1 = [] # 最顶层覆写 (进攻靶区)

    # 3. 遍历文件夹，执行极速分拣
    for file_name in os.listdir(mask_folder):
        if not file_name.endswith('.nii.gz'):
            continue
            
        full_path = os.path.join(mask_folder, file_name)
        roi_class = categorize_roi_name(file_name)
        
        if roi_class == 1:
            masks_class_1.append((file_name, full_path))
        elif roi_class == 2:
            masks_class_2.append((file_name, full_path))
        elif roi_class == 3:
            masks_class_3.append((file_name, full_path))
        else:
            # 被判定为 "DROP" 的文件，静默跳过
            pass

    # ==========================================
    # 4. 暴力覆写 (顺序极其重要：画家算法 3 -> 2 -> 1)
    # ==========================================
    
    # 图层 1：盖印 Class 3 并联器官
    for name, path in masks_class_3:
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
        canvas[mask_arr > 0] = 3

    # 图层 2：盖印 Class 2 串联器官 (如有重叠，强制切掉 Class 3)
    for name, path in masks_class_2:
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
        canvas[mask_arr > 0] = 2

    # 图层 3 (最高优先级)：盖印 Class 1 靶区 (强行压制所有结构)
    for name, path in masks_class_1:
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
        canvas[mask_arr > 0] = 1

    # 5. 封装与保存
    final_img = sitk.GetImageFromArray(canvas)
    final_img.CopyInformation(base_img) # 极其关键：继承物理坐标
    sitk.WriteImage(final_img, output_path)

def process_all_samples(base_dir):
    print(f"开始批量处理目录: {base_dir}")
    if not os.path.exists(base_dir):
        print(f"错误: 目录 {base_dir} 不存在")
        return

    # 获取所有样本文件夹
    sample_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # 遍历所有样本文件夹，加入 tqdm 进度条
    for sample_name in tqdm(sample_names, desc="处理进度"):
        sample_path = os.path.join(base_dir, sample_name)
            
        mask_folder = os.path.join(sample_path, 'masks')
        step4_folder = os.path.join(sample_path, 'step4')
        
        # 匹配 AI 预测的 base_map
        # 文件名格式例如 ai_base_map___00131928.nii.gz 或 ai_base_map__00131928.nii.gz
        # 为了兼容性，使用前缀匹配，或直接用 f"ai_base_map_{sample_name}.nii.gz"
        ai_base_map_path = os.path.join(step4_folder, f'ai_base_map_{sample_name}.nii.gz')
        output_path = os.path.join(sample_path, 'radio_biology_map.nii.gz')
        
        if not os.path.exists(mask_folder):
            # print(f"[-] 跳过 {sample_name}: 缺少 masks 文件夹")
            continue
            
        if not os.path.exists(ai_base_map_path):
            # 尝试搜索 step4 下的 nii.gz 文件
            possible_maps = [f for f in os.listdir(step4_folder) if f.endswith('.nii.gz')] if os.path.exists(step4_folder) else []
            if len(possible_maps) > 0:
                ai_base_map_path = os.path.join(step4_folder, possible_maps[0])
            else:
                # print(f"[-] 跳过 {sample_name}: step4 中未找到 AI 底图")
                continue
                
        # print(f"\n=========================================")
        # print(f"▶ 开始处理样本: {sample_name}")
        step5_clinical_override(ai_base_map_path, mask_folder, output_path)

def process_all_samples_split(dose_mask_root, test_sample_root, output_root):
    """
    批量生成 Radiobiology Map：masks 与 AI 底图分属不同根目录，结果写入 output_root。
    - dose_mask_root/{sample}/masks/
    - test_sample_root/{sample}/step4/
    - output_root/radio_biology_map_{sample}.nii.gz
    """
    print(f"masks 根目录: {dose_mask_root}")
    print(f"AI 底图根目录: {test_sample_root}")
    print(f"输出目录: {output_root}")
    if not os.path.exists(dose_mask_root):
        print(f"错误: 目录 {dose_mask_root} 不存在")
        return
    if not os.path.exists(test_sample_root):
        print(f"错误: 目录 {test_sample_root} 不存在")
        return
    os.makedirs(output_root, exist_ok=True)

    sample_names = sorted(
        d for d in os.listdir(dose_mask_root)
        if os.path.isdir(os.path.join(dose_mask_root, d))
    )

    for sample_name in tqdm(sample_names, desc="Radiobiology Map"):
        mask_folder = os.path.join(dose_mask_root, sample_name, "masks")
        step4_folder = os.path.join(test_sample_root, sample_name, "step4")
        output_path = os.path.join(output_root, f"radio_biology_map_{sample_name}.nii.gz")

        if not os.path.exists(mask_folder):
            continue
        if not os.path.exists(step4_folder):
            continue

        ai_base_map_path = os.path.join(step4_folder, f"ai_base_map_{sample_name}.nii.gz")
        if not os.path.exists(ai_base_map_path):
            possible_maps = [
                f for f in os.listdir(step4_folder) if f.endswith(".nii.gz")
            ]
            if len(possible_maps) > 0:
                ai_base_map_path = os.path.join(step4_folder, possible_maps[0])
            else:
                continue

        step5_clinical_override(ai_base_map_path, mask_folder, output_path)


def process_single_sample(sample_path, output_dir):
    print(f"开始处理单样本: {sample_path}")
    sample_name = os.path.basename(sample_path.rstrip('/'))
    mask_folder = os.path.join(sample_path, 'masks')
    step4_folder = os.path.join(sample_path, 'step4')
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'radio_biology_map_{sample_name}.nii.gz')
    
    ai_base_map_path = os.path.join(step4_folder, f'ai_base_map_{sample_name}.nii.gz')
    
    if not os.path.exists(ai_base_map_path):
        possible_maps = [f for f in os.listdir(step4_folder) if f.endswith('.nii.gz')] if os.path.exists(step4_folder) else []
        if len(possible_maps) > 0:
            ai_base_map_path = os.path.join(step4_folder, possible_maps[0])
        else:
            print(f"[-] 错误: step4 中未找到 AI 底图")
            return
            
    print(f"\n=========================================")
    print(f"▶ 准备将结果输出至: {output_path}")
    step5_clinical_override(ai_base_map_path, mask_folder, output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step5: Radiobiology Map 覆写")
    parser.add_argument(
        "--split",
        action="store_true",
        help="使用分目录模式：dose_mask 与 test_sample 分开，输出到 --out",
    )
    parser.add_argument(
        "--dose-mask",
        default="/data0/zhaojianxiang/dosedata/dose_mask/",
        help="masks 所在根目录：{根}/{样本}/masks/",
    )
    parser.add_argument(
        "--test-sample",
        default="/data0/zhaojianxiang/dosedata/test_sample/",
        help="AI 底图根目录：{根}/{样本}/step4/",
    )
    parser.add_argument(
        "--out",
        default="/data0/zhaojianxiang/dosedata/oras/",
        help="分目录模式下输出目录",
    )
    parser.add_argument(
        "--legacy-base",
        default=None,
        help="旧版单根目录：masks 与 step4 均在同一父目录下（等同原 process_all_samples）",
    )
    args = parser.parse_args()

    if args.split:
        process_all_samples_split(
            os.path.abspath(args.dose_mask),
            os.path.abspath(args.test_sample),
            os.path.abspath(args.out),
        )
    elif args.legacy_base:
        process_all_samples(os.path.abspath(args.legacy_base))
    else:
        # 默认：分目录批量（与当前数据布局一致）
        process_all_samples_split(
            os.path.abspath(args.dose_mask),
            os.path.abspath(args.test_sample),
            os.path.abspath(args.out),
        )
