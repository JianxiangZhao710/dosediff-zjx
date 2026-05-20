import re
import os

def categorize_roi_name(file_name):
    name = file_name.replace('.nii.gz', '').strip().lower()
    
    if name.replace('.', '').isdigit() or any(c in name for c in ['>', '<', '%']):
        return "DROP"
        
    # 拦截包含特定标志的明确垃圾辅助结构
    # 用独立单词匹配或前后带符号，如 "ring", "nt", "sp", "sp03"
    garbage_exact = ['ring', 'nt', 'body', 'external', 'couch', 'iso', 'skin', 'bolus', 'water', 'air']
    if any(g == name or f"{g}_" in name or f"_{g}" in name or f"-{g}" in name or f"{g}-" in name for g in garbage_exact):
        return "DROP"
        
    if re.search(r'^sp\d*$', name) or re.search(r'^bs\d*$', name):
        return "DROP"
        
    garbage_keywords = [
        'mark', 'limit', 'fan', 'ball', 'point',
        'block', 'copy', 'setup', 'temp', 'artifact',
        'dose', 'line', 'contour', 'plan', 'inring'
    ]
    if any(kw in name for kw in garbage_keywords):
        return "DROP"

    if '-' in name and any(kw in name for kw in ['ptv', 'gtv', 'ctv']):
        if 'lung' in name or 'bowel' in name or 'heart' in name or 'cord' in name:
            return "DROP"

    class1_keywords = ['ptv', 'gtv', 'ctv']
    if any(kw in name for kw in class1_keywords):
        return 1

    class2_keywords = ['cord', 'esophagus', 'eso', 'trachea', 'brainstem', 'brain_stem', 'optic', 'chiasm', 'medulla', 'spinal']
    if any(kw in name for kw in class2_keywords):
        return 2

    class3_keywords = ['lung', 'liver', 'heart', 'kidney', 'stomach', 'bowel', 'rectum', 'bladder', 'intestine', 'lens', 'eye', 'thyroid', 'parotid', 'mandible', 'breast']
    if any(kw in name for kw in class3_keywords):
        return 3

    return "DROP"

folder = '/data0/zhaojianxiang/dosedata/test_sample/_00131928/masks'
for f in os.listdir(folder):
    if f.endswith('.nii.gz'):
        print(f'{f}: {categorize_roi_name(f)}')
