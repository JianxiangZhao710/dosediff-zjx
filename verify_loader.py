import sys
sys.path.append("./")
import torch
from guided_diffusion.dose_loader_3d import Dataset_PSDM_3D_Train
from torch.utils.data import DataLoader

def test_loader():
    data_root = '/data0/zhaojianxiang/MedSegDiff_Data_3D/train'
    patch_size = (32, 128, 128)
    
    print(f"Testing loader with root: {data_root}")
    
    try:
        dataset = Dataset_PSDM_3D_Train(data_root=data_root, patch_size=patch_size)
        print(f"Dataset length: {len(dataset)}")
        
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        print("Fetching first batch...")
        for i, (ct, psdm, dose) in enumerate(loader):
            print(f"Batch {i} loaded successfully:")
            print(f"  CT shape:   {ct.shape} (Expected: [2, 1, 32, 128, 128])")
            print(f"  PSDM shape: {psdm.shape} (Expected: [2, 11, 32, 128, 128])")
            print(f"  Dose shape: {dose.shape} (Expected: [2, 1, 32, 128, 128])")
            
            print(f"  CT range:   [{ct.min():.4f}, {ct.max():.4f}]")
            print(f"  PSDM range: [{psdm.min():.4f}, {psdm.max():.4f}]")
            print(f"  Dose range: [{dose.min():.4f}, {dose.max():.4f}]")
            
            break
            
        print("\nVerification PASSED!")
        
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loader()
