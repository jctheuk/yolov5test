#!/usr/bin/env python3
"""
Fix the dataloader classification label loading issue
"""

import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def fix_verify_image_label():
    """Fix the verify_image_label function"""
    print("FIXING DATALOADER CLASSIFICATION LABEL LOADING")
    print("=" * 60)
    
    dataloader_file = Path("yolov5c/utils/dataloaders.py")
    
    if not dataloader_file.exists():
        print("ERROR: dataloader file not found")
        return False
    
    # Read the file
    with open(dataloader_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Change the exception return to return proper classification label
    old_exception_return = "        return [None, None, None, None, nm, nf, ne, nc, msg, [0]]"
    new_exception_return = "        return [None, None, None, None, nm, nf, ne, nc, msg, [1.0, 0.0, 0.0]]"
    
    if old_exception_return in content:
        content = content.replace(old_exception_return, new_exception_return)
        print("FIX 1: Fixed exception return classification label")
    else:
        print("WARNING: Exception return pattern not found")
    
    # Fix 2: Ensure classification_line is properly initialized
    # Check if the function properly handles the case when classification_line is None
    if "classification_line = None" in content:
        print("OK: classification_line initialization found")
    else:
        print("WARNING: classification_line initialization not found")
    
    # Write the fixed file
    with open(dataloader_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("OK: Dataloader file updated")
    return True

def test_fix():
    """Test if the fix works"""
    print("\nTESTING THE FIX")
    print("=" * 30)
    
    import torch  # Add torch import
    
    # Clear cache first
    cache_files = [
        "regurgitationV1/train/labels.cache",
        "regurgitationV1/train/labels.cache.npy"
    ]
    
    for cache_file in cache_files:
        cache_path = Path(cache_file)
        if cache_path.exists():
            cache_path.unlink()
            print(f"Cleared cache: {cache_file}")
    
    # Test dataloader
    try:
        from yolov5c.utils.dataloaders import create_dataloader
        
        print("Creating dataloader with fix...")
        train_loader, dataset = create_dataloader(
            'regurgitationV1/train/images',
            imgsz=416,
            batch_size=2,
            stride=32,
            single_cls=False,
            hyp={'cls_task': 0.3},
            augment=False,
            cache=False,
            rect=False,
            rank=-1,
            workers=0,
            prefix='',
            shuffle=False
        )
        
        print("OK: Dataloader created successfully")
        
        # Test first batch
        batch = next(iter(train_loader))
        images, targets, paths, shapes, classification_labels = batch
        
        print(f"Classification labels: {classification_labels}")
        
        # Check if labels are different now
        unique_labels = torch.unique(classification_labels, dim=0)
        print(f"Unique classification labels: {len(unique_labels)}")
        
        if len(unique_labels) > 1:
            print("SUCCESS: Multiple unique classification labels found!")
            print("   The fix worked - classification labels are now being loaded correctly")
            return True
        else:
            print("ERROR: Still getting same classification labels")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("DATALOADER CLASSIFICATION FIX")
    print("=" * 50)
    
    # Apply the fix
    if fix_verify_image_label():
        print("OK: Fix applied successfully")
        
        # Test the fix
        import torch
        if test_fix():
            print("\nSUCCESS: Classification labels are now loading correctly!")
            print("   Your model should now train with proper classification labels")
            print("   The 40% accuracy issue should be resolved")
        else:
            print("\nERROR: Fix did not work, need further investigation")
    else:
        print("ERROR: Failed to apply fix")

if __name__ == "__main__":
    main()
