#!/usr/bin/env python3
"""
Debug classification label parsing step by step
"""

import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

def debug_single_label_file():
    """Debug parsing of a single label file"""
    print("DEBUG SINGLE LABEL FILE PARSING")
    print("=" * 50)
    
    # Test with a specific label file
    label_file = Path("regurgitationV1/train/labels/a2hiwqVqZ2o=-unnamed_1_1.mp4-0.txt")
    image_file = Path("regurgitationV1/train/images/a2hiwqVqZ2o=-unnamed_1_1.mp4-0.png")
    
    if not label_file.exists():
        print(f"ERROR: Label file not found: {label_file}")
        return
    
    if not image_file.exists():
        print(f"ERROR: Image file not found: {image_file}")
        return
    
    print(f"Testing file: {label_file.name}")
    
    # Read the label file manually
    with open(label_file, 'r') as f:
        lines = f.read().strip().splitlines()
    
    print(f"Raw lines: {lines}")
    
    detection_lines = []
    classification_line = None
    
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        print(f"Processing line: '{line}' -> parts: {parts}")
        if len(parts) == 5:
            detection_lines.append(parts)
            print(f"  -> Detection line: {parts}")
        elif len(parts) == 3:
            classification_line = parts
            print(f"  -> Classification line: {parts}")
        else:
            print(f"  -> Unknown format: {parts}")
    
    print(f"\nResults:")
    print(f"  Detection lines: {detection_lines}")
    print(f"  Classification line: {classification_line}")
    
    if classification_line:
        print(f"  Classification line parsed correctly!")
        
        # Test the processing logic
        try:
            cls_values = [float(x) for x in classification_line]
            print(f"  Converted to float: {cls_values}")
            
            if len(cls_values) == 3:
                print(f"  -> One-hot format: {cls_values}")
                # This should be the final result
                final_classification = cls_values
            else:
                print(f"  -> Not 3 values, converting to one-hot")
                class_idx = int(cls_values[0]) if cls_values else 0
                one_hot = [0.0, 0.0, 0.0]
                if 0 <= class_idx < 3:
                    one_hot[class_idx] = 1.0
                final_classification = one_hot
            
            print(f"  Final classification: {final_classification}")
            
        except Exception as e:
            print(f"  ERROR processing classification line: {e}")
            final_classification = [1.0, 0.0, 0.0]
    else:
        print(f"  ERROR: No classification line found!")
        final_classification = [1.0, 0.0, 0.0]
    
    return final_classification

def debug_verify_image_label():
    """Debug the verify_image_label function directly"""
    print("\nDEBUG verify_image_label FUNCTION")
    print("=" * 50)
    
    try:
        from yolov5c.utils.dataloaders import verify_image_label
        
        # Test with a specific file
        label_file = "regurgitationV1/train/labels/a2hiwqVqZ2o=-unnamed_1_1.mp4-0.txt"
        image_file = "regurgitationV1/train/images/a2hiwqVqZ2o=-unnamed_1_1.mp4-0.png"
        prefix = "DEBUG: "
        
        print(f"Calling verify_image_label with:")
        print(f"  Image: {image_file}")
        print(f"  Label: {label_file}")
        
        result = verify_image_label((image_file, label_file, prefix))
        
        print(f"\nResult: {result}")
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")
        
        if len(result) >= 10:
            im_file, lb, shape, segments, nm, nf, ne, nc, msg, classification_line = result
            print(f"\nUnpacked result:")
            print(f"  im_file: {im_file}")
            print(f"  lb shape: {lb.shape if hasattr(lb, 'shape') else 'N/A'}")
            print(f"  shape: {shape}")
            print(f"  segments: {segments}")
            print(f"  nm, nf, ne, nc: {nm}, {nf}, {ne}, {nc}")
            print(f"  msg: {msg}")
            print(f"  classification_line: {classification_line}")
            print(f"  classification_line type: {type(classification_line)}")
            
            return classification_line
        else:
            print(f"ERROR: Expected 10 values, got {len(result)}")
            return None
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_cache_creation():
    """Debug cache creation process"""
    print("\nDEBUG CACHE CREATION")
    print("=" * 50)
    
    try:
        from yolov5c.utils.dataloaders import create_dataloader
        
        print("Creating dataloader to see cache creation...")
        
        # Create dataloader with cache enabled
        train_loader, dataset = create_dataloader(
            'regurgitationV1/train/images',
            imgsz=416,
            batch_size=1,  # Very small batch
            stride=32,
            single_cls=False,
            hyp={'cls_task': 0.3},
            augment=False,
            cache=True,  # Enable cache
            rect=False,
            rank=-1,
            workers=0,
            prefix='DEBUG: ',
            shuffle=False
        )
        
        print("Dataloader created successfully")
        
        # Check the dataset's classification_labels
        if hasattr(dataset, 'classification_labels'):
            print(f"Dataset classification_labels type: {type(dataset.classification_labels)}")
            print(f"Dataset classification_labels length: {len(dataset.classification_labels)}")
            
            # Check first few entries
            for i in range(min(5, len(dataset.classification_labels))):
                cls_label = dataset.classification_labels[i]
                print(f"  Sample {i}: {cls_label}")
        else:
            print("ERROR: Dataset has no classification_labels attribute")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debug function"""
    print("COMPREHENSIVE CLASSIFICATION PARSING DEBUG")
    print("=" * 60)
    
    # Test 1: Manual parsing
    manual_result = debug_single_label_file()
    
    # Test 2: Function parsing
    function_result = debug_verify_image_label()
    
    # Test 3: Cache creation
    debug_cache_creation()
    
    # Summary
    print("\nDEBUG SUMMARY:")
    print("=" * 30)
    print(f"Manual parsing result: {manual_result}")
    print(f"Function parsing result: {function_result}")
    
    if manual_result == function_result:
        print("OK: Manual and function parsing match")
    else:
        print("ERROR: Manual and function parsing differ!")

if __name__ == "__main__":
    main()
