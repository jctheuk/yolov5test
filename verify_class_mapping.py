#!/usr/bin/env python3
"""
Verify class mapping is correct by checking actual image content
"""

import cv2
import os
import glob

def verify_class_mapping():
    """Verify if class mapping matches actual image content"""
    
    labels_dir = "Regurgitation-YOLODataset-Detection/valid/labels"
    images_dir = "Regurgitation-YOLODataset-Detection/valid/images"
    
    # Class mapping from data.yaml
    cls_names = ['A4C', 'PSAX', 'PLAX']  # [0, 1, 2]
    
    print("=" * 80)
    print("VERIFYING CLASS MAPPING")
    print("=" * 80)
    print("Expected mapping:")
    print("  0 = A4C (Apical 4-Chamber)")
    print("  1 = PSAX (Parasternal Short Axis)")
    print("  2 = PLAX (Parasternal Long Axis)")
    print()
    
    # Check a few samples from each class
    samples_to_check = {
        'A4C': [],
        'PSAX': [],
        'PLAX': []
    }
    
    # Collect samples
    for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) >= 2:
                classification_line = lines[1].strip()
                parts = classification_line.split()
                
                if len(parts) == 3:
                    a4c, psax, plax = map(int, parts)
                    
                    if a4c == 1:
                        samples_to_check['A4C'].append(label_file)
                    elif psax == 1:
                        samples_to_check['PSAX'].append(label_file)
                    elif plax == 1:
                        samples_to_check['PLAX'].append(label_file)
                        
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
    
    # Show samples for manual verification
    for class_name, files in samples_to_check.items():
        print(f"\n{class_name} samples (first 3):")
        for i, label_file in enumerate(files[:3]):
            label_name = os.path.basename(label_file)
            image_name = label_name.replace('.txt', '.png')
            image_path = os.path.join(images_dir, image_name)
            
            print(f"  {i+1}. {label_name}")
            print(f"     Image: {image_path}")
            
            # Show the image
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    # Resize for display
                    height, width = image.shape[:2]
                    if width > 600:
                        scale = 600 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        image = cv2.resize(image, (new_width, new_height))
                    
                    # Show image with class name
                    cv2.imshow(f'{class_name} Sample {i+1}', image)
                    print(f"     Press any key to continue...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print(f"     Could not load image")
            else:
                print(f"     Image file not found")
    
    print("\n" + "=" * 80)
    print("MANUAL VERIFICATION REQUIRED")
    print("=" * 80)
    print("Please verify that:")
    print("1. A4C images show Apical 4-Chamber view (heart from apex)")
    print("2. PSAX images show Parasternal Short Axis view (heart cross-section)")
    print("3. PLAX images show Parasternal Long Axis view (heart lengthwise)")
    print()
    print("If the mapping is wrong, we need to fix the data.yaml file!")

if __name__ == "__main__":
    verify_class_mapping()
