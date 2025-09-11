#!/usr/bin/env python3
"""
Show a PSAX image from the validation set
"""

import cv2
import os
import glob

def show_psax_image():
    # Find PSAX images in validation set
    labels_dir = "Regurgitation-YOLODataset-Detection/valid/labels"
    images_dir = "Regurgitation-YOLODataset-Detection/valid/images"
    
    psax_files = []
    
    # Find PSAX label files
    for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) >= 2:
                classification_line = lines[1].strip()
                parts = classification_line.split()
                
                if len(parts) == 3:
                    a4c, psax, plax = map(int, parts)
                    
                    if psax == 1:  # This is a PSAX image
                        # Get corresponding image file
                        label_name = os.path.basename(label_file)
                        image_name = label_name.replace('.txt', '.png')
                        image_path = os.path.join(images_dir, image_name)
                        
                        if os.path.exists(image_path):
                            psax_files.append((label_name, image_path))
                            
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
    
    if not psax_files:
        print("No PSAX images found!")
        return
    
    # Show the first PSAX image
    label_name, image_path = psax_files[0]
    print(f"Showing PSAX image: {label_name}")
    print(f"Image path: {image_path}")
    
    # Read and display the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Resize image for display (make it smaller)
    height, width = image.shape[:2]
    if width > 800:
        scale = 800 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    # Display the image
    cv2.imshow('PSAX Image', image)
    print("Press any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Also show the label content
    label_path = os.path.join(labels_dir, label_name)
    with open(label_path, 'r') as f:
        content = f.read()
    
    print(f"\nLabel content for {label_name}:")
    print(content)
    
    # Show image info
    print(f"\nImage info:")
    print(f"  Original size: {width}x{height}")
    print(f"  Display size: {image.shape[1]}x{image.shape[0]}")
    print(f"  File size: {os.path.getsize(image_path)} bytes")

if __name__ == "__main__":
    show_psax_image()