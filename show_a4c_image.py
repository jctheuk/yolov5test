import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def show_a4c_image():
    """Display the A4C image with annotations"""
    
    # Image and label paths
    image_path = "Regurgitation-YOLODataset-Detection/train/images/ZmZnwqlqbMKawp0=-unnamed_2_8.mp4-37.png"
    label_path = "Regurgitation-YOLODataset-Detection/train/labels/ZmZnwqlqbMKawp0=-unnamed_2_8.mp4-37.txt"
    
    # Check if files exist
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return
    
    if not Path(label_path).exists():
        print(f"Label file not found: {label_path}")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    
    # Read label file
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Parse detection annotations (first line)
    detection_line = lines[0].strip()
    if detection_line:
        parts = detection_line.split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        bbox_width = float(parts[3])
        bbox_height = float(parts[4])
        
        # Convert normalized coordinates to pixel coordinates
        x_center_px = int(x_center * width)
        y_center_px = int(y_center * height)
        bbox_width_px = int(bbox_width * width)
        bbox_height_px = int(bbox_height * height)
        
        # Calculate bounding box corners
        x1 = x_center_px - bbox_width_px // 2
        y1 = y_center_px - bbox_height_px // 2
        x2 = x_center_px + bbox_width_px // 2
        y2 = y_center_px + bbox_height_px // 2
        
        # Draw bounding box
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add class label
        class_names = ['AR', 'MR', 'PR', 'TR']
        class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
        cv2.putText(image_rgb, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Parse classification (second line)
    classification_line = lines[1].strip()
    if classification_line:
        parts = classification_line.split()
        cls_id = int(parts[0])
        cls_names = ['A4C', 'PLAX', 'PSAX']
        view_name = cls_names[cls_id] if cls_id < len(cls_names) else f'View_{cls_id}'
        
        # Add classification label to image
        cv2.putText(image_rgb, f'View: {view_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display image
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title('A4C Echocardiogram with Annotations')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print information
    print(f"Image: {image_path}")
    print(f"Label: {label_path}")
    print(f"Image size: {width}x{height}")
    if detection_line:
        print(f"Detection: {class_name} at ({x_center:.3f}, {y_center:.3f}) with size ({bbox_width:.3f}, {bbox_height:.3f})")
    if classification_line:
        print(f"Classification: {view_name}")

if __name__ == "__main__":
    show_a4c_image()