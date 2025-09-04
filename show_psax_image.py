import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the PSAX image
image_path = 'Regurgitation-YOLODataset-Detection/train/images/ZmZnwqlqbMKawp0=-unnamed_1_5.mp4-51.png'
label_path = 'Regurgitation-YOLODataset-Detection/train/labels/ZmZnwqlqbMKawp0=-unnamed_1_5.mp4-51.txt'

# Read image
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Read label
with open(label_path, 'r') as f:
    lines = f.readlines()

# Parse detection label (first line)
detection_line = lines[0].strip().split()
class_id = int(detection_line[0])
x_center = float(detection_line[1])
y_center = float(detection_line[2])
width = float(detection_line[3])
height = float(detection_line[4])

# Parse classification label (second line)
classification_line = lines[1].strip().split()
cls_label = [int(x) for x in classification_line]

# Convert normalized coordinates to pixel coordinates
img_height, img_width = img.shape[:2]
x1 = int((x_center - width/2) * img_width)
y1 = int((y_center - height/2) * img_height)
x2 = int((x_center + width/2) * img_width)
y2 = int((y_center + height/2) * img_height)

# Define class names
detection_classes = ['AR', 'MR', 'PR', 'TR']
classification_classes = ['A4C', 'PLAX', 'PSAX']

# Get class names
det_class = detection_classes[class_id] if class_id < len(detection_classes) else f'Class_{class_id}'
cls_class = classification_classes[cls_label.index(1)] if 1 in cls_label else 'Unknown'

print(f"Detection: {det_class} (class {class_id})")
print(f"Classification: {cls_class}")
print(f"Bounding box: ({x1}, {y1}) to ({x2}, {y2})")

# Draw bounding box
cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Add text labels
cv2.putText(img_rgb, f'Detection: {det_class}', (x1, y1-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv2.putText(img_rgb, f'View: {cls_class}', (x1, y2+25), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Display image
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.title(f'PSAX (Parasternal Short-Axis) Ultrasound Image\nDetection: {det_class}, View: {cls_class}')
plt.axis('off')

# Save the image
plt.savefig('psax_annotated.png', dpi=150, bbox_inches='tight')
print("Annotated image saved as 'psax_annotated.png'")

# Show the image
plt.show()
