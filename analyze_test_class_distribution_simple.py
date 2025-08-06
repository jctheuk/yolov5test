import os
import glob
from collections import Counter

def analyze_class_distribution(test_labels_path):
    """
    Analyze class distribution in the test dataset
    """
    # Class names from data.yaml
    detection_classes = ['AR', 'MR', 'PR', 'TR']
    classification_classes = ['PSAX', 'PLAX', 'A4C']
    
    # Counters for detection and classification
    detection_counts = Counter()
    classification_counts = Counter()
    
    # Get all label files
    label_files = glob.glob(os.path.join(test_labels_path, "*.txt"))
    
    print(f"Found {len(label_files)} label files")
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            if len(lines) >= 2:
                # Parse detection annotations (first line)
                detection_line = lines[0].strip()
                if detection_line:
                    parts = detection_line.split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(detection_classes):
                            detection_counts[detection_classes[class_id]] += 1
                
                # Parse classification labels (second line)
                classification_line = lines[1].strip()
                if classification_line:
                    parts = classification_line.split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(classification_classes):
                            classification_counts[classification_classes[class_id]] += 1
                            
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
    
    return detection_counts, classification_counts

def print_distribution(detection_counts, classification_counts):
    """
    Print the class distribution results
    """
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS - TEST DATASET")
    print("="*60)
    
    print("\nDETECTION CLASSES (Valve Regurgitation Types):")
    print("-" * 50)
    total_detections = sum(detection_counts.values())
    for class_name in ['AR', 'MR', 'PR', 'TR']:
        count = detection_counts[class_name]
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"{class_name:>3}: {count:4d} instances ({percentage:5.1f}%)")
    print(f"{'Total':>3}: {total_detections:4d} detections")
    
    print("\nCLASSIFICATION CLASSES (Echocardiogram Views):")
    print("-" * 50)
    total_classifications = sum(classification_counts.values())
    for class_name in ['PSAX', 'PLAX', 'A4C']:
        count = classification_counts[class_name]
        percentage = (count / total_classifications * 100) if total_classifications > 0 else 0
        print(f"{class_name:>4}: {count:4d} instances ({percentage:5.1f}%)")
    print(f"{'Total':>4}: {total_classifications:4d} classifications")
    
    print("\nSUMMARY:")
    print("-" * 50)
    print(f"Total label files processed: {total_detections}")
    print(f"Detection classes present: {len([c for c in detection_counts.values() if c > 0])}/4")
    print(f"Classification classes present: {len([c for c in classification_counts.values() if c > 0])}/3")

if __name__ == "__main__":
    # Path to test labels
    test_labels_path = "Regurgitation-YOLODataset-Detection/test/labels"
    
    # Analyze distribution
    detection_counts, classification_counts = analyze_class_distribution(test_labels_path)
    
    # Print results
    print_distribution(detection_counts, classification_counts) 