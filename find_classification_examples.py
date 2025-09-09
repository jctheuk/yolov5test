import os
from pathlib import Path

def find_classification_examples():
    """Find one example of each classification class (A4C, PLAX, PSAX)"""
    
    dataset_path = "Regurgitation-YOLODataset-Detection"
    sets = ["train", "valid", "test"]
    
    # Classification classes: 0=A4C, 1=PLAX, 2=PSAX
    class_names = ['A4C', 'PLAX', 'PSAX']
    found_examples = {0: None, 1: None, 2: None}  # class_id: (image_path, label_path)
    
    print("Searching for one example of each classification class...")
    print("=" * 60)
    
    for set_name in sets:
        if all(found_examples.values()):
            break  # All classes found
            
        labels_dir = Path(dataset_path) / set_name / "labels"
        images_dir = Path(dataset_path) / set_name / "images"
        
        if not labels_dir.exists() or not images_dir.exists():
            continue
        
        print(f"\nSearching in {set_name.upper()} set...")
        
        # Check each label file
        for label_file in labels_dir.glob("*.txt"):
            if all(found_examples.values()):
                break
                
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                # Check if file has classification line (second line)
                if len(lines) >= 2:
                    classification_line = lines[1].strip()
                    if classification_line:
                        parts = classification_line.split()
                        if len(parts) >= 1:
                            class_id = int(parts[0])
                            
                            # If we haven't found this class yet
                            if class_id in found_examples and found_examples[class_id] is None:
                                image_file = images_dir / (label_file.stem + ".png")
                                if image_file.exists():
                                    found_examples[class_id] = (str(image_file), str(label_file))
                                    print(f"Found {class_names[class_id]} (class {class_id}): {label_file.name}")
                                    
            except Exception as e:
                continue
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION EXAMPLES FOUND:")
    print("=" * 60)
    
    for class_id, example in found_examples.items():
        if example is not None:
            image_path, label_path = example
            print(f"\n{class_names[class_id]} (Class {class_id}):")
            print(f"  Image: {image_path}")
            print(f"  Label: {label_path}")
            
            # Read and display label content
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                print(f"  Label content:")
                for i, line in enumerate(lines):
                    if line.strip():
                        if i == 0:
                            print(f"    Detection: {line.strip()}")
                        elif i == 1:
                            print(f"    Classification: {line.strip()}")
            except Exception as e:
                print(f"  Error reading label: {e}")
        else:
            print(f"\n{class_names[class_id]} (Class {class_id}): NOT FOUND")
    
    # Summary
    found_count = sum(1 for v in found_examples.values() if v is not None)
    print(f"\n" + "=" * 60)
    print(f"SUMMARY: Found {found_count}/3 classification examples")
    
    if found_count == 3:
        print("✅ All classification classes found!")
    else:
        print("⚠ Some classification classes not found")
    
    return found_examples

if __name__ == "__main__":
    find_classification_examples()

