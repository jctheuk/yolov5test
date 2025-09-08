import os
from pathlib import Path

def check_image_label_pairs():
    """Check if each label file has a corresponding image file"""
    
    dataset_path = "Regurgitation-YOLODataset-Detection"
    sets = ["train", "valid", "test"]
    
    total_labels = 0
    total_images = 0
    missing_images = []
    missing_labels = []
    
    print("Checking image-label pairs in Regurgitation-YOLODataset-Detection...")
    print("=" * 60)
    
    for set_name in sets:
        labels_dir = Path(dataset_path) / set_name / "labels"
        images_dir = Path(dataset_path) / set_name / "images"
        
        if not labels_dir.exists():
            print(f"Labels directory not found: {labels_dir}")
            continue
            
        if not images_dir.exists():
            print(f"Images directory not found: {images_dir}")
            continue
        
        print(f"\n{set_name.upper()} SET:")
        print("-" * 30)
        
        # Get all label files
        label_files = list(labels_dir.glob("*.txt"))
        image_files = list(images_dir.glob("*.png"))
        
        set_labels = len(label_files)
        set_images = len(image_files)
        
        print(f"Label files: {set_labels}")
        print(f"Image files: {set_images}")
        
        # Check for missing images
        missing_in_set = []
        for label_file in label_files:
            image_file = images_dir / (label_file.stem + ".png")
            if not image_file.exists():
                missing_in_set.append(label_file.name)
                missing_images.append(f"{set_name}/{label_file.name}")
        
        # Check for missing labels
        missing_labels_in_set = []
        for image_file in image_files:
            label_file = labels_dir / (image_file.stem + ".txt")
            if not label_file.exists():
                missing_labels_in_set.append(image_file.name)
                missing_labels.append(f"{set_name}/{image_file.name}")
        
        if missing_in_set:
            print(f"Missing images for {len(missing_in_set)} labels:")
            for missing in missing_in_set[:5]:  # Show first 5
                print(f"  - {missing}")
            if len(missing_in_set) > 5:
                print(f"  ... and {len(missing_in_set) - 5} more")
        
        if missing_labels_in_set:
            print(f"Missing labels for {len(missing_labels_in_set)} images:")
            for missing in missing_labels_in_set[:5]:  # Show first 5
                print(f"  - {missing}")
            if len(missing_labels_in_set) > 5:
                print(f"  ... and {len(missing_labels_in_set) - 5} more")
        
        if not missing_in_set and not missing_labels_in_set:
            print("✓ All label files have corresponding image files")
        
        total_labels += set_labels
        total_images += set_images
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Total label files: {total_labels}")
    print(f"Total image files: {total_images}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Missing labels: {len(missing_labels)}")
    
    if missing_images:
        print(f"\nFirst 10 missing images:")
        for missing in missing_images[:10]:
            print(f"  - {missing}")
    
    if missing_labels:
        print(f"\nFirst 10 missing labels:")
        for missing in missing_labels[:10]:
            print(f"  - {missing}")
    
    if not missing_images and not missing_labels:
        print("\n✓ Perfect! All label files have corresponding image files")
    else:
        print(f"\n⚠ Found {len(missing_images)} missing images and {len(missing_labels)} missing labels")

if __name__ == "__main__":
    check_image_label_pairs()