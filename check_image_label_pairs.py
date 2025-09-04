import os
import glob

def check_image_label_pairs():
    """Check if each label file has a corresponding image file"""
    
    # Define paths
    base_path = "Regurgitation-YOLODataset-Detection"
    datasets = ["train", "valid", "test"]
    
    total_labels = 0
    total_images = 0
    missing_images = 0
    missing_labels = 0
    
    for dataset in datasets:
        print(f"\n=== Checking {dataset} dataset ===")
        
        # Get all label files
        label_path = os.path.join(base_path, dataset, "labels", "*.txt")
        label_files = glob.glob(label_path)
        
        # Get all image files
        image_path = os.path.join(base_path, dataset, "images", "*.png")
        image_files = glob.glob(image_path)
        
        print(f"Found {len(label_files)} label files")
        print(f"Found {len(image_files)} image files")
        
        # Check for missing images
        missing_count = 0
        for label_file in label_files:
            # Get the base name (without extension)
            base_name = os.path.splitext(os.path.basename(label_file))[0]
            expected_image = os.path.join(base_path, dataset, "images", f"{base_name}.png")
            
            if not os.path.exists(expected_image):
                missing_count += 1
                if missing_count <= 5:  # Show first 5 missing images
                    print(f"  Missing image: {base_name}.png")
        
        if missing_count > 5:
            print(f"  ... and {missing_count - 5} more missing images")
        
        # Check for missing labels
        missing_label_count = 0
        for image_file in image_files:
            # Get the base name (without extension)
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            expected_label = os.path.join(base_path, dataset, "labels", f"{base_name}.txt")
            
            if not os.path.exists(expected_label):
                missing_label_count += 1
                if missing_label_count <= 5:  # Show first 5 missing labels
                    print(f"  Missing label: {base_name}.txt")
        
        if missing_label_count > 5:
            print(f"  ... and {missing_label_count - 5} more missing labels")
        
        # Update totals
        total_labels += len(label_files)
        total_images += len(image_files)
        missing_images += missing_count
        missing_labels += missing_label_count
        
        # Summary for this dataset
        if missing_count == 0 and missing_label_count == 0:
            print(f"✓ {dataset}: All files are paired correctly")
        else:
            print(f"✗ {dataset}: {missing_count} missing images, {missing_label_count} missing labels")
    
    # Overall summary
    print(f"\n=== OVERALL SUMMARY ===")
    print(f"Total label files: {total_labels}")
    print(f"Total image files: {total_images}")
    print(f"Missing images: {missing_images}")
    print(f"Missing labels: {missing_labels}")
    
    if missing_images == 0 and missing_labels == 0:
        print("✓ All datasets are complete - every label has an image and every image has a label")
    else:
        print("✗ Some files are missing - dataset may be incomplete")

if __name__ == "__main__":
    check_image_label_pairs()
