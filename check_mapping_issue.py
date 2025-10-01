"""
Check if there's a class mapping issue between training and validation

The PSAX class has 21.9% representation but only 9% recall.
This suggests a potential mapping bug between class indices and class names.
"""
import yaml
from pathlib import Path

def check_class_mapping():
    print("=" * 60)
    print("CHECKING CLASS MAPPING")
    print("=" * 60)
    
    # Load data.yaml
    with open('regurgitationV1/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("\nFrom data.yaml:")
    print(f"  cls_names: {data_config.get('cls_names', 'NOT FOUND')}")
    print(f"  num_cls: {data_config.get('num_cls', 'NOT FOUND')}")
    
    # Expected mapping
    expected_mapping = {
        0: 'A4C',
        1: 'PSAX',
        2: 'PLAX'
    }
    
    print("\nExpected class index mapping:")
    for idx, name in expected_mapping.items():
        print(f"  {idx} -> {name}")
    
    # Check if data.yaml matches
    cls_names = data_config.get('cls_names', [])
    print("\nActual mapping from data.yaml:")
    for idx, name in enumerate(cls_names):
        print(f"  {idx} -> {name}")
        if expected_mapping[idx] != name:
            print(f"    ERROR: Mismatch! Expected {expected_mapping[idx]}, got {name}")
    
    # Check some label files to verify the mapping
    print("\n" + "=" * 60)
    print("VERIFYING LABEL FILES")
    print("=" * 60)
    
    label_dir = Path("regurgitationV1/train/labels")
    
    # Find files with PSAX labels (one-hot: [0, 1, 0])
    psax_files = []
    for label_file in list(label_dir.glob("*.txt"))[:200]:
        with open(label_file, 'r') as f:
            lines = f.read().strip().split('\n')
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    one_hot = [float(x) for x in parts]
                    if one_hot == [0.0, 1.0, 0.0]:  # PSAX
                        psax_files.append(label_file.name)
                        break
                except:
                    pass
    
    print(f"\nFound {len(psax_files)} files with PSAX labels in first 200 files")
    if psax_files:
        print(f"\nSample PSAX files:")
        for f in psax_files[:5]:
            print(f"  - {f}")
    
    # Check if filename contains any pattern
    print("\nChecking filename patterns for PSAX...")
    for f in psax_files[:10]:
        if 'psax' in f.lower():
            print(f"  {f} - Contains 'psax' in filename")
        elif 'a4c' in f.lower():
            print(f"  {f} - WARNING: Contains 'a4c' but labeled as PSAX!")
        elif 'plax' in f.lower():
            print(f"  {f} - WARNING: Contains 'plax' but labeled as PSAX!")

def check_train_vs_validation_mapping():
    """Check if training and validation use same class mapping"""
    print("\n" + "=" * 60)
    print("CHECKING TRAIN VS VALIDATION MAPPING")
    print("=" * 60)
    
    # The issue might be that training and validation use different class order
    # Let's check if there's any code that reorders classes
    
    print("\nPotential issues to check:")
    print("  1. Does val.py use different class order than train.py?")
    print("  2. Is there a hardcoded class order somewhere?")
    print("  3. Does the model output match the data.yaml order?")
    
    # Check if there's a mismatch in how classes are interpreted
    print("\nFrom your validation output:")
    print("  Class 0 (A4C):  44.1% recall")
    print("  Class 1 (PSAX):  9.1% recall [TERRIBLE!]")
    print("  Class 2 (PLAX): 51.7% recall")
    print("\nThis suggests class 1 is being:")
    print("  - Either mispredicted systematically")
    print("  - Or there's a label/prediction mapping bug")

if __name__ == "__main__":
    check_class_mapping()
    check_train_vs_validation_mapping()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("\nWith 21.9% PSAX in training data but only 9% recall,")
    print("there MUST be a code bug affecting PSAX specifically.")
    print("\nPossible bugs:")
    print("  1. Class mapping mismatch between train and validation")
    print("  2. Loss function bias against class 1")
    print("  3. Model initialization bias")
    print("  4. Gradient clipping affecting class 1 differently")
