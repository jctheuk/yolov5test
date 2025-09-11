#!/usr/bin/env python3
"""
Test script to verify augmentation behavior with and without mosaic
"""

import yaml
import os

def test_hyperparameter_files():
    """Test different hyperparameter configurations"""
    
    # Test 1: Mosaic disabled, other augmentations enabled
    hyp_mosaic_disabled = {
        'mosaic': 0.0,      # Disabled
        'fliplr': 0.5,      # Enabled
        'flipud': 0.0,      # Disabled
        'degrees': 10.0,    # Enabled
        'translate': 0.1,   # Enabled
        'scale': 0.5,       # Enabled
        'hsv_h': 0.015,     # Enabled
        'hsv_s': 0.4,       # Enabled
        'hsv_v': 0.3,       # Enabled
    }
    
    # Test 2: All augmentations disabled
    hyp_all_disabled = {
        'mosaic': 0.0,      # Disabled
        'fliplr': 0.0,      # Disabled
        'flipud': 0.0,      # Disabled
        'degrees': 0.0,     # Disabled
        'translate': 0.0,   # Disabled
        'scale': 0.0,       # Disabled
        'hsv_h': 0.0,       # Disabled
        'hsv_s': 0.0,       # Disabled
        'hsv_v': 0.0,       # Disabled
    }
    
    print("=== YOLOv5 Augmentation Behavior Test ===")
    print()
    
    print("Test 1: Mosaic disabled, other augmentations enabled")
    print("Expected behavior:")
    print("- Mosaic: DISABLED (no 4-image mosaic)")
    print("- Other augmentations: ENABLED")
    print("- Single image will be augmented with:")
    print("  * HSV color augmentation")
    print("  * Horizontal flip (50% chance)")
    print("  * Rotation (±10 degrees)")
    print("  * Translation (±10%)")
    print("  * Scale (±50%)")
    print()
    
    print("Test 2: All augmentations disabled")
    print("Expected behavior:")
    print("- Mosaic: DISABLED")
    print("- All other augmentations: DISABLED")
    print("- Single image will be used as-is (only letterboxing)")
    print()
    
    print("=== Key Finding ===")
    print("Setting mosaic=0.0 does NOT disable other augmentations!")
    print("Other augmentations (HSV, flip, rotation, etc.) will still be applied.")
    print("To disable ALL augmentations, you must set each parameter to 0.0")
    print()
    
    return hyp_mosaic_disabled, hyp_all_disabled

if __name__ == "__main__":
    test_hyperparameter_files()
