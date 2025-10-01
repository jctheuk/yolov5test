#!/usr/bin/env python3
"""
Simple fix for train_classification_task.py to handle trainable models
"""

def fix_train_classification_task():
    """Fix the training script to properly handle trainable models"""
    print("FIXING train_classification_task.py FOR TRAINABLE MODEL")
    print("=" * 60)
    
    # Read the current file
    with open('train_classification_task.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Add special handling for trainable models
    trainable_model_fix = '''        # Special handling for trainable models (our fixed models)
        elif 'trainable' in weights.lower() or 'fixed' in weights.lower():
            LOGGER.info(f'Detected trainable model: {weights}')
            LOGGER.info(f'Loading trainable model with all parameters unfrozen...')
            
            # Load the model state dict directly
            model_state = ckpt['model'].float().state_dict()
            model.load_state_dict(model_state, strict=False)
            
            # Ensure all parameters are trainable
            for name, param in model.named_parameters():
                param.requires_grad = True
            
            LOGGER.info(f'Loaded trainable model with {len(model_state)} parameters')
            LOGGER.info(f'All parameters set to trainable (requires_grad=True)')
        
        # Special handling for yolov5s-cls.pt weights'''
    
    # Find the location to insert the fix
    import re
    pattern = r'(        # Special handling for yolov5s-cls\.pt weights)'
    replacement = trainable_model_fix + '\n' + r'\1'
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print("SUCCESS: Added special handling for trainable models")
    else:
        print("ERROR: Could not find insertion point for trainable model handling")
        return False
    
    # Fix 2: Add verification that parameters are trainable
    verification_fix = '''
    # Verify all parameters are trainable
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    total_count = sum(1 for p in model.parameters())
    LOGGER.info(f'Parameter status: {trainable_count}/{total_count} trainable')
    
    if trainable_count == 0:
        LOGGER.error('ERROR: No trainable parameters found! Model will not learn.')
        raise RuntimeError('Model has no trainable parameters')
    elif trainable_count < total_count:
        LOGGER.warning(f'WARNING: Only {trainable_count}/{total_count} parameters are trainable')
    else:
        LOGGER.info(f'SUCCESS: All {total_count} parameters are trainable')
'''
    
    # Find the location after model loading and before optimizer creation
    pattern = r'(amp = check_amp\(model\)  # check AMP)'
    replacement = r'\1' + verification_fix
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print("SUCCESS: Added parameter trainability verification")
    else:
        print("ERROR: Could not find insertion point for parameter verification")
        return False
    
    # Write the fixed file
    with open('train_classification_task.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("SUCCESS: Successfully fixed train_classification_task.py")
    return True

def create_test_command():
    """Create a test command to verify the fix works"""
    print("\nCREATING TEST COMMAND")
    print("=" * 30)
    
    test_command = """# Test command to verify the fix works
python train_classification_task.py \\
    --data regurgitationV1/data.yaml \\
    --epochs 1 \\
    --batch-size 4 \\
    --device auto \\
    --weights yolov5c/runs/classifybackbone13/weights/last_trainable.pt \\
    --hyp yolov5c/runs/classifybackbone13/hyp.yaml \\
    --name test_trainable_model
"""
    
    with open("test_trainable_fix.txt", "w") as f:
        f.write(test_command.strip())
    
    print("Test command saved to: test_trainable_fix.txt")
    print("\nTo test the fix, run:")
    print("bash test_trainable_fix.txt")
    
    return True

def main():
    """Main fix function"""
    print("FIXING TRAIN_CLASSIFICATION_TASK.PY")
    print("=" * 60)
    
    # Fix the training script
    success = fix_train_classification_task()
    
    if success:
        # Create test command
        create_test_command()
        
        print("\nFINAL SUMMARY:")
        print("=" * 40)
        print("SUCCESS: train_classification_task.py has been fixed!")
        print("   - Added special handling for trainable models")
        print("   - Added parameter trainability verification")
        print("   - Test command created")
        print("\nThe training script will now:")
        print("   1. Detect trainable models automatically")
        print("   2. Verify all parameters are trainable")
        print("   3. Provide clear error messages if issues are found")
        print("\nYou can now use the trainable model safely!")
    else:
        print("ERROR: Failed to fix train_classification_task.py")

if __name__ == "__main__":
    main()
