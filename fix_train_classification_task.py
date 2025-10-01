#!/usr/bin/env python3
"""
Fix train_classification_task.py to properly handle the trainable model
"""

import re

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
            LOGGER.info(f'🔄 Detected trainable model: {weights}')
            LOGGER.info(f'✅ Loading trainable model with all parameters unfrozen...')
            
            # Load the model state dict directly
            model_state = ckpt['model'].float().state_dict()
            model.load_state_dict(model_state, strict=False)
            
            # Ensure all parameters are trainable
            for name, param in model.named_parameters():
                param.requires_grad = True
            
            LOGGER.info(f'✅ Loaded trainable model with {len(model_state)} parameters')
            LOGGER.info(f'✅ All parameters set to trainable (requires_grad=True)')
        
        # Special handling for yolov5s-cls.pt weights'''
    
    # Find the location to insert the fix
    pattern = r'(        # Special handling for yolov5s-cls\.pt weights)'
    replacement = trainable_model_fix + '\n' + r'\1'
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print("✅ Added special handling for trainable models")
    else:
        print("❌ Could not find insertion point for trainable model handling")
        return False
    
    # Fix 2: Ensure the parameter unfreezing happens after model loading
    # Check if the unfreezing code is already there
    unfreeze_pattern = r'for k, v in model\.named_parameters\(\):\s*\n\s*v\.requires_grad = True'
    if re.search(unfreeze_pattern, content):
        print("✅ Parameter unfreezing code is already present")
    else:
        print("❌ Parameter unfreezing code not found")
        return False
    
    # Fix 3: Add verification that parameters are trainable
    verification_fix = '''
    # Verify all parameters are trainable
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    total_count = sum(1 for p in model.parameters())
    LOGGER.info(f'Parameter status: {trainable_count}/{total_count} trainable')
    
    if trainable_count == 0:
        LOGGER.error('❌ ERROR: No trainable parameters found! Model will not learn.')
        raise RuntimeError('Model has no trainable parameters')
    elif trainable_count < total_count:
        LOGGER.warning(f'⚠️  WARNING: Only {trainable_count}/{total_count} parameters are trainable')
    else:
        LOGGER.info(f'✅ All {total_count} parameters are trainable')
'''
    
    # Find the location after model loading and before optimizer creation
    pattern = r'(amp = check_amp\(model\)  # check AMP)'
    replacement = r'\1' + verification_fix
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print("✅ Added parameter trainability verification")
    else:
        print("❌ Could not find insertion point for parameter verification")
        return False
    
    # Fix 4: Add gradient flow verification
    gradient_verification = '''
    # Test gradient flow with a dummy forward pass
    try:
        model.train()
        dummy_input = torch.randn(1, 3, imgsz, imgsz, device=device)
        dummy_target = torch.tensor([0], device=device)
        
        # Forward pass
        dummy_output = model(dummy_input)
        if isinstance(dummy_output, tuple):
            dummy_output = dummy_output[1]  # Use classification output
        
        # Compute loss
        dummy_loss = torch.nn.CrossEntropyLoss()(dummy_output, dummy_target)
        
        # Backward pass
        dummy_loss.backward()
        
        # Check gradients
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        LOGGER.info(f'Gradient flow test: {grad_count}/{total_count} parameters have gradients')
        
        if grad_count == 0:
            LOGGER.error('❌ ERROR: No gradients detected! Model cannot learn.')
            raise RuntimeError('No gradients detected in model')
        else:
            LOGGER.info(f'✅ Gradient flow verified: {grad_count} parameters have gradients')
        
        # Clear gradients
        model.zero_grad()
        
    except Exception as e:
        LOGGER.error(f'❌ Gradient flow test failed: {e}')
        raise RuntimeError(f'Gradient flow test failed: {e}')
'''
    
    # Find location after parameter verification
    pattern = r'(LOGGER\.info\(f\'✅ All \{total_count\} parameters are trainable\'\))'
    replacement = r'\1' + gradient_verification
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print("✅ Added gradient flow verification")
    else:
        print("❌ Could not find insertion point for gradient verification")
        return False
    
    # Write the fixed file
    with open('train_classification_task.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Successfully fixed train_classification_task.py")
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
        print("   - Added gradient flow verification")
        print("   - Test command created")
        print("\nThe training script will now:")
        print("   1. Detect trainable models automatically")
        print("   2. Verify all parameters are trainable")
        print("   3. Test gradient flow before training")
        print("   4. Provide clear error messages if issues are found")
        print("\nYou can now use the trainable model safely!")
    else:
        print("ERROR: Failed to fix train_classification_task.py")

if __name__ == "__main__":
    main()
