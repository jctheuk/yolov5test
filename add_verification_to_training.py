#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of how to add cross-entropy verification to train_classification_task.py
"""

# Add this function to your train_classification_task.py

def verify_crossentropy_during_training(model, images, classification_labels, compute_loss, epoch, batch_idx):
    """
    Verify cross-entropy calculations during training
    
    Add this function to train_classification_task.py and call it occasionally
    """
    from extract_model_outputs import manual_cross_entropy_calculation, parse_model_output
    
    # Only verify occasionally to avoid slowing down training
    if batch_idx % 50 == 0:  # Every 50 batches
        with torch.no_grad():
            # Get model output
            model_output = model(images)
            
            # Parse model output
            detection_outputs, classification_output = parse_model_output(model_output)
            
            if classification_output is not None and classification_labels is not None:
                # Manual cross-entropy calculation
                manual_loss = manual_cross_entropy_calculation(classification_output, classification_labels)
                
                # PyTorch cross-entropy
                pytorch_loss = torch.nn.functional.cross_entropy(classification_output, classification_labels)
                
                # Calculate difference
                diff = abs(manual_loss.item() - pytorch_loss.item())
                
                # Log verification
                if diff < 1e-6:
                    LOGGER.info(f"✅ Cross-entropy verification PASSED (epoch {epoch}, batch {batch_idx})")
                else:
                    LOGGER.warning(f"⚠️ Cross-entropy verification FAILED (epoch {epoch}, batch {batch_idx}) - diff: {diff:.8f}")
                
                # Calculate accuracy
                pred_classes = torch.argmax(classification_output, dim=1)
                correct = (pred_classes == classification_labels).sum().item()
                accuracy = correct / classification_labels.shape[0]
                
                LOGGER.info(f"   Manual CE: {manual_loss.item():.6f}, PyTorch CE: {pytorch_loss.item():.6f}, Acc: {accuracy:.4f}")

# Add this to your training loop in train_classification_task.py
# Insert after line 1107 (after total_loss, loss_items = compute_loss(...))

def training_loop_with_verification():
    """
    Example of how to modify your training loop to include verification
    """
    
    # In your training loop, after computing loss:
    # total_loss, loss_items = compute_loss(model_output, targets, classification_labels, 
    #                                      image_paths=image_paths, class_names=class_names)
    
    # Add this verification call:
    # verify_crossentropy_during_training(model, imgs, classification_labels, compute_loss, epoch, i)
    
    pass

# Alternative: Add verification to your loss function
def enhanced_classification_task_loss():
    """
    Example of how to enhance ClassificationTaskLoss with verification
    """
    
    # In yolov5c/utils/classification_task_loss.py, modify the __call__ method
    # to include verification:
    
    """
    def __call__(self, p, targets, cls_targets=None, image_paths=None, class_names=None):
        # ... existing code ...
        
        # Calculate classification loss
        if classification_output is not None and cls_targets is not None:
            # ... existing loss calculation ...
            
            # Add verification (only occasionally)
            if hasattr(self, '_verify_count'):
                self._verify_count += 1
            else:
                self._verify_count = 0
            
            if self._verify_count % 100 == 0:  # Every 100 calls
                with torch.no_grad():
                    # Manual cross-entropy calculation
                    manual_loss = self.manual_cross_entropy_loss(classification_output, target_indices)
                    
                    # Compare with our calculation
                    diff = abs(lcls_task.item() - manual_loss.item())
                    
                    if diff > 1e-6:
                        print(f"[WARNING] Cross-entropy verification failed! Diff: {diff:.8f}")
                        print(f"  Our calculation: {lcls_task.item():.6f}")
                        print(f"  Manual calculation: {manual_loss.item():.6f}")
        
        # ... rest of existing code ...
    """

if __name__ == "__main__":
    print("📖 This file shows how to add cross-entropy verification to your training script.")
    print("\n🔧 Options:")
    print("1. Add verify_crossentropy_during_training() to your training loop")
    print("2. Modify ClassificationTaskLoss to include verification")
    print("3. Use the standalone verification script: verify_training_loss.py")
    print("\n💡 Recommended: Use verify_training_loss.py first to test your current setup")

