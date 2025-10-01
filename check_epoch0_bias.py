"""
Check if there are epoch 0 or early epoch checkpoints to see initial bias values

This will tell us if bias starts at [0,0,0] or small random values
"""

import torch
from pathlib import Path

def check_all_checkpoints():
    """Check all available checkpoints for bias evolution"""
    print("=" * 60)
    print("CHECKING BIAS EVOLUTION ACROSS CHECKPOINTS")
    print("=" * 60)
    
    # Check classifybackbone13 checkpoints
    weights_dir = Path("yolov5c/runs/classifybackbone13/weights")
    
    if not weights_dir.exists():
        print(f"Weights directory not found: {weights_dir}")
        return
    
    # List all checkpoint files
    checkpoints = list(weights_dir.glob("*.pt"))
    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    for ckpt_path in checkpoints:
        print(f"  - {ckpt_path.name}")
    
    # Check each checkpoint
    for ckpt_path in sorted(checkpoints):
        print(f"\n{'='*50}")
        print(f"Checkpoint: {ckpt_path.name}")
        print(f"{'='*50}")
        
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            
            # Get epoch if available
            epoch = ckpt.get('epoch', 'Unknown')
            print(f"Epoch: {epoch}")
            
            # Find classification head bias
            if 'model' in ckpt:
                model = ckpt['model']
                
                # Check if it's already a model object or state_dict
                if hasattr(model, 'named_modules'):
                    # It's a model object
                    for name, module in model.named_modules():
                        if isinstance(module, torch.nn.Linear) and module.weight.shape[0] == 3:
                            print(f"Classification head: {name}")
                            print(f"  Bias: {module.bias.data}")
                            
                            # Analyze bias pattern
                            bias_vals = module.bias.data
                            print(f"  A4C (0):  {bias_vals[0]:.6f}")
                            print(f"  PSAX (1): {bias_vals[1]:.6f}")
                            print(f"  PLAX (2): {bias_vals[2]:.6f}")
                            
                            if torch.allclose(bias_vals, torch.zeros(3), atol=1e-4):
                                print(f"  ⚠️  Bias is ZERO at epoch {epoch}!")
                            elif abs(bias_vals[1]) > 0.1:
                                print(f"  ⚠️  PSAX bias is large at epoch {epoch}!")
                            break
                elif isinstance(model, dict):
                    # It's a state_dict
                    for key, value in model.items():
                        if 'linear' in key.lower() and 'bias' in key.lower() and value.shape[0] == 3:
                            print(f"Classification bias key: {key}")
                            print(f"  Bias: {value}")
                else:
                    print(f"  Unknown model format: {type(model)}")
            
        except Exception as e:
            print(f"  Error loading checkpoint: {e}")
    
    print("\n" + "=" * 60)
    print("BIAS EVOLUTION ANALYSIS")
    print("=" * 60)
    print("\nIf epoch 0 bias is near zero:")
    print("  → Bias was initialized to zero (code issue)")
    print("\nIf epoch 0 bias is small random:")
    print("  → Bias evolved from random to [-0.263] during training")
    print("  → This is due to class imbalance (use class weights to fix)")

if __name__ == "__main__":
    check_all_checkpoints()

