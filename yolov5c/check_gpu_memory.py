#!/usr/bin/env python3
"""
GPU Memory Monitor for YOLOv5lc Training
Check available memory before starting each fold
"""

import torch
import os
import subprocess
import time

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return None
        
    gpu_id = 0  # Assuming single GPU
    
    # Get memory info
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
    allocated_memory = torch.cuda.memory_allocated(gpu_id) 
    cached_memory = torch.cuda.memory_reserved(gpu_id)
    free_memory = total_memory - allocated_memory
    
    # Convert to GB
    total_gb = total_memory / (1024**3)
    allocated_gb = allocated_memory / (1024**3) 
    cached_gb = cached_memory / (1024**3)
    free_gb = free_memory / (1024**3)
    
    return {
        'total_gb': total_gb,
        'allocated_gb': allocated_gb,
        'cached_gb': cached_gb,
        'free_gb': free_gb,
        'utilization_pct': (allocated_memory / total_memory) * 100
    }

def estimate_yolov5lc_memory_requirement():
    """Estimate memory requirement for YOLOv5lc Large model"""
    
    # Base estimates for batch_size=128, imgsz=416
    estimates = {
        'model_parameters': 2.5,  # GB - Large model with classification
        'activations': 8.0,       # GB - Forward pass activations
        'gradients': 2.5,         # GB - Backward pass gradients  
        'optimizer_states': 5.0,  # GB - Adam optimizer states
        'data_cache': 2.0,        # GB - Image cache (--cache)
        'misc_overhead': 1.0      # GB - PyTorch overhead
    }
    
    total_estimate = sum(estimates.values())
    
    return estimates, total_estimate

def check_memory_safety(fold_name):
    """Check if there's enough memory for training"""
    
    print(f"\n🔍 Memory Check for {fold_name}")
    print("=" * 50)
    
    # Clear any existing cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        time.sleep(1)  # Allow cleanup
    
    memory_info = get_gpu_memory_info()
    if not memory_info:
        print("❌ CUDA not available")
        return False
    
    estimates, total_required = estimate_yolov5lc_memory_requirement()
    
    print(f"Current GPU Memory Status:")
    print(f"  Total:     {memory_info['total_gb']:.1f} GB")
    print(f"  Used:      {memory_info['allocated_gb']:.1f} GB ({memory_info['utilization_pct']:.1f}%)")
    print(f"  Cached:    {memory_info['cached_gb']:.1f} GB")  
    print(f"  Free:      {memory_info['free_gb']:.1f} GB")
    
    print(f"\nEstimated YOLOv5lc Memory Requirements:")
    for component, gb in estimates.items():
        print(f"  {component:20s}: {gb:.1f} GB")
    print(f"  {'TOTAL REQUIRED':20s}: {total_required:.1f} GB")
    
    safety_margin = 2.0  # GB
    available_for_training = memory_info['free_gb'] - safety_margin
    
    print(f"\nMemory Analysis:")
    print(f"  Available for training: {available_for_training:.1f} GB")
    print(f"  Required for YOLOv5lc:  {total_required:.1f} GB")
    print(f"  Safety margin:          {safety_margin:.1f} GB")
    
    if available_for_training >= total_required:
        print(f"✅ SAFE: Sufficient memory for {fold_name}")
        return True
    else:
        shortage = total_required - available_for_training
        print(f"❌ RISK: Memory shortage of {shortage:.1f} GB")
        print(f"⚠️  {fold_name} may cause OOM!")
        
        # Suggest batch size reduction
        current_batch = 128
        safe_batch = int(current_batch * (available_for_training / total_required) * 0.9)
        print(f"💡 SUGGESTION: Reduce batch size to ~{safe_batch}")
        
        return False

def monitor_training_memory(python_cmd):
    """Monitor memory during training execution"""
    
    print(f"\n🚀 Starting training with memory monitoring...")
    print(f"Command: {' '.join(python_cmd)}")
    
    # Start training process
    process = subprocess.Popen(
        python_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Monitor memory every 30 seconds
    last_check = 0
    
    try:
        for line in iter(process.stdout.readline, ''):
            current_time = time.time()
            
            # Print training output
            print(line.strip())
            
            # Memory check every 30 seconds
            if current_time - last_check > 30:
                memory_info = get_gpu_memory_info()
                if memory_info:
                    print(f"\n[MEMORY] GPU: {memory_info['allocated_gb']:.1f}/{memory_info['total_gb']:.1f} GB "
                          f"({memory_info['utilization_pct']:.1f}%)")
                last_check = current_time
            
            # Check for OOM error
            if "out of memory" in line.lower():
                print(f"\n🚨 OOM DETECTED! Terminating training...")
                process.terminate()
                return False
                
        # Wait for completion
        return_code = process.wait()
        return return_code == 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted")
        process.terminate()
        return False

def main():
    """Test memory checking functionality"""
    
    print("🔧 GPU MEMORY CHECKER FOR YOLOv5lc")
    print("=" * 60)
    
    folds = ["V1", "V2", "V3", "V4", "V5"]
    
    for fold in folds:
        safe = check_memory_safety(fold)
        if not safe:
            print(f"\n⚠️  WARNING: {fold} may fail due to insufficient memory")
            print(f"Consider reducing batch size or using smaller model")
            break
        time.sleep(1)
    
    print(f"\n✅ Memory check complete")

if __name__ == "__main__":
    main()
