#!/usr/bin/env python3
"""
Test YOLOv5lc using Original YOLOv5 Large Model Approach
Mimics the proven stability of original YOLOv5l with classification adaptation
"""

import subprocess
import sys
import time
import os

def test_yolov5l_approach(test_epochs=10):
    """
    Test using original YOLOv5 Large model proven approach
    """
    
    print("🔬 TESTING YOLOv5lc with Original YOLOv5l Approach")
    print("=" * 70)
    print("Strategy: Follow original YOLOv5 Large model proven methods")
    print(f"Model: yolov5lc_stable.yaml (reduced from full Large)")
    print(f"Hyperparams: hyp.yolov5l_style.yaml (YOLOv5 proven)")
    print(f"Batch Size: 128 (as requested)")
    print(f"Test Epochs: {test_epochs}")
    print("=" * 70)
    
    # Test configurations based on original YOLOv5 success
    test_configs = [
        {
            "name": "Stable Large P3",
            "model": "models/yolov5lc_stable.yaml",
            "hyp": "data/hyps/hyp.yolov5l_style.yaml",
            "description": "Reduced complexity + YOLOv5 proven hyperparams"
        },
        {
            "name": "Original Large P3", 
            "model": "models/yolov5lc_p3.yaml",
            "hyp": "data/hyps/hyp.yolov5l_style.yaml",
            "description": "Full complexity + YOLOv5 proven hyperparams"
        }
    ]
    
    results = {}
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n{'#'*70}")
        print(f"TEST {i}/{len(test_configs)}: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'#'*70}")
        
        cmd = [
            "python", "train.py",
            "--data", "../regurgitationV4/data.yaml",  # Most stable dataset
            "--cfg", config["model"],
            "--epochs", str(test_epochs),
            "--batch-size", "128",  # Keep as requested
            "--imgsz", "416",
            "--name", f"test_{config['name'].lower().replace(' ', '_')}",
            "--cache",
            "--nosave",
            "--patience", "0",
            "--hyp", config["hyp"],
            "--device", "0"
        ]
        
        print("Command:")
        print(" ".join(cmd))
        print("\n" + "-" * 50)
        
        try:
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                timeout=600,  # 10 minute timeout
                capture_output=True,
                text=True,
                cwd="."
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            success = result.returncode == 0
            results[config['name']] = {
                'success': success,
                'duration': duration,
                'output': result.stdout + result.stderr
            }
            
            print(f"Duration: {duration:.1f}s")
            print(f"Return Code: {result.returncode}")
            
            if success:
                print("✅ SUCCESS: No ConvolutionBackward0 NaN errors!")
            else:
                print("❌ FAILED: Training crashed")
                
                # Check for specific error patterns
                output = result.stdout + result.stderr
                if "ConvolutionBackward0" in output:
                    print("🚨 ConvolutionBackward0 NaN error detected")
                if "cuda out of memory" in output.lower():
                    print("💾 GPU Memory issue")
                if "RuntimeError" in output:
                    print("💥 Runtime error occurred")
                
                # Show error context
                lines = output.split('\n')
                error_lines = []
                for j, line in enumerate(lines):
                    if "error" in line.lower() or "traceback" in line.lower():
                        error_lines.extend(lines[max(0, j-2):j+3])
                
                if error_lines:
                    print("Error context:")
                    for line in error_lines[-10:]:  # Last 10 relevant lines
                        if line.strip():
                            print(f"  {line}")
            
            if not success:
                print(f"❌ {config['name']} FAILED - trying next configuration")
            else:
                print(f"✅ {config['name']} SUCCESS - configuration works!")
                
        except subprocess.TimeoutExpired:
            print("⏰ TIMEOUT: Test exceeded 10 minutes")
            results[config['name']] = {'success': False, 'duration': 600, 'output': 'Timeout'}
            
        except Exception as e:
            print(f"💥 EXCEPTION: {e}")
            results[config['name']] = {'success': False, 'duration': 0, 'output': str(e)}
        
        time.sleep(2)  # Brief pause between tests
    
    # Final analysis
    print(f"\n{'='*70}")
    print("FINAL ANALYSIS - YOLOv5 Large Model Approach")
    print(f"{'='*70}")
    
    successful_configs = []
    for name, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        duration = result['duration']
        print(f"{name:25s}: {status} ({duration:.1f}s)")
        
        if result['success']:
            successful_configs.append(name)
    
    print(f"\nResults: {len(successful_configs)}/{len(results)} configurations successful")
    
    if successful_configs:
        print(f"\n🎉 WORKING CONFIGURATIONS:")
        for config in successful_configs:
            print(f"  ✅ {config}")
        print(f"\n💡 RECOMMENDATION:")
        print(f"Use the successful configuration for full YOLOv5lc training")
        print(f"The Original YOLOv5 Large approach SOLVED the NaN problem!")
    else:
        print(f"\n🚨 ALL CONFIGURATIONS FAILED")
        print(f"Need to investigate further or consider alternative approaches")
    
    # Cleanup test files
    for config in test_configs:
        test_name = f"test_{config['name'].lower().replace(' ', '_')}"
        test_path = f"runs/train/{test_name}"
        if os.path.exists(test_path):
            import shutil
            shutil.rmtree(test_path, ignore_errors=True)
    
    print(f"\n🧹 Cleaned up test files")
    return successful_configs

def main():
    """Run the YOLOv5 Large approach test"""
    
    if not os.path.exists("models"):
        print("❌ Error: Run from yolov5c/ directory")
        sys.exit(1)
    
    print("🚀 YOLOv5 LARGE MODEL APPROACH TEST")
    print("Following Original YOLOv5 Large Model Proven Methods")
    print("=" * 70)
    
    successful_configs = test_yolov5l_approach(test_epochs=5)
    
    if successful_configs:
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Use successful configuration for full K-fold training")
        print(f"2. Update training scripts to use working approach")
        print(f"3. Original YOLOv5 methods work with classification extension!")
    else:
        print(f"\n🔧 TROUBLESHOOTING NEEDED:")
        print(f"Even original YOLOv5 Large approach has issues")
        print(f"Consider hardware/environment factors")

if __name__ == "__main__":
    main()
