import os
import time
import glob
from datetime import datetime

def monitor_training():
    """Monitor YOLOv5 training progress"""
    yolov5c_dir = "yolov5c"
    runs_dir = os.path.join(yolov5c_dir, "runs", "train")
    
    print("Monitoring YOLOv5 training...")
    print("=" * 50)
    
    while True:
        try:
            # Check for training directories
            if os.path.exists(runs_dir):
                training_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
                training_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(runs_dir, x)), reverse=True)
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Found {len(training_dirs)} training directories")
                
                # Check the most recent training directory
                if training_dirs:
                    latest_dir = training_dirs[0]
                    latest_path = os.path.join(runs_dir, latest_dir)
                    
                    print(f"Latest training directory: {latest_dir}")
                    
                    # Check for weight files
                    weights_dir = os.path.join(latest_path, "weights")
                    if os.path.exists(weights_dir):
                        weight_files = glob.glob(os.path.join(weights_dir, "*.pt"))
                        if weight_files:
                            print(f"Found {len(weight_files)} weight files:")
                            for wf in weight_files:
                                size = os.path.getsize(wf) / (1024*1024)  # MB
                                mtime = datetime.fromtimestamp(os.path.getmtime(wf))
                                print(f"  - {os.path.basename(wf)} ({size:.1f} MB, {mtime.strftime('%H:%M:%S')})")
                    
                    # Check for log files
                    log_files = glob.glob(os.path.join(latest_path, "*.log"))
                    if log_files:
                        print(f"Found {len(log_files)} log files")
                        for lf in log_files:
                            mtime = datetime.fromtimestamp(os.path.getmtime(lf))
                            print(f"  - {os.path.basename(lf)} (last modified: {mtime.strftime('%H:%M:%S')})")
                    
                    # Check for results files
                    results_files = glob.glob(os.path.join(latest_path, "results*.png"))
                    if results_files:
                        print(f"Found {len(results_files)} results files")
                
                else:
                    print("No training directories found")
            
            else:
                print("Runs directory not found")
            
            print("-" * 30)
            time.sleep(10)  # Check every 10 seconds
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error monitoring training: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_training() 