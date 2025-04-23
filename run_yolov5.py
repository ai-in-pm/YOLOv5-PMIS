import os
import sys
import argparse
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 Enhanced Launcher")
    parser.add_argument('--mode', type=str, default='detect', choices=['detect', 'enhanced', 'analyze'],
                        help='Operation mode: standard detection, enhanced detection, or database analysis')
    parser.add_argument('--source', type=str, default='',
                        help='Path to source image, video, or directory')
    parser.add_argument('--weights', type=str, default='yolov5s.pt',
                        help='Model weights to use')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results as text files')
    parser.add_argument('--save-db', action='store_true',
                        help='Save results to database (only for enhanced mode)')
    parser.add_argument('--view', action='store_true',
                        help='View results in real-time')
    parser.add_argument('--device', type=str, default='',
                        help='Device to use (cuda device or cpu)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get paths
    current_dir = Path(__file__).parent.absolute()
    yolov5_dir = Path(os.path.join(current_dir.parent, 'yolov5-master'))
    custom_scripts_dir = os.path.join(current_dir, 'custom_scripts')
    
    # Prepare command based on mode
    if args.mode == 'detect':
        # Use standard YOLOv5 detect.py
        script_path = os.path.join(yolov5_dir, 'detect.py')
        cmd = [
            sys.executable, script_path,
            '--weights', args.weights,
        ]
        
        if args.source:
            cmd.extend(['--source', args.source])
        if args.conf != 0.25:
            cmd.extend(['--conf-thres', str(args.conf)])
        if args.save_txt:
            cmd.append('--save-txt')
        if args.view:
            cmd.append('--view-img')
        if args.device:
            cmd.extend(['--device', args.device])
            
    elif args.mode == 'enhanced':
        # Use our enhanced detection script
        script_path = os.path.join(custom_scripts_dir, 'enhanced_detect.py')
        cmd = [
            sys.executable, script_path,
            '--weights', args.weights,
        ]
        
        if args.source:
            cmd.extend(['--source', args.source])
        if args.conf != 0.25:
            cmd.extend(['--conf-thres', str(args.conf)])
        if args.save_txt:
            cmd.append('--save-txt')
        if args.view:
            cmd.append('--view-img')
        if args.device:
            cmd.extend(['--device', args.device])
        if args.save_db:
            cmd.append('--save-to-db')
            
    elif args.mode == 'analyze':
        # Use the enhanced script with analyze flag
        script_path = os.path.join(custom_scripts_dir, 'enhanced_detect.py')
        cmd = [
            sys.executable, script_path,
            '--weights', args.weights,
            '--analyze',
        ]
        
        if args.source:
            cmd.extend(['--source', args.source])
        else:
            # Just need a minimal source for analysis mode
            cmd.extend(['--source', os.path.join(yolov5_dir, 'data', 'images', 'bus.jpg')])
        
        if args.device:
            cmd.extend(['--device', args.device])
        
        # Always save to database in analyze mode
        cmd.append('--save-to-db')
        
    # Print the command
    print(f"\nRunning command: {' '.join(cmd)}\n")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print("\nDetection completed successfully!")
            
            if args.mode == 'detect' or args.mode == 'enhanced':
                print("\nResults saved to the 'results' directory.")
            
            if args.mode == 'enhanced' and args.save_db:
                print("Detection results saved to database.")
                
    except subprocess.CalledProcessError as e:
        print(f"\nError running detection: {e}")
    except KeyboardInterrupt:
        print("\nDetection interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main()
