import os
import sys
import argparse
import yaml
import torch
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Add root path to use YOLOv5
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_path)

class MSProjectModelTuner:
    """Performance tuner for YOLOv5 models on Microsoft Project data"""
    
    def __init__(self, base_model_path=None, output_dir=None):
        self.base_model_path = base_model_path
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), 'models', 'tuned')
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.config_dir = os.path.join(os.path.dirname(__file__), 'configs')
        self.yolov5_dir = os.path.join(root_path, 'yolov5-master')
        
        # Create output dirs
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Find the best model if not specified
        if not self.base_model_path:
            self._find_best_model()
    
    def _find_best_model(self):
        """Find the best available model to use as base for fine-tuning"""
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Check if custom models exist
        if os.path.exists(models_dir):
            # Look for directories in models_dir
            model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
            if model_dirs:
                # Sort by modification time (newest first)
                model_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(models_dir, d)), reverse=True)
                best_model = os.path.join(models_dir, model_dirs[0], 'weights', 'best.pt')
                
                if os.path.exists(best_model):
                    self.base_model_path = best_model
                    print(f"Using best found model at: {best_model}")
                    return
        
        # If no custom model, use standard YOLOv5s
        self.base_model_path = 'yolov5s.pt'
        print(f"No custom model found, will use YOLOv5s")
    
    def create_hyperparameter_config(self, config_name, params=None):
        """Create a hyperparameter tuning config file"""
        # Default hyperparameters for fine-tuning
        default_params = {
            'lr0': 0.01,           # Initial learning rate
            'lrf': 0.01,           # Final learning rate fraction
            'momentum': 0.937,      # SGD momentum/Adam beta1
            'weight_decay': 0.0005, # Optimizer weight decay
            'warmup_epochs': 3.0,   # Warmup epochs
            'warmup_momentum': 0.8, # Warmup initial momentum
            'warmup_bias_lr': 0.1,  # Warmup initial bias lr
            'box': 0.05,           # Box loss gain
            'cls': 0.5,            # Cls loss gain
            'cls_pw': 1.0,         # Cls BCELoss positive_weight
            'obj': 1.0,            # Obj loss gain
            'obj_pw': 1.0,         # Obj BCELoss positive_weight
            'iou_t': 0.20,         # IoU training threshold
            'anchor_t': 4.0,       # Anchor-multiple threshold
            'fl_gamma': 0.0,       # Focal loss gamma
            'hsv_h': 0.015,        # Image HSV-Hue augmentation (fraction)
            'hsv_s': 0.7,          # Image HSV-Saturation augmentation (fraction)
            'hsv_v': 0.4,          # Image HSV-Value augmentation (fraction)
            'degrees': 0.0,        # Image rotation (+/- deg)
            'translate': 0.1,      # Image translation (+/- fraction)
            'scale': 0.5,          # Image scale (+/- gain)
            'shear': 0.0,          # Image shear (+/- deg)
            'perspective': 0.0,     # Image perspective (+/- fraction), range 0-0.001
            'flipud': 0.0,         # Image flip up-down (probability)
            'fliplr': 0.5,         # Image flip left-right (probability)
            'mosaic': 1.0,         # Image mosaic (probability)
            'mixup': 0.0,          # Image mixup (probability)
            'copy_paste': 0.0      # Segment copy-paste (probability)
        }
        
        # Update with provided parameters
        if params:
            default_params.update(params)
        
        # Create the config file
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(default_params, f, sort_keys=False)
        
        print(f"Created hyperparameter config at: {config_path}")
        return config_path
    
    def create_project_specific_config(self, project_type):
        """Create hyperparameter config optimized for specific project types"""
        # Different configs for different project types
        configs = {
            'construction': {
                'lr0': 0.01,
                'translate': 0.2,  # More translation for construction timelines
                'scale': 0.6,      # More scaling for different views
                'mosaic': 1.0,     # Use mosaic augmentation
                'mixup': 0.1       # Use some mixup
            },
            'software': {
                'lr0': 0.005,       # Lower learning rate for detail
                'box': 0.1,        # Higher box loss for precise rectangles
                'obj': 1.2,        # Higher objectness for many similar tasks
                'translate': 0.1, 
                'scale': 0.4,
                'mosaic': 1.0
            },
            'marketing': {
                'lr0': 0.008,
                'translate': 0.15,
                'scale': 0.5,
                'shear': 0.1,      # Some shear for variable layouts
                'mosaic': 1.0
            },
            'research': {
                'lr0': 0.006,
                'box': 0.07,
                'cls': 0.6,       # Higher class loss for research specifics
                'translate': 0.1,
                'scale': 0.4,
                'mosaic': 1.0
            },
            'general': {  # Default balanced config
                'lr0': 0.008,
                'translate': 0.15,
                'scale': 0.5,
                'mosaic': 1.0
            }
        }
        
        # Use general config if project type not found
        config_params = configs.get(project_type.lower(), configs['general'])
        
        return self.create_hyperparameter_config(f"{project_type}_config", config_params)
    
    def fine_tune(self, data_yaml=None, project_type='general', epochs=20, batch_size=16, image_size=640):
        """Fine-tune model with specific hyperparameters for project type"""
        # Find the data YAML if not provided
        if not data_yaml:
            data_yaml = os.path.join(self.data_dir, 'ms_project.yaml')
            if not os.path.exists(data_yaml):
                print(f"Data YAML not found at {data_yaml}")
                return False
        
        # Create project-specific hyperparameter config
        hyp_config = self.create_project_specific_config(project_type)
        
        # Prepare the output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"tuned_{project_type}_{timestamp}"
        
        # Set up training command
        train_script = os.path.join(self.yolov5_dir, 'train.py')
        
        cmd = [
            'python',
            train_script,
            '--img', str(image_size),
            '--batch', str(batch_size),
            '--epochs', str(epochs),
            '--data', data_yaml,
            '--weights', self.base_model_path,
            '--hyp', hyp_config,
            '--project', self.output_dir,
            '--name', output_name,
            '--exist-ok',
            '--cache'
        ]
        
        # Run the training
        print("\nStarting fine-tuning process...")
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            # Run the process and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Print output in real-time
            for line in process.stdout:
                print(line, end='')
            
            # Wait for completion
            process.wait()
            
            if process.returncode == 0:
                tuned_model_path = os.path.join(self.output_dir, output_name, 'weights', 'best.pt')
                print(f"\nFine-tuning completed successfully!")
                print(f"Tuned model saved to: {tuned_model_path}")
                return tuned_model_path
            else:
                print(f"\nFine-tuning failed with return code {process.returncode}")
                return None
            
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            return None
    
    def optimize_model(self, model_path=None, format='onnx'):
        """Optimize model for faster inference"""
        if not model_path:
            model_path = self.base_model_path
        
        print(f"Optimizing model: {model_path}")
        
        # Export to optimized format
        export_script = os.path.join(self.yolov5_dir, 'export.py')
        
        cmd = [
            'python',
            export_script,
            '--weights', model_path,
            '--include', format,
            '--imgsz', '640',
            '--optimize'
        ]
        
        print(f"Running optimization command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            optimized_path = model_path.replace('.pt', f'.{format}')
            
            print(f"Model optimization complete: {optimized_path}")
            return optimized_path
        except Exception as e:
            print(f"Error optimizing model: {e}")
            return None
    
    def benchmark_model(self, model_path, test_images_dir=None):
        """Benchmark model performance on test images"""
        if not test_images_dir:
            test_images_dir = os.path.join(self.data_dir, 'test_images')
            if not os.path.exists(test_images_dir):
                test_images_dir = os.path.join(self.data_dir, 'screenshots')  # Fallback
        
        if not os.path.exists(test_images_dir):
            print(f"Test images directory not found at {test_images_dir}")
            return None
        
        print(f"Benchmarking model: {model_path}")
        print(f"Using test images from: {test_images_dir}")
        
        # Run YOLOv5 val.py for benchmarking
        val_script = os.path.join(self.yolov5_dir, 'val.py')
        
        # Prepare benchmark data.yaml pointing to test images
        benchmark_yaml = os.path.join(self.config_dir, 'benchmark.yaml')
        with open(benchmark_yaml, 'w') as f:
            yaml.dump({
                'path': os.path.dirname(test_images_dir),
                'test': os.path.basename(test_images_dir),
                'nc': 10,  # Number of classes
                'names': ['task', 'milestone', 'resource', 'gantt_bar', 'critical_path',
                          'dependency', 'constraint', 'deadline', 'baseline', 'progress']
            }, f)
        
        cmd = [
            'python',
            val_script,
            '--data', benchmark_yaml,
            '--weights', model_path,
            '--batch-size', '16',
            '--imgsz', '640',
            '--task', 'val',
            '--verbose'
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Parse results
            output = result.stdout
            metrics = {}
            
            # Extract mAP, precision, recall
            for line in output.split('\n'):
                if 'all' in line and 'mAP@0.5' in line:
                    parts = line.split()
                    metrics['mAP50'] = float(parts[3])
                    metrics['mAP50-95'] = float(parts[6])
                    metrics['precision'] = float(parts[9])
                    metrics['recall'] = float(parts[12])
            
            # Extract inference speed
            for line in output.split('\n'):
                if 'Speed:' in line and 'ms' in line:
                    parts = line.split()
                    for part in parts:
                        if 'inference' in part:
                            idx = parts.index(part)
                            metrics['inference_speed'] = float(parts[idx + 1].replace('ms', ''))
            
            print(f"\nBenchmark Results for {os.path.basename(model_path)}:")
            print(f"mAP@0.5: {metrics.get('mAP50', 'N/A'):.4f}")
            print(f"mAP@0.5-0.95: {metrics.get('mAP50-95', 'N/A'):.4f}")
            print(f"Precision: {metrics.get('precision', 'N/A'):.4f}")
            print(f"Recall: {metrics.get('recall', 'N/A'):.4f}")
            print(f"Inference Speed: {metrics.get('inference_speed', 'N/A'):.2f} ms")
            
            # Save benchmark results
            benchmark_path = os.path.join(self.output_dir, 'benchmark_results.json')
            results = {}
            
            # Load existing results if available
            if os.path.exists(benchmark_path):
                with open(benchmark_path, 'r') as f:
                    try:
                        results = json.load(f)
                    except json.JSONDecodeError:
                        results = {}
            
            # Add new results
            model_name = os.path.basename(model_path)
            results[model_name] = {
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                'model_path': model_path
            }
            
            # Save updated results
            with open(benchmark_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Benchmark results saved to: {benchmark_path}")
            return metrics
            
        except Exception as e:
            print(f"Error benchmarking model: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Fine-tune and optimize YOLOv5 models for Microsoft Project")
    parser.add_argument('--model', type=str, default=None, help='Base model path to fine-tune')
    parser.add_argument('--data', type=str, default=None, help='Path to data.yaml configuration')
    parser.add_argument('--output', type=str, default=None, help='Output directory for tuned models')
    parser.add_argument('--type', type=str, default='general', 
                        choices=['general', 'construction', 'software', 'marketing', 'research'],
                        help='Type of project for specialized tuning')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--optimize', action='store_true', help='Optimize model for inference')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark model performance')
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'tflite', 'coreml'],
                        help='Export format for optimization')
    
    args = parser.parse_args()
    
    tuner = MSProjectModelTuner(base_model_path=args.model, output_dir=args.output)
    
    # Fine-tune the model
    tuned_model = tuner.fine_tune(
        data_yaml=args.data,
        project_type=args.type,
        epochs=args.epochs,
        batch_size=args.batch
    )
    
    if not tuned_model:
        print("Fine-tuning failed. Exiting.")
        return
    
    # Optimize if requested
    if args.optimize and tuned_model:
        optimized_model = tuner.optimize_model(tuned_model, format=args.format)
        
        if not optimized_model:
            print("Model optimization failed.")
    
    # Benchmark if requested
    if args.benchmark and tuned_model:
        tuner.benchmark_model(tuned_model)

if __name__ == "__main__":
    main()
