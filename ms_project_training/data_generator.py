import os
import sys
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add root path for shared utilities
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_path)

class MSProjectDataGenerator:
    """Synthetic data generator for Microsoft Project elements"""
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), 'data', 'generated')
        self.label_dir = os.path.join(self.output_dir, 'labels')
        self.image_dir = os.path.join(self.output_dir, 'images')
        
        # Create output directories
        os.makedirs(self.label_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Define class mapping
        self.classes = {
            'task': 0,
            'milestone': 1,
            'resource': 2,
            'gantt_bar': 3,
            'critical_path': 4,
            'dependency': 5,
            'constraint': 6,
            'deadline': 7,
            'baseline': 8,
            'progress': 9
        }
        
        # Load fonts
        self.fonts = {
            'regular': self._load_font('arial.ttf', 12),
            'bold': self._load_font('arialbd.ttf', 12),
            'small': self._load_font('arial.ttf', 10),
            'large': self._load_font('arial.ttf', 14)
        }
        
        # Color schemes for MS Project
        self.color_schemes = {
            'default': {
                'background': (240, 240, 240),
                'grid': (220, 220, 220),
                'text': (0, 0, 0),
                'task': (58, 110, 165),  # Blue
                'milestone': (192, 0, 0),  # Dark red
                'critical_path': (255, 0, 0),  # Red
                'progress': (0, 128, 0),  # Green
                'resource': (128, 64, 0),  # Brown
                'dependency': (120, 120, 120)  # Gray
            },
            'dark': {
                'background': (50, 50, 50),
                'grid': (80, 80, 80),
                'text': (220, 220, 220),
                'task': (100, 149, 237),  # Lighter blue
                'milestone': (255, 99, 71),  # Tomato
                'critical_path': (255, 69, 0),  # Orange-red
                'progress': (50, 205, 50),  # Lime green
                'resource': (210, 180, 140),  # Tan
                'dependency': (169, 169, 169)  # Dark gray
            }
        }
        
        # Task name templates
        self.task_templates = [
            "Task {num}",
            "Design Phase {num}",
            "Development Task {num}",
            "Testing {num}",
            "Review {num}",
            "Implementation {num}",
            "Research {num}",
            "Analysis {num}",
            "Planning {num}",
            "Deployment {num}"
        ]
        
        # Resource name templates
        self.resource_templates = [
            "John Smith",
            "Amy Johnson",
            "Michael Brown",
            "Sarah Davis",
            "Robert Wilson",
            "Developer {num}",
            "Designer {num}",
            "Tester {num}",
            "Analyst {num}",
            "Manager {num}"
        ]
    
    def _load_font(self, font_name, size):
        """Load a font for rendering text in synthetic images"""
        try:
            # Try to load font from system
            font = ImageFont.truetype(font_name, size)
            return font
        except IOError:
            # Fallback to default font
            print(f"Font {font_name} not found, using default")
            return ImageFont.load_default()
    
    def generate_gantt_chart(self, width=1200, height=800, task_count=15, with_resources=True):
        """Generate a synthetic Gantt chart image with labeled elements"""
        # Create blank image with background color
        color_scheme = random.choice(list(self.color_schemes.values()))
        image = Image.new('RGB', (width, height), color_scheme['background'])
        draw = ImageDraw.Draw(image)
        
        # Draw grid lines
        for i in range(0, width, 50):
            draw.line([(i, 0), (i, height)], fill=color_scheme['grid'], width=1)
        for i in range(0, height, 30):
            draw.line([(0, i), (width, i)], fill=color_scheme['grid'], width=1)
        
        # Set up data structures for annotations
        annotations = []
        task_data = self._generate_task_data(task_count, width, height)
        
        # Draw the table header
        header_height = 60
        draw.rectangle([(0, 0), (width, header_height)], fill=(200, 200, 200))
        draw.line([(0, header_height), (width, header_height)], fill=(100, 100, 100), width=2)
        
        # Draw column headers
        columns = [
            {'name': 'ID', 'width': 40},
            {'name': 'Task Name', 'width': 250},
            {'name': 'Duration', 'width': 80},
            {'name': 'Start', 'width': 100},
            {'name': 'Finish', 'width': 100},
            {'name': 'Resources', 'width': 150},
            {'name': 'Gantt Chart', 'width': width - 720}
        ]
        
        x_pos = 0
        for col in columns:
            draw.text((x_pos + 5, 30), col['name'], font=self.fonts['bold'], fill=color_scheme['text'])
            x_pos += col['width']
            draw.line([(x_pos, 0), (x_pos, height)], fill=(150, 150, 150), width=1)
        
        # Calculate the start of the Gantt area
        gantt_start_x = sum(col['width'] for col in columns[:-1])
        
        # Define timeline for Gantt
        start_date = datetime.now()
        days_range = 60
        day_width = (width - gantt_start_x) / days_range
        
        # Draw timeline at top of Gantt area
        for day in range(days_range):
            date = start_date + timedelta(days=day)
            x = gantt_start_x + (day * day_width)
            if day % 7 == 0:  # Highlight weeks
                draw.text((x, 10), date.strftime("%m/%d"), font=self.fonts['small'], fill=color_scheme['text'])
                draw.line([(x, header_height), (x, height)], fill=(180, 180, 180), width=1)
        
        # Draw tasks and gantt bars
        y_pos = header_height + 15
        row_height = 30
        
        for i, task in enumerate(task_data):
            row_y = y_pos + (i * row_height)
            
            # Task ID
            draw.text((5, row_y), str(i+1), font=self.fonts['regular'], fill=color_scheme['text'])
            
            # Task name
            task_name_x = columns[0]['width'] + 5
            draw.text((task_name_x, row_y), task['name'], font=self.fonts['regular'], fill=color_scheme['text'])
            
            # Add task name label
            task_name_width = columns[1]['width'] - 10
            task_name_box = [task_name_x, row_y, task_name_x + task_name_width, row_y + row_height - 5]
            annotations.append({
                'class': 'task',
                'bbox': self._normalize_bbox(task_name_box, width, height)
            })
            
            # Duration
            duration_x = task_name_x + columns[1]['width']
            draw.text((duration_x + 5, row_y), task['duration'], font=self.fonts['regular'], fill=color_scheme['text'])
            
            # Dates
            start_x = duration_x + columns[2]['width']
            finish_x = start_x + columns[3]['width']
            draw.text((start_x + 5, row_y), task['start_date'], font=self.fonts['regular'], fill=color_scheme['text'])
            draw.text((finish_x + 5, row_y), task['end_date'], font=self.fonts['regular'], fill=color_scheme['text'])
            
            # Resources
            if with_resources:
                resource_x = finish_x + columns[4]['width']
                resource_text = task['resource']
                draw.text((resource_x + 5, row_y), resource_text, font=self.fonts['regular'], fill=color_scheme['text'])
                
                # Add resource label
                resource_width = columns[5]['width'] - 10
                resource_box = [resource_x, row_y, resource_x + resource_width, row_y + row_height - 5]
                annotations.append({
                    'class': 'resource',
                    'bbox': self._normalize_bbox(resource_box, width, height)
                })
            
            # Draw Gantt bar
            bar_start = gantt_start_x + (task['day_offset'] * day_width)
            bar_width = task['duration_days'] * day_width
            bar_height = 16
            bar_y = row_y + (row_height - bar_height) / 2
            
            # Determine if critical path
            is_critical = task.get('critical', False)
            bar_color = color_scheme['critical_path'] if is_critical else color_scheme['task']
            
            # Draw task bar
            draw.rectangle(
                [(bar_start, bar_y), (bar_start + bar_width, bar_y + bar_height)],
                fill=bar_color,
                outline=(0, 0, 0)
            )
            
            # Add gantt bar label
            gantt_bar_box = [bar_start, bar_y, bar_start + bar_width, bar_y + bar_height]
            annotations.append({
                'class': 'gantt_bar' if not is_critical else 'critical_path',
                'bbox': self._normalize_bbox(gantt_bar_box, width, height)
            })
            
            # Draw progress if applicable
            if 'progress' in task:
                progress_width = bar_width * (task['progress'] / 100)
                draw.rectangle(
                    [(bar_start, bar_y), (bar_start + progress_width, bar_y + bar_height)],
                    fill=color_scheme['progress']
                )
                
                if progress_width > 10:  # Only add label if progress bar is wide enough
                    progress_box = [bar_start, bar_y, bar_start + progress_width, bar_y + bar_height]
                    annotations.append({
                        'class': 'progress',
                        'bbox': self._normalize_bbox(progress_box, width, height)
                    })
            
            # Draw milestone if applicable
            if task.get('milestone', False):
                milestone_x = bar_start + bar_width
                milestone_size = 12
                milestone_points = [
                    (milestone_x, bar_y + bar_height/2 - milestone_size/2),
                    (milestone_x + milestone_size, bar_y + bar_height/2),
                    (milestone_x, bar_y + bar_height/2 + milestone_size/2)
                ]
                draw.polygon(milestone_points, fill=color_scheme['milestone'])
                
                milestone_box = [
                    milestone_x - milestone_size/2, 
                    bar_y + bar_height/2 - milestone_size/2,
                    milestone_x + milestone_size/2, 
                    bar_y + bar_height/2 + milestone_size/2
                ]
                annotations.append({
                    'class': 'milestone',
                    'bbox': self._normalize_bbox(milestone_box, width, height)
                })
            
            # Draw dependencies occasionally
            if i > 0 and random.random() < 0.7:  # 70% chance of dependency
                prev_task = task_data[i-1]
                prev_bar_end = gantt_start_x + ((prev_task['day_offset'] + prev_task['duration_days']) * day_width)
                prev_bar_y = y_pos + ((i-1) * row_height) + row_height / 2
                
                # Draw dependency arrow
                arrow_points = [
                    (prev_bar_end, prev_bar_y + bar_height/2),
                    (prev_bar_end + 10, prev_bar_y + bar_height/2),
                    (prev_bar_end + 10, bar_y + bar_height/2),
                    (bar_start, bar_y + bar_height/2)
                ]
                draw.line(arrow_points, fill=color_scheme['dependency'], width=1)
                
                # Add arrowhead
                arrow_head = [
                    (bar_start, bar_y + bar_height/2),
                    (bar_start - 5, bar_y + bar_height/2 - 3),
                    (bar_start - 5, bar_y + bar_height/2 + 3)
                ]
                draw.polygon(arrow_head, fill=color_scheme['dependency'])
                
                # Add dependency label
                dependency_box = [
                    prev_bar_end, prev_bar_y + bar_height/2 - 5,
                    bar_start, bar_y + bar_height/2 + 5
                ]
                annotations.append({
                    'class': 'dependency',
                    'bbox': self._normalize_bbox(dependency_box, width, height)
                })
        
        # Apply some random variations to make it more realistic
        image = self._apply_variations(image)
        
        # Save the image and annotations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(self.image_dir, f"synthetic_gantt_{timestamp}.jpg")
        label_path = os.path.join(self.label_dir, f"synthetic_gantt_{timestamp}.txt")
        
        image.save(image_path)
        self._save_yolo_labels(label_path, annotations)
        
        print(f"Generated synthetic Gantt chart with {len(annotations)} annotations: {image_path}")
        return image_path, label_path, annotations
    
    def _generate_task_data(self, count, width, height):
        """Generate random task data for the Gantt chart"""
        tasks = []
        start_date = datetime.now()
        current_day = 0
        
        for i in range(count):
            # Generate random task data
            task_name = random.choice(self.task_templates).format(num=i+1)
            duration_days = random.randint(3, 15)
            day_offset = current_day
            
            # Add some randomness to start days (sometimes tasks overlap or have gaps)
            if i > 0 and random.random() < 0.3:  # 30% chance of non-sequential
                day_offset = max(0, current_day - random.randint(1, 5))  # Overlap
            
            # Calculate dates
            task_start = start_date + timedelta(days=day_offset)
            task_end = task_start + timedelta(days=duration_days)
            
            # Generate task
            task = {
                'name': task_name,
                'duration': f"{duration_days}d",
                'duration_days': duration_days,
                'day_offset': day_offset,
                'start_date': task_start.strftime("%m/%d/%Y"),
                'end_date': task_end.strftime("%m/%d/%Y"),
                'resource': random.choice(self.resource_templates).format(num=random.randint(1, 5)),
                'progress': random.randint(0, 100) if random.random() < 0.7 else 0  # 70% have progress
            }
            
            # Some tasks are critical
            if random.random() < 0.3:  # 30% chance of critical task
                task['critical'] = True
            
            # Some tasks are milestones
            if random.random() < 0.2:  # 20% chance of milestone
                task['milestone'] = True
                task['duration_days'] = 0  # Milestones have zero duration
                task['duration'] = "0d"
            
            tasks.append(task)
            
            # Update current day for next task
            current_day = day_offset + duration_days
        
        return tasks
    
    def _normalize_bbox(self, bbox, width, height):
        """Convert absolute bbox coordinates to normalized YOLOv5 format [x_center, y_center, width, height]"""
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2 / width
        y_center = (y1 + y2) / 2 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        
        return [x_center, y_center, w, h]
    
    def _apply_variations(self, image):
        """Apply random variations to make synthetic images more realistic"""
        # Apply slight blur occasionally
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=1.1))
        
        # Adjust brightness slightly
        brightness_factor = random.uniform(0.9, 1.1)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        
        # Adjust contrast slightly
        contrast_factor = random.uniform(0.9, 1.1)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        
        return image
    
    def _save_yolo_labels(self, path, annotations):
        """Save annotations in YOLOv5 format (class x_center y_center width height)"""
        with open(path, 'w') as f:
            for ann in annotations:
                class_id = self.classes[ann['class']]
                bbox = ann['bbox']
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    def generate_dataset(self, count=50, width=1200, height=800):
        """Generate a full dataset of synthetic images with annotations"""
        print(f"Generating {count} synthetic Microsoft Project images...")
        
        generated_files = []
        for i in range(count):
            # Randomly choose between different view types
            view_type = random.choice(['gantt', 'resource', 'calendar'])  # Could add more view types
            
            if view_type == 'gantt':
                task_count = random.randint(8, 25)
                with_resources = random.random() < 0.8  # 80% chance of resources
                img_path, label_path, _ = self.generate_gantt_chart(
                    width=width, height=height, task_count=task_count, with_resources=with_resources
                )
                generated_files.append((img_path, label_path))
            
            # Could add other view types here
            # elif view_type == 'resource':
            #     img_path, label_path, _ = self.generate_resource_view()
            # elif view_type == 'calendar':
            #     img_path, label_path, _ = self.generate_calendar_view()
            
            # Progress update
            if (i + 1) % 10 == 0 or (i + 1) == count:
                print(f"Generated {i + 1}/{count} images")
        
        return generated_files
    
    def create_data_yaml(self):
        """Create dataset.yaml file for YOLOv5 training"""
        yaml_path = os.path.join(os.path.dirname(self.output_dir), 'ms_project.yaml')
        
        with open(yaml_path, 'w') as f:
            f.write(f"# YOLOv5 MS Project dataset configuration\n")
            f.write(f"path: {os.path.dirname(self.output_dir)}\n\n")
            f.write(f"train: {os.path.join('generated', 'images')}\n")
            f.write(f"val: {os.path.join('generated', 'images')}\n\n")
            
            f.write(f"# Classes\n")
            f.write(f"nc: {len(self.classes)}\n")
            
            # Write class names
            f.write(f"names: [")
            class_names = sorted(self.classes.items(), key=lambda x: x[1])
            for i, (name, _) in enumerate(class_names):
                if i > 0:
                    f.write(", ")
                f.write(f"'{name}'")
            f.write("]\n")
            
        print(f"Created YOLOv5 dataset configuration: {yaml_path}")
        return yaml_path

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic MS Project data for training")
    parser.add_argument('--count', type=int, default=50, help='Number of synthetic images to generate')
    parser.add_argument('--output', type=str, default=None, help='Output directory for synthetic data')
    parser.add_argument('--width', type=int, default=1200, help='Image width')
    parser.add_argument('--height', type=int, default=800, help='Image height')
    parser.add_argument('--create-yaml', action='store_true', help='Create dataset.yaml file for YOLOv5 training')
    
    args = parser.parse_args()
    
    generator = MSProjectDataGenerator(output_dir=args.output)
    generated_files = generator.generate_dataset(count=args.count, width=args.width, height=args.height)
    
    if args.create_yaml:
        generator.create_data_yaml()
    
    print(f"\nGeneration complete! Created {len(generated_files)} image/label pairs.")

if __name__ == "__main__":
    main()
