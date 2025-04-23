import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
import cv2
import numpy as np
from pathlib import Path
import shutil
import argparse

# Add root path for shared utilities
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_path)

class MSProjectLabelingTool:
    """Custom labeling tool for Microsoft Project elements"""
    
    def __init__(self, image_dir=None, output_dir=None):
        self.image_dir = image_dir or os.path.join(os.path.dirname(__file__), 'data', 'screenshots')
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), 'data', 'labeled')
        
        # Create output directories
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'labels'), exist_ok=True)
        
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
        
        # Color mapping for visualization
        self.class_colors = {
            'task': (0, 0, 255),        # Red (BGR)
            'milestone': (0, 0, 128),    # Dark red
            'resource': (0, 255, 0),     # Green
            'gantt_bar': (255, 0, 0),    # Blue
            'critical_path': (0, 128, 255), # Orange
            'dependency': (255, 0, 255),  # Magenta
            'constraint': (255, 255, 0),  # Cyan
            'deadline': (128, 0, 128),    # Purple
            'baseline': (0, 255, 255),    # Yellow
            'progress': (0, 128, 0)       # Dark green
        }
        
        # Internal state
        self.image_files = []
        self.current_image_index = -1
        self.current_image = None
        self.current_image_path = None
        self.current_image_size = (0, 0)
        self.current_annotations = []
        self.current_class = 'task'
        
        # Annotation state
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.drawing = False
        self.dragging = False
        self.selected_box_index = -1
        
        # Setup the GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Initialize the GUI components"""
        self.root = tk.Tk()
        self.root.title("MS Project Labeling Tool")
        self.root.geometry("1280x800")
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_frame, width=200)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Right panel for image display
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image navigation
        nav_frame = ttk.LabelFrame(left_panel, text="Navigation")
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="Load Directory", command=self.load_directory).pack(fill=tk.X, pady=2)
        ttk.Button(nav_frame, text="Previous", command=self.prev_image).pack(fill=tk.X, pady=2)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(fill=tk.X, pady=2)
        
        self.image_counter_label = ttk.Label(nav_frame, text="Image: 0/0")
        self.image_counter_label.pack(fill=tk.X, pady=2)
        
        # Class selection
        class_frame = ttk.LabelFrame(left_panel, text="Class Selection")
        class_frame.pack(fill=tk.X, pady=5)
        
        self.class_var = tk.StringVar(value=self.current_class)
        for class_name in self.classes.keys():
            ttk.Radiobutton(class_frame, text=class_name, value=class_name, 
                          variable=self.class_var, command=self.update_current_class).pack(anchor=tk.W)
        
        # Annotation actions
        action_frame = ttk.LabelFrame(left_panel, text="Actions")
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="Delete Selected", command=self.delete_selected).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Clear All", command=self.clear_annotations).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Save", command=self.save_annotations).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Export to YOLOv5", command=self.export_to_yolo).pack(fill=tk.X, pady=2)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(left_panel, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        # Canvas for image display
        self.canvas_frame = ttk.Frame(right_panel)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#CCCCCC")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(fill=tk.X, side=tk.BOTTOM)
        
        v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Mouse event bindings
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Key bindings
        self.root.bind("<Left>", lambda event: self.prev_image())
        self.root.bind("<Right>", lambda event: self.next_image())
        self.root.bind("<Delete>", lambda event: self.delete_selected())
        self.root.bind("<s>", lambda event: self.save_annotations())
        
        # Set default directory if provided
        if os.path.exists(self.image_dir):
            self.load_directory(self.image_dir)
    
    def load_directory(self, directory=None):
        """Load images from a directory"""
        if directory is None:
            directory = filedialog.askdirectory(initialdir=self.image_dir, title="Select Image Directory")
        
        if directory:  # User selected a directory
            self.image_dir = directory
            self.image_files = []
            
            # Find all images in the directory
            valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
            for file in os.listdir(directory):
                if file.lower().endswith(valid_extensions):
                    self.image_files.append(os.path.join(directory, file))
            
            self.image_files.sort()  # Sort alphabetically
            self.current_image_index = -1
            
            self.status_var.set(f"Loaded {len(self.image_files)} images from {directory}")
            
            if self.image_files:  # If there are images, load the first one
                self.next_image()
            else:
                messagebox.showinfo("No Images", "No image files found in the selected directory.")
    
    def load_image(self, index):
        """Load the image at the specified index"""
        if 0 <= index < len(self.image_files):
            self.current_image_index = index
            self.current_image_path = self.image_files[index]
            
            # Load the image
            self.current_image = Image.open(self.current_image_path)
            self.current_image_size = self.current_image.size
            
            # Prepare the image for display
            self.display_image()
            
            # Load existing annotations if any
            self.load_annotations()
            
            # Update counter
            self.image_counter_label.config(text=f"Image: {index+1}/{len(self.image_files)}")
            self.status_var.set(f"Loaded {os.path.basename(self.current_image_path)}")
    
    def display_image(self):
        """Display the current image on the canvas"""
        if self.current_image:
            # Clear the canvas
            self.canvas.delete("all")
            
            # Convert PIL Image to PhotoImage
            self.tk_image = ImageTk.PhotoImage(self.current_image)
            
            # Reset the canvas scrollregion
            self.canvas.config(scrollregion=(0, 0, self.current_image_size[0], self.current_image_size[1]))
            
            # Create image on canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            
            # Render existing annotations
            self.render_annotations()
    
    def render_annotations(self):
        """Render all annotations on the canvas"""
        for i, ann in enumerate(self.current_annotations):
            # Get class name and color
            class_name = ann['class']
            
            # Convert normalized coordinates to pixel coordinates
            x1, y1, x2, y2 = self.normalize_to_pixel(ann['bbox'])
            
            # Draw the rectangle
            rect_id = self.canvas.create_rectangle(
                x1, y1, x2, y2, 
                outline=self.get_color_string(class_name), 
                width=2,
                tags=(f"box_{i}", "box")
            )
            
            # Draw the label
            self.canvas.create_text(
                x1, y1-10, 
                text=f"{class_name}", 
                fill=self.get_color_string(class_name), 
                anchor=tk.SW,
                tags=(f"label_{i}", "label")
            )
    
    def normalize_to_pixel(self, bbox):
        """Convert normalized coordinates (YOLO format) to pixel coordinates"""
        x_center, y_center, w, h = bbox
        img_w, img_h = self.current_image_size
        
        x1 = int((x_center - w/2) * img_w)
        y1 = int((y_center - h/2) * img_h)
        x2 = int((x_center + w/2) * img_w)
        y2 = int((y_center + h/2) * img_h)
        
        return x1, y1, x2, y2
    
    def pixel_to_normalize(self, x1, y1, x2, y2):
        """Convert pixel coordinates to normalized coordinates (YOLO format)"""
        img_w, img_h = self.current_image_size
        
        x_center = (x1 + x2) / (2 * img_w)
        y_center = (y1 + y2) / (2 * img_h)
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h
        
        return [x_center, y_center, w, h]
    
    def get_color_string(self, class_name):
        """Convert BGR color to tkinter color string"""
        if class_name in self.class_colors:
            b, g, r = self.class_colors[class_name]  # BGR order
            return f"#{r:02x}{g:02x}{b:02x}"
        return "#FF0000"  # Default red
    
    def update_current_class(self):
        """Update the current class selection"""
        self.current_class = self.class_var.get()
        self.status_var.set(f"Current class: {self.current_class}")
    
    def on_mouse_down(self, event):
        """Handle mouse button press"""
        # Get canvas coordinates
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Check if clicking on an existing box
        closest = self.canvas.find_closest(x, y)
        tags = self.canvas.gettags(closest)
        
        if "box" in tags or any(tag.startswith("box_") for tag in tags):
            # Find which box was clicked
            for tag in tags:
                if tag.startswith("box_"):
                    box_index = int(tag.split("_")[1])
                    self.selected_box_index = box_index
                    self.dragging = True
                    
                    # Get the starting coordinates
                    x1, y1, x2, y2 = self.normalize_to_pixel(self.current_annotations[box_index]['bbox'])
                    self.drag_start_x = x
                    self.drag_start_y = y
                    self.drag_orig_coords = (x1, y1, x2, y2)
                    
                    # Highlight the selected box
                    self.canvas.itemconfig(f"box_{box_index}", width=3)
                    return
        
        # Start drawing a new box
        self.drawing = True
        self.start_x = x
        self.start_y = y
        self.current_rect = None
    
    def on_mouse_move(self, event):
        """Handle mouse movement"""
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        if self.drawing and self.start_x is not None:
            # Drawing a new box
            if self.current_rect:
                self.canvas.delete(self.current_rect)
            
            self.current_rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, x, y,
                outline=self.get_color_string(self.current_class),
                width=2
            )
        elif self.dragging and self.selected_box_index >= 0:
            # Moving an existing box
            dx = x - self.drag_start_x
            dy = y - self.drag_start_y
            
            x1, y1, x2, y2 = self.drag_orig_coords
            new_x1, new_y1, new_x2, new_y2 = x1 + dx, y1 + dy, x2 + dx, y2 + dy
            
            # Update the box position on canvas
            self.canvas.coords(f"box_{self.selected_box_index}", new_x1, new_y1, new_x2, new_y2)
            
            # Update the label position
            label_id = self.canvas.find_withtag(f"label_{self.selected_box_index}")
            if label_id:
                self.canvas.coords(label_id, new_x1, new_y1-10)
    
    def on_mouse_up(self, event):
        """Handle mouse button release"""
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        if self.drawing and self.start_x is not None:
            # Finalize the new box
            if self.current_rect:
                # Ensure x1 < x2 and y1 < y2
                x1 = min(self.start_x, x)
                y1 = min(self.start_y, y)
                x2 = max(self.start_x, x)
                y2 = max(self.start_y, y)
                
                # Only add if the box has some size
                if x2 - x1 > 5 and y2 - y1 > 5:
                    # Convert to normalized coordinates
                    bbox = self.pixel_to_normalize(x1, y1, x2, y2)
                    
                    # Add the annotation
                    self.current_annotations.append({
                        'class': self.current_class,
                        'bbox': bbox
                    })
                    
                    # Clear the temporary rectangle
                    self.canvas.delete(self.current_rect)
                    
                    # Redraw all annotations
                    self.render_annotations()
                    
                    self.status_var.set(f"Added {self.current_class} annotation")
                else:
                    self.canvas.delete(self.current_rect)
            
            self.drawing = False
            self.start_x = None
            self.start_y = None
            self.current_rect = None
        
        elif self.dragging and self.selected_box_index >= 0:
            # Finalize the moved box
            x1, y1, x2, y2 = self.canvas.coords(f"box_{self.selected_box_index}")
            bbox = self.pixel_to_normalize(x1, y1, x2, y2)
            
            # Update the annotation
            self.current_annotations[self.selected_box_index]['bbox'] = bbox
            
            # Reset dragging state
            self.dragging = False
            # Keep the box selected for possible deletion
            
            self.status_var.set(f"Moved {self.current_annotations[self.selected_box_index]['class']} annotation")
    
    def delete_selected(self):
        """Delete the selected annotation"""
        if self.selected_box_index >= 0:
            class_name = self.current_annotations[self.selected_box_index]['class']
            del self.current_annotations[self.selected_box_index]
            
            # Redraw all annotations
            self.display_image()
            
            self.selected_box_index = -1
            self.status_var.set(f"Deleted {class_name} annotation")
    
    def clear_annotations(self):
        """Clear all annotations for the current image"""
        if messagebox.askyesno("Clear Annotations", "Are you sure you want to clear all annotations?"): 
            self.current_annotations = []
            self.display_image()
            self.status_var.set("Cleared all annotations")
    
    def load_annotations(self):
        """Load existing annotations for the current image"""
        self.current_annotations = []
        
        # Try to load YOLO format labels
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        label_path = os.path.join(self.output_dir, 'labels', f"{base_name}.txt")
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:  # class_id x_center y_center width height
                        class_id = int(parts[0])
                        bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                        
                        # Get class name from id
                        class_name = None
                        for name, id in self.classes.items():
                            if id == class_id:
                                class_name = name
                                break
                        
                        if class_name:  # Only add if we know the class
                            self.current_annotations.append({
                                'class': class_name,
                                'bbox': bbox
                            })
            
            self.status_var.set(f"Loaded {len(self.current_annotations)} annotations from {label_path}")
    
    def save_annotations(self):
        """Save annotations for the current image"""
        if not self.current_image_path:
            return
        
        # Save in YOLO format
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        label_path = os.path.join(self.output_dir, 'labels', f"{base_name}.txt")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        with open(label_path, 'w') as f:
            for ann in self.current_annotations:
                class_id = self.classes[ann['class']]
                bbox = ann['bbox']  # already normalized
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        
        # Also copy the image to the output directory
        image_output_path = os.path.join(self.output_dir, 'images', os.path.basename(self.current_image_path))
        shutil.copy2(self.current_image_path, image_output_path)
        
        self.status_var.set(f"Saved annotations to {label_path}")
    
    def next_image(self):
        """Load the next image"""
        if self.current_image_index < len(self.image_files) - 1:
            # Save current annotations before moving
            if self.current_image_path:
                self.save_annotations()
            
            self.load_image(self.current_image_index + 1)
    
    def prev_image(self):
        """Load the previous image"""
        if self.current_image_index > 0:
            # Save current annotations before moving
            if self.current_image_path:
                self.save_annotations()
            
            self.load_image(self.current_image_index - 1)
    
    def export_to_yolo(self):
        """Export all annotations to YOLOv5 format"""
        # Save current annotations before export
        if self.current_image_path:
            self.save_annotations()
        
        # Create dataset.yaml for training
        yaml_path = os.path.join(os.path.dirname(self.output_dir), 'ms_project.yaml')
        
        with open(yaml_path, 'w') as f:
            f.write(f"# YOLOv5 MS Project dataset configuration\n")
            f.write(f"path: {os.path.dirname(self.output_dir)}\n\n")
            f.write(f"train: {os.path.basename(self.output_dir)}/images\n")
            f.write(f"val: {os.path.basename(self.output_dir)}/images\n\n")
            
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
        
        # Count the labeled images and annotations
        label_files = os.listdir(os.path.join(self.output_dir, 'labels'))
        total_labels = 0
        for label_file in label_files:
            with open(os.path.join(self.output_dir, 'labels', label_file), 'r') as f:
                total_labels += len(f.readlines())
        
        messagebox.showinfo(
            "Export Complete", 
            f"Export to YOLOv5 format complete!\n\n"
            f"- {len(label_files)} labeled images\n"
            f"- {total_labels} total annotations\n"
            f"- Data configuration saved to {yaml_path}"
        )
        
        self.status_var.set(f"Exported {len(label_files)} images with {total_labels} annotations")
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="MS Project Element Labeling Tool")
    parser.add_argument('--input', type=str, default=None, help='Input directory containing images to label')
    parser.add_argument('--output', type=str, default=None, help='Output directory for labeled data')
    
    args = parser.parse_args()
    
    app = MSProjectLabelingTool(image_dir=args.input, output_dir=args.output)
    app.run()

if __name__ == "__main__":
    main()
