#!/usr/bin/env python
"""
Project Management Integration GUI

This script provides a graphical user interface for the YOLOv5 project management
integration, allowing users to detect and analyze elements in Microsoft Project
and Primavera P6 interfaces.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import logging
from pathlib import Path
from PIL import Image, ImageTk
import cv2
import numpy as np
from datetime import datetime

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(SCRIPT_DIR, 'pm_gui.log'))
    ]
)
logger = logging.getLogger('pm_gui')

# Import detectors
try:
    from msproject.msproject_detector import MSProjectDetector
    HAS_MSPROJECT = True
    logger.info("Microsoft Project detector imported successfully")
except ImportError:
    HAS_MSPROJECT = False
    logger.warning("Microsoft Project detector not available")

try:
    from primavera.primavera_detector import PrimaveraDetector
    HAS_PRIMAVERA = True
    logger.info("Primavera P6 detector imported successfully")
except ImportError:
    HAS_PRIMAVERA = False
    logger.warning("Primavera P6 detector not available")

# Check if OCR is available
try:
    from ocr_utils import get_ocr_processor
    HAS_OCR = True
    logger.info("OCR utilities imported successfully")
except ImportError:
    HAS_OCR = False
    logger.warning("OCR utilities not available")

class RedirectText:
    """Redirect stdout to a tkinter Text widget"""
    
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self.updating = True
        threading.Thread(target=self.update_text_widget, daemon=True).start()
    
    def write(self, string):
        self.queue.put(string)
    
    def flush(self):
        pass
    
    def update_text_widget(self):
        while self.updating:
            try:
                while True:
                    string = self.queue.get_nowait()
                    self.text_widget.configure(state="normal")
                    self.text_widget.insert("end", string)
                    self.text_widget.see("end")
                    self.text_widget.configure(state="disabled")
                    self.queue.task_done()
            except queue.Empty:
                pass
            self.text_widget.update_idletasks()
            self.text_widget.after(100)
    
    def stop(self):
        self.updating = False

class PMIntegrationGUI:
    """GUI for Project Management Integration"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv5 Project Management Integration")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Set up variables
        self.app_type = tk.StringVar(value="msproject")
        self.model_path = tk.StringVar(value="")
        self.conf_thres = tk.DoubleVar(value=0.25)
        self.iou_thres = tk.DoubleVar(value=0.45)
        self.use_ocr = tk.BooleanVar(value=True)
        self.tesseract_path = tk.StringVar(value="")
        
        # Find Tesseract
        self.find_tesseract()
        
        # Set up detectors
        self.msproject_detector = None
        self.primavera_detector = None
        
        # Set up UI
        self.setup_ui()
        
        # Initialize detectors
        self.initialize_detectors()
    
    def find_tesseract(self):
        """Find Tesseract OCR executable"""
        if not HAS_OCR:
            return
        
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'/usr/bin/tesseract',
            r'/usr/local/bin/tesseract'
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                self.tesseract_path.set(path)
                logger.info(f"Found Tesseract at: {path}")
                break
    
    def setup_ui(self):
        """Set up the user interface"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel for controls
        left_panel = ttk.Frame(main_frame, width=300, padding=5)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create right panel for display
        right_panel = ttk.Frame(main_frame, padding=5)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Set up left panel controls
        self.setup_controls(left_panel)
        
        # Set up right panel display
        self.setup_display(right_panel)
    
    def setup_controls(self, parent):
        """Set up control panel"""
        # Application selection
        app_frame = ttk.LabelFrame(parent, text="Application", padding=5)
        app_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(app_frame, text="Microsoft Project", value="msproject", 
                       variable=self.app_type, command=self.on_app_change).pack(anchor=tk.W)
        ttk.Radiobutton(app_frame, text="Primavera P6", value="primavera", 
                       variable=self.app_type, command=self.on_app_change).pack(anchor=tk.W)
        
        # Model settings
        model_frame = ttk.LabelFrame(parent, text="Model Settings", padding=5)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Model Path:").pack(anchor=tk.W)
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path)
        model_entry.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(model_frame, text="Browse...", command=self.browse_model).pack(anchor=tk.E, padx=5, pady=2)
        
        ttk.Label(model_frame, text="Confidence Threshold:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Scale(model_frame, from_=0.1, to=0.9, variable=self.conf_thres, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(model_frame, textvariable=tk.StringVar(value=lambda: f"{self.conf_thres.get():.2f}")).pack(anchor=tk.E)
        
        ttk.Label(model_frame, text="IoU Threshold:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Scale(model_frame, from_=0.1, to=0.9, variable=self.iou_thres, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(model_frame, textvariable=tk.StringVar(value=lambda: f"{self.iou_thres.get():.2f}")).pack(anchor=tk.E)
        
        # OCR settings
        ocr_frame = ttk.LabelFrame(parent, text="OCR Settings", padding=5)
        ocr_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Checkbutton(ocr_frame, text="Use OCR", variable=self.use_ocr, 
                       command=self.on_ocr_change).pack(anchor=tk.W)
        
        ttk.Label(ocr_frame, text="Tesseract Path:").pack(anchor=tk.W)
        tesseract_entry = ttk.Entry(ocr_frame, textvariable=self.tesseract_path)
        tesseract_entry.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(ocr_frame, text="Browse...", command=self.browse_tesseract).pack(anchor=tk.E, padx=5, pady=2)
        
        # Action buttons
        action_frame = ttk.Frame(parent, padding=5)
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(action_frame, text="Start Application", command=self.start_application).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Detect Elements", command=self.detect_elements).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Generate Report", command=self.generate_report).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Save Results", command=self.save_results).pack(fill=tk.X, pady=2)
        
        # Status
        status_frame = ttk.LabelFrame(parent, text="Status", padding=5)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=280).pack(fill=tk.X)
    
    def setup_display(self, parent):
        """Set up display panel"""
        # Create notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Image tab
        image_tab = ttk.Frame(notebook, padding=5)
        notebook.add(image_tab, text="Detection")
        
        # Set up image display
        self.image_canvas = tk.Canvas(image_tab, bg="#CCCCCC")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbars
        h_scrollbar = ttk.Scrollbar(image_tab, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        h_scrollbar.pack(fill=tk.X, side=tk.BOTTOM)
        
        v_scrollbar = ttk.Scrollbar(image_tab, orient=tk.VERTICAL, command=self.image_canvas.yview)
        v_scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        
        self.image_canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Report tab
        report_tab = ttk.Frame(notebook, padding=5)
        notebook.add(report_tab, text="Report")
        
        # Set up report display
        self.report_text = scrolledtext.ScrolledText(report_tab, wrap=tk.WORD, state="disabled")
        self.report_text.pack(fill=tk.BOTH, expand=True)
        
        # Console tab
        console_tab = ttk.Frame(notebook, padding=5)
        notebook.add(console_tab, text="Console")
        
        # Set up console display
        self.console_text = scrolledtext.ScrolledText(console_tab, wrap=tk.WORD, state="disabled")
        self.console_text.pack(fill=tk.BOTH, expand=True)
        
        # Redirect stdout to console
        self.stdout_redirect = RedirectText(self.console_text)
        sys.stdout = self.stdout_redirect
    
    def on_app_change(self):
        """Handle application type change"""
        app_type = self.app_type.get()
        if app_type == "msproject":
            if not HAS_MSPROJECT:
                messagebox.showwarning("Warning", "Microsoft Project detector is not available.")
                self.app_type.set("primavera")
                return
            
            # Set default model path
            model_path = os.path.join(SCRIPT_DIR, "msproject", "models", "msproject_model.pt")
            if os.path.exists(model_path):
                self.model_path.set(model_path)
            else:
                self.model_path.set("msproject_model.pt")
        else:  # primavera
            if not HAS_PRIMAVERA:
                messagebox.showwarning("Warning", "Primavera P6 detector is not available.")
                self.app_type.set("msproject")
                return
            
            # Set default model path
            model_path = os.path.join(SCRIPT_DIR, "primavera", "models", "primavera_model.pt")
            if os.path.exists(model_path):
                self.model_path.set(model_path)
            else:
                self.model_path.set("primavera_model.pt")
    
    def on_ocr_change(self):
        """Handle OCR setting change"""
        if self.use_ocr.get() and not HAS_OCR:
            messagebox.showwarning("Warning", "OCR utilities are not available.")
            self.use_ocr.set(False)
    
    def browse_model(self):
        """Browse for model file"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if file_path:
            self.model_path.set(file_path)
    
    def browse_tesseract(self):
        """Browse for Tesseract executable"""
        file_path = filedialog.askopenfilename(
            title="Select Tesseract Executable",
            filetypes=[("Executable", "*.exe"), ("All Files", "*.*")]
        )
        if file_path:
            self.tesseract_path.set(file_path)
    
    def initialize_detectors(self):
        """Initialize detectors"""
        try:
            # Initialize MS Project detector
            if HAS_MSPROJECT:
                self.msproject_detector = MSProjectDetector(
                    model_path=self.model_path.get() or None,
                    conf_thres=self.conf_thres.get(),
                    iou_thres=self.iou_thres.get(),
                    tesseract_path=self.tesseract_path.get() or None
                )
                logger.info("Microsoft Project detector initialized")
            
            # Initialize Primavera detector
            if HAS_PRIMAVERA:
                self.primavera_detector = PrimaveraDetector(
                    model_path=self.model_path.get() or None,
                    conf_thres=self.conf_thres.get(),
                    iou_thres=self.iou_thres.get(),
                    tesseract_path=self.tesseract_path.get() or None
                )
                logger.info("Primavera P6 detector initialized")
            
            self.status_var.set("Detectors initialized")
        except Exception as e:
            logger.error(f"Error initializing detectors: {e}", exc_info=True)
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Error", f"Failed to initialize detectors: {e}")
    
    def get_current_detector(self):
        """Get the current detector based on app type"""
        app_type = self.app_type.get()
        if app_type == "msproject":
            if not self.msproject_detector:
                self.initialize_detectors()
            return self.msproject_detector
        else:  # primavera
            if not self.primavera_detector:
                self.initialize_detectors()
            return self.primavera_detector
    
    def start_application(self):
        """Start the selected application"""
        detector = self.get_current_detector()
        if not detector:
            messagebox.showerror("Error", "Detector not initialized")
            return
        
        app_type = self.app_type.get()
        
        # Run in a separate thread to avoid freezing the UI
        def run_start():
            try:
                self.status_var.set(f"Starting {app_type}...")
                success = detector.start_application(app_type)
                if success:
                    self.status_var.set(f"{app_type} started successfully")
                else:
                    self.status_var.set(f"Failed to start {app_type}")
                    messagebox.showerror("Error", f"Failed to start {app_type}")
            except Exception as e:
                logger.error(f"Error starting application: {e}", exc_info=True)
                self.status_var.set(f"Error: {e}")
                messagebox.showerror("Error", f"Failed to start application: {e}")
        
        threading.Thread(target=run_start, daemon=True).start()
    
    def detect_elements(self):
        """Detect elements in the current application view"""
        detector = self.get_current_detector()
        if not detector:
            messagebox.showerror("Error", "Detector not initialized")
            return
        
        app_type = self.app_type.get()
        
        # Run in a separate thread to avoid freezing the UI
        def run_detect():
            try:
                self.status_var.set(f"Detecting elements in {app_type}...")
                
                # Update detector settings
                detector.conf_thres = self.conf_thres.get()
                detector.iou_thres = self.iou_thres.get()
                
                # Run detection
                if app_type == "msproject":
                    results = detector.detect_project_elements(with_ocr=self.use_ocr.get())
                else:  # primavera
                    results = detector.detect_primavera_elements(with_ocr=self.use_ocr.get())
                
                if results:
                    self.status_var.set(f"Detected {results['detection_count']} elements")
                    self.display_results(results)
                else:
                    self.status_var.set("No elements detected")
                    messagebox.showinfo("Info", "No elements detected")
            except Exception as e:
                logger.error(f"Error detecting elements: {e}", exc_info=True)
                self.status_var.set(f"Error: {e}")
                messagebox.showerror("Error", f"Failed to detect elements: {e}")
        
        threading.Thread(target=run_detect, daemon=True).start()
    
    def display_results(self, results):
        """Display detection results"""
        # Display annotated image
        if 'image_path' in results and os.path.exists(results['image_path']):
            self.display_image(results['image_path'])
        
        # Store results for later use
        self.current_results = results
    
    def display_image(self, image_path):
        """Display an image on the canvas"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Resize image to fit canvas
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            # Calculate scale factor
            scale_factor = min(canvas_width / image.width, canvas_height / image.height)
            if scale_factor > 1:
                scale_factor = 1  # Don't enlarge small images
            
            # Resize image
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            
            if new_width > 0 and new_height > 0:
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Clear canvas
            self.image_canvas.delete("all")
            
            # Display image
            self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.image_canvas.image = photo  # Keep a reference to prevent garbage collection
            
            # Configure scrollregion
            self.image_canvas.configure(scrollregion=(0, 0, new_width, new_height))
        except Exception as e:
            logger.error(f"Error displaying image: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to display image: {e}")
    
    def generate_report(self):
        """Generate a report from the detection results"""
        detector = self.get_current_detector()
        if not detector:
            messagebox.showerror("Error", "Detector not initialized")
            return
        
        app_type = self.app_type.get()
        
        # Run in a separate thread to avoid freezing the UI
        def run_report():
            try:
                self.status_var.set(f"Generating report for {app_type}...")
                
                # Generate report
                report = detector.generate_project_report()
                
                if report:
                    self.status_var.set("Report generated successfully")
                    self.display_report(report)
                else:
                    self.status_var.set("Failed to generate report")
                    messagebox.showerror("Error", "Failed to generate report")
            except Exception as e:
                logger.error(f"Error generating report: {e}", exc_info=True)
                self.status_var.set(f"Error: {e}")
                messagebox.showerror("Error", f"Failed to generate report: {e}")
        
        threading.Thread(target=run_report, daemon=True).start()
    
    def display_report(self, report):
        """Display a report in the report tab"""
        # Clear report text
        self.report_text.configure(state="normal")
        self.report_text.delete(1.0, tk.END)
        
        # Insert report
        self.report_text.insert(tk.END, report)
        
        # Disable editing
        self.report_text.configure(state="disabled")
    
    def save_results(self):
        """Save detection results to a file"""
        if not hasattr(self, 'current_results'):
            messagebox.showinfo("Info", "No detection results to save")
            return
        
        # Ask for file path
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            import json
            
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(i) for i in obj]
                else:
                    return obj
            
            # Save results to JSON file
            with open(file_path, 'w') as f:
                json.dump(convert_for_json(self.current_results), f, indent=2)
            
            self.status_var.set(f"Results saved to {file_path}")
            messagebox.showinfo("Success", f"Results saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}", exc_info=True)
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Error", f"Failed to save results: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Stop stdout redirect
        if hasattr(self, 'stdout_redirect'):
            self.stdout_redirect.stop()
        
        # Close the window
        self.root.destroy()

def main():
    """Main function"""
    # Create root window
    root = tk.Tk()
    
    # Create GUI
    app = PMIntegrationGUI(root)
    
    # Set up closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()
