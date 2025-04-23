#!/usr/bin/env python
"""
Verify Installation Script

This script verifies that the YOLOv5 Project Management Integration
can be run independently by checking for required dependencies and
directory structure.
"""

import os
import sys
import importlib
import platform

def check_dependency(module_name):
    """Check if a Python module is installed"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def check_directory(directory):
    """Check if a directory exists"""
    return os.path.isdir(directory)

def main():
    """Main verification function"""
    print("YOLOv5 Project Management Integration Verification")
    print("=" * 50)
    
    # Check Python version
    python_version = platform.python_version()
    print(f"Python version: {python_version}")
    if sys.version_info < (3, 8):
        print("WARNING: Python 3.8 or higher is recommended")
    
    # Check required directories
    directories = [
        "integrations",
        "integrations/msproject",
        "integrations/primavera",
        "integrations/ms_project_training",
        "integrations/database",
        "integrations/results",
        "integrations/sample_images"
    ]
    
    print("\nChecking directory structure:")
    for directory in directories:
        exists = check_directory(directory)
        status = "✓" if exists else "✗"
        print(f"  {status} {directory}")
    
    # Check core dependencies
    dependencies = [
        "torch",
        "numpy",
        "opencv-python",
        "PIL",
        "pyautogui",
        "pytesseract"
    ]
    
    print("\nChecking core dependencies:")
    for dependency in dependencies:
        module_name = dependency.replace("-", "_").split("[")[0]
        installed = check_dependency(module_name)
        status = "✓" if installed else "✗"
        print(f"  {status} {dependency}")
    
    # Check for Tesseract OCR
    if platform.system() == "Windows":
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
        tesseract_found = any(os.path.exists(path) for path in tesseract_paths)
    else:
        # On Linux/macOS, check if tesseract is in PATH
        import shutil
        tesseract_found = shutil.which("tesseract") is not None
    
    print("\nChecking Tesseract OCR:")
    status = "✓" if tesseract_found else "✗"
    print(f"  {status} Tesseract OCR")
    
    # Check for core files
    core_files = [
        "integrations/pm_integration.py",
        "integrations/ocr_utils.py",
        "integrations/multi_source_detector.py",
        "integrations/pm_gui.py",
        "integrations/README.md"
    ]
    
    print("\nChecking core files:")
    for file in core_files:
        exists = os.path.isfile(file)
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
    
    print("\nVerification complete!")

if __name__ == "__main__":
    main()
