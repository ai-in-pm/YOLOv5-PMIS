import os
import sys
import subprocess

def launch_microsoft_project():
    """Launch Microsoft Project application"""
    # Common installation paths for Microsoft Project
    possible_paths = [
        r"C:\Program Files\Microsoft Office\root\Office16\WINPROJ.EXE",
        r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINPROJ.EXE",
        r"C:\Program Files\Microsoft Office\Office16\WINPROJ.EXE",
        r"C:\Program Files (x86)\Microsoft Office\Office16\WINPROJ.EXE"
    ]
    
    # Try each path
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found Microsoft Project at: {path}")
            print("Launching Microsoft Project...")
            subprocess.Popen(path)
            return True
    
    # If not found in common locations, try using the Windows Start command
    try:
        print("Attempting to launch Microsoft Project using Windows Start command...")
        subprocess.Popen(['start', 'msproject'], shell=True)
        return True
    except Exception as e:
        print(f"Error launching Microsoft Project: {e}")
        print("Microsoft Project may not be installed on this system.")
        return False

if __name__ == "__main__":
    print("YOLOv5 Microsoft Project Launcher")
    print("=================================")
    launch_microsoft_project()
