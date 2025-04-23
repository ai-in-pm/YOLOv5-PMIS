import os
import sys
import time
import random
import pyautogui
import subprocess
from datetime import datetime
import win32gui
import win32process
import win32con
import psutil
import argparse
import json
from pathlib import Path
import shutil

# Add root path to use shared utilities
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_path)

class AdvancedMSProjectCollector:
    """Advanced screenshot collector for Microsoft Project that captures diverse views and templates"""
    
    def __init__(self, output_dir=None, ms_project_path=None):
        self.ms_project_path = ms_project_path or r"C:\Program Files\Microsoft Office\root\Office16\WINPROJ.EXE"
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), 'data', 'advanced_screenshots')
        self.template_dir = os.path.join(self.output_dir, 'templates')
        self.view_dir = os.path.join(self.output_dir, 'views')
        self.ms_project_hwnd = None
        self.ms_project_pid = None
        
        # Create output directories
        for dir_path in [self.output_dir, self.template_dir, self.view_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Different MS Project views to capture
        self.views = {
            'gantt_chart': {'menu': ['View', 'Gantt Chart'], 'key': 'g'},
            'calendar': {'menu': ['View', 'Calendar'], 'key': 'c'},
            'network_diagram': {'menu': ['View', 'Network Diagram'], 'key': 'n'},
            'task_usage': {'menu': ['View', 'Task Usage'], 'key': 't'},
            'resource_sheet': {'menu': ['View', 'Resource Sheet'], 'key': 'r'},
            'resource_usage': {'menu': ['View', 'Resource Usage'], 'key': 'u'},
            'team_planner': {'menu': ['View', 'Team Planner'], 'key': 'p'},
            'timeline': {'menu': ['View', 'Timeline'], 'key': 'l'},
            'tracking_gantt': {'menu': ['View', 'Tracking Gantt'], 'key': 'k'},
            'resource_graph': {'menu': ['View', 'Resource Graph'], 'key': 'g'}
        }
        
        # Sample project templates to open
        self.templates = [
            "New Product Development",
            "Residential Construction",
            "Software Development",
            "Business Process Improvement",
            "Marketing Campaign",
            "Event Planning",
            "Research Project",
            "IT Implementation",
            "Agile Project",
            "Manufacturing Process"
        ]
        
        print(f"Advanced screenshot collector initialized")
        print(f"Screenshots will be saved to: {self.output_dir}")
    
    def start_ms_project(self):
        """Start Microsoft Project application"""
        if not os.path.exists(self.ms_project_path):
            print(f"Microsoft Project not found at: {self.ms_project_path}")
            print("Searching for Microsoft Project...")
            
            # Try common installation paths
            common_paths = [
                r"C:\Program Files\Microsoft Office\root\Office16\WINPROJ.EXE",
                r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINPROJ.EXE",
                r"C:\Program Files\Microsoft Office\Office16\WINPROJ.EXE",
                r"C:\Program Files (x86)\Microsoft Office\Office16\WINPROJ.EXE"
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    self.ms_project_path = path
                    print(f"Found Microsoft Project at: {path}")
                    break
            
            if not os.path.exists(self.ms_project_path):
                print("Microsoft Project not found. Please specify the correct path.")
                return False
        
        print(f"Starting Microsoft Project from {self.ms_project_path}...")
        try:
            # Start Microsoft Project
            process = subprocess.Popen(self.ms_project_path)
            self.ms_project_pid = process.pid
            
            # Wait for application to start
            print("Waiting for Microsoft Project to start...")
            time.sleep(5)  # Allow MS Project to initialize
            
            # Find the MS Project window
            self._find_ms_project_window()
            
            return True
        except Exception as e:
            print(f"Error starting Microsoft Project: {e}")
            return False
    
    def _find_ms_project_window(self):
        """Find the Microsoft Project window handle"""
        def enum_windows_callback(hwnd, results):
            if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                process = psutil.Process(pid)
                if 'WINPROJ.EXE' in process.name().upper() or 'PROJECT.EXE' in process.name().upper():
                    window_text = win32gui.GetWindowText(hwnd)
                    if 'Project' in window_text:
                        results.append((hwnd, pid))
            return True
        
        results = []
        win32gui.EnumWindows(enum_windows_callback, results)
        
        if results:
            self.ms_project_hwnd, self.ms_project_pid = results[0]
            print(f"Found Microsoft Project window: {win32gui.GetWindowText(self.ms_project_hwnd)}")
            
            # Bring window to foreground and maximize
            self._activate_window()
            return True
        else:
            print("Could not find Microsoft Project window")
            return False
    
    def _activate_window(self):
        """Activate and maximize the Microsoft Project window"""
        if self.ms_project_hwnd:
            # Bring to foreground
            win32gui.SetForegroundWindow(self.ms_project_hwnd)
            
            # Maximize
            win32gui.ShowWindow(self.ms_project_hwnd, win32con.SW_MAXIMIZE)
            time.sleep(1)  # Allow UI to settle
    
    def create_new_project_from_template(self, template_name=None):
        """Create a new project from a template"""
        self._activate_window()
        
        # Press Alt+F to open File menu
        pyautogui.hotkey('alt', 'f')
        time.sleep(0.5)
        
        # Press N for New
        pyautogui.press('n')
        time.sleep(2)  # Wait for templates screen
        
        if template_name is None:
            template_name = random.choice(self.templates)
        
        print(f"Creating new project from template: {template_name}")
        
        # Type the template name to search
        pyautogui.write(template_name)
        time.sleep(1)
        
        # Press Enter to select the first matching template
        pyautogui.press('enter')
        time.sleep(5)  # Wait for template to load
        
        return True
    
    def switch_view(self, view_name):
        """Switch Microsoft Project view"""
        if view_name not in self.views:
            print(f"Unknown view: {view_name}. Available views: {list(self.views.keys())}")
            return False
        
        self._activate_window()
        
        print(f"Switching to {view_name} view...")
        
        # Press Alt+V to open View menu
        pyautogui.hotkey('alt', 'v')
        time.sleep(0.5)
        
        # Press the key associated with the view
        view_key = self.views[view_name]['key']
        pyautogui.press(view_key)
        time.sleep(2)  # Wait for view to change
        
        return True
    
    def capture_screenshot(self, prefix='msproject', subfolder=None):
        """Capture a screenshot of the Microsoft Project window"""
        self._activate_window()
        time.sleep(0.5)  # Ensure window is active
        
        try:
            # Capture the screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.jpg"
            
            # Use subfolder if specified
            if subfolder:
                save_dir = os.path.join(self.output_dir, subfolder)
                os.makedirs(save_dir, exist_ok=True)
            else:
                save_dir = self.output_dir
                
            filepath = os.path.join(save_dir, filename)
            
            # Get window coordinates
            if self.ms_project_hwnd:
                # Get window rect
                rect = win32gui.GetWindowRect(self.ms_project_hwnd)
                x, y, width, height = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
                
                # Capture the specific window
                screenshot = pyautogui.screenshot(region=(x, y, width, height))
            else:
                # Fallback to full screen if window handle not available
                screenshot = pyautogui.screenshot()
                
            screenshot.save(filepath)
            print(f"Screenshot saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None
    
    def navigate_project(self, steps=5):
        """Navigate around the project to get diverse views"""
        self._activate_window()
        
        # Define possible navigation actions
        actions = [
            lambda: pyautogui.press('down', presses=random.randint(1, 5)),  # Scroll down
            lambda: pyautogui.press('up', presses=random.randint(1, 5)),     # Scroll up
            lambda: pyautogui.press('right', presses=random.randint(1, 3)),  # Move right
            lambda: pyautogui.press('left', presses=random.randint(1, 3)),   # Move left
            lambda: pyautogui.press('page_down'),                             # Page down
            lambda: pyautogui.press('page_up'),                               # Page up
            lambda: pyautogui.press('home'),                                  # Go to start
            lambda: pyautogui.press('end'),                                   # Go to end
            lambda: self.zoom_in(),                                           # Zoom in
            lambda: self.zoom_out(),                                          # Zoom out
            lambda: self.expand_collapse_tasks()                              # Expand/collapse tasks
        ]
        
        print(f"Navigating project with {steps} random movements...")
        for i in range(steps):
            action = random.choice(actions)
            action()
            time.sleep(0.5)
    
    def zoom_in(self):
        """Zoom in using Ctrl+Plus"""
        pyautogui.hotkey('ctrl', '+')
    
    def zoom_out(self):
        """Zoom out using Ctrl+Minus"""
        pyautogui.hotkey('ctrl', '-')
    
    def expand_collapse_tasks(self):
        """Randomly expand or collapse tasks"""
        # Press Alt+Shift+- to collapse or Alt+Shift++ to expand
        if random.choice([True, False]):
            pyautogui.hotkey('alt', 'shift', '-')  # Collapse
        else:
            pyautogui.hotkey('alt', 'shift', '+')  # Expand
    
    def collect_template_screenshots(self, count_per_template=2):
        """Collect screenshots across different project templates"""
        if not self._find_ms_project_window():
            if not self.start_ms_project():
                print("Failed to start Microsoft Project. Cannot collect screenshots.")
                return []
        
        screenshots = []
        
        for template in self.templates:
            print(f"\nCollecting screenshots for template: {template}")
            
            # Create a new project from this template
            self.create_new_project_from_template(template)
            
            # Capture screenshots in different views
            for i in range(count_per_template):
                # Select a random view
                view = random.choice(list(self.views.keys()))
                self.switch_view(view)
                
                # Navigate around to get diverse content
                self.navigate_project(steps=random.randint(3, 7))
                
                # Capture and save the screenshot
                prefix = f"{template.replace(' ', '_')}_{view}"
                screenshot_path = self.capture_screenshot(prefix=prefix, subfolder='templates')
                
                if screenshot_path:
                    screenshots.append(screenshot_path)
                
                # Small delay between captures
                time.sleep(1)
        
        return screenshots
    
    def collect_view_screenshots(self, count_per_view=5):
        """Collect screenshots across different project views"""
        if not self._find_ms_project_window():
            if not self.start_ms_project():
                print("Failed to start Microsoft Project. Cannot collect screenshots.")
                return []
        
        screenshots = []
        
        # Create a new project with random template
        self.create_new_project_from_template()
        
        for view_name in self.views.keys():
            print(f"\nCollecting screenshots for view: {view_name}")
            
            # Switch to this view
            self.switch_view(view_name)
            
            # Capture multiple screenshots with different navigation
            for i in range(count_per_view):
                # Navigate around to get diverse content
                self.navigate_project(steps=random.randint(3, 7))
                
                # Capture and save the screenshot
                prefix = f"{view_name}"
                screenshot_path = self.capture_screenshot(prefix=prefix, subfolder='views')
                
                if screenshot_path:
                    screenshots.append(screenshot_path)
                
                # Small delay between captures
                time.sleep(1)
        
        return screenshots
    
    def collect_diverse_screenshots(self, template_count=2, view_count=3):
        """Collect diverse screenshots across templates and views"""
        template_screenshots = self.collect_template_screenshots(count_per_template=template_count)
        view_screenshots = self.collect_view_screenshots(count_per_view=view_count)
        
        all_screenshots = template_screenshots + view_screenshots
        
        print(f"\nCollection complete! Captured {len(all_screenshots)} diverse screenshots:")
        print(f"- {len(template_screenshots)} template-specific screenshots")
        print(f"- {len(view_screenshots)} view-specific screenshots")
        
        return all_screenshots
    
    def close_ms_project(self):
        """Close Microsoft Project"""
        if self.ms_project_hwnd:
            print("Closing Microsoft Project...")
            win32gui.PostMessage(self.ms_project_hwnd, win32con.WM_CLOSE, 0, 0)
            time.sleep(1)
            
            # Check if a save dialog appears and handle it
            def find_save_dialog(hwnd, context):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if "Save" in title or "Microsoft Project" in title:
                        context.append(hwnd)
                return True
            
            dialog_hwnd = []
            win32gui.EnumWindows(find_save_dialog, dialog_hwnd)
            
            if dialog_hwnd:
                # Press Don't Save (Alt+N in English Windows)
                pyautogui.hotkey('alt', 'n')
            
            return True
        return False

def main():
    parser = argparse.ArgumentParser(description="Advanced Microsoft Project screenshot collector")
    parser.add_argument('--output', type=str, default=None, help='Output directory for screenshots')
    parser.add_argument('--templates', type=int, default=2, help='Screenshots per template')
    parser.add_argument('--views', type=int, default=3, help='Screenshots per view')
    
    args = parser.parse_args()
    
    collector = AdvancedMSProjectCollector(output_dir=args.output)
    
    try:
        print("Starting advanced screenshot collection for Microsoft Project...")
        screenshots = collector.collect_diverse_screenshots(
            template_count=args.templates,
            view_count=args.views
        )
        
        print(f"\nCollection complete! Captured {len(screenshots)} screenshots total.")
            
    except KeyboardInterrupt:
        print("\nScreenshot collection interrupted.")
    finally:
        collector.close_ms_project()

if __name__ == "__main__":
    main()
