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

# Add root path to use shared utilities
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_path)

class MSProjectScreenCapture:
    """Automated screenshot collector for Microsoft Project"""
    
    def __init__(self, output_dir=None, ms_project_path=None):
        self.ms_project_path = ms_project_path or r"C:\Program Files\Microsoft Office\root\Office16\WINPROJ.EXE"
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), 'data', 'screenshots')
        self.ms_project_hwnd = None
        self.ms_project_pid = None
        self.view_commands = {
            'gantt_chart': {'menu': ['View', 'Gantt Chart']},
            'calendar': {'menu': ['View', 'Calendar']},
            'network_diagram': {'menu': ['View', 'Network Diagram']},
            'task_usage': {'menu': ['View', 'Task Usage']},
            'resource_sheet': {'menu': ['View', 'Resource Sheet']},
            'resource_usage': {'menu': ['View', 'Resource Usage']},
            'team_planner': {'menu': ['View', 'Team Planner']},
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
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
    
    def open_sample_project(self, project_path=None):
        """Open a sample project file"""
        self._activate_window()
        
        if project_path and os.path.exists(project_path):
            print(f"Opening project file: {project_path}")
            # Simulate keyboard shortuct Ctrl+O
            pyautogui.hotkey('ctrl', 'o')
            time.sleep(1)
            
            # Type the path and press Enter
            pyautogui.write(project_path)
            pyautogui.press('enter')
            time.sleep(3)  # Wait for project to load
            return True
        else:
            # If no project specified, open a new project
            print("Creating a new project")
            pyautogui.hotkey('ctrl', 'n')
            time.sleep(2)
            return True
    
    def switch_view(self, view_name):
        """Switch Microsoft Project view (Gantt Chart, Calendar, etc.)"""
        if view_name not in self.view_commands:
            print(f"Unknown view: {view_name}. Available views: {list(self.view_commands.keys())}")
            return False
        
        self._activate_window()
        
        # Click on View menu
        print(f"Switching to {view_name} view...")
        menu_path = self.view_commands[view_name]['menu']
        
        # First click on the first menu item (usually 'View')
        pyautogui.hotkey('alt')
        time.sleep(0.5)
        pyautogui.press(menu_path[0][0].lower())  # First letter of first menu item
        time.sleep(0.5)
        
        # Now find and click the submenu item
        target_text = menu_path[1]
        
        # For simplicity, we'll use the keyboard to navigate the menu
        # In a real implementation, you'd want to locate the menu item by text recognition
        view_mapping = {
            'Gantt Chart': 'g',
            'Calendar': 'c',
            'Network Diagram': 'n',
            'Task Usage': 't',
            'Resource Sheet': 'r',
            'Resource Usage': 'u',
            'Team Planner': 'p'
        }
        
        if target_text in view_mapping:
            pyautogui.press(view_mapping[target_text])
            time.sleep(2)  # Wait for view to change
            return True
        else:
            print(f"Unknown view submenu: {target_text}")
            return False
    
    def capture_screenshot(self, prefix='ms_project'):
        """Capture a screenshot of the Microsoft Project window"""
        self._activate_window()
        time.sleep(0.5)  # Ensure window is active
        
        try:
            # Capture the screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
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
        """Navigate around the project to get diverse views for screenshots"""
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
    
    def collect_training_screenshots(self, count=10, views=None, navigate_between=True):
        """Collect a batch of diverse screenshots for training"""
        if not self._find_ms_project_window():
            if not self.start_ms_project():
                print("Failed to start Microsoft Project. Cannot collect screenshots.")
                return []
        
        if views is None:
            views = list(self.view_commands.keys())
        
        screenshots = []
        
        print(f"Collecting {count} screenshots across {len(views)} views...")
        
        for i in range(count):
            # Select a random view occasionally
            if i % 3 == 0 or i == 0:
                view = random.choice(views)
                self.switch_view(view)
            
            # Navigate around to get diverse content
            if navigate_between:
                self.navigate_project(steps=random.randint(3, 7))
            
            # Capture and save the screenshot
            prefix = f"ms_project_{view.replace('_', '-')}"
            screenshot_path = self.capture_screenshot(prefix=prefix)
            
            if screenshot_path:
                screenshots.append(screenshot_path)
            
            # Small delay between captures
            time.sleep(1)
        
        return screenshots
    
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
    parser = argparse.ArgumentParser(description="Collect Microsoft Project screenshots for training")
    parser.add_argument('--count', type=int, default=20, help='Number of screenshots to collect')
    parser.add_argument('--output', type=str, default=None, help='Output directory for screenshots')
    parser.add_argument('--project', type=str, default=None, help='Path to a Microsoft Project file to open')
    parser.add_argument('--views', type=str, nargs='+', default=None, 
                        choices=['gantt_chart', 'calendar', 'network_diagram', 'task_usage', 
                                'resource_sheet', 'resource_usage', 'team_planner'],
                        help='Specific views to capture')
    
    args = parser.parse_args()
    
    capture_tool = MSProjectScreenCapture(output_dir=args.output)
    
    try:
        capture_tool.start_ms_project()
        capture_tool.open_sample_project(args.project)
        screenshots = capture_tool.collect_training_screenshots(count=args.count, views=args.views)
        
        print(f"\nCollection complete! Captured {len(screenshots)} screenshots:")
        for path in screenshots:
            print(f"  - {path}")
            
    except KeyboardInterrupt:
        print("\nScreenshot collection interrupted.")
    finally:
        capture_tool.close_ms_project()

if __name__ == "__main__":
    main()
