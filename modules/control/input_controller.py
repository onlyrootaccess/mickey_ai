# Mouse/keyboard command handler
"""
M.I.C.K.E.Y. AI Assistant - Input Control System
Made In Crisis, Keeping Everything Yours

FIFTEENTH FILE IN PIPELINE: Advanced system control module for mouse, keyboard, 
browser automation, and application control with safety confirmations.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import platform
import subprocess
import os

# Import control libraries
import pyautogui
import pyperclip
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

# Import Mickey AI configuration
from config.settings import get_config
from config.constants import (
    SystemConstants, ErrorCodes, ErrorMessages,
    FeatureConstants, PersonalityConstants
)

# Setup logging
logger = logging.getLogger("MickeyControl")


class ControlType(Enum):
    """Types of control actions."""
    MOUSE = "mouse"
    KEYBOARD = "keyboard"
    BROWSER = "browser"
    SYSTEM = "system"
    APPLICATION = "application"


class SafetyLevel(Enum):
    """Safety levels for control actions."""
    SAFE = "safe"           # Read-only or very safe actions
    LOW_RISK = "low_risk"   # Minimal risk actions
    MEDIUM_RISK = "medium_risk"  # Potentially disruptive
    HIGH_RISK = "high_risk" # Destructive or critical actions


@dataclass
class ControlAction:
    """Control action container."""
    action_id: str
    control_type: ControlType
    action: str
    parameters: Dict[str, Any]
    safety_level: SafetyLevel
    requires_confirmation: bool = True
    confirmation_prompt: Optional[str] = None


class MouseController:
    """Advanced mouse control with safety measures."""
    
    def __init__(self):
        self.config = get_config()
        self.safety_enabled = True
        self.mouse_speed = FeatureConstants.MOUSE_MOVE_SPEEDS["normal"]
        
        # Safety boundaries (prevent moving off screen)
        self.screen_width, self.screen_height = pyautogui.size()
        self.safety_margin = 50
        
        # Movement history for undo functionality
        self.movement_history = []
        self.max_history = 10
        
    def set_mouse_speed(self, speed: str):
        """Set mouse movement speed."""
        self.mouse_speed = FeatureConstants.MOUSE_MOVE_SPEEDS.get(
            speed, FeatureConstants.MOUSE_MOVE_SPEEDS["normal"]
        )
        logger.info(f"Mouse speed set to: {speed} ({self.mouse_speed})")
    
    def move_to(self, x: int, y: int, duration: float = None):
        """Move mouse to coordinates with safety checks."""
        try:
            # Apply safety boundaries
            safe_x = max(self.safety_margin, min(x, self.screen_width - self.safety_margin))
            safe_y = max(self.safety_margin, min(y, self.screen_height - self.safety_margin))
            
            # Store current position for history
            current_x, current_y = pyautogui.position()
            self.movement_history.append(("move", current_x, current_y))
            if len(self.movement_history) > self.max_history:
                self.movement_history.pop(0)
            
            # Calculate movement duration based on speed
            if duration is None:
                distance = ((safe_x - current_x)**2 + (safe_y - current_y)**2)**0.5
                duration = distance / (1000 * self.mouse_speed)  # Normalized duration
            
            pyautogui.moveTo(safe_x, safe_y, duration=duration)
            logger.info(f"Mouse moved to: ({safe_x}, {safe_y}) in {duration:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Mouse movement failed: {str(e)}")
            return False
    
    def click(self, button: str = "left", clicks: int = 1):
        """Perform mouse click with safety checks."""
        try:
            if self.safety_enabled and clicks > 2:
                logger.warning(f"Safety: Limited to 2 clicks, requested {clicks}")
                clicks = min(clicks, 2)
            
            pyautogui.click(button=button, clicks=clicks)
            logger.info(f"Mouse {button} click performed ({clicks} clicks)")
            
            return True
            
        except Exception as e:
            logger.error(f"Mouse click failed: {str(e)}")
            return False
    
    def drag_to(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 1.0):
        """Perform drag operation with safety checks."""
        try:
            # Move to start position
            self.move_to(start_x, start_y, duration=0.1)
            
            # Apply safety boundaries to end position
            safe_end_x = max(self.safety_margin, min(end_x, self.screen_width - self.safety_margin))
            safe_end_y = max(self.safety_margin, min(end_y, self.screen_height - self.safety_margin))
            
            pyautogui.dragTo(safe_end_x, safe_end_y, duration=duration, button='left')
            logger.info(f"Drag from ({start_x}, {start_y}) to ({safe_end_x}, {safe_end_y})")
            
            return True
            
        except Exception as e:
            logger.error(f"Drag operation failed: {str(e)}")
            return False
    
    def scroll(self, clicks: int, direction: str = "down"):
        """Perform mouse scroll."""
        try:
            if direction == "up":
                clicks = abs(clicks)
            else:
                clicks = -abs(clicks)
            
            pyautogui.scroll(clicks)
            logger.info(f"Mouse scroll: {direction} {abs(clicks)} clicks")
            
            return True
            
        except Exception as e:
            logger.error(f"Mouse scroll failed: {str(e)}")
            return False
    
    def get_position(self) -> Tuple[int, int]:
        """Get current mouse position."""
        return pyautogui.position()
    
    def undo_last_movement(self) -> bool:
        """Undo last mouse movement."""
        if not self.movement_history:
            return False
        
        try:
            last_action = self.movement_history.pop()
            if last_action[0] == "move":
                _, x, y = last_action
                self.move_to(x, y)
                logger.info("Undid last mouse movement")
                return True
        except Exception as e:
            logger.error(f"Undo movement failed: {str(e)}")
        
        return False


class KeyboardController:
    """Advanced keyboard control with typing simulation."""
    
    def __init__(self):
        self.config = get_config()
        self.safety_enabled = True
        self.typing_delay = FeatureConstants.KEYBOARD_TYPING_DELAYS["normal"]
        
        # Dangerous key combinations to block
        self.dangerous_combinations = [
            ["ctrl", "alt", "delete"],
            ["alt", "f4"],
            ["ctrl", "w"],
            ["ctrl", "q"],
            ["command", "q"],  # Mac
            ["command", "w"],  # Mac
        ]
        
        # Clipboard history
        self.clipboard_history = []
        self.max_clipboard_history = 5
    
    def set_typing_speed(self, speed: str):
        """Set keyboard typing speed."""
        self.typing_delay = FeatureConstants.KEYBOARD_TYPING_DELAYS.get(
            speed, FeatureConstants.KEYBOARD_TYPING_DELAYS["normal"]
        )
        logger.info(f"Typing speed set to: {speed} ({self.typing_delay}s delay)")
    
    def type_text(self, text: str, delay: float = None):
        """Type text with simulated human typing."""
        try:
            if delay is None:
                delay = self.typing_delay
            
            # Safety check for very long text
            if len(text) > 1000 and self.safety_enabled:
                logger.warning("Safety: Text too long, truncating to 1000 characters")
                text = text[:1000]
            
            pyautogui.write(text, interval=delay)
            logger.info(f"Typed {len(text)} characters with {delay}s delay")
            
            return True
            
        except Exception as e:
            logger.error(f"Text typing failed: {str(e)}")
            return False
    
    def press_keys(self, keys: List[str], combination: bool = False):
        """Press keyboard keys with safety checks."""
        try:
            # Convert to lowercase for comparison
            key_list = [k.lower() for k in keys]
            
            # Check for dangerous combinations
            if self.safety_enabled and self._is_dangerous_combination(key_list):
                logger.error(f"Safety: Blocked dangerous key combination: {keys}")
                return False
            
            if combination:
                # Press combination (e.g., Ctrl+C)
                pyautogui.hotkey(*keys)
                logger.info(f"Pressed key combination: {'+'.join(keys)}")
            else:
                # Press keys sequentially
                for key in keys:
                    pyautogui.press(key)
                logger.info(f"Pressed keys sequentially: {keys}")
            
            return True
            
        except Exception as e:
            logger.error(f"Key press failed: {str(e)}")
            return False
    
    def _is_dangerous_combination(self, keys: List[str]) -> bool:
        """Check if key combination is dangerous."""
        for dangerous_combo in self.dangerous_combinations:
            if set(keys) == set(dangerous_combo):
                return True
        return False
    
    def copy_to_clipboard(self, text: str = None) -> bool:
        """Copy text to clipboard."""
        try:
            if text:
                pyperclip.copy(text)
                # Add to history
                self.clipboard_history.append(text)
                if len(self.clipboard_history) > self.max_clipboard_history:
                    self.clipboard_history.pop(0)
                logger.info(f"Copied {len(text)} characters to clipboard")
            else:
                # Copy selected text (Ctrl+C)
                self.press_keys(["ctrl", "c"], combination=True)
                # Small delay to ensure copy completes
                time.sleep(0.1)
                copied_text = pyperclip.paste()
                logger.info(f"Copied selected text to clipboard: {len(copied_text)} characters")
            
            return True
            
        except Exception as e:
            logger.error(f"Clipboard copy failed: {str(e)}")
            return False
    
    def paste_from_clipboard(self) -> bool:
        """Paste from clipboard."""
        try:
            self.press_keys(["ctrl", "v"], combination=True)
            logger.info("Pasted from clipboard")
            return True
            
        except Exception as e:
            logger.error(f"Clipboard paste failed: {str(e)}")
            return False
    
    def get_clipboard_history(self) -> List[str]:
        """Get clipboard history."""
        return self.clipboard_history.copy()


class BrowserController:
    """Web browser automation controller."""
    
    def __init__(self):
        self.config = get_config()
        self.driver = None
        self.current_url = None
        self.browser_actions = []
        
    async def initialize(self):
        """Initialize browser controller."""
        try:
            logger.info("Initializing Browser Controller...")
            
            # Set up Chrome options
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Initialize driver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.info("✅ Browser Controller initialized")
            
        except Exception as e:
            logger.error(f"❌ Browser Controller initialization failed: {str(e)}")
            raise
    
    def open_url(self, url: str) -> bool:
        """Open URL in browser."""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            self.driver.get(url)
            self.current_url = url
            self.browser_actions.append(("navigate", url))
            
            logger.info(f"Opened URL: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open URL {url}: {str(e)}")
            return False
    
    def find_element(self, selector: str, by: str = "css", timeout: int = 10) -> Any:
        """Find element on page."""
        try:
            wait = WebDriverWait(self.driver, timeout)
            
            if by == "css":
                element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            elif by == "xpath":
                element = wait.until(EC.presence_of_element_located((By.XPATH, selector)))
            elif by == "id":
                element = wait.until(EC.presence_of_element_located((By.ID, selector)))
            elif by == "name":
                element = wait.until(EC.presence_of_element_located((By.NAME, selector)))
            else:
                raise ValueError(f"Unsupported selector type: {by}")
            
            return element
            
        except Exception as e:
            logger.error(f"Element not found: {selector} by {by}: {str(e)}")
            return None
    
    def click_element(self, selector: str, by: str = "css") -> bool:
        """Click element on page."""
        try:
            element = self.find_element(selector, by)
            if element:
                element.click()
                self.browser_actions.append(("click", selector))
                logger.info(f"Clicked element: {selector}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to click element {selector}: {str(e)}")
            return False
    
    def type_in_element(self, selector: str, text: str, by: str = "css") -> bool:
        """Type text into input element."""
        try:
            element = self.find_element(selector, by)
            if element:
                element.clear()
                element.send_keys(text)
                self.browser_actions.append(("type", selector, text[:50]))  # Log first 50 chars
                logger.info(f"Typed in element {selector}: {len(text)} characters")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to type in element {selector}: {str(e)}")
            return False
    
    def get_page_text(self) -> str:
        """Get all text from current page."""
        try:
            return self.driver.find_element(By.TAG_NAME, "body").text
        except Exception as e:
            logger.error(f"Failed to get page text: {str(e)}")
            return ""
    
    def take_screenshot(self, save_path: str = None) -> bool:
        """Take screenshot of current page."""
        try:
            if save_path is None:
                save_path = f"screenshot_{int(time.time())}.png"
            
            self.driver.save_screenshot(save_path)
            logger.info(f"Screenshot saved: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to take screenshot: {str(e)}")
            return False
    
    def go_back(self) -> bool:
        """Navigate back in browser history."""
        try:
            self.driver.back()
            self.browser_actions.append(("navigate_back", ""))
            logger.info("Navigated back")
            return True
            
        except Exception as e:
            logger.error(f"Failed to navigate back: {str(e)}")
            return False
    
    def refresh(self) -> bool:
        """Refresh current page."""
        try:
            self.driver.refresh()
            self.browser_actions.append(("refresh", ""))
            logger.info("Page refreshed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh page: {str(e)}")
            return False
    
    def get_current_url(self) -> str:
        """Get current URL."""
        return self.current_url or (self.driver.current_url if self.driver else "")
    
    async def shutdown(self):
        """Shutdown browser controller."""
        try:
            if self.driver:
                self.driver.quit()
                logger.info("✅ Browser Controller shutdown complete")
                
        except Exception as e:
            logger.error(f"Error during browser controller shutdown: {str(e)}")


class SystemController:
    """System-level control and automation."""
    
    def __init__(self):
        self.config = get_config()
        self.system_type = platform.system().lower()
        
    def open_application(self, app_name: str) -> bool:
        """Open system application."""
        try:
            if self.system_type == "windows":
                # Windows
                if app_name.lower() in ["notepad", "notepad.exe"]:
                    subprocess.Popen(["notepad.exe"])
                elif app_name.lower() in ["calculator", "calc.exe"]:
                    subprocess.Popen(["calc.exe"])
                elif app_name.lower() in ["paint", "mspaint.exe"]:
                    subprocess.Popen(["mspaint.exe"])
                else:
                    # Try to open with default program
                    os.startfile(app_name)
                    
            elif self.system_type == "darwin":  # macOS
                if app_name.lower() in ["calculator"]:
                    subprocess.Popen(["open", "-a", "Calculator"])
                elif app_name.lower() in ["textedit"]:
                    subprocess.Popen(["open", "-a", "TextEdit"])
                else:
                    subprocess.Popen(["open", "-a", app_name])
                    
            elif self.system_type == "linux":
                # Linux - use xdg-open
                subprocess.Popen(["xdg-open", app_name])
            
            logger.info(f"Opened application: {app_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open application {app_name}: {str(e)}")
            return False
    
    def close_application(self, app_name: str) -> bool:
        """Close application (simulated - limited capability)."""
        try:
            # This is a simplified implementation
            # In practice, you'd need more sophisticated process management
            if self.system_type == "windows":
                subprocess.Popen(["taskkill", "/f", "/im", app_name])
            elif self.system_type in ["darwin", "linux"]:
                subprocess.Popen(["pkill", "-f", app_name])
            
            logger.info(f"Attempted to close application: {app_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close application {app_name}: {str(e)}")
            return False
    
    def create_file(self, file_path: str, content: str = "") -> bool:
        """Create a new file with content."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Created file: {file_path} ({len(content)} characters)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create file {file_path}: {str(e)}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            info = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "screen_size": pyautogui.size(),
                "mouse_position": pyautogui.position()
            }
            return info
            
        except Exception as e:
            logger.error(f"Failed to get system info: {str(e)}")
            return {}
    
    def execute_command(self, command: str) -> Tuple[bool, str]:
        """Execute system command with safety checks."""
        try:
            # Safety check for dangerous commands
            dangerous_commands = ["rm -rf", "format", "del /f", "shutdown", "restart"]
            if any(cmd in command.lower() for cmd in dangerous_commands):
                logger.error(f"Safety: Blocked dangerous command: {command}")
                return False, "Command blocked for safety reasons"
            
            # Execute command
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Command executed successfully: {command}")
                return True, result.stdout
            else:
                logger.error(f"Command failed: {command} - {result.stderr}")
                return False, result.stderr
                
        except Exception as e:
            logger.error(f"Command execution failed: {command} - {str(e)}")
            return False, str(e)


class InputController:
    """
    Main input controller that orchestrates all control systems.
    Provides unified interface for mouse, keyboard, browser, and system control.
    """
    
    def __init__(self):
        self.config = get_config()
        self.mouse_controller = MouseController()
        self.keyboard_controller = KeyboardController()
        self.browser_controller = BrowserController()
        self.system_controller = SystemController()
        self.is_initialized = False
        
        # Safety and confirmation system
        self.pending_confirmations = {}
        self.confirmation_timeout = 30  # seconds
        
        # Action history for undo functionality
        self.action_history = []
        self.max_action_history = 20
        
    async def initialize(self):
        """Initialize the input controller."""
        try:
            logger.info("Initializing Input Controller...")
            
            # Initialize browser controller
            if self.config.features.enable_browser_control:
                await self.browser_controller.initialize()
            
            self.is_initialized = True
            logger.info("✅ Input Controller initialized")
            
        except Exception as e:
            logger.error(f"❌ Input Controller initialization failed: {str(e)}")
            raise
    
    def _generate_action_id(self) -> str:
        """Generate unique action ID."""
        return f"action_{int(time.time())}_{len(self.action_history)}"
    
    def _assess_safety_level(self, control_type: ControlType, action: str, parameters: Dict) -> SafetyLevel:
        """Assess safety level of an action."""
        # High risk actions
        high_risk_actions = [
            ("system", "execute_command"),
            ("system", "close_application"),
            ("keyboard", "press_keys"),  # When used with dangerous combinations
        ]
        
        # Medium risk actions
        medium_risk_actions = [
            ("browser", "open_url"),
            ("browser", "click_element"),
            ("mouse", "drag_to"),
            ("system", "create_file"),
        ]
        
        action_key = (control_type.value, action)
        
        if action_key in high_risk_actions:
            return SafetyLevel.HIGH_RISK
        elif action_key in medium_risk_actions:
            return SafetyLevel.MEDIUM_RISK
        else:
            return SafetyLevel.LOW_RISK
    
    def _create_confirmation_prompt(self, action: ControlAction) -> str:
        """Create user-friendly confirmation prompt."""
        base_prompt = f"Mickey wants to perform: {action.action}"
        
        if action.control_type == ControlType.MOUSE:
            if "x" in action.parameters and "y" in action.parameters:
                base_prompt += f" at coordinates ({action.parameters['x']}, {action.parameters['y']})"
        
        elif action.control_type == ControlType.KEYBOARD:
            if "text" in action.parameters:
                text = action.parameters["text"]
                preview = text[:50] + "..." if len(text) > 50 else text
                base_prompt += f": '{preview}'"
            elif "keys" in action.parameters:
                base_prompt += f" keys: {action.parameters['keys']}"
        
        elif action.control_type == ControlType.BROWSER:
            if "url" in action.parameters:
                base_prompt += f" URL: {action.parameters['url']}"
            elif "selector" in action.parameters:
                base_prompt += f" element: {action.parameters['selector']}"
        
        elif action.control_type == ControlType.SYSTEM:
            if "app_name" in action.parameters:
                base_prompt += f" application: {action.parameters['app_name']}"
            elif "command" in action.parameters:
                base_prompt += f" command: {action.parameters['command']}"
        
        base_prompt += f"\nSafety Level: {action.safety_level.value.upper()}"
        base_prompt += "\n\nDo you want to proceed? (yes/no)"
        
        return base_prompt
    
    async def execute_command(self, control_type: ControlType, action: str, 
                            parameters: Dict[str, Any], require_confirmation: bool = True) -> Dict[str, Any]:
        """
        Execute control command with safety checks and optional confirmation.
        Main API method for the input controller.
        """
        try:
            if not self.is_initialized:
                return {
                    "success": False,
                    "message": "Input controller not initialized",
                    "action_id": None
                }
            
            # Assess safety level
            safety_level = self._assess_safety_level(control_type, action, parameters)
            
            # Create action object
            action_id = self._generate_action_id()
            control_action = ControlAction(
                action_id=action_id,
                control_type=control_type,
                action=action,
                parameters=parameters,
                safety_level=safety_level,
                requires_confirmation=require_confirmation and safety_level != SafetyLevel.SAFE
            )
            
            # Check if confirmation is required
            if control_action.requires_confirmation:
                confirmation_prompt = self._create_confirmation_prompt(control_action)
                
                # Store pending confirmation
                self.pending_confirmations[action_id] = {
                    "action": control_action,
                    "timestamp": time.time(),
                    "prompt": confirmation_prompt
                }
                
                logger.info(f"Action requires confirmation: {action_id}")
                
                return {
                    "success": True,
                    "requires_confirmation": True,
                    "confirmation_prompt": confirmation_prompt,
                    "action_id": action_id,
                    "safety_level": safety_level.value
                }
            
            # Execute immediately if no confirmation needed
            return await self._execute_action(control_action)
            
        except Exception as e:
            logger.error(f"Command execution failed: {str(e)}")
            return {
                "success": False,
                "message": f"Command execution failed: {str(e)}",
                "action_id": None
            }
    
    async def confirm_action(self, action_id: str, confirmed: bool) -> Dict[str, Any]:
        """Confirm or cancel a pending action."""
        try:
            if action_id not in self.pending_confirmations:
                return {
                    "success": False,
                    "message": "Action not found or expired"
                }
            
            action_data = self.pending_confirmations[action_id]
            
            # Check timeout
            if time.time() - action_data["timestamp"] > self.confirmation_timeout:
                del self.pending_confirmations[action_id]
                return {
                    "success": False,
                    "message": "Confirmation timeout"
                }
            
            if confirmed:
                # Execute the action
                result = await self._execute_action(action_data["action"])
                del self.pending_confirmations[action_id]
                return result
            else:
                # Cancel the action
                del self.pending_confirmations[action_id]
                logger.info(f"Action cancelled: {action_id}")
                return {
                    "success": True,
                    "message": "Action cancelled by user",
                    "action_id": action_id
                }
            
        except Exception as e:
            logger.error(f"Action confirmation failed: {str(e)}")
            return {
                "success": False,
                "message": f"Confirmation failed: {str(e)}"
            }
    
    async def _execute_action(self, action: ControlAction) -> Dict[str, Any]:
        """Execute a control action."""
        try:
            result = None
            
            if action.control_type == ControlType.MOUSE:
                result = await self._execute_mouse_action(action)
            elif action.control_type == ControlType.KEYBOARD:
                result = await self._execute_keyboard_action(action)
            elif action.control_type == ControlType.BROWSER:
                result = await self._execute_browser_action(action)
            elif action.control_type == ControlType.SYSTEM:
                result = await self._execute_system_action(action)
            else:
                result = {"success": False, "message": f"Unknown control type: {action.control_type}"}
            
            # Add to history
            if result.get("success", False):
                self.action_history.append({
                    "action_id": action.action_id,
                    "control_type": action.control_type.value,
                    "action": action.action,
                    "timestamp": time.time(),
                    "parameters": action.parameters
                })
                if len(self.action_history) > self.max_action_history:
                    self.action_history.pop(0)
            
            result["action_id"] = action.action_id
            return result
            
        except Exception as e:
            logger.error(f"Action execution failed: {str(e)}")
            return {
                "success": False,
                "message": f"Action execution failed: {str(e)}",
                "action_id": action.action_id
            }
    
    async def _execute_mouse_action(self, action: ControlAction) -> Dict[str, Any]:
        """Execute mouse action."""
        try:
            if action.action == "move_to":
                success = self.mouse_controller.move_to(
                    action.parameters.get("x", 0),
                    action.parameters.get("y", 0),
                    action.parameters.get("duration")
                )
            elif action.action == "click":
                success = self.mouse_controller.click(
                    action.parameters.get("button", "left"),
                    action.parameters.get("clicks", 1)
                )
            elif action.action == "drag_to":
                success = self.mouse_controller.drag_to(
                    action.parameters.get("start_x", 0),
                    action.parameters.get("start_y", 0),
                    action.parameters.get("end_x", 0),
                    action.parameters.get("end_y", 0),
                    action.parameters.get("duration", 1.0)
                )
            elif action.action == "scroll":
                success = self.mouse_controller.scroll(
                    action.parameters.get("clicks", 1),
                    action.parameters.get("direction", "down")
                )
            elif action.action == "get_position":
                position = self.mouse_controller.get_position()
                return {
                    "success": True,
                    "position": position,
                    "message": f"Mouse position: {position}"
                }
            else:
                return {"success": False, "message": f"Unknown mouse action: {action.action}"}
            
            return {
                "success": success,
                "message": f"Mouse action '{action.action}' {'completed' if success else 'failed'}"
            }
            
        except Exception as e:
            return {"success": False, "message": f"Mouse action failed: {str(e)}"}
    
    async def _execute_keyboard_action(self, action: ControlAction) -> Dict[str, Any]:
        """Execute keyboard action."""
        try:
            if action.action == "type_text":
                success = self.keyboard_controller.type_text(
                    action.parameters.get("text", ""),
                    action.parameters.get("delay")
                )
            elif action.action == "press_keys":
                success = self.keyboard_controller.press_keys(
                    action.parameters.get("keys", []),
                    action.parameters.get("combination", False)
                )
            elif action.action == "copy_to_clipboard":
                success = self.keyboard_controller.copy_to_clipboard(
                    action.parameters.get("text")
                )
            elif action.action == "paste_from_clipboard":
                success = self.keyboard_controller.paste_from_clipboard()
            else:
                return {"success": False, "message": f"Unknown keyboard action: {action.action}"}
            
            return {
                "success": success,
                "message": f"Keyboard action '{action.action}' {'completed' if success else 'failed'}"
            }
            
        except Exception as e:
            return {"success": False, "message": f"Keyboard action failed: {str(e)}"}
    
    async def _execute_browser_action(self, action: ControlAction) -> Dict[str, Any]:
        """Execute browser action."""
        try:
            if action.action == "open_url":
                success = self.browser_controller.open_url(
                    action.parameters.get("url", "")
                )
            elif action.action == "click_element":
                success = self.browser_controller.click_element(
                    action.parameters.get("selector", ""),
                    action.parameters.get("by", "css")
                )
            elif action.action == "type_in_element":
                success = self.browser_controller.type_in_element(
                    action.parameters.get("selector", ""),
                    action.parameters.get("text", ""),
                    action.parameters.get("by", "css")
                )
            elif action.action == "get_page_text":
                text = self.browser_controller.get_page_text()
                return {
                    "success": True,
                    "text": text,
                    "message": f"Retrieved {len(text)} characters from page"
                }
            elif action.action == "get_current_url":
                url = self.browser_controller.get_current_url()
                return {
                    "success": True,
                    "url": url,
                    "message": f"Current URL: {url}"
                }
            else:
                return {"success": False, "message": f"Unknown browser action: {action.action}"}
            
            return {
                "success": success,
                "message": f"Browser action '{action.action}' {'completed' if success else 'failed'}"
            }
            
        except Exception as e:
            return {"success": False, "message": f"Browser action failed: {str(e)}"}
    
    async def _execute_system_action(self, action: ControlAction) -> Dict[str, Any]:
        """Execute system action."""
        try:
            if action.action == "open_application":
                success = self.system_controller.open_application(
                    action.parameters.get("app_name", "")
                )
            elif action.action == "close_application":
                success = self.system_controller.close_application(
                    action.parameters.get("app_name", "")
                )
            elif action.action == "create_file":
                success = self.system_controller.create_file(
                    action.parameters.get("file_path", ""),
                    action.parameters.get("content", "")
                )
            elif action.action == "get_system_info":
                info = self.system_controller.get_system_info()
                return {
                    "success": True,
                    "system_info": info,
                    "message": "System information retrieved"
                }
            elif action.action == "execute_command":
                success, output = self.system_controller.execute_command(
                    action.parameters.get("command", "")
                )
                return {
                    "success": success,
                    "output": output,
                    "message": f"Command {'executed' if success else 'failed'}"
                }
            else:
                return {"success": False, "message": f"Unknown system action: {action.action}"}
            
            return {
                "success": success,
                "message": f"System action '{action.action}' {'completed' if success else 'failed'}"
            }
            
        except Exception as e:
            return {"success": False, "message": f"System action failed: {str(e)}"}
    
    def get_action_history(self, limit: int = 10) -> List[Dict]:
        """Get action history."""
        return self.action_history[-limit:] if self.action_history else []
    
    def get_pending_confirmations(self) -> List[Dict]:
        """Get pending confirmations."""
        return [
            {
                "action_id": action_id,
                "prompt": data["prompt"],
                "timestamp": data["timestamp"]
            }
            for action_id, data in self.pending_confirmations.items()
        ]
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get input controller performance metrics."""
        return {
            "initialized": self.is_initialized,
            "total_actions": len(self.action_history),
            "pending_confirmations": len(self.pending_confirmations),
            "mouse_history": len(self.mouse_controller.movement_history),
            "clipboard_history": len(self.keyboard_controller.clipboard_history),
            "browser_actions": len(self.browser_controller.browser_actions)
        }
    
    async def shutdown(self):
        """Shutdown input controller gracefully."""
        logger.info("Shutting down Input Controller...")
        
        try:
            await self.browser_controller.shutdown()
            logger.info("✅ Input Controller shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during input controller shutdown: {str(e)}")


# Global input controller instance
_input_controller_instance: Optional[InputController] = None


async def get_input_controller() -> InputController:
    """Get or create global input controller instance."""
    global _input_controller_instance
    
    if _input_controller_instance is None:
        _input_controller_instance = InputController()
        await _input_controller_instance.initialize()
    
    return _input_controller_instance


async def main():
    """Command-line testing for input controller."""
    input_controller = await get_input_controller()
    
    # Test performance metrics
    metrics = await input_controller.get_performance_metrics()
    print("Input Controller Status:")
    print(f"Initialized: {metrics['initialized']}")
    print(f"Total Actions: {metrics['total_actions']}")
    print(f"Pending Confirmations: {metrics['pending_confirmations']}")
    
    # Test mouse position
    position = input_controller.mouse_controller.get_position()
    print(f"Current Mouse Position: {position}")
    
    print("\nInput controller ready for commands.")
    print("Use execute_command() method to control system.")


if __name__ == "__main__":
    asyncio.run(main())