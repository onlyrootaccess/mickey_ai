# OS-level operations
"""
Mickey AI - System Integration
OS-level operations and system control for Windows, macOS, and Linux
"""

import logging
import platform
import os
import subprocess
import shutil
import psutil
import time
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import threading

class SystemType(Enum):
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    UNKNOWN = "unknown"

class SystemOperation(Enum):
    SHUTDOWN = "shutdown"
    RESTART = "restart"
    SLEEP = "sleep"
    LOCK = "lock"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    MUTE = "mute"
    UNMUTE = "unmute"
    BRIGHTNESS_UP = "brightness_up"
    BRIGHTNESS_DOWN = "brightness_down"

class SystemIntegration:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Detect operating system
        self.system_type = self._detect_system()
        self.system_info = self._get_system_info()
        
        # System commands based on OS
        self.system_commands = self._get_system_commands()
        
        # Safety confirmation required for critical operations
        self.safety_confirm_required = True
        self.confirmation_callbacks = []
        
        # Performance monitoring
        self.performance_monitor = None
        self.is_monitoring = False
        
        # Mickey's system messages
        self.system_messages = {
            SystemOperation.SHUTDOWN: [
                "Mickey's putting the system to sleep! Goodnight! 💤",
                "Shutting down! See you soon! 👋",
                "System shutdown initiated! Mickey's taking a break! 🐭"
            ],
            SystemOperation.RESTART: [
                "Restarting! Mickey will be back in a flash! ⚡",
                "Rebooting the system! Brb! 🔄",
                "Restart initiated! Fresh start coming up! 🌟"
            ],
            SystemOperation.LOCK: [
                "Locking the system! Mickey's keeping it safe! 🔒",
                "Screen locked! Your secrets are safe with Mickey! 🤫",
                "System locked! Time for a coffee break! ☕"
            ],
            SystemOperation.VOLUME_UP: [
                "Volume increased! Let's pump up the jam! 🔊",
                "Turning up the volume! Mickey's got the beats! 🎵",
                "Volume up! Can you hear me now? 🗣️"
            ]
        }
        
        self.logger.info(f"💻 System Integration initialized for {self.system_type.value}")

    def _detect_system(self) -> SystemType:
        """Detect the current operating system"""
        system = platform.system().lower()
        
        if system == "windows":
            return SystemType.WINDOWS
        elif system == "darwin":
            return SystemType.MACOS
        elif system == "linux":
            return SystemType.LINUX
        else:
            return SystemType.UNKNOWN

    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory information
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk information
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network information
            net_io = psutil.net_io_counters()
            
            # Battery information (if available)
            try:
                battery = psutil.sensors_battery()
                battery_info = {
                    'percent': battery.percent,
                    'power_plugged': battery.power_plugged,
                    'time_left': battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None
                }
            except:
                battery_info = None
            
            return {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.architecture()[0],
                'cpu': {
                    'usage_percent': cpu_percent,
                    'core_count': cpu_count,
                    'frequency': cpu_freq.current if cpu_freq else None
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'percent': memory.percent
                },
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'percent': swap.percent
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                },
                'network': {
                    'bytes_sent': net_io.bytes_sent if net_io else 0,
                    'bytes_recv': net_io.bytes_recv if net_io else 0
                },
                'battery': battery_info
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system info: {str(e)}")
            return {}

    def _get_system_commands(self) -> Dict[SystemOperation, str]:
        """Get system-specific commands for operations"""
        commands = {}
        
        if self.system_type == SystemType.WINDOWS:
            commands = {
                SystemOperation.SHUTDOWN: "shutdown /s /t 0",
                SystemOperation.RESTART: "shutdown /r /t 0",
                SystemOperation.SLEEP: "rundll32.exe powrprof.dll,SetSuspendState 0,1,0",
                SystemOperation.LOCK: "rundll32.exe user32.dll,LockWorkStation",
                SystemOperation.VOLUME_UP: "nircmd.exe changesysvolume 2000",
                SystemOperation.VOLUME_DOWN: "nircmd.exe changesysvolume -2000",
                SystemOperation.MUTE: "nircmd.exe mutesysvolume 1",
                SystemOperation.UNMUTE: "nircmd.exe mutesysvolume 0"
            }
        elif self.system_type == SystemType.MACOS:
            commands = {
                SystemOperation.SHUTDOWN: "osascript -e 'tell application \"System Events\" to shut down'",
                SystemOperation.RESTART: "osascript -e 'tell application \"System Events\" to restart'",
                SystemOperation.SLEEP: "pmset sleepnow",
                SystemOperation.LOCK: "pmset displaysleepnow",
                SystemOperation.VOLUME_UP: "osascript -e 'set volume output volume (output volume of (get volume settings) + 10)'",
                SystemOperation.VOLUME_DOWN: "osascript -e 'set volume output volume (output volume of (get volume settings) - 10)'",
                SystemOperation.MUTE: "osascript -e 'set volume output muted true'",
                SystemOperation.UNMUTE: "osascript -e 'set volume output muted false'"
            }
        elif self.system_type == SystemType.LINUX:
            commands = {
                SystemOperation.SHUTDOWN: "systemctl poweroff",
                SystemOperation.RESTART: "systemctl reboot",
                SystemOperation.SLEEP: "systemctl suspend",
                SystemOperation.LOCK: "gnome-screensaver-command -l",  # GNOME specific
                SystemOperation.VOLUME_UP: "pactl set-sink-volume @DEFAULT_SINK@ +10%",
                SystemOperation.VOLUME_DOWN: "pactl set-sink-volume @DEFAULT_SINK@ -10%",
                SystemOperation.MUTE: "pactl set-sink-mute @DEFAULT_SINK@ 1",
                SystemOperation.UNMUTE: "pactl set-sink-mute @DEFAULT_SINK@ 0"
            }
        
        return commands

    def execute_system_operation(self, operation: SystemOperation, 
                               require_confirmation: bool = True) -> Dict[str, Any]:
        """
        Execute a system operation
        
        Args:
            operation: System operation to perform
            require_confirmation: Whether to require safety confirmation
            
        Returns:
            Dictionary with operation result
        """
        try:
            # Check if operation requires confirmation
            if require_confirmation and self.safety_confirm_required:
                critical_operations = [SystemOperation.SHUTDOWN, SystemOperation.RESTART]
                if operation in critical_operations:
                    return {
                        'success': False,
                        'requires_confirmation': True,
                        'operation': operation.value,
                        'message': f"Safety confirmation required for {operation.value}",
                        'mickey_response': "Mickey needs your confirmation for this important operation! 👀"
                    }
            
            # Execute the operation
            if operation in self.system_commands:
                command = self.system_commands[operation]
                
                # For critical operations, we might want to simulate first
                if operation in [SystemOperation.SHUTDOWN, SystemOperation.RESTART]:
                    self.logger.warning(f"Simulating {operation.value} for safety")
                    result = self._simulate_operation(operation)
                else:
                    result = self._execute_command(command)
                
                self.logger.info(f"System operation executed: {operation.value}")
                
                return {
                    'success': True,
                    'operation': operation.value,
                    'command': command,
                    'result': result,
                    'message': f"Successfully executed {operation.value}",
                    'mickey_response': self._get_system_message(operation)
                }
            else:
                return {
                    'success': False,
                    'operation': operation.value,
                    'message': f"Operation {operation.value} not supported on {self.system_type.value}",
                    'mickey_response': f"Mickey can't do {operation.value} on this system! 🐭"
                }
                
        except Exception as e:
            self.logger.error(f"System operation failed: {str(e)}")
            return {
                'success': False,
                'operation': operation.value,
                'error': str(e),
                'mickey_response': "Oops! Mickey had trouble with that system operation! 😅"
            }

    def _execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a system command"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=30  # 30 second timeout
            )
            
            return {
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                'return_code': -1,
                'stdout': '',
                'stderr': 'Command timed out',
                'success': False
            }
        except Exception as e:
            return {
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'success': False
            }

    def _simulate_operation(self, operation: SystemOperation) -> Dict[str, Any]:
        """Simulate critical operations for safety"""
        return {
            'simulated': True,
            'operation': operation.value,
            'message': 'This was a simulation. Set safety_confirm_required=False to execute for real.',
            'success': True
        }

    def open_application(self, app_name: str, app_path: str = None) -> Dict[str, Any]:
        """
        Open an application
        
        Args:
            app_name: Name of the application
            app_path: Optional path to the application
            
        Returns:
            Dictionary with open result
        """
        try:
            if app_path and os.path.exists(app_path):
                # Use provided path
                command = f'"{app_path}"'
            else:
                # Try to find application
                command = self._find_application_command(app_name)
            
            result = self._execute_command(command)
            
            if result['success']:
                self.logger.info(f"Application opened: {app_name}")
                return {
                    'success': True,
                    'application': app_name,
                    'command': command,
                    'message': f"Successfully opened {app_name}",
                    'mickey_response': f"Mickey opened {app_name}! 🚀"
                }
            else:
                return {
                    'success': False,
                    'application': app_name,
                    'error': result['stderr'],
                    'message': f"Failed to open {app_name}",
                    'mickey_response': f"Mickey couldn't find {app_name}! 🤔"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to open application {app_name}: {str(e)}")
            return {
                'success': False,
                'application': app_name,
                'error': str(e),
                'mickey_response': f"Oops! Mickey had trouble opening {app_name}! 😅"
            }

    def _find_application_command(self, app_name: str) -> str:
        """Find the command to open an application"""
        if self.system_type == SystemType.WINDOWS:
            # Try common Windows applications
            windows_apps = {
                'notepad': 'notepad',
                'calculator': 'calc',
                'paint': 'mspaint',
                'file explorer': 'explorer',
                'command prompt': 'cmd',
                'powershell': 'powershell',
                'browser': 'start msedge'  # Change to preferred browser
            }
            return windows_apps.get(app_name.lower(), app_name)
        
        elif self.system_type == SystemType.MACOS:
            # macOS applications
            mac_apps = {
                'safari': 'open -a Safari',
                'calculator': 'open -a Calculator',
                'textedit': 'open -a TextEdit',
                'finder': 'open -a Finder',
                'terminal': 'open -a Terminal'
            }
            return mac_apps.get(app_name.lower(), f"open -a '{app_name}'")
        
        elif self.system_type == SystemType.LINUX:
            # Linux applications
            linux_apps = {
                'firefox': 'firefox',
                'chrome': 'google-chrome',
                'calculator': 'gnome-calculator',
                'text editor': 'gedit',
                'file manager': 'nautilus',
                'terminal': 'gnome-terminal'
            }
            return linux_apps.get(app_name.lower(), app_name)
        
        return app_name

    def get_running_processes(self, limit: int = 20) -> Dict[str, Any]:
        """Get list of running processes"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
            
            return {
                'success': True,
                'process_count': len(processes),
                'processes': processes[:limit],
                'message': f"Found {len(processes)} running processes"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get running processes: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': "Failed to get running processes"
            }

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information"""
        try:
            # Get current system info
            current_info = self._get_system_info()
            
            # Calculate health scores (0-100)
            cpu_health = 100 - min(current_info['cpu']['usage_percent'], 100)
            memory_health = 100 - current_info['memory']['percent']
            disk_health = 100 - current_info['disk']['percent']
            
            overall_health = (cpu_health + memory_health + disk_health) / 3
            
            # Determine health status
            if overall_health >= 80:
                health_status = "excellent"
                mickey_response = "Mickey says your system is in great shape! 💪"
            elif overall_health >= 60:
                health_status = "good"
                mickey_response = "System health is good! Mickey's happy! 😊"
            elif overall_health >= 40:
                health_status = "fair"
                mickey_response = "System health is fair. Mickey suggests some cleanup! 🧹"
            else:
                health_status = "poor"
                mickey_response = "System health needs attention! Mickey's concerned! 🚨"
            
            return {
                'success': True,
                'health_score': round(overall_health, 1),
                'health_status': health_status,
                'component_scores': {
                    'cpu': round(cpu_health, 1),
                    'memory': round(memory_health, 1),
                    'disk': round(disk_health, 1)
                },
                'system_info': current_info,
                'mickey_response': mickey_response
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': "Failed to get system health"
            }

    def start_performance_monitoring(self, interval: int = 5):
        """Start background performance monitoring"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.performance_monitor = threading.Thread(
            target=self._performance_monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.performance_monitor.start()
        self.logger.info(f"Performance monitoring started (interval: {interval}s)")

    def stop_performance_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.performance_monitor:
            self.performance_monitor.join(timeout=5)
        self.logger.info("Performance monitoring stopped")

    def _performance_monitor_loop(self, interval: int):
        """Background performance monitoring loop"""
        while self.is_monitoring:
            try:
                # Get system health
                health = self.get_system_health()
                
                # Log warnings for poor health
                if health['success'] and health['health_score'] < 40:
                    self.logger.warning(f"Poor system health detected: {health['health_score']}")
                
                # Notify callbacks
                for callback in self.confirmation_callbacks:
                    try:
                        callback('performance_update', health)
                    except Exception as e:
                        self.logger.error(f"Performance callback failed: {str(e)}")
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {str(e)}")
                time.sleep(interval)

    def add_confirmation_callback(self, callback):
        """Add callback for confirmation requests"""
        self.confirmation_callbacks.append(callback)

    def confirm_operation(self, operation: SystemOperation, user_confirmed: bool) -> Dict[str, Any]:
        """
        Confirm a critical operation
        
        Args:
            operation: Operation to confirm
            user_confirmed: Whether user confirmed the operation
            
        Returns:
            Dictionary with confirmation result
        """
        if user_confirmed:
            # Execute the operation without confirmation requirement
            self.safety_confirm_required = False
            result = self.execute_system_operation(operation, require_confirmation=False)
            self.safety_confirm_required = True  # Reset to safe mode
            
            return result
        else:
            return {
                'success': False,
                'operation': operation.value,
                'message': f"Operation {operation.value} cancelled by user",
                'mickey_response': "Mickey cancelled the operation as requested! ✅"
            }

    def _get_system_message(self, operation: SystemOperation) -> str:
        """Get Mickey's message for system operations"""
        import random
        messages = self.system_messages.get(operation, [f"Mickey executed {operation.value}! 🐭"])
        return random.choice(messages)

    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get supported system capabilities"""
        return {
            'system_type': self.system_type.value,
            'system_info': self.system_info,
            'supported_operations': [op.value for op in self.system_commands.keys()],
            'safety_confirm_required': self.safety_confirm_required,
            'performance_monitoring': self.is_monitoring
        }

    def cleanup(self):
        """Cleanup resources"""
        self.stop_performance_monitoring()
        self.confirmation_callbacks.clear()

# Test function
def test_system_integration():
    """Test the system integration"""
    system = SystemIntegration()
    
    print("Testing System Integration...")
    
    # Test system info
    capabilities = system.get_system_capabilities()
    print("System Capabilities:", capabilities)
    
    # Test system health
    health = system.get_system_health()
    print("System Health:", health)
    
    # Test running processes
    processes = system.get_running_processes(limit=5)
    print("Running Processes:", processes['process_count'])
    
    # Test safe operations (volume control)
    volume_result = system.execute_system_operation(SystemOperation.VOLUME_UP)
    print("Volume Operation:", volume_result)
    
    # Test critical operation (should require confirmation)
    shutdown_result = system.execute_system_operation(SystemOperation.SHUTDOWN)
    print("Shutdown Operation:", shutdown_result)
    
    system.cleanup()
    print("System integration test completed!")

if __name__ == "__main__":
    test_system_integration()