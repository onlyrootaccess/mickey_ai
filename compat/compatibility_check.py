# VERY FIRST - Python 3.11.9 + libs verify
"""
M.I.C.K.E.Y. AI Assistant - Compatibility Verification Module
Made In Crisis, Keeping Everything Yours

FIRST FILE IN PIPELINE: Validates system readiness for Mickey AI Assistant.
Checks Python version, core dependencies, hardware capabilities, and OS compatibility.
"""

import sys
import platform
import subprocess
import shutil
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MickeyCompatibility")


class MickeyCompatibilityChecker:
    """
    Comprehensive system compatibility validator for M.I.C.K.E.Y. AI Assistant.
    Ensures all prerequisites are met before launching the assistant.
    """
    
    REQUIRED_PYTHON_VERSION = (3, 11, 9)
    REQUIRED_SYSTEM_PACKAGES = [
        'pip',
        'venv',
        'git'
    ]
    
    # Core Python packages with minimum versions
    CORE_PYTHON_PACKAGES = {
        'fastapi': '0.104.1',
        'uvicorn': '0.24.0',
        'openai-whisper': '20231117',
        'piper-tts': '0.0.2',
        'pyaudio': '0.2.11',
        'opencv-python': '4.8.1.78',
        'customtkinter': '5.2.0',
        'groq': '0.4.0',
        'pyautogui': '0.9.54',
        'requests': '2.31.0',
        'numpy': '1.26.0',
        'pillow': '10.1.0',
        'sqlalchemy': '2.0.23',
        'pydantic': '2.5.0',
        'librosa': '0.10.1',
        'face-recognition': '1.3.0',
        'selenium': '4.15.0',
        'psutil': '5.9.6',
        'nltk': '3.8.1',
        'pygame': '2.5.2'
    }
    
    def __init__(self):
        self.compatibility_report = {
            'python_version': False,
            'operating_system': False,
            'system_packages': {},
            'python_packages': {},
            'hardware': {},
            'overall_status': False
        }
        self.errors = []
        self.warnings = []
    
    def check_python_version(self) -> bool:
        """Verify exact Python 3.11.9 version requirement."""
        try:
            current_version = sys.version_info[:3]  # (major, minor, micro)
            required = self.REQUIRED_PYTHON_VERSION
            
            logger.info(f"Checking Python version: {platform.python_version()}")
            
            if current_version == required:
                self.compatibility_report['python_version'] = True
                logger.info("✅ Python 3.11.9 verified successfully")
                return True
            else:
                error_msg = (
                    f"❌ Python version mismatch! Required: 3.11.9, "
                    f"Found: {platform.python_version()}. "
                    f"Mickey AI requires EXACTLY Python 3.11.9 for optimal performance."
                )
                self.errors.append(error_msg)
                logger.error(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"❌ Failed to check Python version: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return False
    
    def check_operating_system(self) -> bool:
        """Verify supported operating system."""
        try:
            system = platform.system().lower()
            version = platform.version()
            
            logger.info(f"Detected OS: {system} {version}")
            
            supported_systems = ['windows', 'linux', 'darwin']  # Darwin = macOS
            
            if system in supported_systems:
                self.compatibility_report['operating_system'] = True
                logger.info(f"✅ OS {system} supported")
                return True
            else:
                warning_msg = f"⚠️  OS {system} may have limited support. Testing recommended."
                self.warnings.append(warning_msg)
                logger.warning(warning_msg)
                return True  # Not a blocking issue
                
        except Exception as e:
            error_msg = f"❌ Failed to check operating system: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return False
    
    def check_system_packages(self) -> Dict[str, bool]:
        """Verify essential system packages are available."""
        results = {}
        
        for pkg in self.REQUIRED_SYSTEM_PACKAGES:
            try:
                if shutil.which(pkg):
                    results[pkg] = True
                    logger.info(f"✅ System package '{pkg}' found")
                else:
                    results[pkg] = False
                    warning_msg = f"⚠️  System package '{pkg}' not found in PATH"
                    self.warnings.append(warning_msg)
                    logger.warning(warning_msg)
                    
            except Exception as e:
                results[pkg] = False
                error_msg = f"❌ Error checking system package '{pkg}': {str(e)}"
                self.errors.append(error_msg)
                logger.error(error_msg)
        
        self.compatibility_report['system_packages'] = results
        return results
    
    def check_python_packages(self) -> Dict[str, bool]:
        """
        Check if core Python packages are available.
        Note: This checks availability, not exact versions (handled by requirements.txt).
        """
        results = {}
        
        for package, required_version in self.CORE_PYTHON_PACKAGES.items():
            try:
                # Handle special package names that differ from import names
                import_name = package
                if package == 'openai-whisper':
                    import_name = 'whisper'
                elif package == 'piper-tts':
                    import_name = 'piper'
                elif package == 'opencv-python':
                    import_name = 'cv2'
                elif package == 'face-recognition':
                    import_name = 'face_recognition'
                
                __import__(import_name)
                results[package] = True
                logger.info(f"✅ Python package '{package}' available")
                
            except ImportError as e:
                results[package] = False
                error_msg = f"❌ Python package '{package}' not installed: {str(e)}"
                self.errors.append(error_msg)
                logger.error(error_msg)
            except Exception as e:
                results[package] = False
                error_msg = f"❌ Error checking package '{package}': {str(e)}"
                self.errors.append(error_msg)
                logger.error(error_msg)
        
        self.compatibility_report['python_packages'] = results
        return results
    
    def check_audio_hardware(self) -> bool:
        """Verify audio input/output capabilities."""
        try:
            import pyaudio
            
            p = pyaudio.PyAudio()
            
            # Check input devices (microphone)
            input_devices = []
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info.get('maxInputChannels', 0) > 0:
                    input_devices.append(device_info['name'])
            
            # Check output devices (speakers)
            output_devices = []
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info.get('maxOutputChannels', 0) > 0:
                    output_devices.append(device_info['name'])
            
            p.terminate()
            
            audio_status = {
                'input_available': len(input_devices) > 0,
                'output_available': len(output_devices) > 0,
                'input_devices': input_devices,
                'output_devices': output_devices
            }
            
            self.compatibility_report['hardware']['audio'] = audio_status
            
            if audio_status['input_available']:
                logger.info(f"✅ Audio input devices found: {len(input_devices)}")
            else:
                error_msg = "❌ No audio input devices (microphone) found!"
                self.errors.append(error_msg)
                logger.error(error_msg)
            
            if audio_status['output_available']:
                logger.info(f"✅ Audio output devices found: {len(output_devices)}")
            else:
                error_msg = "❌ No audio output devices (speakers) found!"
                self.errors.append(error_msg)
                logger.error(error_msg)
            
            return audio_status['input_available'] and audio_status['output_available']
            
        except ImportError:
            error_msg = "❌ pyaudio not available - cannot check audio hardware"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return False
        except Exception as e:
            error_msg = f"❌ Error checking audio hardware: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return False
    
    def check_camera_hardware(self) -> bool:
        """Verify camera availability for face recognition."""
        try:
            import cv2
            
            camera = cv2.VideoCapture(0)
            if camera.isOpened():
                ret, frame = camera.read()
                camera.release()
                
                if ret:
                    self.compatibility_report['hardware']['camera'] = True
                    logger.info("✅ Camera available for face recognition")
                    return True
                else:
                    self.compatibility_report['hardware']['camera'] = False
                    warning_msg = "⚠️  Camera found but cannot capture frames"
                    self.warnings.append(warning_msg)
                    logger.warning(warning_msg)
                    return False
            else:
                self.compatibility_report['hardware']['camera'] = False
                warning_msg = "⚠️  No camera detected - face recognition disabled"
                self.warnings.append(warning_msg)
                logger.warning(warning_msg)
                return False
                
        except ImportError:
            error_msg = "❌ OpenCV not available - cannot check camera"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return False
        except Exception as e:
            error_msg = f"❌ Error checking camera: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return False
    
    def check_hardware_capabilities(self) -> Dict[str, bool]:
        """Comprehensive hardware verification."""
        hardware_report = {}
        
        hardware_report['audio'] = self.check_audio_hardware()
        hardware_report['camera'] = self.check_camera_hardware()
        
        self.compatibility_report['hardware'] = hardware_report
        return hardware_report
    
    def run_comprehensive_check(self) -> bool:
        """
        Execute all compatibility checks and generate final report.
        Returns True if system is compatible, False otherwise.
        """
        logger.info("🚀 Starting M.I.C.K.E.Y. AI Assistant Compatibility Check...")
        logger.info("Made In Crisis, Keeping Everything Yours")
        logger.info("=" * 60)
        
        # Run all checks
        checks = [
            self.check_python_version(),
            self.check_operating_system(),
            all(self.check_system_packages().values()),  # At least have pip
            self.check_hardware_capabilities().get('audio', False),  # Audio is critical
        ]
        
        # Python packages check - warn but don't block (will be installed later)
        python_pkgs = self.check_python_packages()
        python_pkg_success = sum(python_pkgs.values()) / len(python_pkgs) > 0.5
        
        # Determine overall compatibility
        critical_checks = all(checks)
        self.compatibility_report['overall_status'] = critical_checks
        
        # Generate final report
        self._generate_final_report()
        
        return critical_checks
    
    def _generate_final_report(self):
        """Generate comprehensive compatibility report."""
        logger.info("=" * 60)
        logger.info("📊 M.I.C.K.E.Y. AI COMPATIBILITY REPORT")
        logger.info("=" * 60)
        
        # Python version
        py_status = "✅ PASS" if self.compatibility_report['python_version'] else "❌ FAIL"
        logger.info(f"Python 3.11.9: {py_status}")
        
        # OS
        os_status = "✅ PASS" if self.compatibility_report['operating_system'] else "❌ FAIL"
        logger.info(f"Operating System: {os_status}")
        
        # Audio
        audio_available = self.compatibility_report['hardware'].get('audio', {}).get('input_available', False)
        audio_status = "✅ PASS" if audio_available else "❌ FAIL"
        logger.info(f"Audio Hardware: {audio_status}")
        
        # Camera
        camera_available = self.compatibility_report['hardware'].get('camera', False)
        camera_status = "✅ PASS" if camera_available else "⚠️  OPTIONAL"
        logger.info(f"Camera: {camera_status}")
        
        # Python packages
        python_pkgs = self.compatibility_report['python_packages']
        installed_count = sum(python_pkgs.values())
        total_count = len(python_pkgs)
        logger.info(f"Python Packages: {installed_count}/{total_count} installed")
        
        # Display errors
        if self.errors:
            logger.info("\n❌ CRITICAL ISSUES:")
            for error in self.errors:
                logger.info(f"  - {error}")
        
        # Display warnings
        if self.warnings:
            logger.info("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                logger.info(f"  - {warning}")
        
        # Final verdict
        if self.compatibility_report['overall_status']:
            logger.info("\n🎉 SYSTEM COMPATIBLE! M.I.C.K.E.Y. AI Assistant can be installed.")
            logger.info("Proceed with: python install.py")
        else:
            logger.info("\n💥 SYSTEM INCOMPATIBLE! Please resolve critical issues above.")
        
        logger.info("=" * 60)
    
    def get_report(self) -> Dict:
        """Return comprehensive compatibility report."""
        return self.compatibility_report
    
    def get_errors(self) -> List[str]:
        """Return list of critical errors."""
        return self.errors
    
    def get_warnings(self) -> List[str]:
        """Return list of warnings."""
        return self.warnings


def main():
    """Command-line entry point for compatibility check."""
    checker = MickeyCompatibilityChecker()
    is_compatible = checker.run_comprehensive_check()
    
    # Return appropriate exit code
    sys.exit(0 if is_compatible else 1)


if __name__ == "__main__":
    main()