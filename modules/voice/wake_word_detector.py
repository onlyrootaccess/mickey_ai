# "Hey Mickey" detection
"""
Mickey AI - Wake Word Detector
Detects "Hey Mickey" wake word using Porcupine for always-listening capability
"""

import logging
import threading
import time
import numpy as np
import pvporcupine
import pyaudio
import wave
from typing import Dict, Any, Optional, Callable
from enum import Enum

class WakeWordState(Enum):
    SLEEPING = "sleeping"
    LISTENING = "listening"
    DETECTED = "detected"
    ERROR = "error"

class WakeWordDetector:
    def __init__(self, access_key: str = None, model_path: str = None, keyword_paths: list = None):
        self.logger = logging.getLogger(__name__)
        
        # Wake word configuration
        self.access_key = access_key or "default"  # In production, use environment variable
        self.model_path = model_path
        self.keyword_paths = keyword_paths or []
        
        # Detection state
        self.state = WakeWordState.SLEEPING
        self.detection_callback = None
        self.is_running = False
        self.is_detected = False
        
        # Audio configuration
        self.audio = None
        self.stream = None
        self.porcupine = None
        
        # Performance tracking
        self.detection_count = 0
        self.last_detection_time = 0
        
        # Mickey's wake responses
        self.wake_responses = [
            "I'm listening! What can Mickey do for you? 🐭",
            "Hey there! Mickey's here! How can I help?",
            "Hot dog! You called? Ready for action! 🌭",
            "Mickey's awake and ready! What's up?",
            "Yes? Mickey's all ears! 👂"
        ]
        
        # Threading
        self._detection_thread = None
        self._audio_lock = threading.Lock()
        
        self.logger.info("🔊 Wake Word Detector initialized")

    def initialize(self) -> bool:
        """Initialize Porcupine wake word engine"""
        try:
            self.logger.info("Initializing wake word engine...")
            
            # Create Porcupine instance
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                model_path=self.model_path,
                keyword_paths=self.keyword_paths,
                keywords=["hey mickey"]  # Built-in keyword
            )
            
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            
            # Create audio stream
            self.stream = self.audio.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length,
                stream_callback=self._audio_callback if not self.is_running else None
            )
            
            self.state = WakeWordState.LISTENING
            self.logger.info("✅ Wake word engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize wake word engine: {str(e)}")
            self.state = WakeWordState.ERROR
            return False

    def start_detection(self, callback: Callable = None) -> bool:
        """Start listening for wake word"""
        try:
            if self.state == WakeWordState.ERROR:
                if not self.initialize():
                    return False
            
            if callback:
                self.detection_callback = callback
            
            self.is_running = True
            
            # Start detection thread
            self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self._detection_thread.start()
            
            self.state = WakeWordState.LISTENING
            self.logger.info("🎯 Wake word detection started - Listening for 'Hey Mickey'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start detection: {str(e)}")
            self.state = WakeWordState.ERROR
            return False

    def stop_detection(self) -> bool:
        """Stop listening for wake word"""
        try:
            self.is_running = False
            
            if self._detection_thread and self._detection_thread.is_alive():
                self._detection_thread.join(timeout=5)
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            if self.audio:
                self.audio.terminate()
            
            if self.porcupine:
                self.porcupine.delete()
            
            self.state = WakeWordState.SLEEPING
            self.logger.info("Wake word detection stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop detection: {str(e)}")
            return False

    def _detection_loop(self):
        """Main detection loop"""
        self.logger.info("Starting wake word detection loop...")
        
        try:
            while self.is_running:
                try:
                    # Read audio frame
                    pcm = self.stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                    pcm = np.frombuffer(pcm, dtype=np.int16)
                    
                    # Process with Porcupine
                    keyword_index = self.porcupine.process(pcm)
                    
                    if keyword_index >= 0:
                        self._on_wake_word_detected()
                        
                    # Small delay to prevent CPU overload
                    time.sleep(0.01)
                    
                except Exception as e:
                    self.logger.error(f"Detection loop error: {str(e)}")
                    time.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(f"Detection loop failed: {str(e)}")
            self.state = WakeWordState.ERROR

    def _on_wake_word_detected(self):
        """Handle wake word detection"""
        try:
            current_time = time.time()
            
            # Prevent multiple detections within 2 seconds
            if current_time - self.last_detection_time < 2.0:
                return
            
            self.detection_count += 1
            self.last_detection_time = current_time
            self.state = WakeWordState.DETECTED
            self.is_detected = True
            
            self.logger.info("🎉 Wake word 'Hey Mickey' detected!")
            
            # Execute callback if provided
            if self.detection_callback:
                response = self._get_wake_response()
                self.detection_callback({
                    'wake_word': 'Hey Mickey',
                    'timestamp': current_time,
                    'detection_count': self.detection_count,
                    'mickey_response': response
                })
            
            # Reset detection state after short delay
            threading.Timer(1.0, self._reset_detection).start()
            
        except Exception as e:
            self.logger.error(f"Wake word detection handler failed: {str(e)}")

    def _reset_detection(self):
        """Reset detection state after wake word"""
        self.is_detected = False
        if self.state == WakeWordState.DETECTED:
            self.state = WakeWordState.LISTENING

    def _get_wake_response(self) -> str:
        """Get Mickey's random wake response"""
        import random
        return random.choice(self.wake_responses)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for streaming (alternative approach)"""
        if status:
            self.logger.warning(f"Audio stream status: {status}")
        
        try:
            pcm = np.frombuffer(in_data, dtype=np.int16)
            keyword_index = self.porcupine.process(pcm)
            
            if keyword_index >= 0:
                self._on_wake_word_detected()
                
        except Exception as e:
            self.logger.error(f"Audio callback error: {str(e)}")
        
        return (in_data, pyaudio.paContinue)

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            'state': self.state.value,
            'detection_count': self.detection_count,
            'is_running': self.is_running,
            'is_detected': self.is_detected,
            'last_detection_time': self.last_detection_time,
            'sample_rate': self.porcupine.sample_rate if self.porcupine else None,
            'frame_length': self.porcupine.frame_length if self.porcupine else None
        }

    def set_sensitivity(self, sensitivity: float = 0.5):
        """Set detection sensitivity (if supported)"""
        # Porcupine sensitivity is set during initialization
        # This method is for future compatibility
        self.logger.info(f"Sensitivity set to {sensitivity}")

    def save_audio_sample(self, audio_data: np.ndarray, filename: str = None):
        """Save detected audio sample for analysis"""
        try:
            if filename is None:
                timestamp = int(time.time())
                filename = f"wake_word_{timestamp}.wav"
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.porcupine.sample_rate if self.porcupine else 16000)
                wf.writeframes(audio_data.tobytes())
            
            self.logger.info(f"Audio sample saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save audio sample: {str(e)}")

    def get_status(self) -> Dict[str, Any]:
        """Get current detector status"""
        return {
            'success': self.state != WakeWordState.ERROR,
            'state': self.state.value,
            'is_initialized': self.porcupine is not None,
            'is_listening': self.is_running,
            'detections': self.detection_count,
            'mickey_response': "Mickey's ears are open! 👂" if self.is_running else "Mickey's sleeping 😴"
        }

# Test function
def test_wake_word_detector():
    """Test the wake word detector (requires Porcupine setup)"""
    detector = WakeWordDetector()
    
    def detection_callback(data):
        print(f"Wake word detected! Data: {data}")
    
    try:
        # Try to initialize
        if detector.initialize():
            print("Wake word detector initialized successfully")
            
            # Start detection
            detector.start_detection(detection_callback)
            print("Detection started - say 'Hey Mickey'")
            
            # Run for 10 seconds for testing
            time.sleep(10)
            
            # Stop detection
            detector.stop_detection()
            print("Detection stopped")
            
        else:
            print("Wake word detector initialization failed")
            
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        detector.stop_detection()

if __name__ == "__main__":
    test_wake_word_detector()