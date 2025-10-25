# Format conversion, stream handling
"""
Mickey AI - Audio Utilities
Audio processing, format conversion, and streaming utilities
"""

import logging
import numpy as np
import wave
import audioop
import threading
import queue
import time
from typing import Optional, Callable, List, Tuple
import pyaudio
from scipy import signal
import io

class AudioFormat:
    """Audio format constants"""
    SAMPLE_RATE = 16000
    CHANNELS = 1
    SAMPLE_WIDTH = 2  # 16-bit
    CHUNK_SIZE = 1024
    FORMAT = pyaudio.paInt16

class AudioBuffer:
    """Circular buffer for audio data"""
    def __init__(self, size: int = 44100 * 10):  # 10 seconds buffer
        self.logger = logging.getLogger(__name__)
        self.buffer = np.zeros(size, dtype=np.float32)
        self.size = size
        self.write_pos = 0
        self.read_pos = 0
        self.lock = threading.Lock()
        
    def write(self, data: np.ndarray):
        """Write audio data to buffer"""
        with self.lock:
            data_len = len(data)
            if data_len > self.size:
                data = data[-self.size:]  # Truncate if too long
            
            # Calculate write positions
            end_pos = self.write_pos + data_len
            if end_pos <= self.size:
                self.buffer[self.write_pos:end_pos] = data
            else:
                # Wrap around
                first_part = self.size - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                self.buffer[:data_len - first_part] = data[first_part:]
            
            self.write_pos = end_pos % self.size
            
    def read(self, length: int) -> Optional[np.ndarray]:
        """Read audio data from buffer"""
        with self.lock:
            if self.available_samples < length:
                return None
            
            if self.read_pos + length <= self.size:
                data = self.buffer[self.read_pos:self.read_pos + length]
            else:
                # Wrap around
                first_part = self.size - self.read_pos
                data = np.concatenate([
                    self.buffer[self.read_pos:],
                    self.buffer[:length - first_part]
                ])
            
            self.read_pos = (self.read_pos + length) % self.size
            return data.copy()
    
    @property
    def available_samples(self) -> int:
        """Get number of available samples"""
        if self.write_pos >= self.read_pos:
            return self.write_pos - self.read_pos
        else:
            return self.size - self.read_pos + self.write_pos

class AudioRecorder:
    """Audio recording utility with streaming capabilities"""
    def __init__(self, sample_rate: int = AudioFormat.SAMPLE_RATE, 
                 channels: int = AudioFormat.CHANNELS,
                 chunk_size: int = AudioFormat.CHUNK_SIZE):
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = AudioFormat.FORMAT
        
        self.audio = None
        self.stream = None
        self.is_recording = False
        self.audio_buffer = AudioBuffer()
        self.recording_callbacks = []
        
    def start_recording(self) -> bool:
        """Start audio recording"""
        try:
            self.audio = pyaudio.PyAudio()
            
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.logger.info("🎤 Audio recording started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {str(e)}")
            return False
    
    def stop_recording(self) -> bool:
        """Stop audio recording"""
        try:
            self.is_recording = False
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            if self.audio:
                self.audio.terminate()
            
            self.logger.info("Audio recording stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop recording: {str(e)}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        if status:
            self.logger.warning(f"Audio stream status: {status}")
        
        try:
            # Convert to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Write to buffer
            self.audio_buffer.write(audio_data)
            
            # Notify callbacks
            for callback in self.recording_callbacks:
                try:
                    callback(audio_data)
                except Exception as e:
                    self.logger.error(f"Recording callback error: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Audio callback error: {str(e)}")
        
        return (in_data, pyaudio.paContinue)
    
    def add_callback(self, callback: Callable):
        """Add audio data callback"""
        self.recording_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove audio data callback"""
        if callback in self.recording_callbacks:
            self.recording_callbacks.remove(callback)
    
    def get_audio_data(self, duration: float = 1.0) -> Optional[np.ndarray]:
        """Get audio data for specified duration"""
        samples_needed = int(duration * self.sample_rate)
        return self.audio_buffer.read(samples_needed)

class AudioPlayer:
    """Audio playback utility"""
    def __init__(self, sample_rate: int = AudioFormat.SAMPLE_RATE,
                 channels: int = AudioFormat.CHANNELS):
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = AudioFormat.FORMAT
        
        self.audio = None
        self.stream = None
        self.is_playing = False
        
    def play_audio(self, audio_data: np.ndarray) -> bool:
        """Play audio data"""
        try:
            # Ensure audio data is in correct format
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            self.audio = pyaudio.PyAudio()
            
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )
            
            # Play audio
            self.stream.write(audio_data.tobytes())
            self.is_playing = True
            
            self.logger.info("🔊 Audio playback started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to play audio: {str(e)}")
            return False
    
    def stop_playback(self):
        """Stop audio playback"""
        try:
            self.is_playing = False
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            if self.audio:
                self.audio.terminate()
                
            self.logger.info("Audio playback stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop playback: {str(e)}")

class AudioProcessor:
    """Audio processing utilities"""
    
    @staticmethod
    def resample_audio(audio_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """Resample audio to target rate"""
        if original_rate == target_rate:
            return audio_data
        
        # Calculate number of samples in resampled audio
        num_samples = int(len(audio_data) * target_rate / original_rate)
        
        # Resample using scipy
        resampled_audio = signal.resample(audio_data, num_samples)
        
        return resampled_audio.astype(audio_data.dtype)
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray, target_level: float = 0.1) -> np.ndarray:
        """Normalize audio to target level"""
        if len(audio_data) == 0:
            return audio_data
        
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio_data**2))
        
        if rms < 1e-10:  # Avoid division by zero
            return audio_data
        
        # Calculate gain factor
        gain = target_level / rms
        
        # Apply gain with limiting to prevent clipping
        max_gain = 10.0  # Maximum 20dB gain
        gain = min(gain, max_gain)
        
        normalized_audio = audio_data * gain
        
        # Clip to prevent distortion
        normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
        
        return normalized_audio
    
    @staticmethod
    def remove_silence(audio_data: np.ndarray, threshold: float = 0.01, 
                      min_duration: float = 0.1) -> np.ndarray:
        """Remove silent portions from audio"""
        if len(audio_data) == 0:
            return audio_data
        
        # Calculate energy
        energy = np.abs(audio_data)
        
        # Find non-silent regions
        non_silent = energy > threshold
        
        # Find start and end of non-silent regions
        changes = np.diff(non_silent.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        # Handle edge cases
        if non_silent[0]:
            starts = np.insert(starts, 0, 0)
        if non_silent[-1]:
            ends = np.append(ends, len(audio_data))
        
        # Filter regions by minimum duration
        min_samples = int(min_duration * AudioFormat.SAMPLE_RATE)
        valid_regions = []
        
        for start, end in zip(starts, ends):
            if end - start >= min_samples:
                valid_regions.append((start, end))
        
        # Combine valid regions
        if not valid_regions:
            return np.array([], dtype=audio_data.dtype)
        
        # Extract non-silent audio
        non_silent_audio = np.concatenate([audio_data[start:end] for start, end in valid_regions])
        
        return non_silent_audio
    
    @staticmethod
    def calculate_rms(audio_data: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) of audio"""
        if len(audio_data) == 0:
            return 0.0
        return np.sqrt(np.mean(audio_data**2))
    
    @staticmethod
    def calculate_db(audio_data: np.ndarray) -> float:
        """Calculate audio level in dB"""
        rms = AudioProcessor.calculate_rms(audio_data)
        if rms < 1e-10:
            return -100.0  # Very quiet
        return 20 * np.log10(rms)
    
    @staticmethod
    def convert_to_wav(audio_data: np.ndarray, sample_rate: int = AudioFormat.SAMPLE_RATE) -> bytes:
        """Convert audio data to WAV format bytes"""
        # Ensure correct data type
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(AudioFormat.CHANNELS)
                wav_file.setsampwidth(AudioFormat.SAMPLE_WIDTH)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            return wav_buffer.getvalue()
    
    @staticmethod
    def detect_voice_activity(audio_data: np.ndarray, threshold_db: float = -30.0) -> bool:
        """Detect if audio contains voice activity"""
        db_level = AudioProcessor.calculate_db(audio_data)
        return db_level > threshold_db

# Test function
def test_audio_utils():
    """Test audio utilities"""
    import wave
    
    # Test audio processing
    processor = AudioProcessor()
    
    # Create test audio (sine wave)
    duration = 1.0  # seconds
    frequency = 440  # Hz
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    test_audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    print("Testing Audio Utilities...")
    
    # Test normalization
    normalized = processor.normalize_audio(test_audio)
    print(f"Original RMS: {processor.calculate_rms(test_audio):.4f}")
    print(f"Normalized RMS: {processor.calculate_rms(normalized):.4f}")
    
    # Test RMS and dB calculation
    rms = processor.calculate_rms(test_audio)
    db = processor.calculate_db(test_audio)
    print(f"RMS: {rms:.4f}, dB: {db:.2f}")
    
    # Test voice activity detection
    has_voice = processor.detect_voice_activity(test_audio)
    print(f"Voice detected: {has_voice}")
    
    # Test WAV conversion
    wav_data = processor.convert_to_wav(test_audio)
    print(f"WAV data size: {len(wav_data)} bytes")
    
    print("Audio utilities test completed!")

if __name__ == "__main__":
    test_audio_utils()