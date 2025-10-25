# Audio preprocessing
"""
Mickey AI - Noise Canceler
Advanced noise cancellation and audio enhancement for clear voice processing
"""

import logging
import numpy as np
import threading
import time
from typing import Optional, Callable, Dict, Any
from scipy import signal
from collections import deque

class NoiseCancelMode(Enum):
    AGGRESSIVE = "aggressive"
    MODERATE = "moderate" 
    LIGHT = "light"
    ADAPTIVE = "adaptive"

class NoiseCanceler:
    def __init__(self, sample_rate: int = 16000, mode: NoiseCancelMode = NoiseCancelMode.ADAPTIVE):
        self.logger = logging.getLogger(__name__)
        
        # Audio configuration
        self.sample_rate = sample_rate
        self.mode = mode
        
        # Noise profile and state
        self.noise_profile = None
        self.is_noise_profile_ready = False
        self.noise_profile_frames = 50  # Frames to analyze for noise profile
        self.noise_profile_counter = 0
        
        # Adaptive filtering parameters
        self.learning_rate = 0.01
        self.noise_threshold = 0.02
        self.snr_threshold = 2.0
        
        # Signal processing buffers
        self.audio_buffer = deque(maxlen=sample_rate * 5)  # 5 second buffer
        self.noise_buffer = deque(maxlen=sample_rate * 2)  # 2 second noise buffer
        
        # Frequency domain parameters
        self.fft_size = 512
        self.hop_size = 256
        self.freq_bins = self.fft_size // 2 + 1
        
        # Filter state
        self.noise_psd = np.zeros(self.freq_bins)  # Power Spectral Density of noise
        self.signal_psd = np.zeros(self.freq_bins) # Power Spectral Density of signal
        self.clean_psd = np.zeros(self.freq_bins)  # PSD of cleaned signal
        
        # Statistical noise modeling
        self.noise_mean = None
        self.noise_std = None
        self.signal_mean = None
        self.signal_std = None
        
        # Real-time adaptation
        self.adaptation_rate = 0.98
        self.min_noise_psd = 1e-12  # Minimum noise PSD to avoid division by zero
        
        # Performance tracking
        self.processing_time = 0
        self.noise_reduction_db = 0
        self.enhancement_gain = 1.0
        
        # Callbacks for processed audio
        self.processed_callbacks = []
        
        # Threading
        self._processing_lock = threading.RLock()
        self._is_processing = False
        
        # Mickey's noise cancellation messages
        self.noise_messages = {
            'active': [
                "Mickey's cleaning up the audio! 🧹",
                "Noise cancellation activated! Crystal clear! ✨",
                "Filtering out the noise! Mickey's on it! 🔇",
                "Audio enhancement in progress! 🎵"
            ],
            'calibrating': [
                "Learning the noise profile... Mickey's listening! 👂",
                "Calibrating noise cancellation... Almost ready!",
                "Setting up audio filters... This will help with clarity!",
                "Mickey's tuning the ears for better listening! 🐭"
            ]
        }
        
        self.logger.info("🔇 Noise Canceler initialized - Ready to clean audio!")

    def start_noise_profiling(self, duration: float = 3.0) -> bool:
        """
        Start noise profiling to learn ambient noise characteristics
        
        Args:
            duration: Duration in seconds to profile noise
            
        Returns:
            Boolean indicating success
        """
        try:
            self.logger.info(f"Starting noise profiling for {duration} seconds...")
            
            # Reset noise profile
            self.noise_profile = None
            self.is_noise_profile_ready = False
            self.noise_profile_counter = 0
            
            # Collect noise samples
            def profile_complete():
                self._finalize_noise_profile()
                self.logger.info("Noise profiling completed successfully")
            
            # Schedule profile completion
            threading.Timer(duration, profile_complete).start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Noise profiling failed: {str(e)}")
            return False

    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio data with noise cancellation
        
        Args:
            audio_data: Input audio as numpy array
            
        Returns:
            Cleaned audio as numpy array
        """
        start_time = time.time()
        
        try:
            with self._processing_lock:
                # Convert to float32 if needed
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Update audio buffer
                self.audio_buffer.extend(audio_data)
                
                # Apply noise cancellation based on mode
                if self.mode == NoiseCancelMode.AGGRESSIVE:
                    cleaned_audio = self._aggressive_noise_cancel(audio_data)
                elif self.mode == NoiseCancelMode.MODERATE:
                    cleaned_audio = self._moderate_noise_cancel(audio_data)
                elif self.mode == NoiseCancelMode.LIGHT:
                    cleaned_audio = self._light_noise_cancel(audio_data)
                else:  # ADAPTIVE
                    cleaned_audio = self._adaptive_noise_cancel(audio_data)
                
                # Calculate processing metrics
                self.processing_time = time.time() - start_time
                self._calculate_noise_reduction(audio_data, cleaned_audio)
                
                # Notify callbacks
                self._notify_callbacks(cleaned_audio)
                
                return cleaned_audio
                
        except Exception as e:
            self.logger.error(f"Audio processing failed: {str(e)}")
            return audio_data  # Return original audio on failure

    def _aggressive_noise_cancel(self, audio_data: np.ndarray) -> np.ndarray:
        """Aggressive noise cancellation for very noisy environments"""
        # Spectral subtraction with aggressive parameters
        cleaned_audio = self._spectral_subtraction(audio_data, over_subtraction=1.5, floor=0.002)
        
        # Wiener filtering for additional noise reduction
        cleaned_audio = self._wiener_filter(cleaned_audio, aggressive=True)
        
        # High-pass filter to remove low-frequency noise
        cleaned_audio = self._high_pass_filter(cleaned_audio, cutoff=150)
        
        return cleaned_audio

    def _moderate_noise_cancel(self, audio_data: np.ndarray) -> np.ndarray:
        """Moderate noise cancellation for typical environments"""
        # Spectral subtraction with moderate parameters
        cleaned_audio = self._spectral_subtraction(audio_data, over_subtraction=1.2, floor=0.001)
        
        # Light Wiener filtering
        cleaned_audio = self._wiener_filter(cleaned_audio, aggressive=False)
        
        return cleaned_audio

    def _light_noise_cancel(self, audio_data: np.ndarray) -> np.ndarray:
        """Light noise cancellation for clean environments"""
        # Minimal spectral subtraction
        cleaned_audio = self._spectral_subtraction(audio_data, over_subtraction=1.0, floor=0.0005)
        
        # Simple high-pass filter
        cleaned_audio = self._high_pass_filter(cleaned_audio, cutoff=80)
        
        return cleaned_audio

    def _adaptive_noise_cancel(self, audio_data: np.ndarray) -> np.ndarray:
        """Adaptive noise cancellation that adjusts based on environment"""
        # Calculate current noise level
        current_noise_level = self._calculate_noise_level(audio_data)
        
        # Adjust processing based on noise level
        if current_noise_level > 0.1:  # Very noisy
            return self._aggressive_noise_cancel(audio_data)
        elif current_noise_level > 0.05:  # Moderately noisy
            return self._moderate_noise_cancel(audio_data)
        else:  # Quiet environment
            return self._light_noise_cancel(audio_data)

    def _spectral_subtraction(self, audio_data: np.ndarray, over_subtraction: float = 1.2, 
                            floor: float = 0.001) -> np.ndarray:
        """Spectral subtraction noise cancellation"""
        # Apply STFT (Short-Time Fourier Transform)
        f, t, stft_data = signal.stft(audio_data, fs=self.sample_rate, 
                                    nperseg=self.fft_size, noverlap=self.hop_size)
        
        # Calculate magnitude and phase
        magnitude = np.abs(stft_data)
        phase = np.angle(stft_data)
        
        # Update noise PSD if needed
        if not self.is_noise_profile_ready:
            self._update_noise_psd(magnitude)
        
        # Apply spectral subtraction
        noise_estimate = np.sqrt(self.noise_psd[:, np.newaxis])
        clean_magnitude = magnitude - over_subtraction * noise_estimate
        clean_magnitude = np.maximum(clean_magnitude, floor * np.max(magnitude))
        
        # Reconstruct STFT
        clean_stft = clean_magnitude * np.exp(1j * phase)
        
        # Apply inverse STFT
        t, cleaned_audio = signal.istft(clean_stft, fs=self.sample_rate,
                                      nperseg=self.fft_size, noverlap=self.hop_size)
        
        return cleaned_audio

    def _wiener_filter(self, audio_data: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """Wiener filter for additional noise reduction"""
        # Apply STFT
        f, t, stft_data = signal.stft(audio_data, fs=self.sample_rate,
                                    nperseg=self.fft_size, noverlap=self.hop_size)
        
        magnitude = np.abs(stft_data)
        phase = np.angle(stft_data)
        
        # Calculate Wiener filter parameters
        if aggressive:
            noise_estimate = self.noise_psd[:, np.newaxis] * 2.0
        else:
            noise_estimate = self.noise_psd[:, np.newaxis]
        
        signal_estimate = magnitude ** 2
        wiener_gain = signal_estimate / (signal_estimate + noise_estimate)
        
        # Apply Wiener filter
        clean_magnitude = magnitude * wiener_gain
        
        # Reconstruct STFT
        clean_stft = clean_magnitude * np.exp(1j * phase)
        
        # Apply inverse STFT
        t, cleaned_audio = signal.istft(clean_stft, fs=self.sample_rate,
                                      nperseg=self.fft_size, noverlap=self.hop_size)
        
        return cleaned_audio

    def _high_pass_filter(self, audio_data: np.ndarray, cutoff: float = 100.0) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise"""
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff / nyquist
        
        # Design Butterworth high-pass filter
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio_data)
        
        return filtered_audio

    def _update_noise_psd(self, magnitude: np.ndarray):
        """Update noise power spectral density estimate"""
        if self.noise_profile_counter < self.noise_profile_frames:
            # Learning phase - update noise PSD
            frame_psd = np.mean(magnitude ** 2, axis=1)
            
            if self.noise_profile_counter == 0:
                self.noise_psd = frame_psd
            else:
                self.noise_psd = (self.adaptation_rate * self.noise_psd + 
                                (1 - self.adaptation_rate) * frame_psd)
            
            self.noise_profile_counter += 1
            
            if self.noise_profile_counter >= self.noise_profile_frames:
                self.is_noise_profile_ready = True
                self.logger.info("Noise profile learning completed")

    def _finalize_noise_profile(self):
        """Finalize noise profile after learning period"""
        if not self.is_noise_profile_ready and self.noise_profile_counter > 0:
            self.is_noise_profile_ready = True
            self.logger.info("Noise profile finalized")

    def _calculate_noise_level(self, audio_data: np.ndarray) -> float:
        """Calculate current noise level in audio"""
        # Use RMS as noise level indicator
        rms = np.sqrt(np.mean(audio_data ** 2))
        return min(rms, 1.0)  # Normalize to 0-1 range

    def _calculate_noise_reduction(self, original: np.ndarray, cleaned: np.ndarray):
        """Calculate noise reduction in dB"""
        original_rms = np.sqrt(np.mean(original ** 2))
        cleaned_rms = np.sqrt(np.mean(cleaned ** 2))
        
        if original_rms > 1e-10 and cleaned_rms > 1e-10:
            original_db = 20 * np.log10(original_rms)
            cleaned_db = 20 * np.log10(cleaned_rms)
            self.noise_reduction_db = original_db - cleaned_db
        else:
            self.noise_reduction_db = 0

    def add_processed_callback(self, callback: Callable):
        """Add callback for processed audio data"""
        self.processed_callbacks.append(callback)

    def _notify_callbacks(self, audio_data: np.ndarray):
        """Notify all registered callbacks with processed audio"""
        for callback in self.processed_callbacks:
            try:
                callback(audio_data)
            except Exception as e:
                self.logger.error(f"Processed audio callback failed: {str(e)}")

    def set_mode(self, mode: NoiseCancelMode):
        """Set noise cancellation mode"""
        self.mode = mode
        self.logger.info(f"Noise cancellation mode set to: {mode.value}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for noise cancellation"""
        return {
            'mode': self.mode.value,
            'noise_profile_ready': self.is_noise_profile_ready,
            'noise_reduction_db': self.noise_reduction_db,
            'processing_time_ms': self.processing_time * 1000,
            'noise_profile_frames': self.noise_profile_counter,
            'enhancement_gain': self.enhancement_gain
        }

    def reset_noise_profile(self):
        """Reset noise profile and start fresh learning"""
        self.noise_profile = None
        self.is_noise_profile_ready = False
        self.noise_profile_counter = 0
        self.noise_psd = np.zeros(self.freq_bins)
        self.logger.info("Noise profile reset")

    def enhance_voice(self, audio_data: np.ndarray, boost_db: float = 6.0) -> np.ndarray:
        """Enhance voice frequencies for better clarity"""
        try:
            # Apply band-pass filter for voice frequencies (300Hz - 3400Hz)
            nyquist = self.sample_rate / 2
            low_cut = 300 / nyquist
            high_cut = 3400 / nyquist
            
            # Design band-pass filter
            b, a = signal.butter(4, [low_cut, high_cut], btype='band')
            voice_band = signal.filtfilt(b, a, audio_data)
            
            # Calculate gain based on desired boost
            gain = 10 ** (boost_db / 20)
            
            # Mix original with enhanced voice band
            enhanced_audio = audio_data + (voice_band * (gain - 1))
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(enhanced_audio))
            if max_val > 1.0:
                enhanced_audio = enhanced_audio / max_val
            
            return enhanced_audio
            
        except Exception as e:
            self.logger.error(f"Voice enhancement failed: {str(e)}")
            return audio_data

    def get_status(self) -> Dict[str, Any]:
        """Get current noise canceler status"""
        status_msg = "Mickey's cleaning audio like a pro! 🧹" if self.is_noise_profile_ready else "Mickey's learning the noise environment 👂"
        
        return {
            'success': True,
            'mode': self.mode.value,
            'noise_profile_ready': self.is_noise_profile_ready,
            'noise_reduction_db': f"{self.noise_reduction_db:.1f} dB",
            'processing_time_ms': f"{self.processing_time * 1000:.1f} ms",
            'mickey_response': status_msg
        }

    def cleanup(self):
        """Cleanup resources"""
        self.processed_callbacks.clear()
        self.audio_buffer.clear()
        self.noise_buffer.clear()

# Test function
def test_noise_canceler():
    """Test the noise canceler with sample audio"""
    import wave
    import time
    
    canceler = NoiseCanceler()
    
    print("Testing Noise Canceler...")
    
    # Generate test audio (sine wave with noise)
    duration = 3.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Clean signal (440 Hz sine wave)
    clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Add noise (white noise + low frequency hum)
    noise = 0.1 * np.random.randn(len(t)) + 0.2 * np.sin(2 * np.pi * 60 * t)
    noisy_signal = clean_signal + noise
    
    print(f"Original signal RMS: {np.sqrt(np.mean(clean_signal**2)):.4f}")
    print(f"Noisy signal RMS: {np.sqrt(np.mean(noisy_signal**2)):.4f}")
    
    # Start noise profiling
    canceler.start_noise_profiling(duration=1.0)
    time.sleep(1.1)  # Wait for profiling to complete
    
    # Process audio
    cleaned_signal = canceler.process_audio(noisy_signal)
    
    print(f"Cleaned signal RMS: {np.sqrt(np.mean(cleaned_signal**2)):.4f}")
    
    # Get performance metrics
    metrics = canceler.get_performance_metrics()
    print("Performance Metrics:", metrics)
    
    # Test different modes
    for mode in [NoiseCancelMode.LIGHT, NoiseCancelMode.MODERATE, NoiseCancelMode.AGGRESSIVE]:
        canceler.set_mode(mode)
        cleaned = canceler.process_audio(noisy_signal)
        rms = np.sqrt(np.mean(cleaned**2))
        print(f"Mode {mode.value}: RMS = {rms:.4f}")
    
    canceler.cleanup()
    print("Noise canceler test completed!")

if __name__ == "__main__":
    test_noise_canceler()