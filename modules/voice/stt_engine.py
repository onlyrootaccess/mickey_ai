# FIRST: Whisper-tiny CPU STT
"""
M.I.C.K.E.Y. AI Assistant - Speech-to-Text Engine
Made In Crisis, Keeping Everything Yours

SIXTH FILE IN PIPELINE: Converts speech audio to text using Whisper model.
Provides real-time speech recognition with noise cancellation and voice activity detection.
"""

import asyncio
import logging
import time
import base64
import threading
import queue
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import wave
import io

# Import audio processing libraries
import whisper
import pyaudio
import numpy as np
import librosa
from scipy import signal

# Import Mickey AI configuration
from config.settings import get_config
from config.constants import (
    AudioConstants, SystemConstants, ErrorCodes, ErrorMessages
)

# Setup logging
logger = logging.getLogger("MickeySTT")


@dataclass
class TranscriptionResult:
    """Speech transcription result container."""
    text: str
    confidence: float
    language: str
    duration: float
    word_timestamps: Optional[List[Dict]] = None
    processing_time: float = 0.0
    has_speech: bool = True


class AudioPreprocessor:
    """Audio preprocessing for noise cancellation and voice enhancement."""
    
    def __init__(self):
        self.config = get_config()
        self.sample_rate = AudioConstants.DEFAULT_SAMPLE_RATE
        self.chunk_size = AudioConstants.DEFAULT_CHUNK_SIZE
        
        # Noise reduction parameters
        self.noise_profile = None
        self.noise_reduction_level = self.config.audio.noise_reduction_level
        
    def apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply spectral noise reduction to audio data."""
        try:
            # Convert to frequency domain
            spec = librosa.stft(audio_data.astype(np.float32))
            
            # Calculate noise threshold based on reduction level
            threshold_db = -20 - (self.noise_reduction_level * 5)  # -25dB to -45dB
            
            # Apply spectral gate
            spec_denoised = librosa.decompose.nn_filter(
                spec,
                aggregate=np.median,
                metric='cosine'
            )
            
            # Convert back to time domain
            audio_denoised = librosa.istft(spec_denoised)
            
            return audio_denoised.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {str(e)}")
            return audio_data
    
    def apply_voice_enhancement(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply voice frequency enhancement."""
        try:
            # Design bandpass filter for human voice frequencies (85Hz - 255Hz fundamental, up to 8kHz harmonics)
            nyquist = self.sample_rate / 2
            lowcut = 85.0 / nyquist
            highcut = 4000.0 / nyquist
            
            # Create Butterworth bandpass filter
            b, a = signal.butter(
                4, [lowcut, highcut], btype='band'
            )
            
            # Apply filter
            enhanced_audio = signal.filtfilt(b, a, audio_data)
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Voice enhancement failed: {str(e)}")
            return audio_data
    
    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio levels to consistent volume."""
        try:
            # Calculate RMS (root mean square) for volume normalization
            rms = np.sqrt(np.mean(audio_data**2))
            
            if rms > 0:
                # Target RMS level
                target_rms = 0.1  # Conservative target to avoid clipping
                gain = target_rms / rms
                
                # Apply gain with soft limiting to prevent clipping
                normalized_audio = np.tanh(audio_data * gain)
                return normalized_audio
            else:
                return audio_data
                
        except Exception as e:
            logger.warning(f"Audio normalization failed: {str(e)}")
            return audio_data
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply full audio preprocessing pipeline."""
        try:
            # Convert to float32 for processing
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0  # Convert from int16
            
            # Apply processing chain
            processed_audio = audio_data
            processed_audio = self.apply_noise_reduction(processed_audio)
            processed_audio = self.apply_voice_enhancement(processed_audio)
            processed_audio = self.normalize_audio(processed_audio)
            
            return processed_audio
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            return audio_data


class VoiceActivityDetector:
    """Voice Activity Detection for efficient speech recognition."""
    
    def __init__(self):
        self.config = get_config()
        self.sample_rate = AudioConstants.DEFAULT_SAMPLE_RATE
        
        # VAD parameters
        self.energy_threshold = 0.01
        self.silence_duration = self.config.audio.silence_duration
        self.min_voice_duration = self.config.audio.min_voice_duration
        
        # State tracking
        self.silence_frames = 0
        self.speech_frames = 0
        self.is_speaking = False
        
    def detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains speech."""
        try:
            # Calculate energy (RMS)
            energy = np.sqrt(np.mean(audio_chunk**2))
            
            # Simple energy-based VAD
            has_voice = energy > self.energy_threshold
            
            # Update state machine
            if has_voice:
                self.speech_frames += 1
                self.silence_frames = 0
                
                # Start speaking if we have enough voice frames
                if not self.is_speaking and self.speech_frames >= self.min_voice_duration * (self.sample_rate / len(audio_chunk)):
                    self.is_speaking = True
                    
            else:
                self.silence_frames += 1
                self.speech_frames = 0
                
                # Stop speaking if we have enough silence
                if self.is_speaking and self.silence_frames >= self.silence_duration * (self.sample_rate / len(audio_chunk)):
                    self.is_speaking = False
            
            return self.is_speaking
            
        except Exception as e:
            logger.warning(f"VAD detection failed: {str(e)}")
            return False
    
    def reset(self):
        """Reset VAD state."""
        self.silence_frames = 0
        self.speech_frames = 0
        self.is_speaking = False


class AudioRecorder:
    """Real-time audio recording with VAD and preprocessing."""
    
    def __init__(self):
        self.config = get_config()
        self.sample_rate = AudioConstants.DEFAULT_SAMPLE_RATE
        self.chunk_size = AudioConstants.DEFAULT_CHUNK_SIZE
        self.format = pyaudio.paInt16
        self.channels = 1
        
        self.audio_interface = None
        self.audio_stream = None
        self.is_recording = False
        self.audio_buffer = []
        self.max_buffer_duration = 10  # seconds
        
        self.preprocessor = AudioPreprocessor()
        self.vad = VoiceActivityDetector()
        
    async def initialize(self):
        """Initialize audio recording system."""
        try:
            logger.info("Initializing Audio Recorder...")
            
            self.audio_interface = pyaudio.PyAudio()
            
            # Test audio devices
            input_devices = self._get_input_devices()
            if not input_devices:
                raise RuntimeError("No audio input devices found")
            
            logger.info(f"Found {len(input_devices)} audio input devices")
            
            logger.info("✅ Audio Recorder initialized")
            
        except Exception as e:
            logger.error(f"❌ Audio Recorder initialization failed: {str(e)}")
            raise
    
    def _get_input_devices(self) -> List[Dict]:
        """Get available audio input devices."""
        devices = []
        try:
            for i in range(self.audio_interface.get_device_count()):
                device_info = self.audio_interface.get_device_info_by_index(i)
                if device_info.get('maxInputChannels', 0) > 0:
                    devices.append({
                        'index': i,
                        'name': device_info.get('name', 'Unknown'),
                        'channels': device_info.get('maxInputChannels', 1),
                        'sample_rate': device_info.get('defaultSampleRate', self.sample_rate)
                    })
        except Exception as e:
            logger.error(f"Error getting audio devices: {str(e)}")
        
        return devices
    
    def start_recording(self, device_index: int = None):
        """Start audio recording."""
        try:
            if self.is_recording:
                logger.warning("Audio recording already in progress")
                return
            
            self.audio_buffer = []
            self.vad.reset()
            
            # Open audio stream
            self.audio_stream = self.audio_interface.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=device_index,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            logger.info("🎤 Audio recording started")
            
        except Exception as e:
            logger.error(f"Failed to start audio recording: {str(e)}")
            raise
    
    def stop_recording(self) -> np.ndarray:
        """Stop audio recording and return captured audio."""
        try:
            if not self.is_recording:
                logger.warning("No audio recording in progress")
                return np.array([])
            
            self.is_recording = False
            
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            
            # Combine all audio chunks
            if self.audio_buffer:
                audio_data = np.concatenate(self.audio_buffer)
                logger.info(f"Recording stopped. Captured {len(audio_data)/self.sample_rate:.2f}s of audio")
                return audio_data
            else:
                logger.warning("No audio data captured")
                return np.array([])
            
        except Exception as e:
            logger.error(f"Error stopping audio recording: {str(e)}")
            return np.array([])
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback for real-time processing."""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        if self.is_recording:
            # Convert bytes to numpy array
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)
            
            # Apply VAD
            if self.vad.detect_voice_activity(audio_chunk):
                # Preprocess and add to buffer
                processed_chunk = self.preprocessor.preprocess_audio(audio_chunk)
                self.audio_buffer.append(processed_chunk)
                
                # Limit buffer size
                total_samples = sum(len(chunk) for chunk in self.audio_buffer)
                max_samples = self.max_buffer_duration * self.sample_rate
                
                while total_samples > max_samples and len(self.audio_buffer) > 1:
                    removed_chunk = self.audio_buffer.pop(0)
                    total_samples -= len(removed_chunk)
        
        return (in_data, pyaudio.paContinue)
    
    def record_until_silence(self, device_index: int = None, max_duration: float = 30.0) -> np.ndarray:
        """Record audio until silence is detected or max duration reached."""
        self.start_recording(device_index)
        
        start_time = time.time()
        last_voice_time = time.time()
        
        try:
            while self.is_recording:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check for timeout
                if elapsed > max_duration:
                    logger.info("Maximum recording duration reached")
                    break
                
                # Check for extended silence
                if self.vad.is_speaking:
                    last_voice_time = current_time
                else:
                    silence_duration = current_time - last_voice_time
                    if silence_duration > self.config.audio.silence_duration:
                        logger.info("Silence detected, stopping recording")
                        break
                
                # Small sleep to prevent busy waiting
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Recording interrupted by user")
        
        finally:
            return self.stop_recording()
    
    async def shutdown(self):
        """Shutdown audio recording system."""
        try:
            if self.audio_stream and self.audio_stream.is_active():
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            
            if self.audio_interface:
                self.audio_interface.terminate()
            
            logger.info("✅ Audio Recorder shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during audio recorder shutdown: {str(e)}")


class STTEngine:
    """
    Speech-to-Text engine using OpenAI Whisper.
    Provides both real-time and batch transcription capabilities.
    """
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.config = get_config()
        self.audio_recorder = AudioRecorder()
        self.preprocessor = AudioPreprocessor()
        
        # Performance tracking
        self.total_processing_time = 0.0
        self.total_audio_processed = 0.0
        self.transcription_count = 0
        
    async def initialize(self):
        """Initialize the STT engine and load Whisper model."""
        try:
            logger.info("Initializing STT Engine...")
            
            # Load Whisper model
            model_size = "tiny"  # Fastest model for real-time performance
            logger.info(f"Loading Whisper model: {model_size}")
            
            self.model = whisper.load_model(model_size)
            logger.info(f"✅ Whisper model '{model_size}' loaded successfully")
            
            # Initialize audio recorder
            await self.audio_recorder.initialize()
            
            self.is_initialized = True
            logger.info("✅ STT Engine initialized")
            
        except Exception as e:
            logger.error(f"❌ STT Engine initialization failed: {str(e)}")
            raise
    
    def _audio_to_numpy(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array."""
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 for Whisper
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            return audio_float
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {str(e)}")
            raise
    
    def _prepare_audio_for_whisper(self, audio_data: np.ndarray) -> np.ndarray:
        """Prepare audio data for Whisper processing."""
        try:
            # Resample to 16kHz if necessary (Whisper's expected sample rate)
            if len(audio_data) > 0:
                current_sr = self.config.audio.sample_rate
                target_sr = 16000  # Whisper's sample rate
                
                if current_sr != target_sr:
                    audio_data = librosa.resample(
                        audio_data, 
                        orig_sr=current_sr, 
                        target_sr=target_sr
                    )
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio preparation failed: {str(e)}")
            return audio_data
    
    async def transcribe_audio(self, audio_data: np.ndarray, language: str = "en") -> TranscriptionResult:
        """Transcribe audio data to text using Whisper."""
        start_time = time.time()
        
        try:
            if not self.is_initialized or self.model is None:
                raise RuntimeError("STT Engine not initialized")
            
            # Validate audio data
            if len(audio_data) == 0:
                return TranscriptionResult(
                    text="",
                    confidence=0.0,
                    language=language,
                    duration=0.0,
                    has_speech=False,
                    processing_time=0.0
                )
            
            # Prepare audio for Whisper
            prepared_audio = self._prepare_audio_for_whisper(audio_data)
            audio_duration = len(prepared_audio) / 16000  # Duration in seconds
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                prepared_audio,
                language=language,
                fp16=False,  # Use FP32 for CPU compatibility
                task="transcribe"
            )
            
            # Extract transcription results
            text = result.get("text", "").strip()
            segments = result.get("segments", [])
            
            # Calculate overall confidence
            confidence = 0.0
            if segments:
                confidences = [seg.get("confidence", 0.0) for seg in segments]
                confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Extract word timestamps if available
            word_timestamps = []
            for segment in segments:
                words = segment.get("words", [])
                for word_info in words:
                    word_timestamps.append({
                        "word": word_info.get("word", ""),
                        "start": word_info.get("start", 0.0),
                        "end": word_info.get("end", 0.0),
                        "confidence": word_info.get("probability", 0.0)
                    })
            
            processing_time = time.time() - start_time
            
            # Update performance metrics
            self.total_processing_time += processing_time
            self.total_audio_processed += audio_duration
            self.transcription_count += 1
            
            logger.info(f"Transcription: '{text}' (confidence: {confidence:.3f}, duration: {processing_time:.2f}s)")
            
            return TranscriptionResult(
                text=text,
                confidence=confidence,
                language=language,
                duration=audio_duration,
                word_timestamps=word_timestamps if word_timestamps else None,
                processing_time=processing_time,
                has_speech=len(text) > 0
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            processing_time = time.time() - start_time
            
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=language,
                duration=0.0,
                processing_time=processing_time,
                has_speech=False
            )
    
    async def transcribe(self, audio_data: str, language: str = "en", include_timestamps: bool = False) -> Dict[str, Any]:
        """
        Transcribe base64 encoded audio data to text.
        Main API method for the STT engine.
        """
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)
            audio_np = self._audio_to_numpy(audio_bytes)
            
            # Preprocess audio
            processed_audio = self.preprocessor.preprocess_audio(audio_np)
            
            # Transcribe
            result = await self.transcribe_audio(processed_audio, language)
            
            # Build response
            response = {
                "text": result.text,
                "confidence": result.confidence,
                "language": result.language,
                "duration": result.duration,
                "processing_time": result.processing_time
            }
            
            if include_timestamps and result.word_timestamps:
                response["word_timestamps"] = result.word_timestamps
            
            return response
            
        except Exception as e:
            logger.error(f"STT processing failed: {str(e)}")
            raise
    
    async def record_and_transcribe(self, language: str = "en", max_duration: float = 30.0) -> Dict[str, Any]:
        """Record audio from microphone and transcribe it."""
        try:
            logger.info("Starting voice recording...")
            
            # Record audio until silence
            recorded_audio = self.audio_recorder.record_until_silence(max_duration=max_duration)
            
            if len(recorded_audio) == 0:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "language": language,
                    "duration": 0.0,
                    "processing_time": 0.0,
                    "message": "No audio recorded"
                }
            
            # Transcribe recorded audio
            result = await self.transcribe_audio(recorded_audio, language)
            
            response = {
                "text": result.text,
                "confidence": result.confidence,
                "language": result.language,
                "duration": result.duration,
                "processing_time": result.processing_time,
                "message": "Transcription completed successfully"
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Record and transcribe failed: {str(e)}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": language,
                "duration": 0.0,
                "processing_time": 0.0,
                "message": f"Recording failed: {str(e)}"
            }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get STT engine performance metrics."""
        avg_processing_time = (
            self.total_processing_time / self.transcription_count 
            if self.transcription_count > 0 else 0
        )
        
        real_time_factor = (
            self.total_audio_processed / self.total_processing_time 
            if self.total_processing_time > 0 else 0
        )
        
        return {
            "initialized": self.is_initialized,
            "transcription_count": self.transcription_count,
            "total_audio_processed_seconds": self.total_audio_processed,
            "total_processing_time_seconds": self.total_processing_time,
            "average_processing_time_seconds": avg_processing_time,
            "real_time_factor": real_time_factor,  # >1 means faster than real-time
            "model_loaded": self.model is not None
        }
    
    async def shutdown(self):
        """Shutdown STT engine gracefully."""
        logger.info("Shutting down STT Engine...")
        
        try:
            await self.audio_recorder.shutdown()
            logger.info("✅ STT Engine shutdown complete")
        except Exception as e:
            logger.error(f"Error during STT engine shutdown: {str(e)}")


# Global STT instance
_stt_instance: Optional[STTEngine] = None


async def get_stt_engine() -> STTEngine:
    """Get or create global STT engine instance."""
    global _stt_instance
    
    if _stt_instance is None:
        _stt_instance = STTEngine()
        await _stt_instance.initialize()
    
    return _stt_instance


async def main():
    """Command-line testing for STT engine."""
    stt_engine = await get_stt_engine()
    
    # Test performance metrics
    metrics = await stt_engine.get_performance_metrics()
    print("STT Engine Status:")
    print(f"Initialized: {metrics['initialized']}")
    print(f"Model Loaded: {metrics['model_loaded']}")
    print(f"Ready for transcription")
    
    # Test with a dummy audio (would need real audio for actual transcription)
    print("\nTo test transcription, use the API endpoints or provide audio data.")


if __name__ == "__main__":
    asyncio.run(main())