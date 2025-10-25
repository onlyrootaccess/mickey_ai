# Voice timbre matching
"""
M.I.C.K.E.Y. AI Assistant - Voice Biometrics Engine
Made In Crisis, Keeping Everything Yours

TWELFTH FILE IN PIPELINE: Advanced voice biometrics engine for user authentication.
Uses MFCC features and voice print matching for secure voice recognition.
"""

import asyncio
import logging
import time
import base64
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass

# Import audio processing libraries
import numpy as np
import librosa
from scipy.spatial.distance import cosine, euclidean
from scipy import signal
import pyaudio

# Import Mickey AI configuration
from config.settings import get_config
from config.constants import SecurityConstants, AudioConstants, ErrorCodes, ErrorMessages

# Setup logging
logger = logging.getLogger("MickeyVoiceBiometrics")


@dataclass
class VoiceFeatureResult:
    """Voice feature extraction result container."""
    success: bool
    features: Optional[np.ndarray] = None
    sample_rate: int = 0
    duration: float = 0.0
    processing_time: float = 0.0
    error_message: Optional[str] = None


@dataclass
class VoiceMatchResult:
    """Voice matching result container."""
    success: bool
    match_found: bool
    user_id: Optional[str] = None
    confidence: float = 0.0
    best_match_index: Optional[int] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None


class VoiceFeatureExtractor:
    """Extracts and processes voice features for biometric identification."""
    
    def __init__(self):
        self.config = get_config()
        self.sample_rate = AudioConstants.DEFAULT_SAMPLE_RATE
        
    def extract_mfcc_features(self, audio_data: np.ndarray, sample_rate: int = None) -> np.ndarray:
        """Extract MFCC features from audio data."""
        try:
            if sample_rate is None:
                sample_rate = self.sample_rate
            
            # Ensure audio is in float32 format
            if audio_data.dtype != np.float32:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data
            
            # Extract MFCC features with enhanced parameters
            mfccs = librosa.feature.mfcc(
                y=audio_float,
                sr=sample_rate,
                n_mfcc=SecurityConstants.VOICE_FEATURES_TO_EXTRACT,
                n_fft=2048,
                hop_length=512,
                n_mels=128
            )
            
            # Extract delta features (temporal changes)
            mfcc_delta = librosa.feature.delta(mfccs)
            
            # Extract delta-delta features (acceleration)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Combine all features
            combined_features = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
            
            # Take mean across time to get fixed-length feature vector
            feature_vector = np.mean(combined_features, axis=1)
            
            # Normalize the feature vector
            feature_vector = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"MFCC feature extraction failed: {str(e)}")
            raise
    
    def extract_spectral_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract additional spectral features for enhanced recognition."""
        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate
            )
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate
            )
            
            # Zero crossing rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            
            # Combine spectral features
            spectral_features = np.array([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(zero_crossing_rate),
                np.std(zero_crossing_rate)
            ])
            
            return spectral_features
            
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {str(e)}")
            return np.array([])
    
    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocess audio for better feature extraction."""
        try:
            # High-pass filter to remove low-frequency noise
            nyquist = sample_rate / 2
            highpass_cutoff = 80.0 / nyquist
            
            b, a = signal.butter(4, highpass_cutoff, btype='high')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # Voice activity detection (simple energy-based)
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            
            energy = []
            for i in range(0, len(filtered_audio) - frame_length, hop_length):
                frame = filtered_audio[i:i + frame_length]
                energy.append(np.sum(frame ** 2))
            
            if energy:
                energy_threshold = np.mean(energy) * 0.1
                voice_frames = [e > energy_threshold for e in energy]
                
                # Find voice segments
                voice_segments = []
                in_voice = False
                start_idx = 0
                
                for i, is_voice in enumerate(voice_frames):
                    if is_voice and not in_voice:
                        start_idx = i * hop_length
                        in_voice = True
                    elif not is_voice and in_voice:
                        end_idx = i * hop_length + frame_length
                        voice_segments.append((start_idx, end_idx))
                        in_voice = False
                
                # Use the longest voice segment
                if voice_segments:
                    segments_lengths = [end - start for start, end in voice_segments]
                    longest_segment_idx = np.argmax(segments_lengths)
                    start, end = voice_segments[longest_segment_idx]
                    
                    # Extract voice segment
                    voice_audio = filtered_audio[start:end]
                    
                    # Ensure minimum duration
                    if len(voice_audio) / sample_rate >= SecurityConstants.MIN_VOICE_SAMPLE_SECONDS:
                        return voice_audio
            
            # If no clear voice segments found, return original
            return filtered_audio
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {str(e)}")
            return audio_data
    
    async def extract_voice_features(self, audio_data: bytes) -> VoiceFeatureResult:
        """Extract comprehensive voice features from audio data."""
        start_time = time.time()
        
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_duration = len(audio_np) / self.sample_rate
            
            # Validate audio duration
            if audio_duration < SecurityConstants.MIN_VOICE_SAMPLE_SECONDS:
                return VoiceFeatureResult(
                    success=False,
                    duration=audio_duration,
                    processing_time=time.time() - start_time,
                    error_message=f"Audio too short: {audio_duration:.1f}s (minimum {SecurityConstants.MIN_VOICE_SAMPLE_SECONDS}s)"
                )
            
            if audio_duration > SecurityConstants.MAX_VOICE_SAMPLE_SECONDS:
                return VoiceFeatureResult(
                    success=False,
                    duration=audio_duration,
                    processing_time=time.time() - start_time,
                    error_message=f"Audio too long: {audio_duration:.1f}s (maximum {SecurityConstants.MAX_VOICE_SAMPLE_SECONDS}s)"
                )
            
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_np, self.sample_rate)
            
            # Extract primary MFCC features
            mfcc_features = self.extract_mfcc_features(processed_audio, self.sample_rate)
            
            # Extract additional spectral features
            spectral_features = self.extract_spectral_features(processed_audio, self.sample_rate)
            
            # Combine all features
            if len(spectral_features) > 0:
                combined_features = np.concatenate([mfcc_features, spectral_features])
            else:
                combined_features = mfcc_features
            
            processing_time = time.time() - start_time
            
            return VoiceFeatureResult(
                success=True,
                features=combined_features,
                sample_rate=self.sample_rate,
                duration=audio_duration,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Voice feature extraction failed: {str(e)}")
            return VoiceFeatureResult(
                success=False,
                processing_time=time.time() - start_time,
                error_message=f"Feature extraction error: {str(e)}"
            )


class VoicePrintDatabase:
    """Manages voice print database storage and retrieval."""
    
    def __init__(self):
        self.config = get_config()
        self.voice_prints: Dict[str, List[np.ndarray]] = {}  # user_id -> list of feature vectors
        self.initialized = False
        
    async def initialize(self):
        """Initialize voice print database."""
        try:
            logger.info("Initializing Voice Print Database...")
            
            # Load existing voice prints
            await self._load_voice_prints()
            
            self.initialized = True
            logger.info(f"✅ Voice Print Database initialized with {len(self.voice_prints)} users")
            
        except Exception as e:
            logger.error(f"❌ Voice Print Database initialization failed: {str(e)}")
            raise
    
    async def _load_voice_prints(self):
        """Load voice prints from secure storage."""
        voice_prints_path = Path(get_config().security_dir) / "voice_prints.json"
        
        if not voice_prints_path.exists():
            logger.warning("No existing voice prints found. New prints will be created.")
            return
        
        try:
            with open(voice_prints_path, 'r', encoding='utf-8') as f:
                prints_data = json.load(f)
            
            for user_id, user_data in prints_data.items():
                feature_vectors = []
                for feature_b64 in user_data.get('feature_vectors', []):
                    feature_bytes = base64.b64decode(feature_b64)
                    feature_array = np.frombuffer(feature_bytes, dtype=np.float64)
                    feature_vectors.append(feature_array)
                
                self.voice_prints[user_id] = feature_vectors
            
            logger.info(f"Loaded voice prints for {len(self.voice_prints)} users")
            
        except Exception as e:
            logger.error(f"Failed to load voice prints: {str(e)}")
    
    async def _save_voice_prints(self):
        """Save voice prints to secure storage."""
        try:
            prints_data = {}
            
            for user_id, feature_vectors in self.voice_prints.items():
                features_b64 = []
                for feature_vector in feature_vectors:
                    feature_bytes = feature_vector.tobytes()
                    features_b64.append(base64.b64encode(feature_bytes).decode('utf-8'))
                
                prints_data[user_id] = {
                    'feature_vectors': features_b64,
                    'count': len(feature_vectors),
                    'last_updated': time.time()
                }
            
            voice_prints_path = Path(get_config().security_dir) / "voice_prints.json"
            with open(voice_prints_path, 'w', encoding='utf-8') as f:
                json.dump(prints_data, f, indent=2)
            
            logger.info(f"Saved voice prints for {len(self.voice_prints)} users")
            
        except Exception as e:
            logger.error(f"Failed to save voice prints: {str(e)}")
            raise
    
    def add_voice_print(self, user_id: str, feature_vector: np.ndarray):
        """Add a voice print to the database."""
        if user_id not in self.voice_prints:
            self.voice_prints[user_id] = []
        
        self.voice_prints[user_id].append(feature_vector)
        
        # Limit the number of stored prints per user
        if len(self.voice_prints[user_id]) > 5:
            self.voice_prints[user_id] = self.voice_prints[user_id][-5:]
    
    def get_user_voice_prints(self, user_id: str) -> List[np.ndarray]:
        """Get all voice prints for a user."""
        return self.voice_prints.get(user_id, [])
    
    def get_all_users(self) -> List[str]:
        """Get list of all enrolled users."""
        return list(self.voice_prints.keys())
    
    def remove_user_prints(self, user_id: str) -> bool:
        """Remove all voice prints for a user."""
        if user_id in self.voice_prints:
            del self.voice_prints[user_id]
            return True
        return False


class VoiceBiometricsEngine:
    """
    Advanced voice biometrics engine for user authentication.
    Uses MFCC features and voice print matching for secure voice recognition.
    """
    
    def __init__(self):
        self.config = get_config()
        self.feature_extractor = VoiceFeatureExtractor()
        self.voice_database = VoicePrintDatabase()
        self.is_initialized = False
        
        # Performance tracking
        self.total_enrollments = 0
        self.total_verifications = 0
        self.total_processing_time = 0.0
        
        # Audio recording for live capture
        self.audio_interface = None
        
    async def initialize(self):
        """Initialize the voice biometrics engine."""
        try:
            logger.info("Initializing Voice Biometrics Engine...")
            
            # Initialize voice print database
            await self.voice_database.initialize()
            
            # Initialize audio interface
            await self._initialize_audio_interface()
            
            self.is_initialized = True
            logger.info("✅ Voice Biometrics Engine initialized")
            
        except Exception as e:
            logger.error(f"❌ Voice Biometrics Engine initialization failed: {str(e)}")
            raise
    
    async def _initialize_audio_interface(self):
        """Initialize audio interface for live recording."""
        try:
            self.audio_interface = pyaudio.PyAudio()
            
            # Test microphone availability
            input_devices = self._get_input_devices()
            if input_devices:
                logger.info(f"✅ Microphone available: {len(input_devices)} input devices found")
            else:
                logger.warning("⚠️ No microphone input devices found")
                
        except Exception as e:
            logger.warning(f"Audio interface initialization warning: {str(e)}")
    
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
                        'sample_rate': device_info.get('defaultSampleRate', AudioConstants.DEFAULT_SAMPLE_RATE)
                    })
        except Exception as e:
            logger.error(f"Error getting audio devices: {str(e)}")
        
        return devices
    
    async def enroll_voice(self, audio_data: str, user_id: str) -> Dict[str, Any]:
        """Enroll a new voice for authentication."""
        start_time = time.time()
        
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)
            
            # Extract voice features
            feature_result = await self.feature_extractor.extract_voice_features(audio_bytes)
            
            if not feature_result.success:
                return {
                    'success': False,
                    'user_id': user_id,
                    'confidence': 0.0,
                    'message': feature_result.error_message,
                    'processing_time': time.time() - start_time
                }
            
            # Check voice print quality
            if np.std(feature_result.features) < 0.1:  # Low variance might indicate poor audio
                return {
                    'success': False,
                    'user_id': user_id,
                    'confidence': 0.0,
                    'message': "Poor voice quality detected - please try again with clearer audio",
                    'processing_time': time.time() - start_time
                }
            
            # Check if this voice is already enrolled (basic duplicate check)
            existing_prints = self.voice_database.get_user_voice_prints(user_id)
            if existing_prints:
                similarities = []
                for existing_print in existing_prints:
                    similarity = 1 - cosine(feature_result.features, existing_print)
                    similarities.append(similarity)
                
                avg_similarity = np.mean(similarities) if similarities else 0
                if avg_similarity > 0.8:  # Very similar to existing prints
                    return {
                        'success': False,
                        'user_id': user_id,
                        'confidence': avg_similarity,
                        'message': "Voice print too similar to existing enrollment",
                        'processing_time': time.time() - start_time
                    }
            
            # Add to database
            self.voice_database.add_voice_print(user_id, feature_result.features)
            
            # Save updated database
            await self.voice_database._save_voice_prints()
            
            processing_time = time.time() - start_time
            self.total_enrollments += 1
            self.total_processing_time += processing_time
            
            return {
                'success': True,
                'user_id': user_id,
                'confidence': 1.0,
                'message': "Voice enrolled successfully",
                'duration': feature_result.duration,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Voice enrollment failed: {str(e)}")
            return {
                'success': False,
                'user_id': user_id,
                'confidence': 0.0,
                'message': f"Voice enrollment error: {str(e)}",
                'processing_time': time.time() - start_time
            }
    
    async def recognize_voice(self, audio_data: str) -> VoiceMatchResult:
        """Recognize voice against enrolled profiles."""
        start_time = time.time()
        
        try:
            if not self.voice_database.voice_prints:
                return VoiceMatchResult(
                    success=True,
                    match_found=False,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    error_message="No enrolled voices in system"
                )
            
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)
            
            # Extract voice features
            feature_result = await self.feature_extractor.extract_voice_features(audio_bytes)
            
            if not feature_result.success:
                return VoiceMatchResult(
                    success=False,
                    match_found=False,
                    processing_time=time.time() - start_time,
                    error_message=feature_result.error_message
                )
            
            input_features = feature_result.features
            
            # Compare with all enrolled voice prints
            best_confidence = 0.0
            best_user = None
            best_match_index = None
            
            for user_id, voice_prints in self.voice_database.voice_prints.items():
                user_similarities = []
                
                for print_index, voice_print in enumerate(voice_prints):
                    # Calculate cosine similarity
                    similarity = 1 - cosine(input_features, voice_print)
                    user_similarities.append(similarity)
                
                if user_similarities:
                    user_confidence = max(user_similarities)
                    if user_confidence > best_confidence:
                        best_confidence = user_confidence
                        best_user = user_id
                        best_match_index = user_similarities.index(user_confidence)
            
            processing_time = time.time() - start_time
            self.total_verifications += 1
            self.total_processing_time += processing_time
            
            if best_confidence >= SecurityConstants.VOICE_MATCH_THRESHOLD:
                logger.info(f"Voice recognized: {best_user} (confidence: {best_confidence:.3f})")
                
                return VoiceMatchResult(
                    success=True,
                    match_found=True,
                    user_id=best_user,
                    confidence=best_confidence,
                    best_match_index=best_match_index,
                    processing_time=processing_time
                )
            else:
                logger.info(f"Voice not recognized (best confidence: {best_confidence:.3f})")
                
                return VoiceMatchResult(
                    success=True,
                    match_found=False,
                    confidence=best_confidence,
                    processing_time=processing_time,
                    error_message="Voice not recognized"
                )
            
        except Exception as e:
            logger.error(f"Voice recognition failed: {str(e)}")
            return VoiceMatchResult(
                success=False,
                match_found=False,
                processing_time=time.time() - start_time,
                error_message=f"Voice recognition error: {str(e)}"
            )
    
    async def capture_live_voice(self, duration: float = 5.0, device_index: int = None) -> Optional[str]:
        """Capture voice from live microphone."""
        try:
            if not self.audio_interface:
                logger.error("Audio interface not initialized")
                return None
            
            sample_rate = AudioConstants.DEFAULT_SAMPLE_RATE
            chunk_size = AudioConstants.DEFAULT_CHUNK_SIZE
            format = pyaudio.paInt16
            channels = 1
            
            # Calculate total chunks needed
            total_chunks = int((duration * sample_rate) / chunk_size)
            
            logger.info(f"Starting live voice capture for {duration} seconds...")
            
            stream = self.audio_interface.open(
                format=format,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
                input_device_index=device_index
            )
            
            frames = []
            
            for i in range(total_chunks):
                data = stream.read(chunk_size)
                frames.append(data)
                
                # Simple voice activity detection for early termination
                if len(frames) > 10:  # After 1 second
                    audio_data = b''.join(frames[-10:])
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Check energy level
                    energy = np.sum(audio_np.astype(np.float32) ** 2) / len(audio_np)
                    if energy < 1000:  # Very low energy - probably silence
                        logger.info("Silence detected, stopping capture early")
                        break
            
            stream.stop_stream()
            stream.close()
            
            # Combine all frames
            audio_data = b''.join(frames)
            
            # Convert to base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            logger.info(f"Captured {len(audio_data)/2/sample_rate:.1f}s of audio")
            return audio_b64
            
        except Exception as e:
            logger.error(f"Live voice capture failed: {str(e)}")
            return None
    
    async def verify_voice(self, audio_data: str, expected_user: str) -> Dict[str, Any]:
        """Verify if voice matches specific user."""
        match_result = await self.recognize_voice(audio_data)
        
        if not match_result.success:
            return {
                'success': False,
                'verified': False,
                'confidence': 0.0,
                'message': match_result.error_message
            }
        
        if not match_result.match_found:
            return {
                'success': True,
                'verified': False,
                'confidence': match_result.confidence,
                'message': "Voice not recognized"
            }
        
        verified = (match_result.user_id == expected_user)
        
        return {
            'success': True,
            'verified': verified,
            'user_id': match_result.user_id,
            'confidence': match_result.confidence,
            'message': "Voice verified" if verified else "Voice does not match expected user"
        }
    
    async def get_voice_quality_score(self, audio_data: str) -> Dict[str, Any]:
        """Analyze voice quality for enrollment suitability."""
        try:
            audio_bytes = base64.b64decode(audio_data)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Calculate various quality metrics
            duration = len(audio_np) / self.feature_extractor.sample_rate
            
            # Signal-to-noise ratio (simplified)
            energy = np.sum(audio_np.astype(np.float32) ** 2) / len(audio_np)
            noise_floor = 1000  # Arbitrary threshold
            snr = 10 * np.log10(energy / noise_floor) if energy > noise_floor else 0
            
            # Spectral flatness (measure of noisiness)
            spectral_flatness = librosa.feature.spectral_flatness(y=audio_np.astype(np.float32))[0]
            avg_flatness = np.mean(spectral_flatness)
            
            # Zero crossing rate (speech vs noise indicator)
            zcr = librosa.feature.zero_crossing_rate(audio_np.astype(np.float32))[0]
            avg_zcr = np.mean(zcr)
            
            # Overall quality score (0-100)
            quality_score = min(100, max(0, 
                (duration / SecurityConstants.MAX_VOICE_SAMPLE_SECONDS * 30) +
                (min(snr, 30) / 30 * 40) +  # SNR contribution
                ((1 - avg_flatness) * 20) +  # Lower flatness is better for speech
                (min(avg_zcr * 100, 10))     # ZCR contribution
            ))
            
            return {
                'success': True,
                'quality_score': quality_score,
                'duration': duration,
                'snr_db': snr,
                'spectral_flatness': avg_flatness,
                'zero_crossing_rate': avg_zcr,
                'suitable_for_enrollment': quality_score >= 60
            }
            
        except Exception as e:
            logger.error(f"Voice quality analysis failed: {str(e)}")
            return {
                'success': False,
                'quality_score': 0,
                'suitable_for_enrollment': False,
                'error_message': str(e)
            }
    
    def get_enrolled_users(self) -> List[str]:
        """Get list of enrolled user IDs."""
        return self.voice_database.get_all_users()
    
    async def remove_user_voice(self, user_id: str) -> bool:
        """Remove all voice data for a specific user."""
        try:
            success = self.voice_database.remove_user_prints(user_id)
            if success:
                await self.voice_database._save_voice_prints()
                logger.info(f"Removed voice data for user: {user_id}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to remove user voice: {str(e)}")
            return False
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get voice biometrics engine performance metrics."""
        avg_processing_time = (
            self.total_processing_time / (self.total_enrollments + self.total_verifications)
            if (self.total_enrollments + self.total_verifications) > 0 else 0
        )
        
        return {
            "initialized": self.is_initialized,
            "enrolled_users": len(self.get_enrolled_users()),
            "total_voice_prints": sum(len(prints) for prints in self.voice_database.voice_prints.values()),
            "total_enrollments": self.total_enrollments,
            "total_verifications": self.total_verifications,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time
        }
    
    async def shutdown(self):
        """Shutdown voice biometrics engine gracefully."""
        logger.info("Shutting down Voice Biometrics Engine...")
        
        try:
            # Save any pending data
            await self.voice_database._save_voice_prints()
            
            # Close audio interface
            if self.audio_interface:
                self.audio_interface.terminate()
            
            logger.info("✅ Voice Biometrics Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during voice biometrics shutdown: {str(e)}")


# Global voice biometrics instance
_voice_biometrics_instance: Optional[VoiceBiometricsEngine] = None


async def get_voice_biometrics_engine() -> VoiceBiometricsEngine:
    """Get or create global voice biometrics engine instance."""
    global _voice_biometrics_instance
    
    if _voice_biometrics_instance is None:
        _voice_biometrics_instance = VoiceBiometricsEngine()
        await _voice_biometrics_instance.initialize()
    
    return _voice_biometrics_instance


async def main():
    """Command-line testing for voice biometrics engine."""
    voice_engine = await get_voice_biometrics_engine()
    
    # Test performance metrics
    metrics = await voice_engine.get_performance_metrics()
    print("Voice Biometrics Engine Status:")
    print(f"Initialized: {metrics['initialized']}")
    print(f"Enrolled Users: {metrics['enrolled_users']}")
    print(f"Total Voice Prints: {metrics['total_voice_prints']}")
    print(f"Total Enrollments: {metrics['total_enrollments']}")
    print(f"Total Verifications: {metrics['total_verifications']}")
    
    if metrics['enrolled_users'] == 0:
        print("\n⚠️ No voices enrolled. Use enroll_voice() to add voices.")
    else:
        print(f"\nEnrolled Users: {voice_engine.get_enrolled_users()}")


if __name__ == "__main__":
    asyncio.run(main())