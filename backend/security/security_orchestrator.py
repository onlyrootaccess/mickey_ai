# Unlocks system only after dual auth
"""
M.I.C.K.E.Y. AI Assistant - Security Orchestrator
Made In Crisis, Keeping Everything Yours

FIFTH FILE IN PIPELINE: Core security system handling face recognition, 
voice biometrics, and dual authentication. Protects access to Mickey AI.
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
import threading

# Import security libraries
import cv2
import numpy as np
import face_recognition
import librosa
from scipy.spatial.distance import cosine

# Import Mickey AI configuration
from config.settings import get_config
from config.constants import (
    SecurityConstants, ErrorCodes, ErrorMessages,
    AudioConstants, SystemConstants
)

# Setup logging
logger = logging.getLogger("MickeySecurity")


@dataclass
class FaceAuthResult:
    """Face authentication result container."""
    success: bool
    confidence: float
    user_id: Optional[str] = None
    message: str = ""
    face_detected: bool = False
    processing_time: float = 0.0


@dataclass
class VoiceAuthResult:
    """Voice authentication result container."""
    success: bool
    confidence: float
    user_id: Optional[str] = None
    message: str = ""
    voice_features_extracted: bool = False
    processing_time: float = 0.0


@dataclass
class SecurityProfile:
    """User security profile containing face and voice data."""
    user_id: str
    face_encodings: List[np.ndarray]
    voice_features: List[np.ndarray]
    created_at: float
    last_used: float
    usage_count: int = 0


class FaceRecognitionEngine:
    """Face recognition engine using OpenCV and face_recognition library."""
    
    def __init__(self):
        self.known_faces: List[np.ndarray] = []
        self.known_users: List[str] = []
        self.is_initialized = False
        self.config = get_config()
        
    async def initialize(self):
        """Initialize face recognition engine."""
        try:
            logger.info("Initializing Face Recognition Engine...")
            
            # Load existing face profiles
            await self._load_face_profiles()
            
            self.is_initialized = True
            logger.info("✅ Face Recognition Engine initialized")
            
        except Exception as e:
            logger.error(f"❌ Face Recognition Engine initialization failed: {str(e)}")
            raise
    
    async def _load_face_profiles(self):
        """Load face profiles from secure storage."""
        face_profiles_path = Path(get_config().security_dir) / "face_profiles.json"
        
        if not face_profiles_path.exists():
            logger.warning("No existing face profiles found. New profiles will be created.")
            return
        
        try:
            with open(face_profiles_path, 'r') as f:
                profiles_data = json.load(f)
            
            for user_id, profile_data in profiles_data.items():
                # Convert base64 encoded face encodings back to numpy arrays
                face_encodings = []
                for encoding_b64 in profile_data.get('face_encodings', []):
                    encoding_bytes = base64.b64decode(encoding_b64)
                    encoding_array = np.frombuffer(encoding_bytes, dtype=np.float64)
                    face_encodings.append(encoding_array)
                
                self.known_faces.extend(face_encodings)
                self.known_users.extend([user_id] * len(face_encodings))
            
            logger.info(f"Loaded {len(self.known_faces)} face encodings for {len(set(self.known_users))} users")
            
        except Exception as e:
            logger.error(f"Failed to load face profiles: {str(e)}")
            # Don't raise - start with empty profiles
    
    async def _save_face_profiles(self):
        """Save face profiles to secure storage."""
        try:
            # Group encodings by user
            user_encodings = {}
            for i, user_id in enumerate(self.known_users):
                if user_id not in user_encodings:
                    user_encodings[user_id] = []
                user_encodings[user_id].append(self.known_faces[i])
            
            # Convert to serializable format
            profiles_data = {}
            for user_id, encodings in user_encodings.items():
                encodings_b64 = []
                for encoding in encodings:
                    encoding_bytes = encoding.tobytes()
                    encoding_b64 = base64.b64encode(encoding_bytes).decode('utf-8')
                    encodings_b64.append(encoding_b64)
                
                profiles_data[user_id] = {
                    'face_encodings': encodings_b64,
                    'count': len(encodings)
                }
            
            # Save to file
            face_profiles_path = Path(get_config().security_dir) / "face_profiles.json"
            with open(face_profiles_path, 'w') as f:
                json.dump(profiles_data, f, indent=2)
            
            logger.info(f"Saved {len(self.known_faces)} face encodings to secure storage")
            
        except Exception as e:
            logger.error(f"Failed to save face profiles: {str(e)}")
    
    async def enroll_face(self, image_data: str, user_id: str) -> FaceAuthResult:
        """Enroll a new face for authentication."""
        start_time = time.time()
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return FaceAuthResult(
                    success=False,
                    confidence=0.0,
                    message="Failed to decode image",
                    face_detected=False,
                    processing_time=time.time() - start_time
                )
            
            # Convert BGR to RGB (face_recognition uses RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(
                rgb_image, 
                model=SecurityConstants.FACE_ENCODING_MODEL
            )
            
            if not face_locations:
                return FaceAuthResult(
                    success=False,
                    confidence=0.0,
                    message="No face detected in image",
                    face_detected=False,
                    processing_time=time.time() - start_time
                )
            
            # Extract face encodings
            face_encodings = face_recognition.face_encodings(
                rgb_image, 
                face_locations
            )
            
            if not face_encodings:
                return FaceAuthResult(
                    success=False,
                    confidence=0.0,
                    message="Failed to extract face features",
                    face_detected=True,
                    processing_time=time.time() - start_time
                )
            
            # Use the first face found
            face_encoding = face_encodings[0]
            
            # Store the encoding
            self.known_faces.append(face_encoding)
            self.known_users.append(user_id)
            
            # Save updated profiles
            await self._save_face_profiles()
            
            processing_time = time.time() - start_time
            
            return FaceAuthResult(
                success=True,
                confidence=1.0,
                user_id=user_id,
                message="Face enrolled successfully",
                face_detected=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Face enrollment failed: {str(e)}")
            return FaceAuthResult(
                success=False,
                confidence=0.0,
                message=f"Face enrollment error: {str(e)}",
                face_detected=False,
                processing_time=time.time() - start_time
            )
    
    async def authenticate_face(self, image_data: str) -> FaceAuthResult:
        """Authenticate user using face recognition."""
        start_time = time.time()
        
        try:
            if not self.known_faces:
                return FaceAuthResult(
                    success=False,
                    confidence=0.0,
                    message="No enrolled faces found",
                    face_detected=False,
                    processing_time=time.time() - start_time
                )
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return FaceAuthResult(
                    success=False,
                    confidence=0.0,
                    message="Failed to decode image",
                    face_detected=False,
                    processing_time=time.time() - start_time
                )
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(
                rgb_image,
                model=SecurityConstants.FACE_ENCODING_MODEL
            )
            
            if not face_locations:
                return FaceAuthResult(
                    success=False,
                    confidence=0.0,
                    message="No face detected",
                    face_detected=False,
                    processing_time=time.time() - start_time
                )
            
            # Extract face encodings
            face_encodings = face_recognition.face_encodings(
                rgb_image, 
                face_locations
            )
            
            if not face_encodings:
                return FaceAuthResult(
                    success=False,
                    confidence=0.0,
                    message="Failed to extract face features",
                    face_detected=True,
                    processing_time=time.time() - start_time
                )
            
            # Compare with known faces
            face_encoding = face_encodings[0]  # Use first face
            matches = face_recognition.compare_faces(
                self.known_faces, 
                face_encoding,
                tolerance=SecurityConstants.MAX_FACE_DISTANCE
            )
            
            face_distances = face_recognition.face_distance(self.known_faces, face_encoding)
            
            if True in matches:
                # Find the best match
                best_match_index = np.argmin(face_distances)
                confidence = 1.0 - face_distances[best_match_index]
                user_id = self.known_users[best_match_index]
                
                processing_time = time.time() - start_time
                
                return FaceAuthResult(
                    success=True,
                    confidence=confidence,
                    user_id=user_id,
                    message="Face authentication successful",
                    face_detected=True,
                    processing_time=processing_time
                )
            else:
                processing_time = time.time() - start_time
                
                return FaceAuthResult(
                    success=False,
                    confidence=1.0 - min(face_distances) if len(face_distances) > 0 else 0.0,
                    message="Face not recognized",
                    face_detected=True,
                    processing_time=processing_time
                )
            
        except Exception as e:
            logger.error(f"Face authentication failed: {str(e)}")
            return FaceAuthResult(
                success=False,
                confidence=0.0,
                message=f"Face authentication error: {str(e)}",
                face_detected=False,
                processing_time=time.time() - start_time
            )
    
    def get_enrolled_users(self) -> List[str]:
        """Get list of enrolled user IDs."""
        return list(set(self.known_users))


class VoiceBiometricsEngine:
    """Voice biometrics engine for voice authentication."""
    
    def __init__(self):
        self.voice_profiles: Dict[str, List[np.ndarray]] = {}
        self.is_initialized = False
        self.config = get_config()
    
    async def initialize(self):
        """Initialize voice biometrics engine."""
        try:
            logger.info("Initializing Voice Biometrics Engine...")
            
            # Load existing voice profiles
            await self._load_voice_profiles()
            
            self.is_initialized = True
            logger.info("✅ Voice Biometrics Engine initialized")
            
        except Exception as e:
            logger.error(f"❌ Voice Biometrics Engine initialization failed: {str(e)}")
            raise
    
    async def _load_voice_profiles(self):
        """Load voice profiles from secure storage."""
        voice_profiles_path = Path(get_config().security_dir) / "voice_profiles.json"
        
        if not voice_profiles_path.exists():
            logger.warning("No existing voice profiles found. New profiles will be created.")
            return
        
        try:
            with open(voice_profiles_path, 'r') as f:
                profiles_data = json.load(f)
            
            for user_id, profile_data in profiles_data.items():
                # Convert base64 encoded features back to numpy arrays
                voice_features = []
                for feature_b64 in profile_data.get('voice_features', []):
                    feature_bytes = base64.b64decode(feature_b64)
                    feature_array = np.frombuffer(feature_bytes, dtype=np.float64)
                    voice_features.append(feature_array)
                
                self.voice_profiles[user_id] = voice_features
            
            logger.info(f"Loaded voice profiles for {len(self.voice_profiles)} users")
            
        except Exception as e:
            logger.error(f"Failed to load voice profiles: {str(e)}")
    
    async def _save_voice_profiles(self):
        """Save voice profiles to secure storage."""
        try:
            profiles_data = {}
            
            for user_id, features_list in self.voice_profiles.items():
                features_b64 = []
                for features in features_list:
                    features_bytes = features.tobytes()
                    features_b64.append(base64.b64encode(features_bytes).decode('utf-8'))
                
                profiles_data[user_id] = {
                    'voice_features': features_b64,
                    'count': len(features_list)
                }
            
            voice_profiles_path = Path(get_config().security_dir) / "voice_profiles.json"
            with open(voice_profiles_path, 'w') as f:
                json.dump(profiles_data, f, indent=2)
            
            logger.info(f"Saved voice profiles for {len(self.voice_profiles)} users")
            
        except Exception as e:
            logger.error(f"Failed to save voice profiles: {str(e)}")
    
    def _extract_voice_features(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Extract MFCC features from audio data for voice biometrics."""
        try:
            # Convert bytes to numpy array for librosa
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to floating point for librosa
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_float,
                sr=AudioConstants.DEFAULT_SAMPLE_RATE,
                n_mfcc=SecurityConstants.VOICE_FEATURES_TO_EXTRACT
            )
            
            # Take mean across time to get a fixed-length feature vector
            mfcc_mean = np.mean(mfccs, axis=1)
            
            return mfcc_mean
            
        except Exception as e:
            logger.error(f"Voice feature extraction failed: {str(e)}")
            return None
    
    async def enroll_voice(self, audio_data: str, user_id: str) -> VoiceAuthResult:
        """Enroll a new voice for authentication."""
        start_time = time.time()
        
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)
            
            # Validate audio length
            audio_duration = len(audio_bytes) / (2 * AudioConstants.DEFAULT_SAMPLE_RATE)  # 16-bit audio
            if audio_duration < SecurityConstants.MIN_VOICE_SAMPLE_SECONDS:
                return VoiceAuthResult(
                    success=False,
                    confidence=0.0,
                    message=f"Audio too short: {audio_duration:.1f}s (minimum {SecurityConstants.MIN_VOICE_SAMPLE_SECONDS}s)",
                    voice_features_extracted=False,
                    processing_time=time.time() - start_time
                )
            
            if audio_duration > SecurityConstants.MAX_VOICE_SAMPLE_SECONDS:
                return VoiceAuthResult(
                    success=False,
                    confidence=0.0,
                    message=f"Audio too long: {audio_duration:.1f}s (maximum {SecurityConstants.MAX_VOICE_SAMPLE_SECONDS}s)",
                    voice_features_extracted=False,
                    processing_time=time.time() - start_time
                )
            
            # Extract voice features
            voice_features = self._extract_voice_features(audio_bytes)
            
            if voice_features is None:
                return VoiceAuthResult(
                    success=False,
                    confidence=0.0,
                    message="Failed to extract voice features",
                    voice_features_extracted=False,
                    processing_time=time.time() - start_time
                )
            
            # Store the features
            if user_id not in self.voice_profiles:
                self.voice_profiles[user_id] = []
            
            self.voice_profiles[user_id].append(voice_features)
            
            # Keep only the most recent features (limit storage)
            if len(self.voice_profiles[user_id]) > 5:
                self.voice_profiles[user_id] = self.voice_profiles[user_id][-5:]
            
            # Save updated profiles
            await self._save_voice_profiles()
            
            processing_time = time.time() - start_time
            
            return VoiceAuthResult(
                success=True,
                confidence=1.0,
                user_id=user_id,
                message="Voice enrolled successfully",
                voice_features_extracted=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Voice enrollment failed: {str(e)}")
            return VoiceAuthResult(
                success=False,
                confidence=0.0,
                message=f"Voice enrollment error: {str(e)}",
                voice_features_extracted=False,
                processing_time=time.time() - start_time
            )
    
    async def authenticate_voice(self, audio_data: str) -> VoiceAuthResult:
        """Authenticate user using voice biometrics."""
        start_time = time.time()
        
        try:
            if not self.voice_profiles:
                return VoiceAuthResult(
                    success=False,
                    confidence=0.0,
                    message="No enrolled voice profiles found",
                    voice_features_extracted=False,
                    processing_time=time.time() - start_time
                )
            
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)
            
            # Extract voice features from input
            input_features = self._extract_voice_features(audio_bytes)
            
            if input_features is None:
                return VoiceAuthResult(
                    success=False,
                    confidence=0.0,
                    message="Failed to extract voice features from input",
                    voice_features_extracted=False,
                    processing_time=time.time() - start_time
                )
            
            best_confidence = 0.0
            best_user = None
            
            # Compare with all enrolled profiles
            for user_id, enrolled_features_list in self.voice_profiles.items():
                for enrolled_features in enrolled_features_list:
                    # Calculate cosine similarity (1 - cosine distance)
                    similarity = 1 - cosine(input_features, enrolled_features)
                    
                    if similarity > best_confidence:
                        best_confidence = similarity
                        best_user = user_id
            
            processing_time = time.time() - start_time
            
            if best_confidence >= SecurityConstants.VOICE_MATCH_THRESHOLD:
                return VoiceAuthResult(
                    success=True,
                    confidence=best_confidence,
                    user_id=best_user,
                    message="Voice authentication successful",
                    voice_features_extracted=True,
                    processing_time=processing_time
                )
            else:
                return VoiceAuthResult(
                    success=False,
                    confidence=best_confidence,
                    message="Voice not recognized",
                    voice_features_extracted=True,
                    processing_time=processing_time
                )
            
        except Exception as e:
            logger.error(f"Voice authentication failed: {str(e)}")
            return VoiceAuthResult(
                success=False,
                confidence=0.0,
                message=f"Voice authentication error: {str(e)}",
                voice_features_extracted=False,
                processing_time=time.time() - start_time
            )
    
    def get_enrolled_users(self) -> List[str]:
        """Get list of enrolled user IDs."""
        return list(self.voice_profiles.keys())


class SecurityOrchestrator:
    """
    Main security orchestrator that coordinates face recognition and voice biometrics.
    Handles dual authentication and security level management.
    """
    
    def __init__(self):
        self.face_engine = FaceRecognitionEngine()
        self.voice_engine = VoiceBiometricsEngine()
        self.is_initialized = False
        self.config = get_config()
        self.auth_attempts = 0
        self.lockout_until = 0
        
        # Security event logging
        self.security_events = []
        self.max_security_events = 1000
    
    async def initialize(self):
        """Initialize the security orchestrator and all engines."""
        try:
            logger.info("Initializing Security Orchestrator...")
            
            # Initialize sub-engines
            await self.face_engine.initialize()
            await self.voice_engine.initialize()
            
            self.is_initialized = True
            logger.info("✅ Security Orchestrator initialized")
            
        except Exception as e:
            logger.error(f"❌ Security Orchestrator initialization failed: {str(e)}")
            raise
    
    def _log_security_event(self, event_type: str, success: bool, details: str = ""):
        """Log security event for audit trail."""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'success': success,
            'details': details,
            'ip_address': 'local'  # Could be extended for network scenarios
        }
        
        self.security_events.append(event)
        
        # Limit stored events
        if len(self.security_events) > self.max_security_events:
            self.security_events = self.security_events[-self.max_security_events:]
        
        # Log to security log
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"SECURITY {status}: {event_type} - {details}")
    
    def _check_lockout(self) -> bool:
        """Check if system is in lockout mode due to failed attempts."""
        if time.time() < self.lockout_until:
            remaining = self.lockout_until - time.time()
            logger.warning(f"Security system locked out. {remaining:.1f}s remaining")
            return True
        return False
    
    def _handle_failed_attempt(self):
        """Handle failed authentication attempt and apply lockout if needed."""
        self.auth_attempts += 1
        
        if self.auth_attempts >= SecurityConstants.MAX_AUTH_ATTEMPTS:
            lockout_duration = SecurityConstants.LOCKOUT_DURATION_MINUTES * 60
            self.lockout_until = time.time() + lockout_duration
            logger.warning(f"Too many failed attempts. Locking out for {lockout_duration}s")
            self.auth_attempts = 0
    
    def _handle_successful_attempt(self):
        """Reset failed attempt counter on successful authentication."""
        self.auth_attempts = 0
        self.lockout_until = 0
    
    async def enroll_user(self, user_id: str, face_image: str = None, voice_audio: str = None) -> Dict[str, Any]:
        """Enroll a new user with face and/or voice biometrics."""
        if self._check_lockout():
            return {
                'success': False,
                'message': 'System temporarily locked due to failed attempts',
                'lockout_remaining': self.lockout_until - time.time()
            }
        
        enroll_results = {}
        
        # Enroll face if provided
        if face_image:
            face_result = await self.face_engine.enroll_face(face_image, user_id)
            enroll_results['face'] = {
                'success': face_result.success,
                'confidence': face_result.confidence,
                'message': face_result.message
            }
            self._log_security_event('face_enrollment', face_result.success, f"User: {user_id}")
        
        # Enroll voice if provided
        if voice_audio:
            voice_result = await self.voice_engine.enroll_voice(voice_audio, user_id)
            enroll_results['voice'] = {
                'success': voice_result.success,
                'confidence': voice_result.confidence,
                'message': voice_result.message
            }
            self._log_security_event('voice_enrollment', voice_result.success, f"User: {user_id}")
        
        overall_success = (
            (not face_image or enroll_results.get('face', {}).get('success', False)) and
            (not voice_audio or enroll_results.get('voice', {}).get('success', False))
        )
        
        return {
            'success': overall_success,
            'user_id': user_id,
            'enrollment_results': enroll_results,
            'message': 'User enrollment completed'
        }
    
    async def authenticate_face(self, image_data: str) -> Dict[str, Any]:
        """Authenticate user using face recognition only."""
        if self._check_lockout():
            return {
                'success': False,
                'message': 'System temporarily locked due to failed attempts',
                'lockout_remaining': self.lockout_until - time.time()
            }
        
        face_result = await self.face_engine.authenticate_face(image_data)
        
        if face_result.success:
            self._handle_successful_attempt()
            self._log_security_event('face_auth', True, f"User: {face_result.user_id}")
        else:
            self._handle_failed_attempt()
            self._log_security_event('face_auth', False, face_result.message)
        
        return {
            'success': face_result.success,
            'user_identified': face_result.user_id,
            'confidence': face_result.confidence,
            'message': face_result.message,
            'processing_time': face_result.processing_time,
            'security_level': 'face_only',
            'requires_additional_auth': self.config.security.security_level == 'strict'
        }
    
    async def authenticate_voice(self, audio_data: str) -> Dict[str, Any]:
        """Authenticate user using voice biometrics only."""
        if self._check_lockout():
            return {
                'success': False,
                'message': 'System temporarily locked due to failed attempts',
                'lockout_remaining': self.lockout_until - time.time()
            }
        
        voice_result = await self.voice_engine.authenticate_voice(audio_data)
        
        if voice_result.success:
            self._handle_successful_attempt()
            self._log_security_event('voice_auth', True, f"User: {voice_result.user_id}")
        else:
            self._handle_failed_attempt()
            self._log_security_event('voice_auth', False, voice_result.message)
        
        return {
            'success': voice_result.success,
            'user_identified': voice_result.user_id,
            'confidence': voice_result.confidence,
            'message': voice_result.message,
            'processing_time': voice_result.processing_time,
            'security_level': 'voice_only',
            'requires_additional_auth': self.config.security.security_level in ['standard', 'strict']
        }
    
    async def authenticate_dual(self, image_data: str, audio_data: str) -> Dict[str, Any]:
        """Authenticate user using both face and voice biometrics."""
        if self._check_lockout():
            return {
                'success': False,
                'message': 'System temporarily locked due to failed attempts',
                'lockout_remaining': self.lockout_until - time.time()
            }
        
        # Perform both authentications in parallel
        face_task = asyncio.create_task(self.face_engine.authenticate_face(image_data))
        voice_task = asyncio.create_task(self.voice_engine.authenticate_voice(audio_data))
        
        face_result, voice_result = await asyncio.gather(face_task, voice_task)
        
        # Determine overall success based on security level
        if self.config.security.security_level == 'strict':
            # Both must succeed
            overall_success = face_result.success and voice_result.success
            identified_user = face_result.user_id if face_result.success and voice_result.success else None
        else:
            # Standard - at least one must succeed
            overall_success = face_result.success or voice_result.success
            identified_user = face_result.user_id if face_result.success else voice_result.user_id
        
        # Calculate combined confidence
        combined_confidence = max(face_result.confidence, voice_result.confidence)
        if face_result.success and voice_result.success:
            combined_confidence = (face_result.confidence + voice_result.confidence) / 2
        
        if overall_success:
            self._handle_successful_attempt()
            self._log_security_event('dual_auth', True, f"User: {identified_user}")
        else:
            self._handle_failed_attempt()
            self._log_security_event('dual_auth', False, 
                                   f"Face: {face_result.message}, Voice: {voice_result.message}")
        
        return {
            'success': overall_success,
            'user_identified': identified_user,
            'confidence': combined_confidence,
            'message': f"Face: {face_result.message}, Voice: {voice_result.message}",
            'processing_time': face_result.processing_time + voice_result.processing_time,
            'security_level': 'dual_auth',
            'requires_additional_auth': False,
            'face_result': {
                'success': face_result.success,
                'confidence': face_result.confidence,
                'message': face_result.message
            },
            'voice_result': {
                'success': voice_result.success,
                'confidence': voice_result.confidence,
                'message': voice_result.message
            }
        }
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get current security system status."""
        return {
            'initialized': self.is_initialized,
            'security_level': self.config.security.security_level.value,
            'enrolled_users': {
                'face': self.face_engine.get_enrolled_users(),
                'voice': self.voice_engine.get_enrolled_users()
            },
            'lockout_status': {
                'locked': self._check_lockout(),
                'attempts': self.auth_attempts,
                'lockout_until': self.lockout_until,
                'lockout_remaining': max(0, self.lockout_until - time.time()) if self.lockout_until else 0
            },
            'recent_events': self.security_events[-10:] if self.security_events else []  # Last 10 events
        }
    
    async def reset_security_data(self, user_id: str = None):
        """Reset security data for a specific user or all users."""
        if user_id:
            # Reset specific user
            if user_id in self.face_engine.known_users:
                indices_to_remove = [i for i, uid in enumerate(self.face_engine.known_users) if uid == user_id]
                for index in sorted(indices_to_remove, reverse=True):
                    del self.face_engine.known_faces[index]
                    del self.face_engine.known_users[index]
            
            if user_id in self.voice_engine.voice_profiles:
                del self.voice_engine.voice_profiles[user_id]
            
            await self.face_engine._save_face_profiles()
            await self.voice_engine._save_voice_profiles()
            
            self._log_security_event('data_reset', True, f"User: {user_id}")
            logger.info(f"Security data reset for user: {user_id}")
        else:
            # Reset all users
            self.face_engine.known_faces.clear()
            self.face_engine.known_users.clear()
            self.voice_engine.voice_profiles.clear()
            
            await self.face_engine._save_face_profiles()
            await self.voice_engine._save_voice_profiles()
            
            self._log_security_event('data_reset', True, "All users")
            logger.info("All security data reset")
    
    async def shutdown(self):
        """Shutdown security orchestrator gracefully."""
        logger.info("Shutting down Security Orchestrator...")
        
        # Save any pending data
        try:
            await self.face_engine._save_face_profiles()
            await self.voice_engine._save_voice_profiles()
        except Exception as e:
            logger.error(f"Error during security shutdown: {str(e)}")
        
        logger.info("✅ Security Orchestrator shutdown complete")


# Global security instance
_security_instance: Optional[SecurityOrchestrator] = None


async def get_security_orchestrator() -> SecurityOrchestrator:
    """Get or create global security orchestrator instance."""
    global _security_instance
    
    if _security_instance is None:
        _security_instance = SecurityOrchestrator()
        await _security_instance.initialize()
    
    return _security_instance


async def main():
    """Command-line testing for security orchestrator."""
    security = await get_security_orchestrator()
    
    status = await security.get_security_status()
    print("Security Status:")
    print(json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())