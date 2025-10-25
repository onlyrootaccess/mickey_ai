# OpenCV-based face auth
"""
M.I.C.K.E.Y. AI Assistant - Face Recognition Engine
Made In Crisis, Keeping Everything Yours

ELEVENTH FILE IN PIPELINE: Dedicated face recognition engine using OpenCV 
and face_recognition library. Handles face detection, encoding, and matching.
"""

import asyncio
import logging
import time
import base64
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass

# Import computer vision libraries
import cv2
import numpy as np
import face_recognition

# Import Mickey AI configuration
from config.settings import get_config
from config.constants import SecurityConstants, ErrorCodes, ErrorMessages

# Setup logging
logger = logging.getLogger("MickeyFaceRecognition")


@dataclass
class FaceDetectionResult:
    """Face detection result container."""
    success: bool
    faces_detected: int
    face_locations: List[Tuple]
    face_encodings: List[np.ndarray]
    processing_time: float
    error_message: Optional[str] = None


@dataclass
class FaceMatchResult:
    """Face matching result container."""
    success: bool
    match_found: bool
    user_id: Optional[str] = None
    confidence: float = 0.0
    best_match_index: Optional[int] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None


class FaceRecognitionEngine:
    """
    Advanced face recognition engine using OpenCV and face_recognition library.
    Handles face detection, encoding, matching, and profile management.
    """
    
    def __init__(self):
        self.config = get_config()
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_users: List[str] = []
        self.is_initialized = False
        
        # Performance tracking
        self.total_detections = 0
        self.total_matches = 0
        self.total_processing_time = 0.0
        
        # Cache for recent detections
        self.face_cache = {}
        self.max_cache_size = 50
    
    async def initialize(self):
        """Initialize the face recognition engine."""
        try:
            logger.info("Initializing Face Recognition Engine...")
            
            # Load existing face profiles
            await self._load_face_profiles()
            
            # Initialize camera for real-time detection
            await self._initialize_camera()
            
            self.is_initialized = True
            logger.info(f"✅ Face Recognition Engine initialized with {len(self.known_face_encodings)} known faces")
            
        except Exception as e:
            logger.error(f"❌ Face Recognition Engine initialization failed: {str(e)}")
            raise
    
    async def _initialize_camera(self):
        """Initialize camera for face detection."""
        try:
            # Test camera availability
            camera = cv2.VideoCapture(0)
            if camera.isOpened():
                ret, frame = camera.read()
                camera.release()
                
                if ret:
                    logger.info("✅ Camera available for face recognition")
                else:
                    logger.warning("⚠️ Camera found but cannot capture frames")
            else:
                logger.warning("⚠️ No camera detected - face recognition will use image input only")
                
        except Exception as e:
            logger.warning(f"Camera initialization warning: {str(e)}")
    
    async def _load_face_profiles(self):
        """Load face profiles from secure storage."""
        face_profiles_path = Path(get_config().security_dir) / "face_profiles.json"
        
        if not face_profiles_path.exists():
            logger.warning("No existing face profiles found. New profiles will be created.")
            return
        
        try:
            with open(face_profiles_path, 'r', encoding='utf-8') as f:
                profiles_data = json.load(f)
            
            for user_id, profile_data in profiles_data.items():
                # Convert base64 encoded face encodings back to numpy arrays
                face_encodings = []
                for encoding_b64 in profile_data.get('face_encodings', []):
                    encoding_bytes = base64.b64decode(encoding_b64)
                    encoding_array = np.frombuffer(encoding_bytes, dtype=np.float64)
                    face_encodings.append(encoding_array)
                
                self.known_face_encodings.extend(face_encodings)
                self.known_face_users.extend([user_id] * len(face_encodings))
            
            logger.info(f"Loaded {len(self.known_face_encodings)} face encodings for {len(set(self.known_face_users))} users")
            
        except Exception as e:
            logger.error(f"Failed to load face profiles: {str(e)}")
            # Don't raise - start with empty profiles
    
    async def _save_face_profiles(self):
        """Save face profiles to secure storage."""
        try:
            # Group encodings by user
            user_encodings = {}
            for i, user_id in enumerate(self.known_face_users):
                if user_id not in user_encodings:
                    user_encodings[user_id] = []
                user_encodings[user_id].append(self.known_face_encodings[i])
            
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
                    'count': len(encodings),
                    'last_updated': time.time()
                }
            
            # Save to file
            face_profiles_path = Path(get_config().security_dir) / "face_profiles.json"
            with open(face_profiles_path, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, indent=2)
            
            logger.info(f"Saved {len(self.known_face_encodings)} face encodings to secure storage")
            
        except Exception as e:
            logger.error(f"Failed to save face profiles: {str(e)}")
            raise
    
    def _decode_image(self, image_data: str) -> Optional[np.ndarray]:
        """Decode base64 image data to OpenCV format."""
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            return image
            
        except Exception as e:
            logger.error(f"Image decoding failed: {str(e)}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better face detection."""
        try:
            # Convert BGR to RGB (face_recognition uses RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Optional: Enhance image quality
            # Apply slight sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(rgb_image, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}")
            return image
    
    async def detect_faces(self, image_data: str) -> FaceDetectionResult:
        """Detect faces in image and extract encodings."""
        start_time = time.time()
        
        try:
            # Decode image
            image = self._decode_image(image_data)
            if image is None:
                return FaceDetectionResult(
                    success=False,
                    faces_detected=0,
                    face_locations=[],
                    face_encodings=[],
                    processing_time=time.time() - start_time,
                    error_message="Failed to decode image"
                )
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(
                processed_image,
                model=SecurityConstants.FACE_ENCODING_MODEL,
                number_of_times_to_upsample=1
            )
            
            if not face_locations:
                return FaceDetectionResult(
                    success=True,
                    faces_detected=0,
                    face_locations=[],
                    face_encodings=[],
                    processing_time=time.time() - start_time,
                    error_message="No faces detected in image"
                )
            
            # Extract face encodings
            face_encodings = face_recognition.face_encodings(
                processed_image,
                face_locations,
                model="large"  # Use large model for better accuracy
            )
            
            processing_time = time.time() - start_time
            self.total_detections += 1
            self.total_processing_time += processing_time
            
            logger.info(f"Detected {len(face_locations)} faces in {processing_time:.3f}s")
            
            return FaceDetectionResult(
                success=True,
                faces_detected=len(face_locations),
                face_locations=face_locations,
                face_encodings=face_encodings,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return FaceDetectionResult(
                success=False,
                faces_detected=0,
                face_locations=[],
                face_encodings=[],
                processing_time=time.time() - start_time,
                error_message=f"Face detection error: {str(e)}"
            )
    
    async def enroll_face(self, image_data: str, user_id: str) -> Dict[str, Any]:
        """Enroll a new face for authentication."""
        start_time = time.time()
        
        try:
            # Detect faces in the image
            detection_result = await self.detect_faces(image_data)
            
            if not detection_result.success:
                return {
                    'success': False,
                    'user_id': user_id,
                    'confidence': 0.0,
                    'message': detection_result.error_message,
                    'processing_time': time.time() - start_time
                }
            
            if detection_result.faces_detected == 0:
                return {
                    'success': False,
                    'user_id': user_id,
                    'confidence': 0.0,
                    'message': "No face detected in enrollment image",
                    'processing_time': time.time() - start_time
                }
            
            if detection_result.faces_detected > 1:
                return {
                    'success': False,
                    'user_id': user_id,
                    'confidence': 0.0,
                    'message': "Multiple faces detected - please use image with single face",
                    'processing_time': time.time() - start_time
                }
            
            # Use the first face encoding
            face_encoding = detection_result.face_encodings[0]
            
            # Check if this face is already enrolled
            if self.known_face_encodings:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings,
                    face_encoding,
                    tolerance=SecurityConstants.MAX_FACE_DISTANCE
                )
                
                if any(matches):
                    return {
                        'success': False,
                        'user_id': user_id,
                        'confidence': 0.0,
                        'message': "Face already enrolled in system",
                        'processing_time': time.time() - start_time
                    }
            
            # Store the encoding
            self.known_face_encodings.append(face_encoding)
            self.known_face_users.append(user_id)
            
            # Save updated profiles
            await self._save_face_profiles()
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'user_id': user_id,
                'confidence': 1.0,
                'message': "Face enrolled successfully",
                'faces_detected': detection_result.faces_detected,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Face enrollment failed: {str(e)}")
            return {
                'success': False,
                'user_id': user_id,
                'confidence': 0.0,
                'message': f"Face enrollment error: {str(e)}",
                'processing_time': time.time() - start_time
            }
    
    async def recognize_face(self, image_data: str) -> FaceMatchResult:
        """Recognize face against enrolled profiles."""
        start_time = time.time()
        
        try:
            if not self.known_face_encodings:
                return FaceMatchResult(
                    success=True,
                    match_found=False,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    error_message="No enrolled faces in system"
                )
            
            # Detect faces in the image
            detection_result = await self.detect_faces(image_data)
            
            if not detection_result.success:
                return FaceMatchResult(
                    success=False,
                    match_found=False,
                    processing_time=time.time() - start_time,
                    error_message=detection_result.error_message
                )
            
            if detection_result.faces_detected == 0:
                return FaceMatchResult(
                    success=True,
                    match_found=False,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    error_message="No face detected in image"
                )
            
            # Use the first face encoding for matching
            face_encoding = detection_result.face_encodings[0]
            
            # Compare with known faces
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, 
                face_encoding
            )
            
            # Find the best match
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            
            # Convert distance to confidence (closer distance = higher confidence)
            confidence = 1.0 - min(best_distance, 1.0)
            
            processing_time = time.time() - start_time
            self.total_matches += 1
            
            if confidence >= SecurityConstants.FACE_MATCH_THRESHOLD:
                user_id = self.known_face_users[best_match_index]
                logger.info(f"Face recognized: {user_id} (confidence: {confidence:.3f})")
                
                return FaceMatchResult(
                    success=True,
                    match_found=True,
                    user_id=user_id,
                    confidence=confidence,
                    best_match_index=best_match_index,
                    processing_time=processing_time
                )
            else:
                logger.info(f"Face not recognized (best confidence: {confidence:.3f})")
                
                return FaceMatchResult(
                    success=True,
                    match_found=False,
                    confidence=confidence,
                    best_match_index=best_match_index,
                    processing_time=processing_time,
                    error_message="Face not recognized"
                )
            
        except Exception as e:
            logger.error(f"Face recognition failed: {str(e)}")
            return FaceMatchResult(
                success=False,
                match_found=False,
                processing_time=time.time() - start_time,
                error_message=f"Face recognition error: {str(e)}"
            )
    
    async def capture_live_face(self, timeout: int = 30) -> Optional[str]:
        """Capture face from live camera feed."""
        try:
            camera = cv2.VideoCapture(0)
            start_time = time.time()
            
            logger.info("Starting live face capture...")
            
            while (time.time() - start_time) < timeout:
                ret, frame = camera.read()
                if not ret:
                    continue
                
                # Convert frame to base64 for processing
                _, buffer = cv2.imencode('.jpg', frame)
                image_data = base64.b64encode(buffer).decode('utf-8')
                
                # Check for faces
                detection_result = await self.detect_faces(image_data)
                
                if detection_result.success and detection_result.faces_detected > 0:
                    camera.release()
                    logger.info("Face captured from live camera")
                    return image_data
                
                # Small delay to prevent excessive processing
                await asyncio.sleep(0.1)
            
            camera.release()
            logger.warning("Face capture timeout - no face detected")
            return None
            
        except Exception as e:
            logger.error(f"Live face capture failed: {str(e)}")
            if 'camera' in locals():
                camera.release()
            return None
    
    async def verify_face(self, image_data: str, expected_user: str) -> Dict[str, Any]:
        """Verify if face matches specific user."""
        match_result = await self.recognize_face(image_data)
        
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
                'message': "Face not recognized"
            }
        
        verified = (match_result.user_id == expected_user)
        
        return {
            'success': True,
            'verified': verified,
            'user_id': match_result.user_id,
            'confidence': match_result.confidence,
            'message': "Face verified" if verified else "Face does not match expected user"
        }
    
    def get_enrolled_users(self) -> List[str]:
        """Get list of enrolled user IDs."""
        return list(set(self.known_face_users))
    
    async def remove_user_faces(self, user_id: str) -> bool:
        """Remove all face data for a specific user."""
        try:
            # Find indices to remove
            indices_to_remove = [
                i for i, uid in enumerate(self.known_face_users) 
                if uid == user_id
            ]
            
            # Remove in reverse order to avoid index issues
            for index in sorted(indices_to_remove, reverse=True):
                del self.known_face_encodings[index]
                del self.known_face_users[index]
            
            # Save updated profiles
            await self._save_face_profiles()
            
            logger.info(f"Removed face data for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove user faces: {str(e)}")
            return False
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get face recognition engine performance metrics."""
        avg_processing_time = (
            self.total_processing_time / (self.total_detections + self.total_matches)
            if (self.total_detections + self.total_matches) > 0 else 0
        )
        
        return {
            "initialized": self.is_initialized,
            "enrolled_users": len(self.get_enrolled_users()),
            "total_face_encodings": len(self.known_face_encodings),
            "total_detections": self.total_detections,
            "total_matches": self.total_matches,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "cache_size": len(self.face_cache)
        }
    
    async def shutdown(self):
        """Shutdown face recognition engine gracefully."""
        logger.info("Shutting down Face Recognition Engine...")
        
        try:
            # Save any pending data
            await self._save_face_profiles()
            
            # Clear caches
            self.face_cache.clear()
            
            logger.info("✅ Face Recognition Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during face recognition shutdown: {str(e)}")


# Global face recognition instance
_face_recognition_instance: Optional[FaceRecognitionEngine] = None


async def get_face_recognition_engine() -> FaceRecognitionEngine:
    """Get or create global face recognition engine instance."""
    global _face_recognition_instance
    
    if _face_recognition_instance is None:
        _face_recognition_instance = FaceRecognitionEngine()
        await _face_recognition_instance.initialize()
    
    return _face_recognition_instance


async def main():
    """Command-line testing for face recognition engine."""
    face_engine = await get_face_recognition_engine()
    
    # Test performance metrics
    metrics = await face_engine.get_performance_metrics()
    print("Face Recognition Engine Status:")
    print(f"Initialized: {metrics['initialized']}")
    print(f"Enrolled Users: {metrics['enrolled_users']}")
    print(f"Total Face Encodings: {metrics['total_face_encodings']}")
    print(f"Total Detections: {metrics['total_detections']}")
    print(f"Total Matches: {metrics['total_matches']}")
    
    if metrics['enrolled_users'] == 0:
        print("\n⚠️ No faces enrolled. Use enroll_face() to add faces.")
    else:
        print(f"\nEnrolled Users: {face_engine.get_enrolled_users()}")


if __name__ == "__main__":
    asyncio.run(main())