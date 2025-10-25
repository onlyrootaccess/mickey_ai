# SECOND: Piper TTS female voice
"""
M.I.C.K.E.Y. AI Assistant - Text-to-Speech Synthesizer
Made In Crisis, Keeping Everything Yours

SEVENTH FILE IN PIPELINE: Generates ultra-realistic female voice using Piper TTS.
Provides emotional prosody, variable speed, and high-quality speech synthesis.
"""

import asyncio
import logging
import time
import base64
import json
import threading
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import TTS and audio libraries
import piper
import pyaudio
import numpy as np
import onnxruntime as ort
from scipy import signal

# Import Mickey AI configuration
from config.settings import get_config
from config.constants import (
    AudioConstants, SystemConstants, ErrorCodes, ErrorMessages,
    PersonalityConstants
)

# Setup logging
logger = logging.getLogger("MickeyTTS")


class VoiceEmotion(Enum):
    """Emotional states for voice synthesis."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited"
    CALM = "calm"
    SERIOUS = "serious"
    WARM = "warm"
    PLAYFUL = "playful"
    EMPATHETIC = "empathetic"
    CONFIDENT = "confident"


@dataclass
class SynthesisResult:
    """Speech synthesis result container."""
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    voice_model: str
    emotion: VoiceEmotion
    speed: float
    processing_time: float = 0.0


class VoiceModelManager:
    """Manages Piper voice models and configurations."""
    
    def __init__(self):
        self.config = get_config()
        self.models_dir = Path(get_config().models_dir)
        self.available_voices: Dict[str, Dict] = {}
        self.loaded_models: Dict[str, Any] = {}
        
    async def initialize(self):
        """Initialize voice model manager and discover available voices."""
        try:
            logger.info("Initializing Voice Model Manager...")
            
            # Discover available voice models
            await self._discover_voice_models()
            
            if not self.available_voices:
                logger.warning("No voice models found. Please download voice models.")
                # We'll proceed but synthesis will fail until models are available
            
            logger.info("✅ Voice Model Manager initialized")
            
        except Exception as e:
            logger.error(f"❌ Voice Model Manager initialization failed: {str(e)}")
            raise
    
    async def _discover_voice_models(self):
        """Discover available Piper voice models in models directory."""
        try:
            voice_files = list(self.models_dir.glob("*.onnx"))
            
            for voice_file in voice_files:
                voice_name = voice_file.stem
                config_file = voice_file.with_suffix('.json')
                
                if config_file.exists():
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                        
                        self.available_voices[voice_name] = {
                            'model_path': str(voice_file),
                            'config_path': str(config_file),
                            'config': config_data,
                            'sample_rate': config_data.get('audio', {}).get('sample_rate', 22050),
                            'language': config_data.get('language', {}).get('code', 'en'),
                            'description': config_data.get('description', 'Unknown voice')
                        }
                        
                        logger.info(f"Discovered voice model: {voice_name}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to load config for {voice_name}: {str(e)}")
                else:
                    logger.warning(f"No config file found for voice model: {voice_name}")
            
            logger.info(f"Found {len(self.available_voices)} voice models")
            
        except Exception as e:
            logger.error(f"Voice model discovery failed: {str(e)}")
    
    def get_voice_model(self, voice_name: str) -> Optional[Dict]:
        """Get voice model configuration by name."""
        return self.available_voices.get(voice_name)
    
    def get_default_voice(self) -> Optional[str]:
        """Get the default voice model name."""
        default_voice = self.config.audio.tts_voice
        
        if default_voice in self.available_voices:
            return default_voice
        
        # Fallback to any available voice
        if self.available_voices:
            return list(self.available_voices.keys())[0]
        
        return None
    
    def list_available_voices(self) -> List[Dict]:
        """List all available voice models with details."""
        voices = []
        for name, details in self.available_voices.items():
            voices.append({
                'name': name,
                'sample_rate': details['sample_rate'],
                'language': details['language'],
                'description': details['description']
            })
        return voices


class EmotionalProsody:
    """Manages emotional prosody and voice characteristics."""
    
    def __init__(self):
        self.config = get_config()
        
        # Emotional parameters mapping
        self.emotion_parameters = {
            VoiceEmotion.NEUTRAL: {
                'speed': 1.0,
                'pitch_variation': 0.5,
                'energy': 0.5,
                'pause_duration': 0.1
            },
            VoiceEmotion.HAPPY: {
                'speed': 1.1,
                'pitch_variation': 0.8,
                'energy': 0.8,
                'pause_duration': 0.05
            },
            VoiceEmotion.EXCITED: {
                'speed': 1.2,
                'pitch_variation': 0.9,
                'energy': 0.9,
                'pause_duration': 0.03
            },
            VoiceEmotion.CALM: {
                'speed': 0.9,
                'pitch_variation': 0.3,
                'energy': 0.4,
                'pause_duration': 0.15
            },
            VoiceEmotion.SERIOUS: {
                'speed': 0.95,
                'pitch_variation': 0.2,
                'energy': 0.6,
                'pause_duration': 0.2
            },
            VoiceEmotion.WARM: {
                'speed': 1.0,
                'pitch_variation': 0.6,
                'energy': 0.7,
                'pause_duration': 0.1
            },
            VoiceEmotion.PLAYFUL: {
                'speed': 1.15,
                'pitch_variation': 0.85,
                'energy': 0.75,
                'pause_duration': 0.07
            },
            VoiceEmotion.EMPATHETIC: {
                'speed': 0.85,
                'pitch_variation': 0.4,
                'energy': 0.5,
                'pause_duration': 0.18
            },
            VoiceEmotion.CONFIDENT: {
                'speed': 1.05,
                'pitch_variation': 0.45,
                'energy': 0.8,
                'pause_duration': 0.12
            }
        }
    
    def get_emotion_parameters(self, emotion: VoiceEmotion) -> Dict[str, float]:
        """Get synthesis parameters for specific emotion."""
        return self.emotion_parameters.get(emotion, self.emotion_parameters[VoiceEmotion.NEUTRAL])
    
    def adjust_text_for_emotion(self, text: str, emotion: VoiceEmotion) -> str:
        """Adjust text with emotional markers and pauses."""
        params = self.get_emotion_parameters(emotion)
        
        # Add emotional emphasis markers (SSML-like but simplified for Piper)
        if emotion == VoiceEmotion.EXCITED:
            # Add exclamation for excitement
            if not text.endswith(('!', '?', '.')):
                text += '!'
        elif emotion == VoiceEmotion.SERIOUS:
            # Ensure proper punctuation for serious tone
            if not text.endswith(('.', '!', '?')):
                text += '.'
        
        # Add pause markers based on emotion
        pause_duration = params['pause_duration']
        if pause_duration > 0.15:  # Long pauses for serious/empathetic
            # Add comma pauses for longer sentences
            sentences = text.split('. ')
            if len(sentences) > 1:
                text = '. '.join([s.strip() for s in sentences])
        
        return text
    
    def calculate_synthesis_speed(self, base_speed: float, emotion: VoiceEmotion) -> float:
        """Calculate final synthesis speed considering emotion and base speed."""
        emotion_params = self.get_emotion_parameters(emotion)
        emotion_speed = emotion_params['speed']
        
        # Combine base speed and emotion speed
        final_speed = base_speed * emotion_speed
        
        # Clamp to reasonable range
        return max(0.5, min(2.0, final_speed))


class AudioPostProcessor:
    """Post-processes synthesized audio for enhanced quality."""
    
    def __init__(self):
        self.config = get_config()
    
    def apply_voice_enhancement(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply audio enhancement to improve voice quality."""
        try:
            # High-pass filter to remove low-frequency noise
            nyquist = sample_rate / 2
            highpass_cutoff = 80.0 / nyquist  # 80Hz high-pass
            
            b, a = signal.butter(2, highpass_cutoff, btype='high')
            enhanced_audio = signal.filtfilt(b, a, audio_data)
            
            # Gentle compression to even out volume
            compression_factor = 0.8
            enhanced_audio = np.tanh(enhanced_audio * compression_factor)
            
            # Normalize audio to prevent clipping
            max_amplitude = np.max(np.abs(enhanced_audio))
            if max_amplitude > 0:
                enhanced_audio = enhanced_audio / max_amplitude * 0.95  # Leave 5% headroom
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Voice enhancement failed: {str(e)}")
            return audio_data
    
    def adjust_pitch_subtly(self, audio_data: np.ndarray, sample_rate: int, 
                          pitch_shift: float) -> np.ndarray:
        """
        Apply subtle pitch shifting for emotional variation.
        pitch_shift: -1.0 to 1.0, where negative lowers pitch, positive raises
        """
        try:
            if abs(pitch_shift) < 0.1:  # No significant shift needed
                return audio_data
            
            # Simple resampling for pitch shift (crude but effective for small shifts)
            shift_factor = 1.0 + (pitch_shift * 0.1)  # Max 10% shift
            
            # Calculate new length
            new_length = int(len(audio_data) / shift_factor)
            
            # Resample
            if shift_factor > 1.0:
                # Higher pitch - interpolate
                x_old = np.linspace(0, 1, len(audio_data))
                x_new = np.linspace(0, 1, new_length)
                pitched_audio = np.interp(x_new, x_old, audio_data)
            else:
                # Lower pitch - decimate (with anti-aliasing)
                pitched_audio = signal.resample(audio_data, new_length)
            
            return pitched_audio
            
        except Exception as e:
            logger.warning(f"Pitch adjustment failed: {str(e)}")
            return audio_data
    
    def apply_emotional_characteristics(self, audio_data: np.ndarray, sample_rate: int,
                                      emotion: VoiceEmotion) -> np.ndarray:
        """Apply emotion-specific audio processing."""
        try:
            processed_audio = audio_data
            
            emotion_params = EmotionalProsody().get_emotion_parameters(emotion)
            pitch_variation = emotion_params['pitch_variation']
            energy = emotion_params['energy']
            
            # Convert pitch variation to shift amount (-0.1 to 0.1)
            pitch_shift = (pitch_variation - 0.5) * 0.2
            
            # Apply pitch shift
            processed_audio = self.adjust_pitch_subtly(processed_audio, sample_rate, pitch_shift)
            
            # Apply energy (volume) adjustment
            if energy > 0.7:  # High energy
                processed_audio = processed_audio * 1.1
            elif energy < 0.4:  # Low energy
                processed_audio = processed_audio * 0.9
            
            return processed_audio
            
        except Exception as e:
            logger.warning(f"Emotional processing failed: {str(e)}")
            return audio_data


class AudioPlayer:
    """Handles audio playback for synthesized speech."""
    
    def __init__(self):
        self.config = get_config()
        self.audio_interface = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize audio playback system."""
        try:
            logger.info("Initializing Audio Player...")
            
            self.audio_interface = pyaudio.PyAudio()
            
            # Test output devices
            output_devices = self._get_output_devices()
            if not output_devices:
                logger.warning("No audio output devices found")
            else:
                logger.info(f"Found {len(output_devices)} audio output devices")
            
            self.is_initialized = True
            logger.info("✅ Audio Player initialized")
            
        except Exception as e:
            logger.error(f"❌ Audio Player initialization failed: {str(e)}")
            raise
    
    def _get_output_devices(self) -> List[Dict]:
        """Get available audio output devices."""
        devices = []
        try:
            for i in range(self.audio_interface.get_device_count()):
                device_info = self.audio_interface.get_device_info_by_index(i)
                if device_info.get('maxOutputChannels', 0) > 0:
                    devices.append({
                        'index': i,
                        'name': device_info.get('name', 'Unknown'),
                        'channels': device_info.get('maxOutputChannels', 2),
                        'sample_rate': device_info.get('defaultSampleRate', 22050)
                    })
        except Exception as e:
            logger.error(f"Error getting audio devices: {str(e)}")
        
        return devices
    
    def play_audio(self, audio_data: np.ndarray, sample_rate: int, device_index: int = None):
        """Play audio data through specified output device."""
        try:
            if not self.is_initialized:
                raise RuntimeError("Audio Player not initialized")
            
            # Ensure audio data is in correct format (16-bit PCM)
            if audio_data.dtype != np.int16:
                # Convert float32 to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data
            
            # Open audio stream
            stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
                output_device_index=device_index
            )
            
            # Play audio
            stream.write(audio_int16.tobytes())
            
            # Close stream
            stream.stop_stream()
            stream.close()
            
            logger.debug(f"Audio playback completed: {len(audio_data)/sample_rate:.2f}s")
            
        except Exception as e:
            logger.error(f"Audio playback failed: {str(e)}")
            raise
    
    async def play_audio_async(self, audio_data: np.ndarray, sample_rate: int, device_index: int = None):
        """Play audio asynchronously to avoid blocking."""
        def _play():
            self.play_audio(audio_data, sample_rate, device_index)
        
        # Run in thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _play)
    
    async def shutdown(self):
        """Shutdown audio playback system."""
        try:
            if self.audio_interface:
                self.audio_interface.terminate()
            
            logger.info("✅ Audio Player shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during audio player shutdown: {str(e)}")


class TTSSynthesizer:
    """
    Text-to-Speech engine using Piper TTS.
    Provides high-quality, emotional speech synthesis for Mickey's voice.
    """
    
    def __init__(self):
        self.model_manager = VoiceModelManager()
        self.emotional_prosody = EmotionalProsody()
        self.audio_postprocessor = AudioPostProcessor()
        self.audio_player = AudioPlayer()
        
        self.current_voice: Optional[str] = None
        self.piper_voice: Optional[Any] = None
        self.is_initialized = False
        
        # Performance tracking
        self.total_synthesis_time = 0.0
        self.total_audio_generated = 0.0
        self.synthesis_count = 0
        
    async def initialize(self):
        """Initialize the TTS synthesizer and load default voice."""
        try:
            logger.info("Initializing TTS Synthesizer...")
            
            # Initialize components
            await self.model_manager.initialize()
            await self.audio_player.initialize()
            
            # Load default voice
            default_voice = self.model_manager.get_default_voice()
            if default_voice:
                await self.load_voice(default_voice)
                logger.info(f"✅ Default voice loaded: {default_voice}")
            else:
                logger.warning("No voice models available. Synthesis will fail.")
            
            self.is_initialized = True
            logger.info("✅ TTS Synthesizer initialized")
            
        except Exception as e:
            logger.error(f"❌ TTS Synthesizer initialization failed: {str(e)}")
            raise
    
    async def load_voice(self, voice_name: str) -> bool:
        """Load a specific voice model for synthesis."""
        try:
            voice_model = self.model_manager.get_voice_model(voice_name)
            if not voice_model:
                logger.error(f"Voice model not found: {voice_name}")
                return False
            
            # Initialize Piper voice
            self.piper_voice = piper.PiperVoice.load(
                model_path=voice_model['model_path'],
                config_path=voice_model['config_path'],
                use_cuda=False  # CPU-only for compatibility
            )
            
            self.current_voice = voice_name
            logger.info(f"✅ Voice model loaded: {voice_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load voice model {voice_name}: {str(e)}")
            return False
    
    def _text_to_emotion(self, text: str, explicit_emotion: str = None) -> VoiceEmotion:
        """Determine appropriate emotion for text content."""
        if explicit_emotion:
            try:
                return VoiceEmotion(explicit_emotion.lower())
            except ValueError:
                logger.warning(f"Unknown emotion: {explicit_emotion}, using neutral")
        
        # Auto-detect emotion from text content
        text_lower = text.lower()
        
        # Emotion detection heuristics
        if any(word in text_lower for word in ['happy', 'great', 'wonderful', 'excited', '!']):
            return VoiceEmotion.HAPPY
        elif any(word in text_lower for word in ['urgent', 'important', 'serious', 'critical']):
            return VoiceEmotion.SERIOUS
        elif any(word in text_lower for word in ['sorry', 'sad', 'unfortunate', 'unfortunately']):
            return VoiceEmotion.EMPATHETIC
        elif any(word in text_lower for word in ['joke', 'funny', 'laugh', 'haha']):
            return VoiceEmotion.PLAYFUL
        elif any(word in text_lower for word in ['confident', 'sure', 'certain', 'definitely']):
            return VoiceEmotion.CONFIDENT
        elif any(word in text_lower for word in ['calm', 'relax', 'peaceful', 'quiet']):
            return VoiceEmotion.CALM
        else:
            return VoiceEmotion.NEUTRAL
    
    async def synthesize_speech(self, text: str, voice_model: str = None, 
                              speed: float = None, emotion: str = None) -> SynthesisResult:
        """Synthesize speech from text using Piper TTS."""
        start_time = time.time()
        
        try:
            if not self.is_initialized or self.piper_voice is None:
                raise RuntimeError("TTS Synthesizer not properly initialized")
            
            # Validate and prepare inputs
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            voice_to_use = voice_model or self.current_voice
            if not voice_to_use:
                raise RuntimeError("No voice model loaded")
            
            speed_to_use = speed or self.config.audio.tts_speed
            detected_emotion = self._text_to_emotion(text, emotion)
            
            # Adjust text and parameters for emotion
            emotional_text = self.emotional_prosody.adjust_text_for_emotion(text, detected_emotion)
            final_speed = self.emotional_prosody.calculate_synthesis_speed(speed_to_use, detected_emotion)
            
            # Synthesize speech using Piper
            audio_bytes = bytearray()
            sample_rate = self.piper_voice.config.sample_rate
            
            # Piper synthesis
            self.piper_voice.synthesize(
                emotional_text,
                audio_bytes,
                speaker_id=None,  # Use default speaker
                length_scale=1.0 / final_speed,  # Piper uses inverse scale
                noise_scale=0.667,
                noise_w=0.8
            )
            
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Apply post-processing
            processed_audio = self.audio_postprocessor.apply_voice_enhancement(
                audio_float, sample_rate
            )
            processed_audio = self.audio_postprocessor.apply_emotional_characteristics(
                processed_audio, sample_rate, detected_emotion
            )
            
            # Convert back to int16 for playback
            final_audio = (processed_audio * 32767).astype(np.int16)
            
            processing_time = time.time() - start_time
            audio_duration = len(final_audio) / sample_rate
            
            # Update performance metrics
            self.total_synthesis_time += processing_time
            self.total_audio_generated += audio_duration
            self.synthesis_count += 1
            
            logger.info(f"Synthesized: '{text[:50]}...' (emotion: {detected_emotion.value}, "
                       f"duration: {audio_duration:.2f}s, processing: {processing_time:.2f}s)")
            
            return SynthesisResult(
                audio_data=final_audio,
                sample_rate=sample_rate,
                duration=audio_duration,
                voice_model=voice_to_use,
                emotion=detected_emotion,
                speed=final_speed,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {str(e)}")
            processing_time = time.time() - start_time
            raise
    
    async def synthesize(self, text: str, voice_model: str = None, 
                        speed: float = None, emotion: str = None) -> Dict[str, Any]:
        """
        Synthesize speech and return as base64 encoded audio.
        Main API method for the TTS engine.
        """
        try:
            result = await self.synthesize_speech(text, voice_model, speed, emotion)
            
            # Convert audio to base64
            audio_bytes = result.audio_data.tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            response = {
                "audio_data": audio_b64,
                "duration": result.duration,
                "voice_model": result.voice_model,
                "sample_rate": result.sample_rate,
                "emotion": result.emotion.value,
                "speed": result.speed,
                "processing_time": result.processing_time
            }
            
            return response
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {str(e)}")
            raise
    
    async def speak_text(self, text: str, voice_model: str = None, 
                        speed: float = None, emotion: str = None,
                        device_index: int = None) -> Dict[str, Any]:
        """Synthesize and immediately play speech."""
        try:
            # Synthesize speech
            result = await self.synthesize_speech(text, voice_model, speed, emotion)
            
            # Play audio
            await self.audio_player.play_audio_async(
                result.audio_data, 
                result.sample_rate, 
                device_index
            )
            
            return {
                "success": True,
                "text": text,
                "duration": result.duration,
                "voice_model": result.voice_model,
                "emotion": result.emotion.value,
                "speed": result.speed,
                "processing_time": result.processing_time,
                "message": "Speech synthesized and played successfully"
            }
            
        except Exception as e:
            logger.error(f"Speak text failed: {str(e)}")
            return {
                "success": False,
                "text": text,
                "duration": 0.0,
                "processing_time": 0.0,
                "message": f"Speech synthesis failed: {str(e)}"
            }
    
    async def get_available_voices(self) -> List[Dict]:
        """Get list of available voice models."""
        return self.model_manager.list_available_voices()
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get TTS engine performance metrics."""
        avg_synthesis_time = (
            self.total_synthesis_time / self.synthesis_count 
            if self.synthesis_count > 0 else 0
        )
        
        real_time_factor = (
            self.total_audio_generated / self.total_synthesis_time 
            if self.total_synthesis_time > 0 else 0
        )
        
        return {
            "initialized": self.is_initialized,
            "current_voice": self.current_voice,
            "synthesis_count": self.synthesis_count,
            "total_audio_generated_seconds": self.total_audio_generated,
            "total_synthesis_time_seconds": self.total_synthesis_time,
            "average_synthesis_time_seconds": avg_synthesis_time,
            "real_time_factor": real_time_factor,
            "voice_loaded": self.piper_voice is not None
        }
    
    async def shutdown(self):
        """Shutdown TTS engine gracefully."""
        logger.info("Shutting down TTS Synthesizer...")
        
        try:
            await self.audio_player.shutdown()
            
            # Clean up Piper voice
            if self.piper_voice:
                # Piper doesn't have explicit cleanup, but we can dereference
                self.piper_voice = None
            
            logger.info("✅ TTS Synthesizer shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during TTS engine shutdown: {str(e)}")


# Global TTS instance
_tts_instance: Optional[TTSSynthesizer] = None


async def get_tts_engine() -> TTSSynthesizer:
    """Get or create global TTS engine instance."""
    global _tts_instance
    
    if _tts_instance is None:
        _tts_instance = TTSSynthesizer()
        await _tts_instance.initialize()
    
    return _tts_instance


async def main():
    """Command-line testing for TTS engine."""
    tts_engine = await get_tts_engine()
    
    # Test available voices
    voices = await tts_engine.get_available_voices()
    print("Available Voices:")
    for voice in voices:
        print(f"  - {voice['name']}: {voice['description']} ({voice['sample_rate']}Hz)")
    
    # Test performance metrics
    metrics = await tts_engine.get_performance_metrics()
    print(f"\nTTS Engine Status:")
    print(f"Initialized: {metrics['initialized']}")
    print(f"Voice Loaded: {metrics['voice_loaded']}")
    print(f"Current Voice: {metrics['current_voice']}")
    
    if voices:
        print(f"\nReady for synthesis. Use synthesize() or speak_text() methods.")
    else:
        print(f"\n⚠️  No voice models available. Please download voice models to the models directory.")


if __name__ == "__main__":
    asyncio.run(main())