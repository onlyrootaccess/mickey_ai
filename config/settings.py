# Centralized configuration manager
"""
M.I.C.K.E.Y. AI Assistant - Centralized Configuration Manager
Made In Crisis, Keeping Everything Yours

SECOND FILE IN PIPELINE: Manages all settings, API keys, paths, and constants.
Provides single source of truth for entire Mickey AI ecosystem.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

# Setup logging
logger = logging.getLogger("MickeyConfig")


class VoiceModelQuality(Enum):
    """TTS Model quality levels - balance between speed and quality"""
    ULTRA_FAST = "ultra_fast"    # Lowest latency, lower quality
    BALANCED = "balanced"        # Best trade-off
    HIGH_QUALITY = "high_quality" # Best quality, higher latency


class SecurityLevel(Enum):
    """Security configuration levels"""
    MINIMAL = "minimal"      # Wake word only
    STANDARD = "standard"    # Wake word + voice biometrics  
    STRICT = "strict"        # Wake word + voice + face recognition


@dataclass
class AudioSettings:
    """Audio processing configuration"""
    # Input settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: int = 8  # pyaudio.paInt16 equivalent
    
    # Noise cancellation
    noise_reduction_level: int = 3  # 1-5 scale
    echo_cancellation: bool = True
    auto_gain_control: bool = True
    
    # Voice activity detection
    vad_aggressiveness: int = 2  # 0-3 scale
    silence_duration: float = 0.8  # seconds
    min_voice_duration: float = 0.3  # seconds
    
    # TTS settings
    tts_voice: str = "female_crystal"  # Default female voice
    tts_speed: float = 1.0  # 0.5 to 2.0
    tts_quality: VoiceModelQuality = VoiceModelQuality.BALANCED
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self).items()}


@dataclass
class LLMSettings:
    """Groq LLM configuration"""
    api_key: str = ""  # Will be set from environment
    model: str = "llama3-70b-8192"  # Groq's fastest 70B model
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    timeout: int = 30  # seconds
    
    # Response tuning
    enable_humor: bool = True
    enable_emotions: bool = True
    personality_strength: float = 0.8  # 0.0 to 1.0
    
    # Fallback models
    fallback_models: List[str] = None
    
    def __post_init__(self):
        if self.fallback_models is None:
            self.fallback_models = ["mixtral-8x7b-32768", "llama3-8b-8192"]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GUISettings:
    """Frontend GUI configuration"""
    # Window settings
    window_width: int = 800
    window_height: int = 600
    transparency: float = 0.85  # 0.0 to 1.0
    always_on_top: bool = True
    
    # Visual theme - "Wire Gucci" aesthetic
    primary_color: str = "#00FFFF"  # Cyan wireframe
    secondary_color: str = "#FF00FF"  # Magenta accents
    background_color: str = "#0A0A0A"  # Near-black
    text_color: str = "#FFFFFF"
    
    # Animation settings
    enable_animations: bool = True
    animation_speed: float = 1.0  # 0.5 to 2.0
    pulse_effect: bool = True
    
    # HUD elements
    show_voice_visualizer: bool = True
    show_system_stats: bool = True
    show_response_text: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SecuritySettings:
    """Security and authentication configuration"""
    security_level: SecurityLevel = SecurityLevel.STANDARD
    
    # Wake word
    wake_word: str = "hey mickey"
    wake_word_sensitivity: float = 0.8  # 0.0 to 1.0
    
    # Biometrics
    voice_match_threshold: float = 0.7  # Voice similarity threshold
    face_match_threshold: float = 0.6   # Face recognition threshold
    max_auth_attempts: int = 3
    
    # Privacy
    store_voice_samples: bool = False
    store_face_data: bool = True  # Required for recognition
    auto_logout_minutes: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self).items()}


@dataclass
class FeatureSettings:
    """Feature toggle and configuration"""
    # Core features
    enable_web_search: bool = True
    enable_email: bool = True
    enable_reminders: bool = True
    enable_weather: bool = True
    enable_news: bool = True
    enable_music: bool = True
    
    # System control
    enable_mouse_control: bool = True
    enable_keyboard_control: bool = True
    enable_browser_control: bool = True
    
    # Learning and adaptation
    enable_learning: bool = True
    enable_personalization: bool = True
    
    # External APIs
    weather_api_key: str = ""
    news_api_key: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MickeyConfig:
    """
    Main configuration manager for M.I.C.K.E.Y. AI Assistant.
    Handles loading, saving, and accessing all settings.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        # Determine configuration directory
        if config_dir is None:
            self.config_dir = self._get_default_config_dir()
        else:
            self.config_dir = Path(config_dir)
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration file path
        self.config_file = self.config_dir / "mickey_config.json"
        
        # Initialize default settings
        self.audio = AudioSettings()
        self.llm = LLMSettings()
        self.gui = GUISettings()
        self.security = SecuritySettings()
        self.features = FeatureSettings()
        
        # System paths
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.data_dir / "models"
        self.security_dir = self.data_dir / "security_profiles"
        self.logs_dir = self.base_dir / "logs"
        
        # Create necessary directories
        self._create_directories()
        
        # Load existing configuration or create default
        self.load_config()
        
        # Load API keys from environment
        self._load_environment_keys()
    
    def _get_default_config_dir(self) -> Path:
        """Get platform-specific configuration directory."""
        system = os.name
        
        if system == 'nt':  # Windows
            appdata = os.getenv('APPDATA')
            if appdata:
                return Path(appdata) / "MickeyAI"
            else:
                return Path.home() / "AppData" / "Roaming" / "MickeyAI"
        
        elif system == 'posix':  # Linux/macOS
            xdg_config = os.getenv('XDG_CONFIG_HOME')
            if xdg_config:
                return Path(xdg_config) / "mickey-ai"
            else:
                return Path.home() / ".config" / "mickey-ai"
        
        else:  # Fallback
            return Path.cwd() / "config"
    
    def _create_directories(self):
        """Create all necessary directories for Mickey AI."""
        directories = [
            self.data_dir,
            self.models_dir,
            self.security_dir,
            self.logs_dir,
            self.config_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def _load_environment_keys(self):
        """Load API keys from environment variables."""
        # Groq API Key
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            self.llm.api_key = groq_key
            logger.info("Loaded Groq API key from environment")
        else:
            logger.warning("GROQ_API_KEY not found in environment")
        
        # Weather API Key
        weather_key = os.getenv("WEATHER_API_KEY")
        if weather_key:
            self.features.weather_api_key = weather_key
        
        # News API Key  
        news_key = os.getenv("NEWS_API_KEY")
        if news_key:
            self.features.news_api_key = news_key
    
    def load_config(self) -> bool:
        """
        Load configuration from JSON file.
        Returns True if successful, False otherwise.
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Update settings from loaded data
                self._update_from_dict(config_data)
                logger.info(f"Configuration loaded from {self.config_file}")
                return True
            else:
                logger.info("No existing config found, using defaults")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return False
    
    def save_config(self) -> bool:
        """
        Save current configuration to JSON file.
        Returns True if successful, False otherwise.
        """
        try:
            config_data = self.to_dict()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            return False
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update settings from dictionary data."""
        # Audio settings
        if 'audio' in config_data:
            audio_data = config_data['audio']
            for key, value in audio_data.items():
                if hasattr(self.audio, key):
                    # Handle enum values
                    if key == 'tts_quality' and isinstance(value, str):
                        value = VoiceModelQuality(value)
                    setattr(self.audio, key, value)
        
        # LLM settings
        if 'llm' in config_data:
            llm_data = config_data['llm']
            for key, value in llm_data.items():
                if hasattr(self.llm, key):
                    setattr(self.llm, key, value)
        
        # GUI settings
        if 'gui' in config_data:
            gui_data = config_data['gui']
            for key, value in gui_data.items():
                if hasattr(self.gui, key):
                    setattr(self.gui, key, value)
        
        # Security settings
        if 'security' in config_data:
            security_data = config_data['security']
            for key, value in security_data.items():
                if hasattr(self.security, key):
                    # Handle enum values
                    if key == 'security_level' and isinstance(value, str):
                        value = SecurityLevel(value)
                    setattr(self.security, key, value)
        
        # Feature settings
        if 'features' in config_data:
            feature_data = config_data['features']
            for key, value in feature_data.items():
                if hasattr(self.features, key):
                    setattr(self.features, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all settings to dictionary."""
        return {
            'audio': self.audio.to_dict(),
            'llm': self.llm.to_dict(),
            'gui': self.gui.to_dict(),
            'security': self.security.to_dict(),
            'features': self.features.to_dict(),
            'paths': {
                'base_dir': str(self.base_dir),
                'data_dir': str(self.data_dir),
                'config_dir': str(self.config_dir)
            }
        }
    
    def validate(self) -> List[str]:
        """
        Validate current configuration.
        Returns list of validation errors.
        """
        errors = []
        
        # Check Groq API key
        if not self.llm.api_key:
            errors.append("Groq API key is required. Set GROQ_API_KEY environment variable.")
        
        # Check audio settings
        if self.audio.sample_rate not in [8000, 16000, 22050, 44100]:
            errors.append("Sample rate must be one of: 8000, 16000, 22050, 44100")
        
        # Check security thresholds
        if not (0 <= self.security.voice_match_threshold <= 1):
            errors.append("Voice match threshold must be between 0 and 1")
        
        if not (0 <= self.security.face_match_threshold <= 1):
            errors.append("Face match threshold must be between 0 and 1")
        
        # Check GUI transparency
        if not (0 <= self.gui.transparency <= 1):
            errors.append("GUI transparency must be between 0 and 1")
        
        return errors
    
    def get_model_paths(self) -> Dict[str, Path]:
        """Get paths to all model files."""
        return {
            'whisper_model': self.models_dir / "whisper-tiny.pt",
            'piper_voice': self.models_dir / f"{self.audio.tts_voice}.onnx",
            'face_encodings': self.security_dir / "face_encodings.dat",
            'voice_profile': self.security_dir / "voice_profile.dat"
        }
    
    def reset_to_defaults(self):
        """Reset all settings to default values."""
        self.audio = AudioSettings()
        self.llm = LLMSettings()
        self.gui = GUISettings()
        self.security = SecuritySettings()
        self.features = FeatureSettings()
        
        logger.info("Configuration reset to defaults")


# Global configuration instance
_config_instance: Optional[MickeyConfig] = None


def get_config(config_dir: Optional[str] = None) -> MickeyConfig:
    """
    Get or create global configuration instance.
    This is the preferred way to access configuration throughout the application.
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = MickeyConfig(config_dir)
    
    return _config_instance


def reload_config(config_dir: Optional[str] = None) -> MickeyConfig:
    """
    Reload configuration from disk and return new instance.
    Useful for development and testing.
    """
    global _config_instance
    _config_instance = MickeyConfig(config_dir)
    return _config_instance


def main():
    """Command-line utility for configuration management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mickey AI Configuration Manager")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--reset", action="store_true", help="Reset to defaults")
    parser.add_argument("--save", action="store_true", help="Save current configuration")
    
    args = parser.parse_args()
    
    config = get_config()
    
    if args.show:
        print("Current Mickey AI Configuration:")
        print(json.dumps(config.to_dict(), indent=2))
    
    if args.validate:
        errors = config.validate()
        if errors:
            print("Configuration errors found:")
            for error in errors:
                print(f"  ❌ {error}")
        else:
            print("✅ Configuration is valid")
    
    if args.reset:
        config.reset_to_defaults()
        print("Configuration reset to defaults")
    
    if args.save:
        if config.save_config():
            print("Configuration saved successfully")
        else:
            print("Failed to save configuration")


if __name__ == "__main__":
    main()