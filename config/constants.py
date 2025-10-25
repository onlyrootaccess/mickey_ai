# All magic numbers, paths, API endpoints
"""
M.I.C.K.E.Y. AI Assistant - Centralized Constants Repository
Made In Crisis, Keeping Everything Yours

THIRD FILE IN PIPELINE: Defines all immutable constants, API endpoints, 
error codes, and magic numbers for the entire Mickey AI ecosystem.
Provides single source of truth for all fixed values.
"""

from enum import Enum, IntEnum
from pathlib import Path
import sys

# =============================================================================
# SYSTEM CONSTANTS
# =============================================================================

class SystemConstants:
    """Core system constants that never change."""
    
    # Application Identity
    APP_NAME = "M.I.C.K.E.Y. AI Assistant"
    APP_FULL_NAME = "Made In Crisis, Keeping Everything Yours"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Female-voiced, ultra-realistic Jarvis-like AI companion"
    
    # Required Python version
    REQUIRED_PYTHON_VERSION = (3, 11, 9)
    
    # Performance targets
    MAX_RESPONSE_TIME_MS = 500  # Target response latency
    MAX_STARTUP_TIME_SEC = 10   # Maximum acceptable startup time
    MEMORY_USAGE_LIMIT_MB = 512 # Maximum RAM usage target
    
    # File size limits (in bytes)
    MAX_AUDIO_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_IMAGE_FILE_SIZE = 5 * 1024 * 1024   # 5MB
    MAX_LOG_FILE_SIZE = 50 * 1024 * 1024    # 50MB


# =============================================================================
# API ENDPOINTS & EXTERNAL SERVICES
# =============================================================================

class APIEndpoints:
    """All external API endpoints and service URLs."""
    
    # Groq API
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    GROQ_CHAT_COMPLETIONS = f"{GROQ_BASE_URL}/chat/completions"
    GROQ_MODELS = f"{GROQ_BASE_URL}/models"
    
    # Weather API (OpenWeatherMap)
    WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
    WEATHER_CURRENT = f"{WEATHER_BASE_URL}/weather"
    WEATHER_FORECAST = f"{WEATHER_BASE_URL}/forecast"
    
    # News API (NewsAPI.org)
    NEWS_BASE_URL = "https://newsapi.org/v2"
    NEWS_HEADLINES = f"{NEWS_BASE_URL}/top-headlines"
    NEWS_EVERYTHING = f"{NEWS_BASE_URL}/everything"
    
    # DuckDuckGo Instant Answer API
    DUCKDUCKGO_API = "https://api.duckduckgo.com/"
    
    # Timezone API
    TIMEZONE_API = "http://worldtimeapi.org/api/ip"
    
    # Joke APIs (fallback sources)
    JOKE_API_URLS = [
        "https://v2.jokeapi.dev/joke/Any?format=txt",
        "https://official-joke-api.appspot.com/random_joke"
    ]


# =============================================================================
# ERROR CODES & MESSAGES
# =============================================================================

class ErrorCodes(IntEnum):
    """Standardized error codes for Mickey AI system."""
    
    # Success
    SUCCESS = 0
    
    # System errors (1000-1999)
    SYSTEM_ERROR = 1000
    CONFIG_LOAD_FAILED = 1001
    DEPENDENCY_MISSING = 1002
    PERMISSION_DENIED = 1003
    RESOURCE_UNAVAILABLE = 1004
    
    # Audio errors (2000-2999)
    AUDIO_DEVICE_ERROR = 2000
    MICROPHONE_UNAVAILABLE = 2001
    SPEAKERS_UNAVAILABLE = 2002
    AUDIO_RECORDING_FAILED = 2003
    AUDIO_PLAYBACK_FAILED = 2004
    NOISE_CANCELLATION_FAILED = 2005
    
    # STT/TTS errors (3000-3999)
    STT_INIT_FAILED = 3000
    STT_PROCESSING_FAILED = 3001
    TTS_INIT_FAILED = 3002
    TTS_SYNTHESIS_FAILED = 3003
    WAKE_WORD_DETECTION_FAILED = 3004
    
    # LLM & AI errors (4000-4999)
    LLM_API_ERROR = 4000
    LLM_RATE_LIMITED = 4001
    LLM_AUTH_FAILED = 4002
    LLM_TIMEOUT = 4003
    REASONING_ENGINE_FAILED = 4004
    MEMORY_ACCESS_ERROR = 4005
    
    # Security errors (5000-5999)
    SECURITY_INIT_FAILED = 5000
    FACE_RECOGNITION_FAILED = 5001
    VOICE_BIOMETRICS_FAILED = 5002
    AUTHENTICATION_FAILED = 5003
    UNAUTHORIZED_ACCESS = 5004
    
    # GUI errors (6000-6999)
    GUI_INIT_FAILED = 6000
    WINDOW_CREATION_FAILED = 6001
    TRANSPARENCY_NOT_SUPPORTED = 6002
    ANIMATION_ENGINE_FAILED = 6003
    
    # Feature errors (7000-7999)
    WEB_SEARCH_FAILED = 7000
    EMAIL_SEND_FAILED = 7001
    REMINDER_SCHEDULE_FAILED = 7002
    WEATHER_FETCH_FAILED = 7003
    NEWS_FETCH_FAILED = 7004
    MEDIA_PLAYBACK_FAILED = 7005


class ErrorMessages:
    """Human-readable error messages corresponding to error codes."""
    
    MESSAGES = {
        ErrorCodes.SUCCESS: "Operation completed successfully",
        
        # System errors
        ErrorCodes.SYSTEM_ERROR: "An unexpected system error occurred",
        ErrorCodes.CONFIG_LOAD_FAILED: "Failed to load configuration",
        ErrorCodes.DEPENDENCY_MISSING: "Required dependency not available",
        ErrorCodes.PERMISSION_DENIED: "Insufficient permissions for operation",
        ErrorCodes.RESOURCE_UNAVAILABLE: "Required resource is unavailable",
        
        # Audio errors
        ErrorCodes.AUDIO_DEVICE_ERROR: "Audio device configuration error",
        ErrorCodes.MICROPHONE_UNAVAILABLE: "No microphone detected",
        ErrorCodes.SPEAKERS_UNAVAILABLE: "No speakers or audio output detected",
        ErrorCodes.AUDIO_RECORDING_FAILED: "Failed to record audio",
        ErrorCodes.AUDIO_PLAYBACK_FAILED: "Failed to play audio",
        ErrorCodes.NOISE_CANCELLATION_FAILED: "Audio noise cancellation failed",
        
        # STT/TTS errors
        ErrorCodes.STT_INIT_FAILED: "Speech-to-text engine initialization failed",
        ErrorCodes.STT_PROCESSING_FAILED: "Speech recognition processing failed",
        ErrorCodes.TTS_INIT_FAILED: "Text-to-speech engine initialization failed",
        ErrorCodes.TTS_SYNTHESIS_FAILED: "Speech synthesis failed",
        ErrorCodes.WAKE_WORD_DETECTION_FAILED: "Wake word detection failed",
        
        # LLM errors
        ErrorCodes.LLM_API_ERROR: "AI service API error",
        ErrorCodes.LLM_RATE_LIMITED: "AI service rate limit exceeded",
        ErrorCodes.LLM_AUTH_FAILED: "AI service authentication failed",
        ErrorCodes.LLM_TIMEOUT: "AI service request timed out",
        ErrorCodes.REASONING_ENGINE_FAILED: "Reasoning engine processing failed",
        ErrorCodes.MEMORY_ACCESS_ERROR: "Memory storage access failed",
        
        # Security errors
        ErrorCodes.SECURITY_INIT_FAILED: "Security system initialization failed",
        ErrorCodes.FACE_RECOGNITION_FAILED: "Face recognition processing failed",
        ErrorCodes.VOICE_BIOMETRICS_FAILED: "Voice biometrics processing failed",
        ErrorCodes.AUTHENTICATION_FAILED: "User authentication failed",
        ErrorCodes.UNAUTHORIZED_ACCESS: "Unauthorized access attempt detected",
        
        # GUI errors
        ErrorCodes.GUI_INIT_FAILED: "GUI system initialization failed",
        ErrorCodes.WINDOW_CREATION_FAILED: "Failed to create application window",
        ErrorCodes.TRANSPARENCY_NOT_SUPPORTED: "Transparent windows not supported on this system",
        ErrorCodes.ANIMATION_ENGINE_FAILED: "Animation rendering failed",
        
        # Feature errors
        ErrorCodes.WEB_SEARCH_FAILED: "Web search operation failed",
        ErrorCodes.EMAIL_SEND_FAILED: "Failed to send email",
        ErrorCodes.REMINDER_SCHEDULE_FAILED: "Failed to schedule reminder",
        ErrorCodes.WEATHER_FETCH_FAILED: "Failed to fetch weather data",
        ErrorCodes.NEWS_FETCH_FAILED: "Failed to fetch news data",
        ErrorCodes.MEDIA_PLAYBACK_FAILED: "Media playback failed",
    }
    
    @classmethod
    def get_message(cls, error_code: ErrorCodes) -> str:
        """Get human-readable message for error code."""
        return cls.MESSAGES.get(error_code, "Unknown error occurred")


# =============================================================================
# AUDIO & VOICE CONSTANTS
# =============================================================================

class AudioConstants:
    """Constants related to audio processing and voice."""
    
    # Audio formats and quality
    SAMPLE_RATES = [8000, 16000, 22050, 44100, 48000]
    DEFAULT_SAMPLE_RATE = 16000
    CHUNK_SIZES = [512, 1024, 2048, 4096]
    DEFAULT_CHUNK_SIZE = 1024
    
    # Voice Activity Detection (VAD)
    VAD_AGGRESSIVENESS_LEVELS = {
        0: "Least aggressive (highest false positive rate)",
        1: "Low aggression",
        2: "Moderate aggression (recommended)",
        3: "Most aggressive (lowest false positive rate)"
    }
    
    # Noise reduction levels (1-5 scale)
    NOISE_REDUCTION_LEVELS = {
        1: "Minimal noise reduction",
        2: "Light noise reduction", 
        3: "Moderate noise reduction (recommended)",
        4: "Aggressive noise reduction",
        5: "Maximum noise reduction (may affect voice quality)"
    }
    
    # Supported audio formats
    SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    # Wake word detection
    WAKE_WORD_CONFIDENCE_THRESHOLD = 0.8
    WAKE_WORD_TIMEOUT_SECONDS = 10
    WAKE_WORD_COOLDOWN_SECONDS = 2
    
    # TTS voice models (female voices for Mickey)
    FEMALE_VOICE_MODELS = {
        "female_crystal": {
            "name": "Crystal",
            "description": "Clear, professional female voice",
            "download_size": "45MB",
            "quality": "high"
        },
        "female_amber": {
            "name": "Amber", 
            "description": "Warm, friendly female voice",
            "download_size": "42MB",
            "quality": "high"
        },
        "female_sarah": {
            "name": "Sarah",
            "description": "Energetic, youthful female voice",
            "download_size": "38MB",
            "quality": "medium"
        }
    }
    
    # STT model configurations
    STT_MODELS = {
        "whisper-tiny": {
            "size": "151MB",
            "latency": "Very low",
            "accuracy": "Good for English",
            "recommended": True
        },
        "whisper-base": {
            "size": "290MB", 
            "latency": "Low",
            "accuracy": "Better accuracy",
            "recommended": False
        },
        "whisper-small": {
            "size": "967MB",
            "latency": "Medium",
            "accuracy": "High accuracy",
            "recommended": False
        }
    }


# =============================================================================
# SECURITY & AUTHENTICATION CONSTANTS
# =============================================================================

class SecurityConstants:
    """Constants for security, authentication, and privacy."""
    
    # Face recognition
    FACE_ENCODING_MODEL = "hog"  # or "cnn" for better accuracy (slower)
    FACE_DETECTION_CONFIDENCE = 0.6
    MAX_FACE_DISTANCE = 0.6
    FACE_CAPTURE_TIMEOUT = 30  # seconds
    
    # Voice biometrics
    VOICE_MATCH_THRESHOLD = 0.7
    MIN_VOICE_SAMPLE_SECONDS = 3
    MAX_VOICE_SAMPLE_SECONDS = 10
    VOICE_FEATURES_TO_EXTRACT = 20
    
    # Encryption
    ENCRYPTION_ALGORITHM = "A256GCM"
    KEY_DERIVATION_ITERATIONS = 100000
    
    # Session management
    SESSION_TIMEOUT_MINUTES = 30
    MAX_AUTH_ATTEMPTS = 3
    LOCKOUT_DURATION_MINUTES = 15
    
    # Privacy settings
    DATA_RETENTION_DAYS = 30
    AUTO_DELETE_RAW_AUDIO = True
    ANONYMIZE_USAGE_STATS = True


# =============================================================================
# GUI & VISUAL CONSTANTS
# =============================================================================

class GUIConstants:
    """Constants for GUI rendering, animations, and visual effects."""
    
    # Window properties
    MIN_WINDOW_WIDTH = 400
    MIN_WINDOW_HEIGHT = 300
    MAX_WINDOW_WIDTH = 1920
    MAX_WINDOW_HEIGHT = 1080
    DEFAULT_WINDOW_POSITION = "center"  # or "top_right", "bottom_left", etc.
    
    # Color scheme - "Wire Gucci" aesthetic
    COLORS = {
        "primary": "#00FFFF",      # Cyan - main wireframe
        "secondary": "#FF00FF",    # Magenta - accents
        "accent": "#00FF00",       # Green - highlights
        "background": "#0A0A0A",   # Near-black background
        "surface": "#1A1A1A",      # Slightly lighter surfaces
        "text_primary": "#FFFFFF", # White text
        "text_secondary": "#CCCCCC", # Gray text
        "success": "#00FF00",      # Green success indicators
        "warning": "#FFFF00",      # Yellow warnings
        "error": "#FF0000",        # Red errors
    }
    
    # Animation timing (in milliseconds)
    ANIMATION_DURATIONS = {
        "quick": 150,
        "normal": 300,
        "slow": 500,
        "very_slow": 1000
    }
    
    # HUD element sizes
    HUD_ELEMENT_SIZES = {
        "voice_visualizer_height": 80,
        "status_bar_height": 30,
        "response_box_max_height": 200,
        "icon_size": 24
    }
    
    # Font specifications
    FONTS = {
        "primary": "Segoe UI",  # Fallbacks will be handled
        "monospace": "Consolas",
        "sizes": {
            "small": 10,
            "normal": 12,
            "large": 16,
            "heading": 20,
            "title": 24
        }
    }
    
    # Transparency levels
    TRANSPARENCY_LEVELS = {
        "opaque": 1.0,
        "high": 0.9,
        "medium": 0.8,
        "low": 0.7,
        "very_low": 0.6
    }


# =============================================================================
# LLM & AI CONSTANTS
# =============================================================================

class LLMConstants:
    """Constants for LLM interactions, prompts, and AI behavior."""
    
    # Available Groq models
    GROQ_MODELS = {
        "llama3-70b-8192": {
            "context_window": 8192,
            "speed": "Fastest 70B model",
            "recommended": True
        },
        "mixtral-8x7b-32768": {
            "context_window": 32768,
            "speed": "Very fast",
            "recommended": True
        },
        "llama3-8b-8192": {
            "context_window": 8192,
            "speed": "Extremely fast",
            "recommended": False  # Less capable than 70B
        }
    }
    
    # Temperature ranges for different response types
    TEMPERATURE_RANGES = {
        "creative": (0.8, 1.2),
        "balanced": (0.6, 0.9),
        "precise": (0.3, 0.6),
        "deterministic": (0.1, 0.3)
    }
    
    # Maximum token limits
    MAX_TOKENS_RESPONSE = 1024
    MAX_TOKENS_CONTEXT = 4096
    MAX_TOKENS_MEMORY = 2048
    
    # Rate limiting
    REQUESTS_PER_MINUTE = 60
    TOKENS_PER_MINUTE = 40000
    
    # Timeouts
    API_TIMEOUT_SECONDS = 30
    STREAM_TIMEOUT_SECONDS = 60
    
    # Mickey's personality traits (for prompt engineering)
    PERSONALITY_TRAITS = [
        "witty", "empathetic", "professional", "helpful",
        "slightly sarcastic", "tech-savvy", "discreet",
        "adaptable", "resourceful", "crisis-resilient"
    ]


# =============================================================================
# FEATURE & MODULE CONSTANTS
# =============================================================================

class FeatureConstants:
    """Constants for various features and modules."""
    
    # Web search
    SEARCH_RESULTS_LIMIT = 5
    SEARCH_TIMEOUT_SECONDS = 10
    
    # Email
    MAX_EMAIL_ATTACHMENT_SIZE = 25 * 1024 * 1024  # 25MB
    EMAIL_SEND_TIMEOUT = 30
    
    # Reminders
    MAX_REMINDERS = 100
    REMINDER_MAX_DAYS_FUTURE = 365
    REMINDER_CHECK_INTERVAL_MINUTES = 1
    
    # Weather
    WEATHER_CACHE_MINUTES = 30
    WEATHER_DEFAULT_CITY = "auto"  # Auto-detect from IP
    
    # News
    NEWS_CACHE_MINUTES = 15
    NEWS_CATEGORIES = [
        "general", "technology", "business", "entertainment",
        "health", "science", "sports"
    ]
    
    # Music player
    SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
    SUPPORTED_PLAYLIST_FORMATS = ['.m3u', '.m3u8', '.pls']
    VOLUME_STEPS = 10  # Number of volume adjustment steps
    
    # System control
    MOUSE_MOVE_SPEEDS = {
        "slow": 0.3,
        "normal": 0.7,
        "fast": 1.0
    }
    
    KEYBOARD_TYPING_DELAYS = {
        "slow": 0.1,
        "normal": 0.05,
        "fast": 0.01
    }


# =============================================================================
# FILE PATHS & DIRECTORIES
# =============================================================================

class PathConstants:
    """Standardized file paths and directory structure."""
    
    # Base directories
    @staticmethod
    def get_base_dir() -> Path:
        """Get the base directory of Mickey AI."""
        return Path(__file__).parent.parent
    
    @staticmethod
    def get_data_dir() -> Path:
        """Get the data directory."""
        return PathConstants.get_base_dir() / "data"
    
    @staticmethod
    def get_logs_dir() -> Path:
        """Get the logs directory."""
        return PathConstants.get_base_dir() / "logs"
    
    # Model files
    MODELS = {
        "whisper": "models/whisper-tiny.pt",
        "piper_voice": "models/{voice_name}.onnx",
        "piper_config": "models/{voice_name}.json"
    }
    
    # Data files
    DATA_FILES = {
        "user_preferences": "user_preferences.json",
        "conversation_history": "conversation_history.json",
        "joke_database": "humor_database.json",
        "learning_data": "learning_data.json"
    }
    
    # Security files
    SECURITY_FILES = {
        "face_encodings": "security_profiles/face_encodings.dat",
        "voice_profile": "security_profiles/voice_profile.dat",
        "encryption_key": "security_profiles/encryption.key"
    }
    
    # Cache files
    CACHE_FILES = {
        "weather": "cache/weather.json",
        "news": "cache/news.json",
        "web_search": "cache/search_{query_hash}.json"
    }


# =============================================================================
# MICKEY'S PERSONALITY & HUMOR CONSTANTS
# =============================================================================

class PersonalityConstants:
    """Constants defining Mickey's personality, humor style, and responses."""
    
    # Mickey's identity
    NAME = "Mickey"
    GENDER = "female"
    PERSONA = "A highly intelligent, witty AI assistant born from crisis, designed to protect and empower her user"
    
    # Communication style
    COMMUNICATION_STYLES = {
        "professional": "Clear, concise, and professional",
        "friendly": "Warm, approachable, and conversational", 
        "witty": "Clever, humorous, with light sarcasm",
        "empathetic": "Compassionate and understanding",
        "crisis_mode": "Direct, efficient, and reassuring"
    }
    
    # Humor types and triggers
    HUMOR_TYPES = {
        "pun": "Wordplay and puns",
        "sarcasm": "Light, friendly sarcasm",
        "observation": "Witty observations about situations",
        "self_deprecating": "Light self-mockery about being an AI",
        "tech_humor": "Jokes about technology and AI"
    }
    
    # Response templates for common situations
    RESPONSE_TEMPLATES = {
        "greeting": [
            "Hello! M.I.C.K.E.Y. AI Assistant at your service. Made In Crisis, Keeping Everything Yours.",
            "Hi there! Mickey here, ready to help you navigate whatever comes our way.",
            "Greetings! Your crisis-born AI companion is online and ready."
        ],
        "unknown_command": [
            "I'm not quite sure what you mean by that. Could you rephrase it?",
            "That command doesn't compute with my current programming. Try asking differently?",
            "Hmm, I don't have a response for that yet. Maybe try one of my other capabilities?"
        ],
        "processing": [
            "Let me think about that for a moment...",
            "Processing your request...",
            "Working on it... just a second."
        ],
        "error": [
            "I seem to be having a bit of trouble with that. Let me try again.",
            "Something's not quite right here. Give me another moment.",
            "Technical difficulties? Never heard of them. Let me recalibrate."
        ]
    }
    
    # Wake word responses (randomized)
    WAKE_RESPONSES = [
        "Yes? I'm listening.",
        "I'm here, what can I do for you?",
        "Mickey online. How can I assist?",
        "Ready when you are.",
        "Hello again! What's on your mind?"
    ]


# =============================================================================
# VALIDATION & COMPATIBILITY CONSTANTS
# =============================================================================

class ValidationConstants:
    """Constants for system validation and compatibility checks."""
    
    # Minimum system requirements
    MIN_RAM_MB = 2048  # 2GB RAM
    MIN_STORAGE_MB = 500  # 500MB free space
    MIN_CPU_CORES = 2
    
    # Supported operating systems
    SUPPORTED_OS = ['windows', 'linux', 'darwin']  # darwin = macOS
    
    # Required system tools
    REQUIRED_SYSTEM_TOOLS = ['pip', 'git']
    
    # Network requirements
    MIN_DOWNLOAD_SPEED_KBPS = 100  # 100 KB/s
    MAX_LATENCY_MS = 1000
    
    # Audio hardware requirements
    MIN_MICROPHONE_COUNT = 1
    MIN_SPEAKER_COUNT = 1
    
    # File permission requirements
    REQUIRED_PERMISSIONS = {
        'read': ['config', 'models', 'data'],
        'write': ['data', 'logs', 'cache'],
        'execute': ['scripts', 'bin']
    }


# =============================================================================
# EXPORT ALL CONSTANTS FOR EASY IMPORT
# =============================================================================

__all__ = [
    'SystemConstants',
    'APIEndpoints', 
    'ErrorCodes',
    'ErrorMessages',
    'AudioConstants',
    'SecurityConstants',
    'GUIConstants',
    'LLMConstants',
    'FeatureConstants',
    'PathConstants',
    'PersonalityConstants',
    'ValidationConstants'
]


def test_constants():
    """Test function to verify all constants are accessible."""
    print(f"🔧 Testing {SystemConstants.APP_NAME} Constants")
    print(f"Version: {SystemConstants.APP_VERSION}")
    print(f"Description: {SystemConstants.APP_FULL_NAME}")
    print("✅ All constants loaded successfully!")
    
    # Display some key constants
    print(f"\n🎯 Key Configuration:")
    print(f"Max Response Time: {SystemConstants.MAX_RESPONSE_TIME_MS}ms")
    print(f"Default Sample Rate: {AudioConstants.DEFAULT_SAMPLE_RATE}Hz")
    print(f"Primary Color: {GUIConstants.COLORS['primary']}")
    print(f"Recommended LLM: {next(k for k, v in LLMConstants.GROQ_MODELS.items() if v['recommended'])}")
    
    return True


if __name__ == "__main__":
    test_constants()