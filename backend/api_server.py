# Main FastAPI app + endpoint routing
"""
M.I.C.K.E.Y. AI Assistant - Core FastAPI Server
Made In Crisis, Keeping Everything Yours

FOURTH FILE IN PIPELINE: Central API server that orchestrates all Mickey AI modules.
Provides RESTful endpoints for frontend communication, module coordination, and system management.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import Mickey AI configuration
from config.settings import get_config
from config.constants import (
    SystemConstants, ErrorCodes, ErrorMessages, 
    APIEndpoints, LLMConstants, SecurityConstants
)

# Setup logging
logger = logging.getLogger("MickeyAPIServer")

# Global state for the API server
class MickeyServerState:
    """Global state management for Mickey API server."""
    
    def __init__(self):
        self.is_initialized = False
        self.startup_time = None
        self.active_connections = 0
        self.request_count = 0
        self.system_status = "booting"
        self.security_unlocked = False
        self.current_user = None
        self.module_health = {}
        
        # Performance tracking
        self.avg_response_time = 0.0
        self.total_processing_time = 0.0
        
        # Module references (will be populated as modules are initialized)
        self.modules = {
            'security': None,
            'voice_stt': None,
            'voice_tts': None,
            'llm': None,
            'memory': None,
            'control': None,
            'gui': None
        }

# Global instances
server_state = MickeyServerState()
config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting M.I.C.K.E.Y. AI API Server...")
    logger.info(f"Made In Crisis, Keeping Everything Yours - v{SystemConstants.APP_VERSION}")
    
    server_state.startup_time = time.time()
    server_state.system_status = "initializing"
    
    try:
        # Initialize core systems
        await initialize_core_systems()
        server_state.is_initialized = True
        server_state.system_status = "ready"
        
        startup_duration = time.time() - server_state.startup_time
        logger.info(f"✅ Mickey API Server initialized in {startup_duration:.2f}s")
        
        yield  # Server runs here
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {str(e)}")
        server_state.system_status = "error"
        raise
    
    finally:
        # Shutdown
        logger.info("🛑 Shutting down M.I.C.K.E.Y. AI API Server...")
        await shutdown_core_systems()
        server_state.is_initialized = False
        server_state.system_status = "shutdown"


async def initialize_core_systems():
    """Initialize core Mickey AI systems."""
    logger.info("Initializing core systems...")
    
    # Validate configuration
    config_errors = config.validate()
    if config_errors:
        error_msg = f"Configuration validation failed: {', '.join(config_errors)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Initialize module health tracking
    for module_name in server_state.modules.keys():
        server_state.module_health[module_name] = {
            'status': 'not_loaded',
            'last_check': time.time(),
            'error_count': 0
        }
    
    # Note: Individual modules will be initialized when their endpoints are first called
    # This prevents circular dependencies during startup
    
    logger.info("✅ Core systems initialized")


async def shutdown_core_systems():
    """Shutdown core Mickey AI systems gracefully."""
    logger.info("Shutting down core systems...")
    
    # Shutdown modules in reverse dependency order
    modules_to_shutdown = ['gui', 'control', 'llm', 'memory', 'voice_tts', 'voice_stt', 'security']
    
    for module_name in modules_to_shutdown:
        module = server_state.modules.get(module_name)
        if module and hasattr(module, 'shutdown'):
            try:
                if asyncio.iscoroutinefunction(module.shutdown):
                    await module.shutdown()
                else:
                    module.shutdown()
                logger.info(f"✅ {module_name} shutdown complete")
            except Exception as e:
                logger.error(f"❌ {module_name} shutdown failed: {str(e)}")
    
    logger.info("✅ Core systems shutdown complete")


# Pydantic models for request/response schemas
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    uptime: float
    active_connections: int
    request_count: int
    avg_response_time: float
    system_status: str
    security_unlocked: bool
    module_health: Dict[str, Any]

class SecurityAuthRequest(BaseModel):
    """Security authentication request model."""
    auth_type: str = Field(..., description="Type of authentication: 'face', 'voice', or 'both'")
    image_data: Optional[str] = Field(None, description="Base64 encoded image for face auth")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio for voice auth")

class SecurityAuthResponse(BaseModel):
    """Security authentication response model."""
    success: bool
    message: str
    user_identified: Optional[str] = None
    security_level: str
    requires_additional_auth: bool = False

class STTRequest(BaseModel):
    """Speech-to-Text request model."""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    language: str = Field("en", description="Language code for transcription")
    enable_timestamps: bool = Field(False, description="Include word-level timestamps")

class STTResponse(BaseModel):
    """Speech-to-Text response model."""
    text: str
    confidence: float
    language: str
    duration: float
    word_timestamps: Optional[List[Dict]] = None

class TTSRequest(BaseModel):
    """Text-to-Speech request model."""
    text: str = Field(..., description="Text to synthesize")
    voice_model: str = Field(None, description="Voice model to use")
    speed: float = Field(1.0, description="Speech speed multiplier")
    emotion: str = Field("neutral", description="Emotional tone")

class TTSResponse(BaseModel):
    """Text-to-Speech response model."""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    duration: float
    voice_model: str
    sample_rate: int

class LLMRequest(BaseModel):
    """LLM query request model."""
    message: str = Field(..., description="User message to process")
    conversation_id: Optional[str] = Field(None, description="Conversation session ID")
    enable_memory: bool = Field(True, description="Use conversation memory")
    enable_humor: bool = Field(True, description="Enable humorous responses")
    temperature: float = Field(None, description="LLM temperature override")

class LLMResponse(BaseModel):
    """LLM query response model."""
    response: str
    conversation_id: str
    processing_time: float
    tokens_used: int
    model_used: str
    emotion_detected: Optional[str] = None
    contains_humor: bool = False

class CommandRequest(BaseModel):
    """System command request model."""
    command_type: str = Field(..., description="Type of command: 'mouse', 'keyboard', 'browser', 'system'")
    action: str = Field(..., description="Specific action to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Command parameters")
    require_confirmation: bool = Field(True, description="Require user confirmation")

class CommandResponse(BaseModel):
    """System command response model."""
    success: bool
    message: str
    action_performed: str
    requires_confirmation: bool
    confirmation_prompt: Optional[str] = None

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error_code: int
    error_message: str
    details: Optional[Dict[str, Any]] = None


# Create FastAPI application
app = FastAPI(
    title=SystemConstants.APP_NAME,
    description=SystemConstants.APP_FULL_NAME,
    version=SystemConstants.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for security validation
async def validate_security():
    """Validate that security requirements are met for protected endpoints."""
    if not server_state.security_unlocked and config.security.security_level != "minimal":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Security authentication required"
        )
    return True


# Utility functions
def calculate_response_time(start_time: float) -> float:
    """Calculate and update average response time."""
    response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Update running average
    server_state.request_count += 1
    server_state.total_processing_time += response_time
    server_state.avg_response_time = server_state.total_processing_time / server_state.request_count
    
    return response_time


async def initialize_module(module_name: str):
    """Lazy initialization of modules."""
    if server_state.modules[module_name] is not None:
        return server_state.modules[module_name]
    
    logger.info(f"Initializing module: {module_name}")
    
    try:
        # Import and initialize the module
        if module_name == 'security':
            from backend.security.security_orchestrator import SecurityOrchestrator
            module = SecurityOrchestrator()
        elif module_name == 'voice_stt':
            from modules.voice.stt_engine import STTEngine
            module = STTEngine()
        elif module_name == 'voice_tts':
            from modules.voice.tts_synthesizer import TTSSynthesizer
            module = TTSSynthesizer()
        elif module_name == 'llm':
            from backend.intelligence.groq_client import GroqClient
            module = GroqClient()
        elif module_name == 'memory':
            from backend.intelligence.memory_manager import MemoryManager
            module = MemoryManager()
        elif module_name == 'control':
            from modules.control.input_controller import InputController
            module = InputController()
        elif module_name == 'gui':
            from frontend.gui_main import MickeyGUI
            module = MickeyGUI()
        else:
            raise ValueError(f"Unknown module: {module_name}")
        
        # Initialize the module
        if asyncio.iscoroutinefunction(module.initialize):
            await module.initialize()
        elif hasattr(module, 'initialize'):
            module.initialize()
        
        server_state.modules[module_name] = module
        server_state.module_health[module_name]['status'] = 'healthy'
        server_state.module_health[module_name]['last_check'] = time.time()
        
        logger.info(f"✅ Module {module_name} initialized successfully")
        return module
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize module {module_name}: {str(e)}")
        server_state.module_health[module_name]['status'] = 'error'
        server_state.module_health[module_name]['last_check'] = time.time()
        server_state.module_health[module_name]['error_count'] += 1
        raise


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint - API information."""
    return {
        "message": f"Welcome to {SystemConstants.APP_NAME}",
        "version": SystemConstants.APP_VERSION,
        "description": SystemConstants.APP_FULL_NAME,
        "status": "operational"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    uptime = time.time() - server_state.startup_time if server_state.startup_time else 0
    
    return HealthResponse(
        status="healthy" if server_state.is_initialized else "initializing",
        version=SystemConstants.APP_VERSION,
        uptime=uptime,
        active_connections=server_state.active_connections,
        request_count=server_state.request_count,
        avg_response_time=server_state.avg_response_time,
        system_status=server_state.system_status,
        security_unlocked=server_state.security_unlocked,
        module_health=server_state.module_health
    )


@app.post("/api/security/authenticate", response_model=SecurityAuthResponse)
async def authenticate_user(request: SecurityAuthRequest):
    """
    Authenticate user using face recognition, voice biometrics, or both.
    This is the first step in the security flow.
    """
    start_time = time.time()
    
    try:
        security_module = await initialize_module('security')
        
        # Perform authentication based on type
        if request.auth_type == 'face':
            result = await security_module.authenticate_face(request.image_data)
        elif request.auth_type == 'voice':
            result = await security_module.authenticate_voice(request.audio_data)
        elif request.auth_type == 'both':
            result = await security_module.authenticate_dual(request.image_data, request.audio_data)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid authentication type. Use 'face', 'voice', or 'both'"
            )
        
        # Update server state if authentication successful
        if result['success']:
            server_state.security_unlocked = True
            server_state.current_user = result.get('user_id')
        
        response_time = calculate_response_time(start_time)
        logger.info(f"Authentication completed in {response_time:.2f}ms - Success: {result['success']}")
        
        return SecurityAuthResponse(**result)
        
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}"
        )


@app.post("/api/voice/stt", response_model=STTResponse)
async def speech_to_text(request: STTRequest, _: bool = Depends(validate_security)):
    """Convert speech audio to text using Whisper STT."""
    start_time = time.time()
    
    try:
        stt_module = await initialize_module('voice_stt')
        
        # Process audio through STT engine
        result = await stt_module.transcribe(
            audio_data=request.audio_data,
            language=request.language,
            include_timestamps=request.enable_timestamps
        )
        
        response_time = calculate_response_time(start_time)
        logger.info(f"STT completed in {response_time:.2f}ms - Text: {result['text'][:50]}...")
        
        return STTResponse(**result)
        
    except Exception as e:
        logger.error(f"STT processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech recognition failed: {str(e)}"
        )


@app.post("/api/voice/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest, _: bool = Depends(validate_security)):
    """Convert text to speech using Piper TTS."""
    start_time = time.time()
    
    try:
        tts_module = await initialize_module('voice_tts')
        
        # Synthesize speech
        result = await tts_module.synthesize(
            text=request.text,
            voice_model=request.voice_model or config.audio.tts_voice,
            speed=request.speed,
            emotion=request.emotion
        )
        
        response_time = calculate_response_time(start_time)
        logger.info(f"TTS completed in {response_time:.2f}ms - Text length: {len(request.text)}")
        
        return TTSResponse(**result)
        
    except Exception as e:
        logger.error(f"TTS synthesis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis failed: {str(e)}"
        )


@app.post("/api/llm/query", response_model=LLMResponse)
async def query_llm(request: LLMRequest, _: bool = Depends(validate_security)):
    """Process user message through Groq LLM with reasoning and memory."""
    start_time = time.time()
    
    try:
        llm_module = await initialize_module('llm')
        memory_module = await initialize_module('memory')
        
        # Get conversation context if memory is enabled
        context = None
        if request.enable_memory and request.conversation_id:
            context = await memory_module.get_conversation_context(request.conversation_id)
        
        # Process through LLM
        result = await llm_module.process_query(
            message=request.message,
            context=context,
            enable_humor=request.enable_humor,
            temperature=request.temperature
        )
        
        # Update conversation memory
        if request.enable_memory:
            if not request.conversation_id:
                # Create new conversation
                conversation_id = await memory_module.create_conversation()
            else:
                conversation_id = request.conversation_id
            
            await memory_module.add_interaction(
                conversation_id=conversation_id,
                user_message=request.message,
                ai_response=result['response']
            )
            result['conversation_id'] = conversation_id
        
        response_time = calculate_response_time(start_time)
        logger.info(f"LLM query completed in {response_time:.2f}ms - Tokens: {result.get('tokens_used', 0)}")
        
        return LLMResponse(**result)
        
    except Exception as e:
        logger.error(f"LLM query failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI processing failed: {str(e)}"
        )


@app.post("/api/control/command", response_model=CommandResponse)
async def execute_command(request: CommandRequest, _: bool = Depends(validate_security)):
    """Execute system control commands (mouse, keyboard, browser, etc.)."""
    start_time = time.time()
    
    try:
        control_module = await initialize_module('control')
        
        # Execute the command
        result = await control_module.execute_command(
            command_type=request.command_type,
            action=request.action,
            parameters=request.parameters
        )
        
        response_time = calculate_response_time(start_time)
        logger.info(f"Command executed in {response_time:.2f}ms - {request.command_type}.{request.action}")
        
        return CommandResponse(
            success=result['success'],
            message=result['message'],
            action_performed=request.action,
            requires_confirmation=request.require_confirmation,
            confirmation_prompt=result.get('confirmation_prompt')
        )
        
    except Exception as e:
        logger.error(f"Command execution failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Command execution failed: {str(e)}"
        )


@app.get("/api/system/status")
async def system_status():
    """Get detailed system status information."""
    return {
        "server": {
            "initialized": server_state.is_initialized,
            "status": server_state.system_status,
            "uptime": time.time() - server_state.startup_time if server_state.startup_time else 0,
            "active_connections": server_state.active_connections,
            "request_count": server_state.request_count,
            "avg_response_time": server_state.avg_response_time
        },
        "security": {
            "unlocked": server_state.security_unlocked,
            "current_user": server_state.current_user,
            "security_level": config.security.security_level.value
        },
        "performance": {
            "max_response_time_ms": SystemConstants.MAX_RESPONSE_TIME_MS,
            "memory_usage_limit_mb": SystemConstants.MEMORY_USAGE_LIMIT_MB
        },
        "modules": server_state.module_health
    }


@app.post("/api/system/shutdown")
async def shutdown_system(background_tasks: BackgroundTasks, _: bool = Depends(validate_security)):
    """Initiate system shutdown (graceful)."""
    logger.info("Shutdown command received via API")
    
    async def shutdown_background():
        await asyncio.sleep(1)  # Give time for response to be sent
        # This would typically trigger the main application shutdown
        # For now, we'll just log and update state
        server_state.system_status = "shutting_down"
        logger.info("System shutdown initiated via API")
    
    background_tasks.add_task(shutdown_background)
    
    return {
        "message": "Shutdown sequence initiated",
        "status": "shutting_down"
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with standardized error response."""
    logger.warning(f"HTTPException: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=exc.status_code,
            error_message=exc.detail,
            details={"path": request.url.path}
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions with standardized error response."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_code=ErrorCodes.SYSTEM_ERROR,
            error_message="An internal server error occurred",
            details={"exception": str(exc)}
        ).dict()
    )


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request, call_next):
    """Middleware to track request metrics."""
    server_state.active_connections += 1
    start_time = time.time()
    
    try:
        response = await call_next(request)
        return response
    finally:
        server_state.active_connections -= 1
        # Response time is calculated in individual endpoints for accuracy


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the Mickey API server."""
    uvicorn.run(
        "backend.api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    # Command-line execution for development
    import argparse
    
    parser = argparse.ArgumentParser(description="Mickey AI API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting {SystemConstants.APP_NAME} API Server...")
    print(f"📡 Endpoint: http://{args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    print(f"🔧 Reload: {args.reload}")
    
    run_server(host=args.host, port=args.port, reload=args.reload)