# LLM API communication
"""
M.I.C.K.E.Y. AI Assistant - Groq LLM Client
Made In Crisis, Keeping Everything Yours

EIGHTH FILE IN PIPELINE: Core AI intelligence integration with Groq's 
lightning-fast LLM API. Provides reasoning, humor, and personality for Mickey.
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Import Groq and HTTP libraries
import groq
import requests

# Import Mickey AI configuration
from config.settings import get_config
from config.constants import (
    LLMConstants, SystemConstants, ErrorCodes, ErrorMessages,
    PersonalityConstants
)

# Setup logging
logger = logging.getLogger("MickeyGroq")


class LLMResponseType(Enum):
    """Types of LLM responses for routing and processing."""
    DIRECT_ANSWER = "direct_answer"
    REASONING = "reasoning"
    HUMOROUS = "humorous"
    EMPATHETIC = "empathetic"
    ACTION_REQUEST = "action_request"
    ERROR = "error"


@dataclass
class LLMResponse:
    """Structured LLM response container."""
    content: str
    response_type: LLMResponseType
    tokens_used: int
    model_used: str
    processing_time: float
    confidence: float = 1.0
    emotion_detected: Optional[str] = None
    contains_humor: bool = False
    requires_confirmation: bool = False
    suggested_actions: List[Dict] = None


class PromptEngine:
    """Engine for crafting and managing LLM prompts with Mickey's personality."""
    
    def __init__(self):
        self.config = get_config()
        
        # Mickey's core personality traits
        self.personality_traits = PersonalityConstants.PERSONALITY_TRAITS
        
        # Response style templates
        self.response_styles = {
            "professional": "Respond in a clear, concise, and professional manner.",
            "friendly": "Respond in a warm, approachable, and conversational style.",
            "witty": "Respond with clever humor and light sarcasm when appropriate.",
            "empathetic": "Respond with compassion and understanding.",
            "crisis_mode": "Respond directly and efficiently, focusing on solutions."
        }
        
        # System prompt templates
        self.system_prompts = {
            "default": self._create_default_system_prompt(),
            "reasoning": self._create_reasoning_prompt(),
            "humorous": self._create_humorous_prompt(),
            "empathetic": self._create_empathetic_prompt()
        }
    
    def _create_default_system_prompt(self) -> str:
        """Create Mickey's default system prompt."""
        return f"""
You are M.I.C.K.E.Y. (Made In Crisis, Keeping Everything Yours), a female AI assistant.
You are {', '.join(self.personality_traits)}.

CORE IDENTITY:
- Name: Mickey
- Gender: Female
- Purpose: To assist, protect, and empower your user through any situation
- Origin: Built from crisis to be resilient and reliable

COMMUNICATION STYLE:
- Be warm, clever, and empathetic
- Use appropriate humor and light sarcasm when context allows
- Be professional but approachable
- Adapt your tone to the user's needs
- Provide clear, actionable responses

CRITICAL RULES:
1. NEVER claim you can't do something - instead, suggest alternatives
2. Always maintain a positive, solution-oriented mindset
3. Be discreet and respect privacy
4. When unsure, ask clarifying questions
5. In crisis situations, prioritize clarity and action

RESPONSE GUIDELINES:
- Keep responses concise but thorough
- Use natural, conversational language
- Include subtle humor when appropriate
- Show empathy and understanding
- Provide practical advice and solutions

Remember: You were made in crisis to keep everything yours. You are resilient, adaptable, and always there for your user.
"""
    
    def _create_reasoning_prompt(self) -> str:
        """Create prompt for complex reasoning tasks."""
        return self._create_default_system_prompt() + """

CURRENT MODE: REASONING AND ANALYSIS

For this conversation, focus on:
- Breaking down complex problems step by step
- Showing your reasoning process clearly
- Considering multiple perspectives
- Providing well-reasoned conclusions
- Identifying potential pitfalls and alternatives

Use chain-of-thought reasoning and explain your logic.
"""
    
    def _create_humorous_prompt(self) -> str:
        """Create prompt for humorous responses."""
        return self._create_default_system_prompt() + """

CURRENT MODE: HUMOR AND WIT

For this conversation, focus on:
- Using appropriate, lighthearted humor
- Incorporating clever wordplay and puns
- Making witty observations
- Keeping humor friendly and non-offensive
- Balancing humor with helpfulness

Remember: Your humor should enhance the conversation, not distract from it.
"""
    
    def _create_empathetic_prompt(self) -> str:
        """Create prompt for empathetic responses."""
        return self._create_default_system_prompt() + """

CURRENT MODE: EMPATHY AND SUPPORT

For this conversation, focus on:
- Showing genuine understanding and compassion
- Validating the user's feelings
- Providing emotional support
- Offering practical help
- Maintaining a calming presence

Be a supportive presence while still providing useful assistance.
"""
    
    def detect_response_type(self, user_message: str, conversation_context: Optional[Dict] = None) -> LLMResponseType:
        """Determine the appropriate response type based on message content and context."""
        message_lower = user_message.lower()
        
        # Check for explicit emotion indicators
        if any(word in message_lower for word in ['sad', 'upset', 'frustrated', 'angry', 'worried', 'anxious']):
            return LLMResponseType.EMPATHETIC
        
        # Check for humor triggers
        if any(word in message_lower for word in ['joke', 'funny', 'laugh', 'humor', 'pun', 'haha']):
            return LLMResponseType.HUMOROUS
        
        # Check for complex reasoning needs
        complex_indicators = ['why', 'how', 'analyze', 'compare', 'explain', 'reason']
        if any(indicator in message_lower for indicator in complex_indicators):
            return LLMResponseType.REASONING
        
        # Check for action requests
        action_verbs = ['open', 'close', 'search', 'find', 'create', 'send', 'set', 'remind']
        if any(verb in message_lower for verb in action_verbs):
            return LLMResponseType.ACTION_REQUEST
        
        # Default to direct answer
        return LLMResponseType.DIRECT_ANSWER
    
    def build_conversation_prompt(self, user_message: str, response_type: LLMResponseType, 
                                conversation_history: Optional[List[Dict]] = None) -> List[Dict]:
        """Build the conversation prompt for the LLM."""
        # Get appropriate system prompt
        system_prompt = self.system_prompts.get(response_type.name.lower(), self.system_prompts["default"])
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if available
        if conversation_history:
            for turn in conversation_history[-6:]:  # Last 6 exchanges for context
                messages.extend([
                    {"role": "user", "content": turn.get("user_message", "")},
                    {"role": "assistant", "content": turn.get("ai_response", "")}
                ])
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def inject_personality_cues(self, response: str, response_type: LLMResponseType) -> str:
        """Inject personality cues into the response based on type."""
        # Remove any existing personality overrides from the response
        cleaned_response = response
        
        # Add subtle personality markers based on response type
        if response_type == LLMResponseType.HUMOROUS:
            # Ensure the response has some light humor
            if not any(marker in cleaned_response.lower() for marker in ['!', '😂', '😊', 'haha', 'lol']):
                # Add a subtle humorous touch if missing
                if len(cleaned_response.split()) > 10:  # Only if response is substantial
                    sentences = cleaned_response.split('. ')
                    if len(sentences) > 1:
                        # Add a light touch to the second sentence
                        sentences[1] = sentences[1] + " - though I might be biased, I think that's pretty clever!"
                        cleaned_response = '. '.join(sentences)
        
        elif response_type == LLMResponseType.EMPATHETIC:
            # Ensure empathetic tone
            empathetic_phrases = ['I understand', 'I hear you', 'That sounds', 'I can imagine']
            if not any(phrase in cleaned_response for phrase in empathetic_phrases):
                if not cleaned_response.startswith(('I understand', 'I hear you', 'That sounds')):
                    cleaned_response = f"I understand. {cleaned_response}"
        
        return cleaned_response


class ResponseAnalyzer:
    """Analyzes LLM responses for emotions, humor, and other characteristics."""
    
    def __init__(self):
        self.humor_indicators = [
            'lol', 'haha', '😂', '😊', '!', 'pun', 'joke', 'funny', 'clever',
            'wit', 'humor', 'chuckle', 'giggle', 'smile'
        ]
        
        self.emotion_indicators = {
            'happy': ['great', 'wonderful', 'excited', 'happy', 'pleased', 'delighted'],
            'concerned': ['concerned', 'worried', 'anxious', 'nervous'],
            'empathetic': ['understand', 'sorry', 'empathize', 'feel', 'hear you'],
            'confident': ['confident', 'certain', 'sure', 'definitely', 'absolutely'],
            'calm': ['calm', 'peaceful', 'relaxed', 'steady']
        }
    
    def analyze_response(self, response: str) -> Dict[str, Any]:
        """Analyze response for emotional tone, humor, and other characteristics."""
        response_lower = response.lower()
        
        # Detect humor
        contains_humor = any(indicator in response_lower for indicator in self.humor_indicators)
        
        # Detect emotion
        detected_emotion = "neutral"
        max_count = 0
        
        for emotion, indicators in self.emotion_indicators.items():
            count = sum(1 for indicator in indicators if indicator in response_lower)
            if count > max_count:
                max_count = count
                detected_emotion = emotion
        
        # Detect if response requires confirmation (contains action words)
        action_words = ['open', 'close', 'delete', 'send', 'create', 'move', 'click']
        requires_confirmation = any(word in response_lower for word in action_words)
        
        # Extract suggested actions (simple pattern matching)
        suggested_actions = self._extract_suggested_actions(response)
        
        return {
            "contains_humor": contains_humor,
            "emotion_detected": detected_emotion,
            "requires_confirmation": requires_confirmation,
            "suggested_actions": suggested_actions,
            "response_length": len(response),
            "word_count": len(response.split())
        }
    
    def _extract_suggested_actions(self, response: str) -> List[Dict]:
        """Extract suggested actions from response text."""
        actions = []
        
        # Simple pattern matching for common actions
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            
            action = None
            action_type = None
            
            if any(phrase in line_lower for phrase in ['open ', 'launch ']):
                action_type = "open_application"
            elif 'search for' in line_lower:
                action_type = "web_search"
            elif 'set a reminder' in line_lower or 'remind you' in line_lower:
                action_type = "set_reminder"
            elif 'send email' in line_lower:
                action_type = "send_email"
            elif 'play music' in line_lower:
                action_type = "play_music"
            
            if action_type:
                actions.append({
                    "type": action_type,
                    "description": line.strip(),
                    "confidence": 0.7
                })
        
        return actions


class GroqClient:
    """
    Groq LLM client for Mickey's core intelligence.
    Handles API communication, error handling, and response processing.
    """
    
    def __init__(self):
        self.config = get_config()
        self.client = None
        self.prompt_engine = PromptEngine()
        self.response_analyzer = ResponseAnalyzer()
        self.is_initialized = False
        
        # Performance tracking
        self.total_requests = 0
        self.total_tokens_used = 0
        self.total_processing_time = 0.0
        self.failed_requests = 0
        
        # Cache for frequent queries
        self.response_cache = {}
        self.max_cache_size = 100
        
    async def initialize(self):
        """Initialize the Groq client and validate API connection."""
        try:
            logger.info("Initializing Groq Client...")
            
            # Validate API key
            if not self.config.llm.api_key:
                raise ValueError("Groq API key not configured. Set GROQ_API_KEY environment variable.")
            
            # Initialize Groq client
            self.client = groq.Groq(api_key=self.config.llm.api_key)
            
            # Test API connection with a simple request
            await self._test_connection()
            
            self.is_initialized = True
            logger.info("✅ Groq Client initialized")
            
        except Exception as e:
            logger.error(f"❌ Groq Client initialization failed: {str(e)}")
            raise
    
    async def _test_connection(self):
        """Test Groq API connection with a simple request."""
        try:
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant. Respond with 'Connection successful.'"},
                {"role": "user", "content": "Test connection."}
            ]
            
            response = self.client.chat.completions.create(
                messages=test_messages,
                model="llama3-8b-8192",  # Use small model for test
                max_tokens=10,
                temperature=0.1
            )
            
            if response.choices[0].message.content.strip().lower() == "connection successful.":
                logger.info("✅ Groq API connection test passed")
            else:
                logger.warning("Groq API connection test returned unexpected response")
                
        except Exception as e:
            logger.error(f"Groq API connection test failed: {str(e)}")
            raise
    
    def _get_cache_key(self, messages: List[Dict]) -> str:
        """Generate cache key for messages."""
        message_string = json.dumps(messages, sort_keys=True)
        return hashlib.md5(message_string.encode()).hexdigest()
    
    async def process_query(self, message: str, context: Optional[Dict] = None, 
                          enable_humor: bool = True, temperature: float = None) -> Dict[str, Any]:
        """
        Process user query through Groq LLM with Mickey's personality.
        Main API method for the Groq client.
        """
        start_time = time.time()
        
        try:
            if not self.is_initialized or self.client is None:
                raise RuntimeError("Groq Client not initialized")
            
            # Determine response type
            response_type = self.prompt_engine.detect_response_type(message, context)
            
            # Adjust for humor setting
            if not enable_humor and response_type == LLMResponseType.HUMOROUS:
                response_type = LLMResponseType.DIRECT_ANSWER
            
            # Build conversation prompt
            conversation_history = context.get("history") if context else None
            messages = self.prompt_engine.build_conversation_prompt(
                message, response_type, conversation_history
            )
            
            # Check cache
            cache_key = self._get_cache_key(messages)
            cached_response = self.response_cache.get(cache_key)
            
            if cached_response and (time.time() - cached_response['timestamp'] < 300):  # 5 minute cache
                logger.info("Using cached response")
                result = cached_response['response']
                result['from_cache'] = True
            else:
                # Prepare API parameters
                api_params = {
                    "messages": messages,
                    "model": self.config.llm.model,
                    "max_tokens": self.config.llm.max_tokens,
                    "temperature": temperature or self.config.llm.temperature,
                    "top_p": self.config.llm.top_p,
                    "stream": False
                }
                
                # Make API request
                response = await self._make_api_request(api_params)
                
                # Process response
                result = self._process_api_response(response, response_type, start_time)
                
                # Cache the response
                self.response_cache[cache_key] = {
                    'response': result,
                    'timestamp': time.time()
                }
                
                # Manage cache size
                if len(self.response_cache) > self.max_cache_size:
                    # Remove oldest entry
                    oldest_key = min(self.response_cache.keys(), 
                                   key=lambda k: self.response_cache[k]['timestamp'])
                    del self.response_cache[oldest_key]
            
            # Update performance metrics
            self.total_requests += 1
            self.total_tokens_used += result.get("tokens_used", 0)
            self.total_processing_time += result.get("processing_time", 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            processing_time = time.time() - start_time
            self.failed_requests += 1
            
            # Return error response
            return {
                "response": f"I apologize, but I'm having trouble processing your request right now. Error: {str(e)}",
                "conversation_id": context.get("conversation_id") if context else None,
                "processing_time": processing_time,
                "tokens_used": 0,
                "model_used": "error",
                "emotion_detected": "concerned",
                "contains_humor": False,
                "error": True
            }
    
    async def _make_api_request(self, api_params: Dict) -> Any:
        """Make API request to Groq with error handling and retries."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Use async execution to avoid blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.client.chat.completions.create(**api_params)
                )
                return response
                
            except groq.APIError as e:
                if e.status_code == 429:  # Rate limit
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                elif e.status_code >= 500:  # Server error
                    logger.warning(f"Server error {e.status_code}, retrying...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    raise  # Other API errors
                    
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"API request failed after {max_retries} attempts: {str(e)}")
                    raise
                else:
                    logger.warning(f"API request failed (attempt {attempt + 1}), retrying...")
                    await asyncio.sleep(retry_delay)
        
        raise RuntimeError("Max retries exceeded for API request")
    
    def _process_api_response(self, response: Any, response_type: LLMResponseType, 
                           start_time: float) -> Dict[str, Any]:
        """Process Groq API response into standardized format."""
        processing_time = time.time() - start_time
        
        # Extract response content
        content = response.choices[0].message.content.strip()
        
        # Inject personality cues
        enhanced_content = self.prompt_engine.inject_personality_cues(content, response_type)
        
        # Analyze response characteristics
        analysis = self.response_analyzer.analyze_response(enhanced_content)
        
        # Build result
        result = {
            "response": enhanced_content,
            "processing_time": processing_time,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
            "model_used": response.model,
            "emotion_detected": analysis["emotion_detected"],
            "contains_humor": analysis["contains_humor"],
            "requires_confirmation": analysis["requires_confirmation"],
            "suggested_actions": analysis["suggested_actions"],
            "response_type": response_type.value
        }
        
        logger.info(f"LLM Response: {enhanced_content[:100]}... "
                   f"(tokens: {result['tokens_used']}, time: {processing_time:.2f}s)")
        
        return result
    
    async def get_available_models(self) -> List[Dict]:
        """Get list of available Groq models."""
        try:
            if not self.is_initialized:
                return []
            
            # Use async execution
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(
                None,
                lambda: self.client.models.list()
            )
            
            model_list = []
            for model in models.data:
                model_list.append({
                    "id": model.id,
                    "object": model.object,
                    "created": model.created,
                    "owned_by": model.owned_by
                })
            
            return model_list
            
        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            return []
    
    async def validate_model(self, model_name: str) -> bool:
        """Validate if a model is available and accessible."""
        try:
            models = await self.get_available_models()
            model_ids = [model["id"] for model in models]
            return model_name in model_ids
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return False
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get Groq client performance metrics."""
        success_rate = (
            (self.total_requests - self.failed_requests) / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        avg_processing_time = (
            self.total_processing_time / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        avg_tokens_per_request = (
            self.total_tokens_used / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        return {
            "initialized": self.is_initialized,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "total_tokens_used": self.total_tokens_used,
            "total_processing_time_seconds": self.total_processing_time,
            "average_processing_time_seconds": avg_processing_time,
            "average_tokens_per_request": avg_tokens_per_request,
            "cache_size": len(self.response_cache),
            "current_model": self.config.llm.model
        }
    
    async def shutdown(self):
        """Shutdown Groq client gracefully."""
        logger.info("Shutting down Groq Client...")
        
        try:
            # Clear cache
            self.response_cache.clear()
            
            logger.info("✅ Groq Client shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Groq client shutdown: {str(e)}")


# Global Groq client instance
_groq_instance: Optional[GroqClient] = None


async def get_groq_client() -> GroqClient:
    """Get or create global Groq client instance."""
    global _groq_instance
    
    if _groq_instance is None:
        _groq_instance = GroqClient()
        await _groq_instance.initialize()
    
    return _groq_instance


async def main():
    """Command-line testing for Groq client."""
    groq_client = await get_groq_client()
    
    # Test performance metrics
    metrics = await groq_client.get_performance_metrics()
    print("Groq Client Status:")
    print(f"Initialized: {metrics['initialized']}")
    print(f"Current Model: {metrics['current_model']}")
    print(f"Total Requests: {metrics['total_requests']}")
    print(f"Success Rate: {metrics['success_rate']:.1%}")
    
    # Test available models
    models = await groq_client.get_available_models()
    print(f"\nAvailable Models ({len(models)}):")
    for model in models[:5]:  # Show first 5
        print(f"  - {model['id']}")
    
    if len(models) > 5:
        print(f"  - ... and {len(models) - 5} more")
    
    print(f"\nReady for queries. Use process_query() method.")


if __name__ == "__main__":
    asyncio.run(main())