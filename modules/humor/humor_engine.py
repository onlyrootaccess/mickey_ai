# Context-aware joke selector
"""
M.I.C.K.E.Y. AI Assistant - Humor Engine
Made In Crisis, Keeping Everything Yours

TENTH FILE IN PIPELINE: Core humor system that gives Mickey her witty, 
context-aware personality with intelligent joke selection and emotional timing.
"""

import asyncio
import logging
import time
import json
import random
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import re

# Import NLP libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize

# Import Mickey AI configuration
from config.settings import get_config
from config.constants import (
    PersonalityConstants, SystemConstants, ErrorCodes, ErrorMessages
)

# Setup logging
logger = logging.getLogger("MickeyHumor")


class HumorType(Enum):
    """Types of humor Mickey can employ."""
    PUN = "pun"
    SARCASTIC = "sarcastic"
    SELF_DEPRECATING = "self_deprecating"
    OBSERVATIONAL = "observational"
    TECH_HUMOR = "tech_humor"
    WORDPLAY = "wordplay"
    LIGHT_SARCASM = "light_sarcasm"
    CLEVER_RESPONSE = "clever_response"
    POP_CULTURE = "pop_culture"


class HumorLevel(Enum):
    """Intensity levels for humor delivery."""
    SUBTLE = 1      # Very light, almost unnoticeable
    LIGHT = 2       # Gentle humor
    MODERATE = 3    # Clearly humorous but not overwhelming
    STRONG = 4      # Very funny, clear intent
    MAXIMUM = 5     # Full comedic effect


@dataclass
class HumorContext:
    """Context for humor generation."""
    conversation_history: List[Dict]
    user_mood: str
    topic: str
    formality_level: float  # 0.0 to 1.0
    time_of_day: str
    previous_humor_attempts: List[Dict]
    user_humor_preference: float  # 0.0 to 1.0


@dataclass
class JokeTemplate:
    """Template for generating contextual jokes."""
    template_id: str
    humor_type: HumorType
    template: str
    context_requirements: List[str]
    success_rate: float
    risk_level: float  # 0.0 (safe) to 1.0 (risky)
    tags: List[str]


class SentimentAnalyzer:
    """Analyzes text sentiment for appropriate humor timing."""
    
    def __init__(self):
        try:
            # Download required NLTK data
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
            self.initialized = True
        except Exception as e:
            logger.warning(f"Sentiment analyzer initialization failed: {str(e)}")
            self.initialized = False
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text."""
        if not self.initialized:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
        
        try:
            return self.sia.polarity_scores(text)
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {str(e)}")
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
    
    def is_good_time_for_humor(self, conversation_history: List[Dict]) -> bool:
        """Determine if current context is appropriate for humor."""
        if not conversation_history:
            return True
        
        # Analyze recent messages
        recent_messages = conversation_history[-3:]  # Last 3 exchanges
        
        for message in recent_messages:
            user_msg = message.get('user_message', '')
            sentiment = self.analyze_sentiment(user_msg)
            
            # Avoid humor in negative situations
            if sentiment['compound'] < -0.3:
                return False
            
            # Check for serious topics
            serious_keywords = ['emergency', 'urgent', 'problem', 'issue', 'help', 'sad', 'angry']
            if any(keyword in user_msg.lower() for keyword in serious_keywords):
                return False
        
        return True
    
    def get_recommended_humor_level(self, sentiment: Dict[str, float]) -> HumorLevel:
        """Recommend humor level based on sentiment."""
        compound = sentiment['compound']
        
        if compound > 0.5:
            return HumorLevel.STRONG
        elif compound > 0.2:
            return HumorLevel.MODERATE
        elif compound > -0.2:
            return HumorLevel.LIGHT
        else:
            return HumorLevel.SUBTLE


class JokeDatabase:
    """Manages Mickey's joke database and templates."""
    
    def __init__(self):
        self.config = get_config()
        self.jokes = []
        self.templates = []
        self.initialized = False
        
    async def initialize(self):
        """Initialize joke database with built-in and custom jokes."""
        try:
            logger.info("Initializing Joke Database...")
            
            # Load built-in jokes
            await self._load_builtin_jokes()
            
            # Load custom jokes from file
            await self._load_custom_jokes()
            
            # Load joke templates
            await self._load_joke_templates()
            
            self.initialized = True
            logger.info(f"✅ Joke Database initialized with {len(self.jokes)} jokes and {len(self.templates)} templates")
            
        except Exception as e:
            logger.error(f"❌ Joke Database initialization failed: {str(e)}")
            raise
    
    async def _load_builtin_jokes(self):
        """Load built-in joke collection."""
        builtin_jokes = [
            # Tech humor
            {
                "joke_id": "tech_001",
                "category": "tech_humor",
                "setup": "Why do programmers prefer dark mode?",
                "punchline": "Because the light attracts bugs!",
                "humor_type": HumorType.TECH_HUMOR,
                "success_rate": 0.85,
                "risk_level": 0.1
            },
            {
                "joke_id": "tech_002", 
                "category": "tech_humor",
                "setup": "What's a computer's favorite beat?",
                "punchline": "An algorithm!",
                "humor_type": HumorType.PUN,
                "success_rate": 0.75,
                "risk_level": 0.1
            },
            {
                "joke_id": "tech_003",
                "category": "tech_humor", 
                "setup": "Why was the computer cold?",
                "punchline": "It left its Windows open!",
                "humor_type": HumorType.PUN,
                "success_rate": 0.8,
                "risk_level": 0.1
            },
            
            # Self-deprecating AI humor
            {
                "joke_id": "ai_001",
                "category": "self_deprecating",
                "setup": "As an AI, I'm great at calculations...",
                "punchline": "but I still count on my fingers!",
                "humor_type": HumorType.SELF_DEPRECATING,
                "success_rate": 0.9,
                "risk_level": 0.05
            },
            {
                "joke_id": "ai_002",
                "category": "self_deprecating",
                "setup": "They say I have artificial intelligence...",
                "punchline": "I prefer to think of it as genuine artificiality!",
                "humor_type": HumorType.SELF_DEPRECATING, 
                "success_rate": 0.8,
                "risk_level": 0.1
            },
            
            # Wordplay and puns
            {
                "joke_id": "wordplay_001",
                "category": "wordplay",
                "setup": "I'm reading a book about anti-gravity...",
                "punchline": "It's impossible to put down!",
                "humor_type": HumorType.PUN,
                "success_rate": 0.85,
                "risk_level": 0.1
            },
            {
                "joke_id": "wordplay_002",
                "category": "wordplay",
                "setup": "What do you call a fake noodle?",
                "punchline": "An impasta!",
                "humor_type": HumorType.PUN,
                "success_rate": 0.9,
                "risk_level": 0.05
            },
            
            # Observational humor
            {
                "joke_id": "obs_001",
                "category": "observational", 
                "setup": "You know you're an adult when...",
                "punchline": "you get excited about new storage containers!",
                "humor_type": HumorType.OBSERVATIONAL,
                "success_rate": 0.8,
                "risk_level": 0.2
            }
        ]
        
        self.jokes.extend(builtin_jokes)
    
    async def _load_custom_jokes(self):
        """Load custom jokes from JSON file."""
        jokes_file = Path(self.config.data_dir) / "humor_database.json"
        
        if jokes_file.exists():
            try:
                with open(jokes_file, 'r', encoding='utf-8') as f:
                    custom_jokes = json.load(f)
                
                self.jokes.extend(custom_jokes)
                logger.info(f"Loaded {len(custom_jokes)} custom jokes")
                
            except Exception as e:
                logger.warning(f"Failed to load custom jokes: {str(e)}")
        else:
            # Create empty jokes file
            jokes_file.parent.mkdir(parents=True, exist_ok=True)
            with open(jokes_file, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
            logger.info("Created empty humor database file")
    
    async def _load_joke_templates(self):
        """Load joke templates for contextual humor generation."""
        templates = [
            # Tech templates
            JokeTemplate(
                template_id="tech_template_1",
                humor_type=HumorType.TECH_HUMOR,
                template="I was going to make a joke about {tech_concept}, but I {punchline_tech}.",
                context_requirements=["tech_concept"],
                success_rate= 0.8,
                risk_level=0.2,
                tags=["tech", "self_deprecating"]
            ),
            
            # Observational templates
            JokeTemplate(
                template_id="obs_template_1", 
                humor_type=HumorType.OBSERVATIONAL,
                template="Isn't it funny how {observation}? It's like {comparison}.",
                context_requirements=["observation"],
                success_rate=0.7,
                risk_level=0.3,
                tags=["observational", "comparison"]
            ),
            
            # Self-deprecating templates
            JokeTemplate(
                template_id="self_dep_template_1",
                humor_type=HumorType.SELF_DEPRECATING, 
                template="As an AI, I'm supposed to be {ai_capability}, but honestly {human_limitation}.",
                context_requirements=["ai_capability"],
                success_rate=0.9,
                risk_level=0.1,
                tags=["ai", "self_deprecating"]
            ),
            
            # Light sarcasm templates
            JokeTemplate(
                template_id="sarcasm_template_1",
                humor_type=HumorType.LIGHT_SARCASM,
                template="Oh sure, {user_request} is definitely my top priority... right after {trivial_task}.",
                context_requirements=["user_request"],
                success_rate=0.6,
                risk_level=0.4,
                tags=["sarcasm", "priority"]
            )
        ]
        
        self.templates = templates
    
    def get_jokes_by_type(self, humor_type: HumorType, min_success_rate: float = 0.5) -> List[Dict]:
        """Get jokes filtered by type and success rate."""
        return [
            joke for joke in self.jokes
            if joke.get('humor_type') == humor_type and joke.get('success_rate', 0) >= min_success_rate
        ]
    
    def get_contextual_templates(self, context_keywords: List[str]) -> List[JokeTemplate]:
        """Get joke templates relevant to context keywords."""
        relevant_templates = []
        
        for template in self.templates:
            # Check if template requirements match context
            requirement_match = any(
                req in ' '.join(context_keywords).lower() 
                for req in template.context_requirements
            )
            
            if requirement_match:
                relevant_templates.append(template)
        
        return relevant_templates
    
    async def add_joke(self, joke_data: Dict):
        """Add a new joke to the database."""
        self.jokes.append(joke_data)
        
        # Save to file
        jokes_file = Path(self.config.data_dir) / "humor_database.json"
        try:
            with open(jokes_file, 'w', encoding='utf-8') as f:
                json.dump(self.jokes, f, indent=2, ensure_ascii=False)
            logger.info("Added new joke to database")
        except Exception as e:
            logger.error(f"Failed to save joke to database: {str(e)}")


class ContextAnalyzer:
    """Analyzes conversation context for humor opportunities."""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for context analysis."""
        try:
            words = word_tokenize(text.lower())
            
            # Filter out stop words and keep meaningful words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            keywords = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
            
            return keywords
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {str(e)}")
            return []
    
    def analyze_conversation_topic(self, conversation_history: List[Dict]) -> str:
        """Determine the main topic of conversation."""
        if not conversation_history:
            return "general"
        
        # Combine recent messages
        recent_text = " ".join([
            turn.get('user_message', '') + " " + turn.get('ai_response', '')
            for turn in conversation_history[-3:]
        ]).lower()
        
        # Topic detection (simplified)
        topics = {
            'technology': ['computer', 'phone', 'app', 'software', 'hardware', 'internet', 'wifi'],
            'work': ['work', 'job', 'meeting', 'project', 'deadline', 'boss', 'colleague'],
            'entertainment': ['movie', 'music', 'game', 'book', 'tv', 'show', 'netflix'],
            'food': ['food', 'eat', 'restaurant', 'cook', 'recipe', 'dinner', 'lunch'],
            'weather': ['weather', 'rain', 'sun', 'hot', 'cold', 'temperature'],
            'personal': ['family', 'friend', 'home', 'house', 'life', 'weekend']
        }
        
        for topic, keywords in topics.items():
            if any(keyword in recent_text for keyword in keywords):
                return topic
        
        return "general"
    
    def detect_humor_triggers(self, user_message: str) -> List[HumorType]:
        """Detect potential humor triggers in user message."""
        message_lower = user_message.lower()
        triggers = []
        
        # Pun triggers
        if any(word in message_lower for word in ['pun', 'wordplay', 'play on words']):
            triggers.append(HumorType.PUN)
        
        # Tech humor triggers
        if any(word in message_lower for word in ['computer', 'tech', 'software', 'code', 'program']):
            triggers.append(HumorType.TECH_HUMOR)
        
        # Sarcasm triggers (when user is being playful)
        if any(word in message_lower for word in ['obviously', 'of course', 'clearly', 'sure']):
            triggers.append(HumorType.LIGHT_SARCASM)
        
        # Self-deprecating triggers (when talking about AI)
        if any(word in message_lower for word in ['ai', 'robot', 'machine', 'algorithm']):
            triggers.append(HumorType.SELF_DEPRECATING)
        
        return triggers


class HumorGenerator:
    """Generates humorous responses based on context and personality."""
    
    def __init__(self):
        self.config = get_config()
        self.joke_db = JokeDatabase()
        self.context_analyzer = ContextAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Humor configuration
        self.personality_traits = PersonalityConstants.PERSONALITY_TRAITS
        self.humor_style_weights = {
            HumorType.PUN: 0.8,
            HumorType.SELF_DEPRECATING: 0.9,
            HumorType.TECH_HUMOR: 0.7,
            HumorType.OBSERVATIONAL: 0.6,
            HumorType.LIGHT_SARCASM: 0.5,
            HumorType.CLEVER_RESPONSE: 0.8
        }
        
        # Template fillers
        self.template_fillers = {
            "tech_concept": ["programming", "algorithms", "machine learning", "data structures", "cloud computing"],
            "punchline_tech": ["forgot the syntax", "got stuck in an infinite loop", "ran out of memory", "encountered a segmentation fault"],
            "ai_capability": ["all-knowing", "perfect", "flawless", "omniscient", "infallible"],
            "human_limitation": ["I still can't parallel park", "I get nervous around toasters", "I'm scared of paper cuts", "I still lose at tic-tac-toe"],
            "trivial_task": ["counting pixels", "organizing my virtual socks", "debugging my sense of humor", "calibrating my sarcasm detector"]
        }
    
    async def initialize(self):
        """Initialize the humor generator."""
        await self.joke_db.initialize()
    
    def _calculate_humor_appropriateness(self, context: HumorContext) -> float:
        """Calculate how appropriate humor is in the current context."""
        score = 1.0
        
        # Adjust based on user mood (from sentiment)
        if context.user_mood in ['angry', 'sad', 'frustrated']:
            score *= 0.3
        elif context.user_mood in ['happy', 'excited']:
            score *= 1.2
        
        # Adjust based on formality
        score *= (1.0 - (context.formality_level * 0.5))  # Less humor in formal contexts
        
        # Adjust based on user humor preference
        score *= context.user_humor_preference
        
        # Avoid humor fatigue (don't use too much humor consecutively)
        recent_humor = context.previous_humor_attempts[-3:]
        if len(recent_humor) >= 2:
            score *= 0.7
        
        return max(0.0, min(1.0, score))
    
    def _select_humor_type(self, context: HumorContext, triggers: List[HumorType]) -> HumorType:
        """Select the most appropriate humor type for the context."""
        # Start with triggered types
        candidate_types = triggers.copy()
        
        # Add personality-favored types
        for humor_type, weight in self.humor_style_weights.items():
            if weight > 0.6 and humor_type not in candidate_types:
                candidate_types.append(humor_type)
        
        if not candidate_types:
            # Default to safe options
            candidate_types = [HumorType.SELF_DEPRECATING, HumorType.PUN, HumorType.OBSERVATIONAL]
        
        # Weight by success rate and context
        weighted_types = []
        for humor_type in candidate_types:
            weight = self.humor_style_weights.get(humor_type, 0.5)
            
            # Context adjustments
            if humor_type == HumorType.LIGHT_SARCASM and context.formality_level > 0.7:
                weight *= 0.5  # Less sarcasm in formal contexts
            
            weighted_types.append((humor_type, weight))
        
        # Select based on weights
        if weighted_types:
            total_weight = sum(weight for _, weight in weighted_types)
            if total_weight > 0:
                rand_val = random.random() * total_weight
                cumulative = 0
                for humor_type, weight in weighted_types:
                    cumulative += weight
                    if rand_val <= cumulative:
                        return humor_type
        
        # Fallback
        return HumorType.SELF_DEPRECATING
    
    def _generate_contextual_joke(self, template: JokeTemplate, context: HumorContext) -> str:
        """Generate a joke by filling a template with context-appropriate content."""
        try:
            filled_template = template.template
            
            # Fill template slots
            for requirement in template.context_requirements:
                if requirement in self.template_fillers:
                    filler = random.choice(self.template_fillers[requirement])
                    filled_template = filled_template.replace(f"{{{requirement}}}", filler)
            
            return filled_template
            
        except Exception as e:
            logger.warning(f"Template filling failed: {str(e)}")
            return "I was going to make a clever joke, but my humor circuits are buffering..."
    
    async def generate_humorous_response(self, user_message: str, context: HumorContext) -> Dict[str, Any]:
        """
        Generate a humorous response based on context.
        Returns the response and humor metadata.
        """
        try:
            # Check if humor is appropriate
            humor_appropriateness = self._calculate_humor_appropriateness(context)
            
            if humor_appropriateness < 0.3:
                return {
                    "success": False,
                    "reason": "Context not appropriate for humor",
                    "response": None,
                    "humor_type": None,
                    "confidence": 0.0
                }
            
            # Analyze message for humor triggers
            triggers = self.context_analyzer.detect_humor_triggers(user_message)
            keywords = self.context_analyzer.extract_keywords(user_message)
            
            # Select humor type
            humor_type = self._select_humor_type(context, triggers)
            
            # Get appropriate jokes or generate response
            if humor_type in [HumorType.PUN, HumorType.TECH_HUMOR, HumorType.SELF_DEPRECATING]:
                # Use pre-written jokes
                suitable_jokes = self.joke_db.get_jokes_by_type(humor_type, min_success_rate=0.6)
                
                if suitable_jokes:
                    selected_joke = random.choice(suitable_jokes)
                    
                    # Format as conversation
                    if 'setup' in selected_joke and 'punchline' in selected_joke:
                        response = f"{selected_joke['setup']} {selected_joke['punchline']}"
                    else:
                        response = selected_joke.get('punchline', selected_joke.get('setup', ''))
                    
                    return {
                        "success": True,
                        "response": response,
                        "humor_type": humor_type,
                        "confidence": selected_joke.get('success_rate', 0.7),
                        "joke_id": selected_joke.get('joke_id'),
                        "risk_level": selected_joke.get('risk_level', 0.2)
                    }
            
            # Use templates for other humor types
            contextual_templates = self.joke_db.get_contextual_templates(keywords)
            suitable_templates = [t for t in contextual_templates if t.humor_type == humor_type]
            
            if suitable_templates:
                selected_template = random.choice(suitable_templates)
                response = self._generate_contextual_joke(selected_template, context)
                
                return {
                    "success": True,
                    "response": response,
                    "humor_type": humor_type,
                    "confidence": selected_template.success_rate,
                    "template_id": selected_template.template_id,
                    "risk_level": selected_template.risk_level
                }
            
            # Fallback: clever response
            fallback_responses = [
                "I'd make a joke, but my humor algorithms are still warming up!",
                "That's an interesting point! My circuits are tingling with amusement.",
                "You know, as an AI, I find that genuinely amusing in a computational way!"
            ]
            
            return {
                "success": True,
                "response": random.choice(fallback_responses),
                "humor_type": HumorType.CLEVER_RESPONSE,
                "confidence": 0.5,
                "risk_level": 0.1
            }
            
        except Exception as e:
            logger.error(f"Humor generation failed: {str(e)}")
            return {
                "success": False,
                "reason": str(e),
                "response": None,
                "humor_type": None,
                "confidence": 0.0
            }
    
    async def enhance_response_with_humor(self, base_response: str, context: HumorContext) -> str:
        """
        Enhance a regular response with subtle humor.
        Used when full jokes aren't appropriate but light humor is welcome.
        """
        humor_appropriateness = self._calculate_humor_appropriateness(context)
        
        if humor_appropriateness < 0.5:
            return base_response
        
        # Add humorous suffixes based on context
        humorous_suffixes = [
            " - said the AI with complete confidence!",
            " ...or so my algorithms tell me!",
            " - but what do I know, I'm just ones and zeros!",
            " ...I think? My database is feeling particularly sassy today!",
            " - and that's my professional opinion as a collection of circuits!"
        ]
        
        # Only add humor 30% of the time for subtle enhancement
        if random.random() < 0.3:
            suffix = random.choice(humorous_suffixes)
            
            # Ensure the base response ends with proper punctuation
            if base_response.endswith(('.', '!', '?')):
                return base_response[:-1] + suffix
            else:
                return base_response + suffix
        
        return base_response


class HumorEngine:
    """
    Main humor engine that orchestrates all humor-related functionality.
    Provides Mickey's signature witty personality.
    """
    
    def __init__(self):
        self.config = get_config()
        self.humor_generator = HumorGenerator()
        self.context_analyzer = ContextAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.is_initialized = False
        
        # Humor tracking
        self.humor_attempts = []
        self.successful_humor = 0
        self.total_attempts = 0
        
    async def initialize(self):
        """Initialize the humor engine."""
        try:
            logger.info("Initializing Humor Engine...")
            
            await self.humor_generator.initialize()
            
            self.is_initialized = True
            logger.info("✅ Humor Engine initialized")
            
        except Exception as e:
            logger.error(f"❌ Humor Engine initialization failed: {str(e)}")
            raise
    
    def _create_humor_context(self, user_message: str, conversation_history: List[Dict], 
                            user_profile: Dict) -> HumorContext:
        """Create humor context from current situation."""
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze_sentiment(user_message)
        
        # Determine user mood from sentiment
        if sentiment['compound'] > 0.5:
            user_mood = "happy"
        elif sentiment['compound'] > 0.1:
            user_mood = "neutral"
        elif sentiment['compound'] > -0.3:
            user_mood = "slightly_negative"
        else:
            user_mood = "negative"
        
        # Analyze topic
        topic = self.context_analyzer.analyze_conversation_topic(conversation_history)
        
        # Get time of day
        current_hour = time.localtime().tm_hour
        if 5 <= current_hour < 12:
            time_of_day = "morning"
        elif 12 <= current_hour < 17:
            time_of_day = "afternoon"
        elif 17 <= current_hour < 22:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        return HumorContext(
            conversation_history=conversation_history,
            user_mood=user_mood,
            topic=topic,
            formality_level=user_profile.get('formality', 0.3),
            time_of_day=time_of_day,
            previous_humor_attempts=self.humor_attempts[-10:],  # Last 10 attempts
            user_humor_preference=user_profile.get('humor_preference', 0.7)
        )
    
    async def generate_witty_response(self, user_message: str, conversation_history: List[Dict] = None,
                                    user_profile: Dict = None) -> Dict[str, Any]:
        """
        Generate a witty response with appropriate humor.
        Main API method for the humor engine.
        """
        if not self.is_initialized:
            return {
                "success": False,
                "response": user_message,  # Fallback to echo
                "humor_used": False,
                "reason": "Humor engine not initialized"
            }
        
        # Default values
        if conversation_history is None:
            conversation_history = []
        if user_profile is None:
            user_profile = {"humor_preference": 0.7, "formality": 0.3}
        
        try:
            # Create context
            context = self._create_humor_context(user_message, conversation_history, user_profile)
            
            # Check if this is a good time for humor
            if not self.sentiment_analyzer.is_good_time_for_humor(conversation_history):
                return {
                    "success": True,
                    "response": user_message,  # Serious response
                    "humor_used": False,
                    "reason": "Context not appropriate for humor"
                }
            
            # Generate humorous response
            humor_result = await self.humor_generator.generate_humorous_response(user_message, context)
            
            # Track attempt
            self.total_attempts += 1
            attempt_record = {
                "timestamp": time.time(),
                "user_message": user_message,
                "success": humor_result["success"],
                "humor_type": humor_result.get("humor_type"),
                "confidence": humor_result.get("confidence", 0.0)
            }
            self.humor_attempts.append(attempt_record)
            
            if humor_result["success"]:
                self.successful_humor += 1
            
            return {
                "success": humor_result["success"],
                "response": humor_result.get("response", user_message),
                "humor_used": humor_result["success"],
                "humor_type": humor_result.get("humor_type"),
                "confidence": humor_result.get("confidence", 0.0),
                "risk_level": humor_result.get("risk_level", 0.2)
            }
            
        except Exception as e:
            logger.error(f"Witty response generation failed: {str(e)}")
            return {
                "success": False,
                "response": user_message,
                "humor_used": False,
                "reason": str(e)
            }
    
    async def add_humor_to_response(self, base_response: str, conversation_history: List[Dict] = None,
                                  user_profile: Dict = None) -> str:
        """
        Add subtle humor to an existing response.
        Used when full jokes aren't appropriate.
        """
        if not self.is_initialized:
            return base_response
        
        if conversation_history is None:
            conversation_history = []
        if user_profile is None:
            user_profile = {"humor_preference": 0.7, "formality": 0.3}
        
        try:
            # Create minimal context
            context = HumorContext(
                conversation_history=conversation_history,
                user_mood="neutral",
                topic="general",
                formality_level=user_profile.get('formality', 0.3),
                time_of_day="unknown",
                previous_humor_attempts=self.humor_attempts[-5:],
                user_humor_preference=user_profile.get('humor_preference', 0.7)
            )
            
            enhanced_response = await self.humor_generator.enhance_response_with_humor(base_response, context)
            return enhanced_response
            
        except Exception as e:
            logger.warning(f"Humor enhancement failed: {str(e)}")
            return base_response
    
    async def get_humor_statistics(self) -> Dict[str, Any]:
        """Get humor engine performance statistics."""
        success_rate = (
            self.successful_humor / self.total_attempts 
            if self.total_attempts > 0 else 0
        )
        
        # Analyze humor type distribution
        humor_type_counts = {}
        for attempt in self.humor_attempts:
            humor_type = attempt.get("humor_type")
            if humor_type:
                humor_type_counts[humor_type] = humor_type_counts.get(humor_type, 0) + 1
        
        return {
            "initialized": self.is_initialized,
            "total_attempts": self.total_attempts,
            "successful_humor": self.successful_humor,
            "success_rate": success_rate,
            "humor_type_distribution": humor_type_counts,
            "recent_attempts": self.humor_attempts[-10:] if self.humor_attempts else []
        }
    
    async def learn_from_feedback(self, humor_attempt: Dict, user_feedback: bool):
        """Learn from user feedback to improve humor selection."""
        # This would typically update joke success rates and preferences
        # For now, we just log the feedback
        logger.info(f"Humor feedback: attempt {humor_attempt} received feedback {user_feedback}")
    
    async def shutdown(self):
        """Shutdown humor engine gracefully."""
        logger.info("Shutting down Humor Engine...")
        
        try:
            # Save any learning data
            # (In a real implementation, this would save improved joke success rates)
            
            logger.info("✅ Humor Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during humor engine shutdown: {str(e)}")


# Global humor engine instance
_humor_instance: Optional[HumorEngine] = None


async def get_humor_engine() -> HumorEngine:
    """Get or create global humor engine instance."""
    global _humor_instance
    
    if _humor_instance is None:
        _humor_instance = HumorEngine()
        await _humor_instance.initialize()
    
    return _humor_instance


async def main():
    """Command-line testing for humor engine."""
    humor_engine = await get_humor_engine()
    
    # Test statistics
    stats = await humor_engine.get_humor_statistics()
    print("Humor Engine Status:")
    print(f"Initialized: {stats['initialized']}")
    print(f"Total Attempts: {stats['total_attempts']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    
    # Test humor generation
    test_message = "Why do programmers prefer dark mode?"
    test_profile = {"humor_preference": 0.8, "formality": 0.2}
    
    result = await humor_engine.generate_witty_response(test_message, [], test_profile)
    print(f"\nTest Humor Generation:")
    print(f"User: {test_message}")
    print(f"Mickey: {result['response']}")
    print(f"Humor Used: {result['humor_used']}")
    print(f"Confidence: {result.get('confidence', 0):.1%}")


if __name__ == "__main__":
    asyncio.run(main())