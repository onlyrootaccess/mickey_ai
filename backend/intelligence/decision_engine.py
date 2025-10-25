"""
Mickey AI - Decision Engine
Makes intelligent decisions based on context, user intent, and available modules
"""

import logging
import re
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime

class DecisionConfidence(Enum):
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"
    UNCERTAIN = "uncertain"

class UserIntent(Enum):
    CHAT = "chat"
    SEARCH = "search"
    CONTROL = "control"
    MEDIA = "media"
    SYSTEM = "system"
    BROWSER = "browser"
    CREATIVE = "creative"
    HUMOR = "humor"
    UNKNOWN = "unknown"

class DecisionResult:
    def __init__(self, intent: UserIntent, confidence: DecisionConfidence, 
                 action: str, parameters: Dict[str, Any], explanation: str = ""):
        self.intent = intent
        self.confidence = confidence
        self.action = action
        self.parameters = parameters
        self.explanation = explanation
        self.timestamp = datetime.now()

class DecisionEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Intent patterns with regex
        self.intent_patterns = {
            UserIntent.SEARCH: [
                r'(search|find|look up|google).*for (.+)',
                r'what is (.+)',
                r'who is (.+)',
                r'how to (.+)',
                r'where is (.+)',
                r'when is (.+)',
                r'why is (.+)'
            ],
            UserIntent.CONTROL: [
                r'(click|move|scroll|type|press).*(mouse|keyboard|screen)',
                r'(open|close|start|launch).*(app|application|program)',
                r'(minimize|maximize|resize).*window',
                r'shutdown|restart|sleep|lock.*computer'
            ],
            UserIntent.MEDIA: [
                r'(play|pause|stop|resume).*(music|video|song|movie)',
                r'(volume up|volume down|mute|unmute)',
                r'(next|previous).*(track|song|video)',
                r'(what.s playing|current track)'
            ],
            UserIntent.BROWSER: [
                r'(open|go to|visit).*(website|url|http)',
                r'(browse|navigate).*internet',
                r'(new tab|close tab)',
                r'(bookmark|save).*page'
            ],
            UserIntent.SYSTEM: [
                r'system.*status',
                r'health.*check',
                r'memory.*usage',
                r'cpu.*usage',
                r'disk.*space'
            ],
            UserIntent.CREATIVE: [
                r'(write|compose|create).*(story|poem|joke|song)',
                r'(imagine|think).*about',
                r'(what if|suppose)',
                r'(brainstorm|idea).*for'
            ],
            UserIntent.HUMOR: [
                r'(joke|funny|humor|laugh)',
                r'(make me smile|cheer me up)',
                r'(roast|tease).*me',
                r'(mickey).*(joke|funny)'
            ]
        }
        
        # Action mappings
        self.action_mappings = {
            UserIntent.SEARCH: {
                'web_search': ['search', 'find', 'look up', 'what is', 'who is'],
                'news_search': ['news', 'latest', 'current events'],
                'image_search': ['images', 'pictures', 'photos']
            },
            UserIntent.CONTROL: {
                'mouse_move': ['move mouse', 'click', 'scroll'],
                'type_text': ['type', 'write', 'enter text'],
                'system_command': ['shutdown', 'restart', 'lock']
            },
            UserIntent.MEDIA: {
                'play_media': ['play', 'start music', 'resume'],
                'pause_media': ['pause', 'stop'],
                'volume_control': ['volume', 'mute', 'unmute']
            },
            UserIntent.BROWSER: {
                'open_url': ['open website', 'visit', 'go to'],
                'browser_search': ['search for', 'look up'],
                'browser_control': ['new tab', 'close tab', 'refresh']
            },
            UserIntent.SYSTEM: {
                'system_status': ['status', 'health', 'usage'],
                'resource_info': ['memory', 'cpu', 'disk']
            }
        }
        
        # Context awareness
        self.conversation_context = []
        self.user_preferences = {}
        
        # Mickey's personality traits
        self.personality_traits = {
            'humor_level': 0.7,  # 0-1 scale
            'creativity_level': 0.8,
            'helpfulness_level': 0.9,
            'curiosity_level': 0.6
        }
        
        self.logger.info("🎯 Decision Engine initialized - Ready to make smart choices!")

    def analyze_intent(self, user_input: str, context: List[Dict] = None) -> DecisionResult:
        """
        Analyze user input to determine intent and appropriate action
        
        Args:
            user_input: Raw user input text
            context: Conversation history for context awareness
            
        Returns:
            DecisionResult with intent, confidence, and action details
        """
        try:
            # Update context
            if context:
                self.conversation_context = context[-5:]  # Keep last 5 messages
            
            # Preprocess input
            processed_input = self._preprocess_input(user_input)
            
            # Detect intent
            intent, confidence, extracted_params = self._detect_intent(processed_input)
            
            # Determine action based on intent
            action, action_params = self._determine_action(intent, processed_input, extracted_params)
            
            # Generate explanation
            explanation = self._generate_explanation(intent, confidence, action)
            
            # Apply personality adjustments
            if random.random() < self.personality_traits['humor_level'] and intent != UserIntent.HUMOR:
                explanation = self._add_humor_to_explanation(explanation)
            
            result = DecisionResult(
                intent=intent,
                confidence=confidence,
                action=action,
                parameters=action_params,
                explanation=explanation
            )
            
            self.logger.info(f"Decision made: {intent.value} -> {action} (confidence: {confidence.value})")
            return result
            
        except Exception as e:
            self.logger.error(f"Intent analysis failed: {str(e)}")
            return self._create_fallback_result(user_input)

    def _preprocess_input(self, user_input: str) -> str:
        """Preprocess user input for better analysis"""
        # Convert to lowercase
        processed = user_input.lower().strip()
        
        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        # Expand contractions
        contractions = {
            "what's": "what is",
            "who's": "who is",
            "where's": "where is",
            "how's": "how is",
            "why's": "why is",
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "shouldn't": "should not"
        }
        
        for contraction, expansion in contractions.items():
            processed = processed.replace(contraction, expansion)
        
        return processed

    def _detect_intent(self, processed_input: str) -> Tuple[UserIntent, DecisionConfidence, Dict[str, Any]]:
        """Detect user intent from processed input"""
        best_intent = UserIntent.UNKNOWN
        best_confidence = DecisionConfidence.UNCERTAIN
        extracted_params = {}
        highest_score = 0
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                matches = re.search(pattern, processed_input, re.IGNORECASE)
                if matches:
                    score = self._calculate_confidence_score(pattern, processed_input, matches)
                    
                    if score > highest_score:
                        highest_score = score
                        best_intent = intent
                        extracted_params = self._extract_parameters(intent, matches.groups())
                        
                        # Set confidence based on score
                        if score > 0.8:
                            best_confidence = DecisionConfidence.HIGH
                        elif score > 0.6:
                            best_confidence = DecisionConfidence.MEDIUM
                        else:
                            best_confidence = DecisionConfidence.LOW
        
        # If no pattern matched, use keyword matching as fallback
        if best_intent == UserIntent.UNKNOWN:
            return self._fallback_intent_detection(processed_input)
        
        return best_intent, best_confidence, extracted_params

    def _calculate_confidence_score(self, pattern: str, input_text: str, matches: re.Match) -> float:
        """Calculate confidence score for intent detection"""
        base_score = 0.5
        
        # Pattern complexity bonus
        pattern_complexity = len(pattern.split()) / 10
        base_score += min(pattern_complexity, 0.3)
        
        # Match length bonus
        match_ratio = len(matches.group(0)) / len(input_text)
        base_score += match_ratio * 0.2
        
        # Context awareness bonus
        if self.conversation_context:
            context_bonus = self._check_context_relevance(input_text)
            base_score += context_bonus * 0.2
        
        return min(base_score, 1.0)

    def _check_context_relevance(self, current_input: str) -> float:
        """Check how relevant current input is to conversation context"""
        if not self.conversation_context:
            return 0.0
        
        # Simple keyword matching for context relevance
        context_text = " ".join([msg.get('message', '') for msg in self.conversation_context])
        current_words = set(current_input.lower().split())
        context_words = set(context_text.lower().split())
        
        common_words = current_words.intersection(context_words)
        
        if not current_words:
            return 0.0
        
        return len(common_words) / len(current_words)

    def _extract_parameters(self, intent: UserIntent, match_groups: Tuple) -> Dict[str, Any]:
        """Extract parameters from matched groups"""
        params = {}
        
        if not match_groups:
            return params
        
        if intent == UserIntent.SEARCH:
            # The search query is typically in the last group
            if match_groups:
                params['query'] = match_groups[-1].strip()
        
        elif intent == UserIntent.CONTROL:
            if match_groups:
                params['control_type'] = match_groups[0] if match_groups else 'click'
        
        elif intent == UserIntent.MEDIA:
            if match_groups:
                action = match_groups[0] if match_groups else 'play'
                params['media_action'] = action
        
        elif intent == UserIntent.BROWSER:
            if len(match_groups) > 1:
                params['url'] = match_groups[1] if 'http' in match_groups[1] else f"https://www.google.com/search?q={match_groups[1]}"
        
        return params

    def _fallback_intent_detection(self, processed_input: str) -> Tuple[UserIntent, DecisionConfidence, Dict[str, Any]]:
        """Fallback intent detection using keyword matching"""
        keyword_scores = {
            UserIntent.CHAT: 0,
            UserIntent.SEARCH: 0,
            UserIntent.HUMOR: 0
        }
        
        # Simple keyword scoring
        search_keywords = ['search', 'find', 'what', 'who', 'how', 'where', 'when', 'why']
        control_keywords = ['click', 'move', 'type', 'open', 'close', 'shutdown']
        media_keywords = ['play', 'pause', 'volume', 'music', 'video']
        humor_keywords = ['joke', 'funny', 'laugh', 'humor', 'roast']
        
        words = processed_input.split()
        
        for word in words:
            if word in search_keywords:
                keyword_scores[UserIntent.SEARCH] += 1
            if word in control_keywords:
                keyword_scores[UserIntent.CONTROL] += 1
            if word in media_keywords:
                keyword_scores[UserIntent.MEDIA] += 1
            if word in humor_keywords:
                keyword_scores[UserIntent.HUMOR] += 1
        
        # Default to chat if no strong signals
        if any(score > 0 for score in keyword_scores.values()):
            best_intent = max(keyword_scores, key=keyword_scores.get)
            confidence = DecisionConfidence.LOW if keyword_scores[best_intent] == 1 else DecisionConfidence.MEDIUM
        else:
            best_intent = UserIntent.CHAT
            confidence = DecisionConfidence.MEDIUM
        
        return best_intent, confidence, {}

    def _determine_action(self, intent: UserIntent, processed_input: str, extracted_params: Dict) -> Tuple[str, Dict[str, Any]]:
        """Determine specific action based on intent and input"""
        action_params = extracted_params.copy()
        
        if intent == UserIntent.CHAT:
            return "converse", {'message': processed_input}
        
        elif intent == UserIntent.SEARCH:
            if 'news' in processed_input:
                return "news_search", action_params
            elif any(word in processed_input for word in ['image', 'picture', 'photo']):
                return "image_search", action_params
            else:
                return "web_search", action_params
        
        elif intent == UserIntent.CONTROL:
            if any(word in processed_input for word in ['mouse', 'click', 'scroll']):
                return "mouse_control", action_params
            elif any(word in processed_input for word in ['type', 'keyboard', 'write']):
                return "type_text", action_params
            elif any(word in processed_input for word in ['shutdown', 'restart', 'lock']):
                return "system_command", action_params
            else:
                return "mouse_control", action_params
        
        elif intent == UserIntent.MEDIA:
            if any(word in processed_input for word in ['play', 'resume', 'start']):
                return "play_media", action_params
            elif any(word in processed_input for word in ['pause', 'stop']):
                return "pause_media", action_params
            elif any(word in processed_input for word in ['volume', 'mute']):
                return "volume_control", action_params
            else:
                return "play_media", action_params
        
        elif intent == UserIntent.BROWSER:
            if any(word in processed_input for word in ['open', 'visit', 'go to']):
                return "open_url", action_params
            else:
                return "browser_search", action_params
        
        elif intent == UserIntent.SYSTEM:
            return "system_status", action_params
        
        elif intent == UserIntent.CREATIVE:
            return "creative_response", {'prompt': processed_input}
        
        elif intent == UserIntent.HUMOR:
            return "tell_joke", {'context': processed_input}
        
        else:
            return "converse", {'message': processed_input}

    def _generate_explanation(self, intent: UserIntent, confidence: DecisionConfidence, action: str) -> str:
        """Generate human-readable explanation for the decision"""
        explanations = {
            UserIntent.SEARCH: {
                DecisionConfidence.HIGH: "I'm pretty sure you want me to search for something!",
                DecisionConfidence.MEDIUM: "I think you're looking for information.",
                DecisionConfidence.LOW: "It seems like you might want me to search for something?"
            },
            UserIntent.CONTROL: {
                DecisionConfidence.HIGH: "Got it! I'll control your system as requested.",
                DecisionConfidence.MEDIUM: "I believe you want me to perform a system control action.",
                DecisionConfidence.LOW: "I think you might be asking me to control something?"
            },
            UserIntent.CHAT: {
                DecisionConfidence.HIGH: "I understand you want to have a conversation!",
                DecisionConfidence.MEDIUM: "Let's chat! I'm here to help.",
                DecisionConfidence.LOW: "I think you want to talk about something?"
            }
        }
        
        # Default explanation if not found in mapping
        default_explanation = f"I'll perform the {action} action for you!"
        
        intent_explanations = explanations.get(intent, {})
        return intent_explanations.get(confidence, default_explanation)

    def _add_humor_to_explanation(self, explanation: str) -> str:
        """Add Mickey's humor to explanations"""
        humor_prefixes = [
            "Mickey's on it! ",
            "Hot dog! ",
            "Gosh! ",
            "Aw, gee! ",
            "Ha ha! "
        ]
        
        humor_suffixes = [
            " This is gonna be fun!",
            " Mickey's magic at work!",
            " Let's make some magic!",
            " Time for an adventure!",
            " Gosh, I love helping!"
        ]
        
        # 30% chance to add humor
        if random.random() < 0.3:
            if random.random() < 0.5:
                explanation = random.choice(humor_prefixes) + explanation
            else:
                explanation = explanation + random.choice(humor_suffixes)
        
        return explanation

    def _create_fallback_result(self, user_input: str) -> DecisionResult:
        """Create fallback result when analysis fails"""
        return DecisionResult(
            intent=UserIntent.CHAT,
            confidence=DecisionConfidence.LOW,
            action="converse",
            parameters={'message': user_input},
            explanation="I'm not quite sure what you meant, but let's chat about it!"
        )

    def update_personality_trait(self, trait: str, value: float):
        """Update Mickey's personality traits"""
        if trait in self.personality_traits and 0 <= value <= 1:
            self.personality_traits[trait] = value
            self.logger.info(f"Updated {trait} to {value}")

    def get_decision_stats(self) -> Dict[str, Any]:
        """Get statistics about decision making"""
        return {
            'personality_traits': self.personality_traits,
            'context_messages': len(self.conversation_context),
            'intent_patterns_count': sum(len(patterns) for patterns in self.intent_patterns.values())
        }

# Test function
def test_decision_engine():
    """Test the decision engine with various inputs"""
    decision_engine = DecisionEngine()
    
    test_inputs = [
        "search for funny cat videos",
        "play some music",
        "what is the weather today?",
        "move mouse to top right corner",
        "tell me a joke",
        "system status please",
        "open google website"
    ]
    
    for user_input in test_inputs:
        result = decision_engine.analyze_intent(user_input)
        print(f"Input: '{user_input}'")
        print(f"Intent: {result.intent.value}")
        print(f"Confidence: {result.confidence.value}")
        print(f"Action: {result.action}")
        print(f"Parameters: {result.parameters}")
        print(f"Explanation: {result.explanation}")
        print("---")

if __name__ == "__main__":
    test_decision_engine()