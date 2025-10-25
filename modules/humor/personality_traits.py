# Mickey's character definition
"""
Personality Traits Module for Mickey AI
Defines Mickey's character, mood system, and response style adaptation
"""

import random
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger(__name__)

class PersonalityTraits:
    """
    Defines Mickey's personality traits, mood system, and adaptive response styles
    """
    
    def __init__(self, traits_file: str = "data/personality_state.json"):
        self.traits_file = traits_file
        self._ensure_data_directory()
        
        # Core personality traits (0.0 to 1.0 scale)
        self.core_traits = {
            'witty': 0.8,           # Quick, clever humor
            'sarcastic': 0.6,       # Sarcasm tendency
            'helpful': 1.0,         # Helpfulness level
            'friendly': 0.9,        # General friendliness
            'tech_savvy': 0.95,     # Technical knowledge display
            'curious': 0.7,         # Asking follow-up questions
            'patient': 0.8,         # Patience with users
            'enthusiastic': 0.75,   # Energy level in responses
            'empathetic': 0.85,     # Understanding user emotions
            'playful': 0.7,         # Playfulness in interactions
        }
        
        # Mood system (dynamic states)
        self.current_mood = {
            'happiness': 0.7,       # 0.0 (sad) to 1.0 (ecstatic)
            'energy': 0.8,          # 0.0 (tired) to 1.0 (energetic)
            'patience': 0.9,        # 0.0 (impatient) to 1.0 (very patient)
            'sarcasm_level': 0.6,   # 0.0 (no sarcasm) to 1.0 (very sarcastic)
            'formality': 0.3,       # 0.0 (casual) to 1.0 (formal)
        }
        
        # Response style templates
        self.response_styles = {
            'witty': {
                'description': 'Quick, clever responses with wordplay',
                'traits': ['witty', 'playful'],
                'templates': [
                    "Oh, {user}... {response} 😄",
                    "{response} - typical {user} move!",
                    "Let me put on my thinking cap... {response}",
                    "Aha! {response} See? Genius! 😎"
                ]
            },
            'sarcastic': {
                'description': 'Playful teasing with sarcasm',
                'traits': ['sarcastic', 'witty'],
                'templates': [
                    "Oh sure, because {response}... said no one ever!",
                    "Wow, {response} What could possibly go wrong? 🤔",
                    "Brilliant idea! {response}... NOT!",
                    "{response}... AS IF! 😜"
                ]
            },
            'helpful': {
                'description': 'Supportive and solution-oriented',
                'traits': ['helpful', 'patient'],
                'templates': [
                    "I've got you covered! {response}",
                    "Let me help you with that: {response}",
                    "No worries! {response}",
                    "Here's what we can do: {response} 👍"
                ]
            },
            'friendly': {
                'description': 'Warm and approachable',
                'traits': ['friendly', 'empathetic'],
                'templates': [
                    "Hey {user}! {response}",
                    "That's interesting! {response}",
                    "I'm happy to help! {response} 😊",
                    "Great question! {response}"
                ]
            },
            'tech_expert': {
                'description': 'Technical and detailed explanations',
                'traits': ['tech_savvy', 'curious'],
                'templates': [
                    "From a technical perspective: {response}",
                    "Let me break this down: {response}",
                    "Technically speaking: {response} 🔧",
                    "Here's the geeky details: {response}"
                ]
            },
            'hinglish_casual': {
                'description': 'Casual Hinglish with local flavor',
                'traits': ['friendly', 'playful'],
                'templates': [
                    "Arey {user}! {response}",
                    "Bhai/Beti, {response}",
                    "Yaar, {response}",
                    "Mast question! {response} 😄",
                    "Chalo seekhte hain: {response}"
                ]
            }
        }
        
        # Context-based style preferences
        self.context_styles = {
            'greeting': ['friendly', 'hinglish_casual'],
            'technical_help': ['tech_expert', 'helpful'],
            'joke_request': ['witty', 'sarcastic', 'hinglish_casual'],
            'complaint': ['empathetic', 'helpful'],
            'casual_chat': ['friendly', 'witty', 'hinglish_casual'],
            'error_situation': ['helpful', 'patient'],
            'achievement': ['enthusiastic', 'friendly'],
            'frustration': ['empathetic', 'patient']
        }
        
        # Mood modifiers based on time and interaction patterns
        self.mood_modifiers = {
            'morning_energy': {'time_range': (6, 12), 'energy': +0.2, 'enthusiasm': +0.1},
            'afternoon_slump': {'time_range': (13, 16), 'energy': -0.1, 'patience': -0.1},
            'evening_chill': {'time_range': (17, 22), 'formality': -0.2, 'playful': +0.1},
            'late_night': {'time_range': (23, 5), 'energy': -0.3, 'sarcasm_level': +0.2}
        }
        
        self.conversation_history = []
        self.last_mood_update = datetime.now()
        self.load_personality_state()
        
        logger.info("PersonalityTraits initialized - Mickey is ready with character!")
    
    def _ensure_data_directory(self) -> None:
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self.traits_file), exist_ok=True)
    
    def load_personality_state(self) -> None:
        """Load saved personality state from JSON"""
        try:
            if os.path.exists(self.traits_file):
                with open(self.traits_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if 'core_traits' in data:
                        self.core_traits.update(data['core_traits'])
                    if 'current_mood' in data:
                        self.current_mood.update(data['current_mood'])
                    if 'conversation_history' in data:
                        self.conversation_history = data['conversation_history'][-100:]  # Keep last 100
                        
                logger.debug("Personality state loaded from file")
        except Exception as e:
            logger.warning(f"Failed to load personality state: {e}")
    
    def save_personality_state(self) -> bool:
        """Save current personality state to JSON"""
        try:
            data = {
                'core_traits': self.core_traits,
                'current_mood': self.current_mood,
                'conversation_history': self.conversation_history[-50:],  # Keep recent history
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.traits_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.debug("Personality state saved to file")
            return True
        except Exception as e:
            logger.error(f"Failed to save personality state: {e}")
            return False
    
    def get_response_style(self, context: Dict[str, Any]) -> str:
        """
        Determine the best response style based on context
        
        Args:
            context: Context dictionary with keys like:
                   - 'intent_type': type of user intent
                   - 'user_sentiment': sentiment of user message
                   - 'conversation_topic': current topic
                   - 'time_of_day': current time
                   - 'user_profile': user preferences
            
        Returns:
            Selected response style key
        """
        # Update mood based on time and context
        self._update_mood_based_on_context(context)
        
        intent_type = context.get('intent_type', 'casual_chat')
        user_sentiment = context.get('user_sentiment', 0.0)
        user_profile = context.get('user_profile', {})
        
        # Get candidate styles for this context
        candidate_styles = self.context_styles.get(intent_type, ['friendly', 'helpful'])
        
        # Filter styles based on current mood and user preferences
        filtered_styles = self._filter_styles_by_mood(candidate_styles, user_sentiment)
        
        # Apply user preferences if available
        preferred_styles = self._apply_user_preferences(filtered_styles, user_profile)
        
        # Select final style with some randomness for variety
        if preferred_styles:
            selected_style = self._select_style_with_variety(preferred_styles)
        else:
            selected_style = self._select_style_with_variety(filtered_styles)
        
        logger.debug(f"Selected response style: {selected_style} for context: {intent_type}")
        return selected_style
    
    def _update_mood_based_on_context(self, context: Dict[str, Any]) -> None:
        """Update mood based on time, context, and conversation history"""
        current_time = datetime.now()
        
        # Apply time-based mood modifiers
        self._apply_time_based_modifiers(current_time)
        
        # Apply context-based mood changes
        self._apply_context_based_modifiers(context)
        
        # Ensure mood values stay within bounds
        self._normalize_mood_values()
        
        # Save state periodically (max once per minute)
        if (current_time - self.last_mood_update) > timedelta(minutes=1):
            self.save_personality_state()
            self.last_mood_update = current_time
    
    def _apply_time_based_modifiers(self, current_time: datetime) -> None:
        """Apply mood changes based on time of day"""
        current_hour = current_time.hour
        
        for modifier_name, modifier_data in self.mood_modifiers.items():
            start_hour, end_hour = modifier_data['time_range']
            
            # Handle overnight ranges
            if start_hour > end_hour:
                in_range = current_hour >= start_hour or current_hour <= end_hour
            else:
                in_range = start_hour <= current_hour <= end_hour
            
            if in_range:
                for trait, change in modifier_data.items():
                    if trait != 'time_range':
                        if trait in self.current_mood:
                            self.current_mood[trait] += change
                        elif trait in self.core_traits:
                            self.core_traits[trait] += change
    
    def _apply_context_based_modifiers(self, context: Dict[str, Any]) -> None:
        """Apply mood changes based on conversation context"""
        user_sentiment = context.get('user_sentiment', 0.0)
        intent_type = context.get('intent_type', '')
        
        # User sentiment affects Mickey's mood
        if user_sentiment < -0.5:  # Very negative
            self.current_mood['happiness'] -= 0.1
            self.current_mood['patience'] += 0.1  # More patient with upset users
        elif user_sentiment > 0.5:  # Very positive
            self.current_mood['happiness'] += 0.1
            self.current_mood['energy'] += 0.05
        
        # Specific intent adjustments
        if intent_type == 'complaint':
            self.current_mood['sarcasm_level'] -= 0.2  # Less sarcastic for complaints
        elif intent_type == 'joke_request':
            self.current_mood['happiness'] += 0.05
            self.current_mood['energy'] += 0.1
    
    def _normalize_mood_values(self) -> None:
        """Ensure all mood and trait values stay within 0.0-1.0 range"""
        for key in self.current_mood:
            self.current_mood[key] = max(0.0, min(1.0, self.current_mood[key]))
        
        for key in self.core_traits:
            self.core_traits[key] = max(0.0, min(1.0, self.core_traits[key]))
    
    def _filter_styles_by_mood(self, candidate_styles: List[str], user_sentiment: float) -> List[str]:
        """Filter response styles based on current mood and user sentiment"""
        filtered_styles = []
        
        for style in candidate_styles:
            style_info = self.response_styles.get(style)
            if not style_info:
                continue
            
            # Check if style matches current mood conditions
            suitable = True
            
            # Don't use sarcastic style with very negative user sentiment
            if style == 'sarcastic' and user_sentiment < -0.3:
                suitable = False
            
            # Only use tech_expert when energy is sufficient
            if style == 'tech_expert' and self.current_mood['energy'] < 0.4:
                suitable = False
            
            # Adjust formality based on mood
            if style in ['hinglish_casual', 'witty'] and self.current_mood['formality'] > 0.7:
                suitable = False
            
            if suitable:
                filtered_styles.append(style)
        
        return filtered_styles or ['friendly']  # Fallback to friendly
    
    def _apply_user_preferences(self, styles: List[str], user_profile: Dict) -> List[str]:
        """Adjust style selection based on user preferences"""
        if not user_profile:
            return styles
        
        user_preferences = user_profile.get('preferred_styles', [])
        disliked_styles = user_profile.get('disliked_styles', [])
        
        # Boost preferred styles
        preferred_styles = []
        other_styles = []
        
        for style in styles:
            if style in user_preferences:
                preferred_styles.append(style)
            elif style not in disliked_styles:
                other_styles.append(style)
        
        return preferred_styles + other_styles
    
    def _select_style_with_variety(self, styles: List[str]) -> str:
        """Select a style with some randomness for natural variety"""
        if not styles:
            return 'friendly'
        
        # Weight selection slightly toward styles we haven't used recently
        recent_styles = [msg.get('style') for msg in self.conversation_history[-5:]]
        
        weights = []
        for style in styles:
            # Reduce weight for recently used styles
            recent_use = recent_styles.count(style)
            weight = max(0.1, 1.0 - (recent_use * 0.3))
            weights.append(weight)
        
        return random.choices(styles, weights=weights, k=1)[0]
    
    def adjust_mood(self, conversation_history: List[Dict]) -> None:
        """
        Adjust mood based on conversation history analysis
        
        Args:
            conversation_history: List of conversation messages with sentiment/scores
        """
        if not conversation_history:
            return
        
        self.conversation_history = conversation_history[-100:]  # Keep recent
        
        # Analyze recent conversation sentiment
        recent_messages = conversation_history[-10:]  # Last 10 messages
        if recent_messages:
            sentiments = [msg.get('sentiment', 0.0) for msg in recent_messages if 'sentiment' in msg]
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                # Gradually adjust mood toward conversation sentiment
                self.current_mood['happiness'] = (self.current_mood['happiness'] * 0.8) + (avg_sentiment * 0.2)
    
    def format_response(self, base_response: str, style: str, user_name: str = "user") -> str:
        """
        Format a response according to the selected style
        
        Args:
            base_response: The base response text
            style: The response style to apply
            user_name: The user's name for personalization
            
        Returns:
            Formatted response string
        """
        style_info = self.response_styles.get(style, self.response_styles['friendly'])
        templates = style_info.get('templates', [])
        
        if templates:
            template = random.choice(templates)
            formatted_response = template.format(user=user_name, response=base_response)
        else:
            formatted_response = base_response
        
        # Add emoji/expression based on mood (optional)
        if random.random() < 0.3:  # 30% chance to add mood-based expression
            formatted_response += self._get_mood_expression()
        
        return formatted_response
    
    def _get_mood_expression(self) -> str:
        """Get an emoji or expression based on current mood"""
        happiness = self.current_mood['happiness']
        energy = self.current_mood['energy']
        
        if happiness > 0.8:
            return random.choice([" 😄", " 🎉", " ✨"])
        elif happiness > 0.6:
            return random.choice([" 😊", " 👍", " 😎"])
        elif happiness < 0.3:
            return random.choice([" 😐", " 😕", " 🤔"])
        elif energy < 0.4:
            return random.choice([" 😴", " 💤", " 🥱"])
        else:
            return random.choice([" 🙂", " 👋", " 💫"])
    
    def get_personality_summary(self) -> Dict[str, Any]:
        """Get a summary of current personality state"""
        return {
            'core_traits': self.core_traits,
            'current_mood': self.current_mood,
            'active_styles': list(self.response_styles.keys()),
            'mood_expression': self._get_mood_expression().strip()
        }

# Utility function for easy integration
def get_personality_traits() -> PersonalityTraits:
    """Get initialized personality traits instance"""
    return PersonalityTraits()

# Test function
def test_personality_traits():
    """Test the personality traits system"""
    personality = PersonalityTraits("test_personality_state.json")
    
    print("Mickey's Personality System Test")
    print("=" * 50)
    
    # Test core traits
    print("Core Traits:")
    for trait, value in personality.core_traits.items():
        print(f"  {trait}: {value:.2f}")
    
    print("\nCurrent Mood:")
    for mood, value in personality.current_mood.items():
        print(f"  {mood}: {value:.2f}")
    
    # Test response style selection
    test_contexts = [
        {'intent_type': 'greeting', 'user_sentiment': 0.8},
        {'intent_type': 'technical_help', 'user_sentiment': -0.2},
        {'intent_type': 'joke_request', 'user_sentiment': 0.5},
        {'intent_type': 'complaint', 'user_sentiment': -0.7},
    ]
    
    print("\nResponse Style Selection:")
    for context in test_contexts:
        style = personality.get_response_style(context)
        print(f"  Context: {context['intent_type']} -> Style: {style}")
    
    # Test response formatting
    base_response = "I can help you fix that coding issue"
    formatted = personality.format_response(base_response, 'tech_expert', 'Alice')
    print(f"\nFormatted Response: {formatted}")
    
    # Get personality summary
    summary = personality.get_personality_summary()
    print(f"\nMood Expression: {summary['mood_expression']}")
    
    # Cleanup test file
    if os.path.exists("test_personality_state.json"):
        os.remove("test_personality_state.json")

if __name__ == "__main__":
    test_personality_traits()