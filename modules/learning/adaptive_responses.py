# Personalized reply generation
"""
Adaptive Response Engine for Mickey AI
Personalizes responses based on user profiles, conversation history, and preferences
"""

import json
import os
import random
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class AdaptiveResponseEngine:
    """
    Personalizes Mickey's responses based on user profiles and conversation context
    """
    
    def __init__(self, personality_file: str = "data/adaptive_personality.json"):
        self.personality_file = personality_file
        self._ensure_data_directory()
        
        # Response adaptation strategies
        self.adaptation_strategies = {
            'humor_boost': {
                'description': 'Add humor to responses',
                'triggers': ['high_engagement', 'positive_sentiment', 'joke_lover'],
                'weight': 0.3
            },
            'detail_level': {
                'description': 'Adjust technical detail level',
                'triggers': ['tech_savvy_user', 'learning_mode', 'complex_topic'],
                'weight': 0.25
            },
            'formality_adjust': {
                'description': 'Adjust formality level',
                'triggers': ['formal_context', 'professional_user', 'first_interaction'],
                'weight': 0.15
            },
            'empathy_boost': {
                'description': 'Increase empathy in responses',
                'triggers': ['negative_sentiment', 'personal_topic', 'user_stressed'],
                'weight': 0.2
            },
            'cultural_context': {
                'description': 'Add cultural references',
                'triggers': ['hinglish_user', 'indian_context', 'cultural_topic'],
                'weight': 0.1
            }
        }
        
        # Response templates for different adaptation types
        self.adaptive_templates = {
            'humor': [
                "😂 {response}",
                "{response} - Just kidding! Well, mostly...",
                "🤔 {response} But what do I know, I'm just an AI!",
                "{response} *winks*",
                "Serious answer: {response} Funny answer: Just Google it! 😜"
            ],
            'detailed': [
                "Let me break this down for you: {response}",
                "Here's a comprehensive explanation: {response}",
                "From a technical perspective: {response}",
                "Let me give you the full picture: {response}",
                "Detailed explanation: {response}"
            ],
            'simplified': [
                "In simple terms: {response}",
                "Here's the easy version: {response}",
                "Let me make this simple: {response}",
                "No complicated jargon: {response}",
                "Straight to the point: {response}"
            ],
            'empathetic': [
                "I understand how you feel. {response}",
                "That sounds challenging. {response}",
                "I'm here to help. {response}",
                "That must be difficult. {response}",
                "I can imagine how that feels. {response}"
            ],
            'cultural': [
                "Arey! {response}",
                "Yaar, {response}",
                "Bhai/Beti, {response}",
                "Mast question! {response}",
                "Desi style mein: {response} 😄"
            ],
            'enthusiastic': [
                "Awesome! {response} 🎉",
                "Fantastic question! {response}",
                "I love this! {response} ✨",
                "Great topic! {response}",
                "This is exciting! {response} 🚀"
            ]
        }
        
        # User preference mappings
        self.preference_mappings = {
            'joke_lover': {'humor_boost': 0.8, 'cultural_context': 0.3},
            'tech_savvy': {'detail_level': 0.9, 'humor_boost': 0.1},
            'casual_user': {'formality_adjust': -0.7, 'cultural_context': 0.6},
            'professional_user': {'formality_adjust': 0.8, 'detail_level': 0.7},
            'new_user': {'formality_adjust': 0.5, 'empathy_boost': 0.4},
            'stressed_user': {'empathy_boost': 0.9, 'humor_boost': -0.5},
            'indian_user': {'cultural_context': 0.8, 'humor_boost': 0.4}
        }
        
        self.adaptation_history = {}
        self.load_adaptation_data()
        
        logger.info("AdaptiveResponseEngine initialized")
    
    def _ensure_data_directory(self) -> None:
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self.personality_file), exist_ok=True)
    
    def load_adaptation_data(self) -> None:
        """Load adaptation data from JSON file"""
        try:
            if os.path.exists(self.personality_file):
                with open(self.personality_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.adaptation_history = data.get('adaptation_history', {})
                logger.debug("Adaptation data loaded successfully")
            else:
                logger.info("No existing adaptation data found")
                self.adaptation_history = {}
                
        except Exception as e:
            logger.error(f"Error loading adaptation data: {e}")
            self.adaptation_history = {}
    
    def save_adaptation_data(self) -> bool:
        """Save adaptation data to JSON file"""
        try:
            data = {
                'adaptation_history': self.adaptation_history,
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.personality_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug("Adaptation data saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving adaptation data: {e}")
            return False
    
    def personalize_response(self, base_response: str, user_profile: Dict, 
                           conversation_context: Dict = None) -> str:
        """
        Personalize a response based on user profile and context
        
        Args:
            base_response: The base LLM-generated response
            user_profile: User preference and profile data
            conversation_context: Current conversation context
            
        Returns:
            Personalized response string
        """
        if not base_response or not user_profile:
            return base_response
        
        conversation_context = conversation_context or {}
        
        # Calculate adaptation factors
        adaptation_factors = self._calculate_adaptation_factors(user_profile, conversation_context)
        
        # Apply adaptations
        personalized_response = self._apply_adaptations(base_response, adaptation_factors, user_profile)
        
        # Update adaptation history
        self._update_adaptation_history(user_profile.get('user_id', 'default'), adaptation_factors)
        
        logger.debug(f"Personalized response with factors: {adaptation_factors}")
        return personalized_response
    
    def _calculate_adaptation_factors(self, user_profile: Dict, context: Dict) -> Dict[str, float]:
        """
        Calculate how much to adapt the response based on profile and context
        
        Args:
            user_profile: User profile data
            context: Conversation context
            
        Returns:
            Dictionary of adaptation factors (0.0 to 1.0)
        """
        factors = {
            'humor_boost': 0.0,
            'detail_level': 0.0,
            'formality_adjust': 0.0,
            'empathy_boost': 0.0,
            'cultural_context': 0.0,
            'enthusiasm_boost': 0.0
        }
        
        # Apply user preference mappings
        user_preferences = user_profile.get('preferences', {})
        user_traits = self._extract_user_traits(user_profile)
        
        for trait, trait_adaptations in self.preference_mappings.items():
            if trait in user_traits:
                for adaptation, weight in trait_adaptations.items():
                    if adaptation in factors:
                        factors[adaptation] += weight
        
        # Apply context-based adjustments
        context_adjustments = self._get_context_adjustments(context)
        for adaptation, adjustment in context_adjustments.items():
            if adaptation in factors:
                factors[adaptation] += adjustment
        
        # Apply learned preferences from history
        history_adjustments = self._get_history_adjustments(user_profile.get('user_id', 'default'))
        for adaptation, adjustment in history_adjustments.items():
            if adaptation in factors:
                factors[adaptation] = (factors[adaptation] + adjustment) / 2  # Blend with current
        
        # Normalize factors to 0.0-1.0 range
        for key in factors:
            factors[key] = max(-1.0, min(1.0, factors[key]))
        
        return factors
    
    def _extract_user_traits(self, user_profile: Dict) -> List[str]:
        """Extract user traits from profile data"""
        traits = []
        
        # Check favorite topics
        favorite_topics = user_profile.get('favorite_topics', [])
        if 'jokes' in favorite_topics or user_profile.get('preferred_style') == 'humorous':
            traits.append('joke_lover')
        
        if 'tech' in favorite_topics or user_profile.get('interaction_count', 0) > 50:
            traits.append('tech_savvy')
        
        # Check interaction style
        preferred_style = user_profile.get('preferred_style', '')
        if preferred_style in ['casual', 'humorous']:
            traits.append('casual_user')
        elif preferred_style in ['technical', 'detailed']:
            traits.append('professional_user')
        
        # Check for new user
        if user_profile.get('total_interactions', 0) < 5:
            traits.append('new_user')
        
        # Check for cultural context
        if user_profile.get('hinglish_ratio', 0) > 0.3:
            traits.append('indian_user')
        
        return traits
    
    def _get_context_adjustments(self, context: Dict) -> Dict[str, float]:
        """Get adaptation adjustments based on conversation context"""
        adjustments = {}
        
        # Sentiment-based adjustments
        sentiment = context.get('sentiment', 0.0)
        if sentiment < -0.3:
            adjustments['empathy_boost'] = 0.7
            adjustments['humor_boost'] = -0.5
        elif sentiment > 0.5:
            adjustments['humor_boost'] = 0.4
            adjustments['enthusiasm_boost'] = 0.3
        
        # Engagement-based adjustments
        engagement = context.get('engagement', 0.5)
        if engagement > 0.7:
            adjustments['detail_level'] = 0.3
            adjustments['enthusiasm_boost'] = 0.4
        
        # Topic-based adjustments
        topic = context.get('topic', '')
        if topic in ['tech', 'programming', 'science']:
            adjustments['detail_level'] = 0.6
        elif topic in ['personal', 'emotions']:
            adjustments['empathy_boost'] = 0.8
        elif topic in ['culture', 'india', 'bollywood']:
            adjustments['cultural_context'] = 0.7
        
        # Time-based adjustments
        hour = datetime.now().hour
        if 23 <= hour <= 5:  # Late night
            adjustments['humor_boost'] = 0.2
            adjustments['formality_adjust'] = -0.3
        
        return adjustments
    
    def _get_history_adjustments(self, user_id: str) -> Dict[str, float]:
        """Get adjustments based on adaptation history"""
        if user_id not in self.adaptation_history:
            return {}
        
        user_history = self.adaptation_history[user_id]
        if not user_history or len(user_history) < 5:
            return {}
        
        # Calculate average successful adaptations
        recent_history = user_history[-10:]  # Last 10 adaptations
        successful_adaptations = [adapt for adapt in recent_history if adapt.get('successful', False)]
        
        if not successful_adaptations:
            return {}
        
        # Calculate average factors for successful adaptations
        avg_factors = {}
        for adaptation in successful_adaptations:
            factors = adaptation.get('factors', {})
            for factor, value in factors.items():
                avg_factors[factor] = avg_factors.get(factor, 0.0) + value
        
        for factor in avg_factors:
            avg_factors[factor] /= len(successful_adaptations)
        
        return avg_factors
    
    def _apply_adaptations(self, response: str, factors: Dict, user_profile: Dict) -> str:
        """
        Apply adaptations to the base response
        
        Args:
            response: Base response
            factors: Adaptation factors
            user_profile: User profile data
            
        Returns:
            Adapted response
        """
        adapted_response = response
        
        # Apply humor adaptation
        if factors['humor_boost'] > 0.3:
            if random.random() < factors['humor_boost']:
                adapted_response = self._apply_template(adapted_response, 'humor')
        
        # Apply detail level adaptation
        if abs(factors['detail_level']) > 0.2:
            if factors['detail_level'] > 0:
                adapted_response = self._apply_template(adapted_response, 'detailed')
            else:
                adapted_response = self._apply_template(adapted_response, 'simplified')
        
        # Apply empathy adaptation
        if factors['empathy_boost'] > 0.4:
            adapted_response = self._apply_template(adapted_response, 'empathetic')
        
        # Apply cultural adaptation
        if factors['cultural_context'] > 0.3:
            if random.random() < factors['cultural_context']:
                adapted_response = self._apply_template(adapted_response, 'cultural')
        
        # Apply enthusiasm adaptation
        if factors['enthusiasm_boost'] > 0.3:
            adapted_response = self._apply_template(adapted_response, 'enthusiastic')
        
        # Apply formality adjustment (simpler approach)
        if factors['formality_adjust'] > 0.5:
            # Make more formal by removing casual elements
            adapted_response = adapted_response.replace('!', '.').replace('😄', '').replace('😂', '')
        elif factors['formality_adjust'] < -0.5:
            # Make more casual by adding friendly elements
            if not any(emoji in adapted_response for emoji in ['😄', '😊', '😂']):
                adapted_response += random.choice([' 😊', ' 😄', ' :)'])
        
        return adapted_response
    
    def _apply_template(self, response: str, template_type: str) -> str:
        """Apply a response template"""
        templates = self.adaptive_templates.get(template_type, [])
        if templates:
            template = random.choice(templates)
            return template.format(response=response)
        return response
    
    def _update_adaptation_history(self, user_id: str, factors: Dict) -> None:
        """Update adaptation history for learning"""
        if user_id not in self.adaptation_history:
            self.adaptation_history[user_id] = []
        
        adaptation_record = {
            'timestamp': datetime.now().isoformat(),
            'factors': factors,
            'successful': True,  # This would be set based on user feedback
            'context': 'auto_generated'  # In real implementation, would include actual context
        }
        
        self.adaptation_history[user_id].append(adaptation_record)
        
        # Keep only last 50 records per user
        if len(self.adaptation_history[user_id]) > 50:
            self.adaptation_history[user_id] = self.adaptation_history[user_id][-50:]
        
        # Save periodically
        if len(self.adaptation_history[user_id]) % 10 == 0:
            self.save_adaptation_data()
    
    def learn_from_feedback(self, user_id: str, original_response: str, 
                          user_feedback: Dict, adapted_factors: Dict) -> None:
        """
        Learn from user feedback to improve future adaptations
        
        Args:
            user_id: User identifier
            original_response: The response that was sent
            user_feedback: Feedback data (e.g., {'rating': 5, 'comment': 'great!'})
            adapted_factors: The adaptation factors that were used
        """
        if user_id not in self.adaptation_history:
            return
        
        # Find the most recent adaptation for this user
        user_history = self.adaptation_history[user_id]
        if not user_history:
            return
        
        latest_adaptation = user_history[-1]
        
        # Determine if adaptation was successful based on feedback
        rating = user_feedback.get('rating', 0)
        successful = rating >= 4  # Consider 4-5 stars as successful
        
        # Update the success flag
        latest_adaptation['successful'] = successful
        latest_adaptation['feedback'] = user_feedback
        latest_adaptation['feedback_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Learned from feedback: adaptation was {'successful' if successful else 'unsuccessful'}")
        self.save_adaptation_data()
    
    def get_adaptation_stats(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get adaptation statistics
        
        Args:
            user_id: Specific user or None for global stats
            
        Returns:
            Adaptation statistics dictionary
        """
        if user_id:
            history = self.adaptation_history.get(user_id, [])
            total = len(history)
            successful = sum(1 for adapt in history if adapt.get('successful', False))
            
            return {
                'user_id': user_id,
                'total_adaptations': total,
                'successful_adaptations': successful,
                'success_rate': successful / total if total > 0 else 0,
                'preferred_adaptations': self._get_preferred_adaptations(history)
            }
        else:
            # Global stats
            total_adaptations = sum(len(history) for history in self.adaptation_history.values())
            total_users = len(self.adaptation_history)
            
            return {
                'total_users': total_users,
                'total_adaptations': total_adaptations,
                'average_adaptations_per_user': total_adaptations / total_users if total_users > 0 else 0,
                'active_users': sum(1 for history in self.adaptation_history.values() if len(history) > 5)
            }
    
    def _get_preferred_adaptations(self, history: List[Dict]) -> List[str]:
        """Get list of preferred adaptation types from history"""
        if not history:
            return []
        
        successful_adaptations = [adapt for adapt in history if adapt.get('successful', False)]
        if not successful_adaptations:
            return []
        
        # Count which factors were most present in successful adaptations
        factor_scores = {}
        for adaptation in successful_adaptations:
            factors = adaptation.get('factors', {})
            for factor, value in factors.items():
                if abs(value) > 0.2: # Only count significant adaptations
                    factor_scores[factor] = factor_scores.get(factor, 0) + abs(value)
        
        return [factor for factor, score in sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)[:3]]

# Utility function for easy integration
def get_adaptive_engine() -> AdaptiveResponseEngine:
    """Get initialized adaptive response engine instance"""
    return AdaptiveResponseEngine()

# Test function
def test_adaptive_responses():
    """Test the adaptive response engine"""
    engine = AdaptiveResponseEngine("test_adaptive_personality.json")
    
    # Test user profiles
    test_profiles = [
        {
            'user_id': 'tech_user',
            'preferred_style': 'technical',
            'favorite_topics': ['tech', 'programming'],
            'total_interactions': 100,
            'hinglish_ratio': 0.1
        },
        {
            'user_id': 'casual_user',
            'preferred_style': 'humorous',
            'favorite_topics': ['jokes', 'entertainment'],
            'total_interactions': 20,
            'hinglish_ratio': 0.6
        },
        {
            'user_id': 'new_user',
            'preferred_style': '',
            'favorite_topics': [],
            'total_interactions': 2,
            'hinglish_ratio': 0.2
        }
    ]
    
    base_response = "The weather today will be sunny with a high of 30 degrees."
    
    print("Adaptive Response Engine Test:")
    print("=" * 50)
    
    for profile in test_profiles:
        context = {'sentiment': 0.7, 'engagement': 0.8, 'topic': 'weather'}
        personalized = engine.personalize_response(base_response, profile, context)
        
        print(f"\nUser: {profile['user_id']}")
        print(f"Original: {base_response}")
        print(f"Personalized: {personalized}")
    
    # Test adaptation stats
    stats = engine.get_adaptation_stats()
    print(f"\nGlobal Stats: {stats}")
    
    # Cleanup test file
    if os.path.exists("test_adaptive_personality.json"):
        os.remove("test_adaptive_personality.json")

if __name__ == "__main__":
    test_adaptive_responses()