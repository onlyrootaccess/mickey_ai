# User habit learning
"""
Preference Tracker Module for Mickey AI
Tracks and learns user habits, preferences, and behavior patterns
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from collections import defaultdict, Counter

# Configure logging
logger = logging.getLogger(__name__)

class PreferenceTracker:
    """
    Tracks user preferences, habits, and behavior patterns over time
    """
    
    def __init__(self, data_file: str = "data/user_preferences.json"):
        self.data_file = data_file
        self._ensure_data_directory()
        self.user_profiles = {}
        self.session_start_time = datetime.now()
        
        # Preference categories to track
        self.preference_categories = {
            'topics': {
                'weights': {'weather': 1, 'jokes': 1, 'tech': 1, 'news': 1, 'music': 1},
                'max_items': 10
            },
            'interaction_style': {
                'weights': {'formal': 0, 'casual': 0, 'humorous': 0, 'technical': 0},
                'max_items': 5
            },
            'time_patterns': {
                'weights': defaultdict(int),
                'max_items': 24  # Hours in day
            },
            'command_frequency': {
                'weights': Counter(),
                'max_items': 20
            },
            'response_preferences': {
                'weights': {'detailed': 0, 'concise': 0, 'humorous': 0, 'empathetic': 0},
                'max_items': 5
            }
        }
        
        self.load_preferences()
        logger.info("PreferenceTracker initialized")
    
    def _ensure_data_directory(self) -> None:
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
    
    def load_preferences(self) -> None:
        """Load user preferences from JSON file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.user_profiles = data.get('user_profiles', {})
                    
                    # Convert back to Counter and defaultdict if needed
                    for user_id, profile in self.user_profiles.items():
                        if 'command_frequency' in profile:
                            profile['command_frequency'] = Counter(profile['command_frequency'])
                        
                logger.debug(f"Loaded preferences for {len(self.user_profiles)} users")
            else:
                logger.info("No existing preferences file found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading preferences: {e}")
            self.user_profiles = {}
    
    def save_preferences(self) -> bool:
        """Save user preferences to JSON file"""
        try:
            # Convert Counter and defaultdict to serializable formats
            save_data = {'user_profiles': {}}
            
            for user_id, profile in self.user_profiles.items():
                save_profile = profile.copy()
                
                if 'command_frequency' in save_profile:
                    save_profile['command_frequency'] = dict(save_profile['command_frequency'])
                
                save_data['user_profiles'][user_id] = save_profile
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.debug("Preferences saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
            return False
    
    def log_preference(self, user_id: str, action: str, metadata: Optional[Dict] = None) -> None:
        """
        Log a user action to learn their preferences
        
        Args:
            user_id: Unique user identifier
            action: The action performed (e.g., 'request_joke', 'ask_weather')
            metadata: Additional context about the action
        """
        metadata = metadata or {}
        
        # Ensure user profile exists
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = self._create_new_user_profile()
        
        profile = self.user_profiles[user_id]
        current_time = datetime.now()
        
        # Update last active timestamp
        profile['last_active'] = current_time.isoformat()
        
        # Update interaction count
        profile['total_interactions'] = profile.get('total_interactions', 0) + 1
        
        # Update command frequency
        profile['command_frequency'][action] += 1
        
        # Update time pattern
        hour = current_time.hour
        profile['time_patterns'][str(hour)] = profile['time_patterns'].get(str(hour), 0) + 1
        
        # Extract topic from action or metadata
        topic = self._extract_topic(action, metadata)
        if topic:
            profile['topics'][topic] = profile['topics'].get(topic, 0) + 1
        
        # Update interaction style preference
        style = metadata.get('interaction_style')
        if style and style in profile['response_preferences']:
            profile['response_preferences'][style] += 1
        
        # Limit the size of collections
        self._limit_preference_collections(profile)
        
        # Save periodically (every 10 interactions)
        if profile['total_interactions'] % 10 == 0:
            self.save_preferences()
        
        logger.debug(f"Logged preference for user {user_id}: {action}")
    
    def _extract_topic(self, action: str, metadata: Dict) -> Optional[str]:
        """Extract topic from action and metadata"""
        # Map actions to topics
        action_to_topic = {
            'request_joke': 'jokes',
            'ask_weather': 'weather',
            'tech_help': 'tech',
            'search_web': 'web_search',
            'play_music': 'music',
            'get_news': 'news',
            'calculate': 'math',
            'set_reminder': 'productivity',
            'ask_definition': 'education'
        }
        
        # Check if action maps to a topic
        if action in action_to_topic:
            return action_to_topic[action]
        
        # Check metadata for topic
        return metadata.get('topic')
    
    def _create_new_user_profile(self) -> Dict[str, Any]:
        """Create a new user profile with default structure"""
        return {
            'created_at': datetime.now().isoformat(),
            'last_active': datetime.now().isoformat(),
            'total_interactions': 0,
            'topics': {},
            'time_patterns': {},
            'command_frequency': Counter(),
            'response_preferences': {
                'detailed': 0,
                'concise': 0, 
                'humorous': 0,
                'empathetic': 0,
                'technical': 0
            },
            'personal_info': {},
            'session_history': []
        }
    
    def _limit_preference_collections(self, profile: Dict[str, Any]) -> None:
        """Limit the size of preference collections to prevent unbounded growth"""
        # Limit topics to top N
        if len(profile['topics']) > self.preference_categories['topics']['max_items']:
            sorted_topics = sorted(profile['topics'].items(), key=lambda x: x[1], reverse=True)
            profile['topics'] = dict(sorted_topics[:self.preference_categories['topics']['max_items']])
        
        # Limit time patterns (already limited by 24 hours)
        if len(profile['time_patterns']) > 24:
            sorted_times = sorted(profile['time_patterns'].items(), key=lambda x: x[1], reverse=True)
            profile['time_patterns'] = dict(sorted_times[:24])
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive user profile with preferences
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            User profile dictionary
        """
        profile = self.user_profiles.get(user_id, self._create_new_user_profile())
        
        # Calculate derived preferences
        enhanced_profile = self._enhance_profile_with_insights(profile)
        return enhanced_profile
    
    def _enhance_profile_with_insights(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Add calculated insights to the profile"""
        enhanced = profile.copy()
        
        # Calculate favorite topics
        if enhanced['topics']:
            enhanced['favorite_topics'] = [
                topic for topic, count in 
                sorted(enhanced['topics'].items(), key=lambda x: x[1], reverse=True)[:3]
            ]
        else:
            enhanced['favorite_topics'] = []
        
        # Calculate most used commands
        if enhanced['command_frequency']:
            enhanced['favorite_commands'] = [
                cmd for cmd, count in 
                enhanced['command_frequency'].most_common(3)
            ]
        else:
            enhanced['favorite_commands'] = []
        
        # Calculate active hours
        if enhanced['time_patterns']:
            enhanced['active_hours'] = [
                int(hour) for hour, count in 
                sorted(enhanced['time_patterns'].items(), key=lambda x: x[1], reverse=True)[:3]
            ]
        else:
            enhanced['active_hours'] = []
        
        # Calculate preferred response style
        if enhanced['response_preferences']:
            preferred_style = max(enhanced['response_preferences'].items(), key=lambda x: x[1])
            enhanced['preferred_style'] = preferred_style[0]
        else:
            enhanced['preferred_style'] = 'balanced'
        
        # Calculate interaction frequency
        if enhanced['total_interactions'] > 0:
            created_at = datetime.fromisoformat(enhanced['created_at'])
            days_active = (datetime.now() - created_at).days or 1
            enhanced['interactions_per_day'] = enhanced['total_interactions'] / days_active
        else:
            enhanced['interactions_per_day'] = 0
        
        return enhanced
    
    def get_user_preference(self, user_id: str, preference_type: str) -> Any:
        """
        Get specific preference for a user
        
        Args:
            user_id: Unique user identifier
            preference_type: Type of preference to retrieve
            
        Returns:
            Preference value or None
        """
        profile = self.get_user_profile(user_id)
        return profile.get(preference_type)
    
    def update_personal_info(self, user_id: str, info_key: str, info_value: Any) -> bool:
        """
        Update personal information for a user
        
        Args:
            user_id: Unique user identifier
            info_key: Information key (e.g., 'name', 'location')
            info_value: Information value
            
        Returns:
            Success status
        """
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = self._create_new_user_profile()
        
        if 'personal_info' not in self.user_profiles[user_id]:
            self.user_profiles[user_id]['personal_info'] = {}
        
        self.user_profiles[user_id]['personal_info'][info_key] = info_value
        return self.save_preferences()
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get overall usage statistics across all users
        
        Returns:
            Usage statistics dictionary
        """
        total_users = len(self.user_profiles)
        total_interactions = sum(profile.get('total_interactions', 0) for profile in self.user_profiles.values())
        
        # Most popular topics across all users
        all_topics = Counter()
        for profile in self.user_profiles.values():
            all_topics.update(profile.get('topics', {}))
        
        popular_topics = all_topics.most_common(5)
        
        return {
            'total_users': total_users,
            'total_interactions': total_interactions,
            'popular_topics': dict(popular_topics),
            'average_interactions_per_user': total_interactions / total_users if total_users > 0 else 0
        }

# Utility function for easy integration
def get_preference_tracker() -> PreferenceTracker:
    """Get initialized preference tracker instance"""
    return PreferenceTracker()

# Test function
def test_preference_tracker():
    """Test the preference tracker functionality"""
    tracker = PreferenceTracker("test_user_preferences.json")
    
    # Test logging preferences
    test_user = "test_user_123"
    
    # Simulate user interactions
    interactions = [
        ('request_joke', {'interaction_style': 'humorous'}),
        ('ask_weather', {'topic': 'weather'}),
        ('tech_help', {'topic': 'tech', 'interaction_style': 'technical'}),
        ('request_joke', {'interaction_style': 'humorous'}),
        ('ask_weather', {'topic': 'weather'}),
    ]
    
    for action, metadata in interactions:
        tracker.log_preference(test_user, action, metadata)
    
    # Get user profile
    profile = tracker.get_user_profile(test_user)
    
    print("Preference Tracker Test Results:")
    print("=" * 50)
    print(f"User: {test_user}")
    print(f"Total Interactions: {profile['total_interactions']}")
    print(f"Favorite Topics: {profile.get('favorite_topics', [])}")
    print(f"Favorite Commands: {profile.get('favorite_commands', [])}")
    print(f"Preferred Style: {profile.get('preferred_style', 'unknown')}")
    print(f"Active Hours: {profile.get('active_hours', [])}")
    
    # Test usage statistics
    stats = tracker.get_usage_statistics()
    print(f"\nOverall Statistics:")
    print(f"Total Users: {stats['total_users']}")
    print(f"Total Interactions: {stats['total_interactions']}")
    print(f"Popular Topics: {stats['popular_topics']}")
    
    # Cleanup test file
    if os.path.exists("test_user_preferences.json"):
        os.remove("test_user_preferences.json")

if __name__ == "__main__":
    test_preference_tracker()