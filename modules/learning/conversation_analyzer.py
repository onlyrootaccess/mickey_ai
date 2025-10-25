# Dialog pattern recognition
"""
Conversation Analyzer Module for Mickey AI
Analyzes dialog patterns, topics, sentiment trends, and engagement metrics
"""

import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
import logging

try:
    from textblob import TextBlob
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK/TextBlob not available - using fallback methods")

# Configure logging
logger = logging.getLogger(__name__)

class ConversationAnalyzer:
    """
    Analyzes conversation patterns, topics, sentiment, and engagement metrics
    """
    
    def __init__(self, data_file: str = "data/conversation_history.json"):
        self.data_file = data_file
        self._ensure_data_directory()
        
        # Topic categories and keywords
        self.topic_categories = {
            'technology': ['code', 'programming', 'python', 'java', 'bug', 'error', 'computer', 'software', 'app', 'website'],
            'weather': ['weather', 'temperature', 'rain', 'sunny', 'forecast', 'humidity', 'season'],
            'entertainment': ['movie', 'music', 'game', 'video', 'song', 'film', 'netflix', 'youtube'],
            'news': ['news', 'headlines', 'current', 'events', 'politics', 'world'],
            'personal': ['how are you', 'feeling', 'family', 'friends', 'relationship', 'personal'],
            'humor': ['joke', 'funny', 'laugh', 'humor', 'comedy', 'hilarious'],
            'productivity': ['reminder', 'task', 'schedule', 'calendar', 'meeting', 'work', 'project'],
            'education': ['learn', 'study', 'teach', 'course', 'tutorial', 'knowledge', 'explain'],
            'food': ['food', 'restaurant', 'cooking', 'recipe', 'meal', 'dinner', 'lunch'],
            'travel': ['travel', 'vacation', 'trip', 'flight', 'hotel', 'destination']
        }
        
        # Engagement indicators
        self.engagement_indicators = {
            'high_engagement': ['?', '!', 'tell me more', 'explain', 'why', 'how', 'what if'],
            'low_engagement': ['ok', 'thanks', 'bye', 'goodbye', 'see you', 'got it']
        }
        
        # Hinglish specific patterns
        self.hinglish_patterns = {
            'greetings': ['namaste', 'kaise ho', 'kya haal', 'suna', 'batao'],
            'agreement': ['theek hai', 'accha', 'sahi hai', 'bilkul', 'haan'],
            'disagreement': ['nahi', 'galat', 'aisa nahi', 'kyun'],
            'surprise': ['arey', 'wah', 'sach mein', 'accha'],
            'confusion': ['samjha nahi', 'phir se bolo', 'kya matlab']
        }
        
        self.conversation_history = []
        self.load_conversation_history()
        
        logger.info("ConversationAnalyzer initialized")
    
    def _ensure_data_directory(self) -> None:
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
    
    def load_conversation_history(self) -> None:
        """Load conversation history from JSON file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversation_history = data.get('conversations', [])
                logger.debug(f"Loaded {len(self.conversation_history)} conversations from history")
            else:
                logger.info("No existing conversation history found")
                self.conversation_history = []
                
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
            self.conversation_history = []
    
    def save_conversation_history(self) -> bool:
        """Save conversation history to JSON file"""
        try:
            # Keep only last 1000 conversations to prevent file from growing too large
            if len(self.conversation_history) > 1000:
                self.conversation_history = self.conversation_history[-1000:]
            
            data = {
                'conversations': self.conversation_history,
                'last_updated': datetime.now().isoformat(),
                'total_sessions': len(self.conversation_history)
            }
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug("Conversation history saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")
            return False
    
    def add_conversation(self, user_message: str, ai_response: str, user_id: str = "default", 
                        metadata: Optional[Dict] = None) -> None:
        """
        Add a new conversation to history
        
        Args:
            user_message: User's input message
            ai_response: Mickey's response
            user_id: User identifier
            metadata: Additional conversation metadata
        """
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_message': user_message,
            'ai_response': ai_response,
            'metadata': metadata or {},
            'analysis': self._analyze_single_conversation(user_message, ai_response)
        }
        
        self.conversation_history.append(conversation)
        
        # Save periodically (every 10 conversations)
        if len(self.conversation_history) % 10 == 0:
            self.save_conversation_history()
    
    def _analyze_single_conversation(self, user_message: str, ai_response: str) -> Dict[str, Any]:
        """
        Analyze a single conversation turn
        
        Args:
            user_message: User's message
            ai_response: AI's response
            
        Returns:
            Analysis results dictionary
        """
        analysis = {
            'user_sentiment': self._analyze_sentiment(user_message),
            'message_length': len(user_message),
            'topics': self._extract_topics(user_message),
            'engagement_score': self._calculate_engagement(user_message),
            'hinglish_elements': self._detect_hinglish(user_message),
            'question_count': self._count_questions(user_message),
            'response_time_estimate': len(ai_response) / 50  # Rough typing time estimate
        }
        
        return analysis
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using TextBlob or fallback method
        
        Args:
            text: Input text
            
        Returns:
            Sentiment score between -1.0 (negative) and 1.0 (positive)
        """
        if not text:
            return 0.0
        
        if NLTK_AVAILABLE:
            try:
                blob = TextBlob(text)
                return blob.sentiment.polarity
            except Exception as e:
                logger.warning(f"TextBlob sentiment analysis failed: {e}")
        
        # Fallback sentiment analysis using keyword matching
        return self._fallback_sentiment_analysis(text)
    
    def _fallback_sentiment_analysis(self, text: str) -> float:
        """Fallback sentiment analysis using keyword matching"""
        positive_words = ['good', 'great', 'excellent', 'awesome', 'amazing', 'happy', 'thanks', 'love', 'nice', 'wonderful',
                         'badiya', 'mast', 'shandaar', 'kamaal', 'wah']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'angry', 'sad', 'frustrated', 'problem', 'error', 'bug',
                         'bekar', 'kharab', 'problem', 'dikkat']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract topics from text using keyword matching
        
        Args:
            text: Input text
            
        Returns:
            List of detected topics
        """
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in self.topic_categories.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics
    
    def _calculate_engagement(self, text: str) -> float:
        """
        Calculate engagement score for user message
        
        Args:
            text: User message
            
        Returns:
            Engagement score between 0.0 (low) and 1.0 (high)
        """
        score = 0.5  # Base score
        
        # Length factor
        if len(text) > 50:
            score += 0.2
        elif len(text) < 10:
            score -= 0.2
        
        # Question factor
        question_count = self._count_questions(text)
        score += min(0.3, question_count * 0.1)
        
        # Engagement indicators
        text_lower = text.lower()
        high_engagement_terms = sum(1 for term in self.engagement_indicators['high_engagement'] if term in text_lower)
        low_engagement_terms = sum(1 for term in self.engagement_indicators['low_engagement'] if term in text_lower)
        
        score += high_engagement_terms * 0.05
        score -= low_engagement_terms * 0.1
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))
    
    def _detect_hinglish(self, text: str) -> Dict[str, Any]:
        """
        Detect Hinglish elements in text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of Hinglish analysis
        """
        analysis = {
            'contains_hinglish': False,
            'patterns_found': [],
            'hinglish_ratio': 0.0
        }
        
        hinglish_terms = []
        for category, terms in self.hinglish_patterns.items():
            found_terms = [term for term in terms if term in text.lower()]
            if found_terms:
                analysis['patterns_found'].append(category)
                hinglish_terms.extend(found_terms)
        
        # Calculate Hinglish ratio (rough estimate)
        words = text.lower().split()
        if words:
            analysis['hinglish_ratio'] = len(hinglish_terms) / len(words)
            analysis['contains_hinglish'] = len(hinglish_terms) > 0
        
        return analysis
    
    def _count_questions(self, text: str) -> int:
        """Count number of questions in text"""
        question_indicators = ['?', 'what', 'why', 'how', 'when', 'where', 'who', 'which', 'can you', 'could you', 'kya', 'kaise', 'kab']
        text_lower = text.lower()
        return sum(1 for indicator in question_indicators if indicator in text_lower)
    
    def analyze_session(self, history: List[Dict]) -> Dict[str, Any]:
        """
        Analyze a conversation session for patterns and insights
        
        Args:
            history: List of conversation messages
            
        Returns:
            Analysis results dictionary
        """
        if not history:
            return {
                'topics': [],
                'engagement': 0.0,
                'sentiment_trend': 'neutral',
                'session_duration': 0,
                'message_count': 0,
                'average_response_length': 0
            }
        
        # Extract analysis from history
        analyses = [conv.get('analysis', {}) for conv in history if 'analysis' in conv]
        
        if not analyses:
            # Analyze on the fly if no pre-computed analysis
            analyses = []
            for conv in history:
                analysis = self._analyze_single_conversation(
                    conv.get('user_message', ''), 
                    conv.get('ai_response', '')
                )
                analyses.append(analysis)
        
        # Calculate session metrics
        topics = Counter()
        sentiment_scores = []
        engagement_scores = []
        message_lengths = []
        
        for analysis in analyses:
            topics.update(analysis.get('topics', []))
            sentiment_scores.append(analysis.get('user_sentiment', 0.0))
            engagement_scores.append(analysis.get('engagement_score', 0.5))
            message_lengths.append(analysis.get('message_length', 0))
        
        # Calculate session duration
        if len(history) > 1:
            first_time = datetime.fromisoformat(history[0]['timestamp'])
            last_time = datetime.fromisoformat(history[-1]['timestamp'])
            session_duration = (last_time - first_time).total_seconds()
        else:
            session_duration = 0
        
        # Determine sentiment trend
        if len(sentiment_scores) >= 3:
            first_half = sum(sentiment_scores[:len(sentiment_scores)//2]) / (len(sentiment_scores)//2)
            second_half = sum(sentiment_scores[len(sentiment_scores)//2:]) / (len(sentiment_scores) - len(sentiment_scores)//2)
            
            if second_half > first_half + 0.1:
                sentiment_trend = 'improving'
            elif second_half < first_half - 0.1:
                sentiment_trend = 'declining'
            else:
                sentiment_trend = 'stable'
        else:
            sentiment_trend = 'neutral'
        
        return {
            'topics': [topic for topic, count in topics.most_common(5)],
            'engagement': sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0.0,
            'sentiment_trend': sentiment_trend,
            'average_sentiment': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0,
            'session_duration': session_duration,
            'message_count': len(history),
            'average_response_length': sum(message_lengths) / len(message_lengths) if message_lengths else 0,
            'dominant_topic': topics.most_common(1)[0][0] if topics else 'general',
            'question_ratio': sum(1 for analysis in analyses if analysis.get('question_count', 0) > 0) / len(analyses)
        }
    
    def get_conversation_patterns(self, user_id: str = None, days_back: int = 30) -> Dict[str, Any]:
        """
        Get conversation patterns for a user or all users
        
        Args:
            user_id: Specific user ID or None for all users
            days_back: Number of days to look back
            
        Returns:
            Conversation patterns dictionary
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Filter conversations
        if user_id:
            conversations = [
                conv for conv in self.conversation_history 
                if conv.get('user_id') == user_id and 
                datetime.fromisoformat(conv['timestamp']) >= cutoff_date
            ]
        else:
            conversations = [
                conv for conv in self.conversation_history 
                if datetime.fromisoformat(conv['timestamp']) >= cutoff_date
            ]
        
        if not conversations:
            return {'error': 'No conversations found in the specified period'}
        
        # Analyze patterns
        time_patterns = defaultdict(int)
        topic_frequency = Counter()
        sentiment_by_hour = defaultdict(list)
        
        for conv in conversations:
            # Time patterns
            hour = datetime.fromisoformat(conv['timestamp']).hour
            time_patterns[hour] += 1
            
            # Topic frequency
            analysis = conv.get('analysis', {})
            topic_frequency.update(analysis.get('topics', []))
            
            # Sentiment by hour
            sentiment = analysis.get('user_sentiment', 0.0)
            sentiment_by_hour[hour].append(sentiment)
        
        # Calculate average sentiment by hour
        avg_sentiment_by_hour = {}
        for hour, sentiments in sentiment_by_hour.items():
            avg_sentiment_by_hour[hour] = sum(sentiments) / len(sentiments)
        
        return {
            'total_conversations': len(conversations),
            'active_hours': dict(sorted(time_patterns.items(), key=lambda x: x[1], reverse=True)[:5]),
            'popular_topics': dict(topic_frequency.most_common(5)),
            'avg_sentiment_by_hour': avg_sentiment_by_hour,
            'most_engaged_hour': max(time_patterns.items(), key=lambda x: x[1])[0] if time_patterns else None,
            'conversations_per_day': len(conversations) / days_back
        }
    
    def get_user_engagement_metrics(self, user_id: str) -> Dict[str, Any]:
        """
        Get engagement metrics for a specific user
        
        Args:
            user_id: User identifier
            
        Returns:
            Engagement metrics dictionary
        """
        user_conversations = [
            conv for conv in self.conversation_history 
            if conv.get('user_id') == user_id
        ]
        
        if not user_conversations:
            return {'error': 'No conversations found for user'}
        
        analyses = [conv.get('analysis', {}) for conv in user_conversations]
        engagement_scores = [analysis.get('engagement_score', 0.5) for analysis in analyses]
        sentiment_scores = [analysis.get('user_sentiment', 0.0) for analysis in analyses]
        
        return {
            'total_interactions': len(user_conversations),
            'average_engagement': sum(engagement_scores) / len(engagement_scores),
            'average_sentiment': sum(sentiment_scores) / len(sentiment_scores),
            'engagement_trend': 'increasing' if len(engagement_scores) > 5 and engagement_scores[-1] > engagement_scores[0] else 'stable',
            'preferred_topics': Counter([topic for analysis in analyses for topic in analysis.get('topics', [])]).most_common(3),
            'last_active': max(datetime.fromisoformat(conv['timestamp']) for conv in user_conversations).isoformat()
        }

# Utility function for easy integration
def get_conversation_analyzer() -> ConversationAnalyzer:
    """Get initialized conversation analyzer instance"""
    return ConversationAnalyzer()

# Test function
def test_conversation_analyzer():
    """Test the conversation analyzer functionality"""
    analyzer = ConversationAnalyzer("test_conversation_history.json")
    
    # Test adding conversations
    test_conversations = [
        ("Hello Mickey, how are you today?", "I'm doing great! How can I help you?", "user123"),
        ("Can you tell me a joke?", "Why don't scientists trust atoms? Because they make up everything!", "user123"),
        ("That was funny! What's the weather like?", "I can check the weather for you. What's your location?", "user123"),
        ("I'm in Delhi", "It's sunny in Delhi with a high of 32°C. Perfect weather!", "user123"),
        ("Thanks Mickey, you're awesome!", "You're welcome! Always happy to help.", "user123"),
    ]
    
    for user_msg, ai_resp, user_id in test_conversations:
        analyzer.add_conversation(user_msg, ai_resp, user_id)
    
    # Test session analysis
    session_analysis = analyzer.analyze_session(analyzer.conversation_history)
    
    print("Conversation Analyzer Test Results:")
    print("=" * 50)
    print(f"Topics: {session_analysis['topics']}")
    print(f"Engagement: {session_analysis['engagement']:.2f}")
    print(f"Sentiment Trend: {session_analysis['sentiment_trend']}")
    print(f"Message Count: {session_analysis['message_count']}")
    print(f"Dominant Topic: {session_analysis['dominant_topic']}")
    
    # Test pattern analysis
    patterns = analyzer.get_conversation_patterns("user123", 7)
    print(f"\nUser Patterns:")
    print(f"Active Hours: {patterns['active_hours']}")
    print(f"Popular Topics: {patterns['popular_topics']}")
    
    # Test engagement metrics
    engagement = analyzer.get_user_engagement_metrics("user123")
    print(f"\nEngagement Metrics:")
    print(f"Average Engagement: {engagement['average_engagement']:.2f}")
    print(f"Preferred Topics: {engagement['preferred_topics']}")
    
    # Cleanup test file
    if os.path.exists("test_conversation_history.json"):
        os.remove("test_conversation_history.json")

if __name__ == "__main__":
    test_conversation_analyzer()