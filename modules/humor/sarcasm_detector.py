# Tone analysis for sarcasm triggers
"""
Sarcasm Detector Module for Mickey AI
Detects sarcasm in user input using sentiment analysis and pattern matching
with Hinglish support
"""

import re
from typing import Tuple, Dict, List
from textblob import TextBlob
import logging

# Configure logging
logger = logging.getLogger(__name__)

class SarcasmDetector:
    """
    Detects sarcastic intent in text using multiple approaches:
    - Sentiment analysis (TextBlob)
    - Pattern matching for sarcasm triggers
    - Hinglish-specific markers
    - Contextual clues
    """
    
    def __init__(self):
        self.sarcasm_indicators = {
            # English sarcasm markers
            'english': [
                r'\b(yeah|sure|right|of course|obviously|clearly)\b.*\?',
                r'\b(as if|in your dreams|dream on)\b',
                r'\b(wow|amazing|brilliant|fantastic)\b.*\!+',
                r'\b(I love how|I really enjoy)\b.*\b(but|actually|never)\b',
                r'\b(tell me more|please continue)\b',
                r'\b(that was so|that is totally)\b.*\b(not|never)\b',
                r'\b(sarcasm|sarcastic)\b',
                r'\b(slow clap|standing ovation)\b'
            ],
            
            # Hinglish sarcasm markers  
            'hinglish': [
                r'\b(haan|theek hai|sach mein|waah|kya baat hai)\b.*\?',
                r'\b(badiya|shabaash|maza aa gaya)\b',
                r'\b(zyada ho gaya|abey saale|arey waah)\b',
                r'\b(accha ji|thik hai bhai)\b',
                r'\b(kya genius hai|kya idea hai)\b',
                r'\s+(yaar|boss|bhai)\s+.*\?$',
                r'\b(aisa kyun|kaise soch liya)\b',
                r'\b(masterstroke|genius level)\b',
                r'\b(jhakaas|baut heavy)\b.*\!'
            ],
            
            # Universal sarcasm patterns
            'universal': [
                r'(\?|\!){2,}',  # Multiple punctuation
                r'[A-Z]{3,}',    # ALL CAPS
                r'😂+|🤣+',      # Excessive emojis
                r'\*slow clap\*|\*facepalm\*',  # Action descriptions
            ]
        }
        
        # Positive words that might indicate sarcasm when overused
        self.overly_positive = [
            'wonderful', 'perfect', 'amazing', 'brilliant', 'fantastic',
            'awesome', 'great', 'excellent', 'outstanding', 'superb',
            'badiya', 'mast', 'jhakaas', 'awesome', 'wah'
        ]
        
        self.confidence_threshold = 0.7
        logger.info("SarcasmDetector initialized")
    
    def detect_sarcasm(self, text: str) -> Tuple[bool, float]:
        """
        Main function to detect sarcasm in input text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (is_sarcastic: bool, confidence: float)
        """
        if not text or len(text.strip()) < 3:
            return False, 0.0
        
        text = text.strip()
        logger.debug(f"Analyzing text for sarcasm: {text}")
        
        # Calculate individual scores
        pattern_score = self._pattern_analysis(text)
        sentiment_score = self._sentiment_analysis(text)
        linguistic_score = self._linguistic_analysis(text)
        
        # Combine scores with weights
        final_confidence = (
            pattern_score * 0.5 +
            sentiment_score * 0.3 + 
            linguistic_score * 0.2
        )
        
        is_sarcastic = final_confidence >= self.confidence_threshold
        
        logger.debug(f"Sarcasm detection: {is_sarcastic} (confidence: {final_confidence:.2f})")
        return is_sarcastic, final_confidence
    
    def _pattern_analysis(self, text: str) -> float:
        """
        Analyze text for sarcasm patterns using regex
        
        Args:
            text: Input text
            
        Returns:
            Pattern match score (0.0-1.0)
        """
        score = 0.0
        matches_found = 0
        total_patterns = 0
        
        for category, patterns in self.sarcasm_indicators.items():
            for pattern in patterns:
                total_patterns += 1
                if re.search(pattern, text, re.IGNORECASE):
                    matches_found += 1
                    # Different weights for different categories
                    if category == 'hinglish':
                        score += 0.8  # Hinglish patterns are strong indicators
                    elif category == 'english':
                        score += 0.7
                    else:
                        score += 0.6
        
        # Normalize score
        if total_patterns > 0:
            pattern_score = min(1.0, score / 3.0)  # Divide by max possible category weight
        else:
            pattern_score = 0.0
            
        return pattern_score
    
    def _sentiment_analysis(self, text: str) -> float:
        """
        Analyze sentiment contradictions that might indicate sarcasm
        
        Args:
            text: Input text
            
        Returns:
            Sentiment-based sarcasm score (0.0-1.0)
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            sarcasm_score = 0.0
            
            # High positivity with low subjectivity might indicate sarcasm
            if polarity > 0.6 and subjectivity < 0.3:
                sarcasm_score += 0.7
            
            # Negative sentiment with positive words is suspicious
            if polarity < -0.2:
                positive_word_count = sum(1 for word in self.overly_positive 
                                        if word.lower() in text.lower())
                if positive_word_count > 0:
                    sarcasm_score += 0.6
            
            # Check for sentiment mismatch with punctuation
            if polarity > 0.3 and ('?' in text or '!' in text):
                sarcasm_score += 0.4
            
            return min(1.0, sarcasm_score)
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return 0.0
    
    def _linguistic_analysis(self, text: str) -> float:
        """
        Analyze linguistic features for sarcasm indicators
        
        Args:
            text: Input text
            
        Returns:
            Linguistic analysis score (0.0-1.0)
        """
        score = 0.0
        
        # Check for excessive punctuation
        if text.count('!') + text.count('?') > 2:
            score += 0.6
        
        # Check for ALL CAPS words
        if re.search(r'\b[A-Z]{4,}\b', text):
            score += 0.5
        
        # Check for excessive positive adjectives
        positive_adjectives = sum(1 for word in self.overly_positive 
                                if word.lower() in text.lower())
        if positive_adjectives >= 2:
            score += 0.7
        
        # Check for contrastive conjunctions (but, however, etc.)
        contrast_words = ['but', 'however', 'although', 'though', 'yet']
        if any(word in text.lower() for word in contrast_words):
            score += 0.4
        
        # Hinglish specific: Check for mixed language with exaggerated praise
        hinglish_praise = ['kya baat hai', 'badiya', 'maza aa gaya', 'wah wah']
        if any(phrase in text.lower() for phrase in hinglish_praise):
            score += 0.5
        
        return min(1.0, score)
    
    def analyze_conversation_context(self, current_text: str, previous_text: str = "") -> float:
        """
        Analyze sarcasm in context of conversation history
        
        Args:
            current_text: Current message
            previous_text: Previous message in conversation
            
        Returns:
            Contextual sarcasm score (0.0-1.0)
        """
        if not previous_text:
            return 0.0
        
        context_score = 0.0
        
        # If previous message was a question and current is overly positive
        if previous_text.strip().endswith('?') and any(word in current_text.lower() 
                                                     for word in self.overly_positive):
            context_score += 0.8
        
        # If there's a sentiment shift from negative to overly positive
        try:
            prev_blob = TextBlob(previous_text)
            curr_blob = TextBlob(current_text)
            
            if prev_blob.sentiment.polarity < -0.3 and curr_blob.sentiment.polarity > 0.6:
                context_score += 0.7
        except:
            pass
        
        return min(1.0, context_score)

# Integration with Hinglish Processor
try:
    from modules.humor.hinglish_processor import HinglishProcessor
except ImportError:
    HinglishProcessor = None
    logger.warning("HinglishProcessor not available - proceeding without Hinglish support")

class EnhancedSarcasmDetector(SarcasmDetector):
    """
    Enhanced detector with Hinglish processing integration
    """
    
    def __init__(self):
        super().__init__()
        self.hinglish_processor = None
        if HinglishProcessor:
            try:
                self.hinglish_processor = HinglishProcessor()
            except:
                logger.warning("Failed to initialize HinglishProcessor")
    
    def detect_sarcasm_with_context(self, text: str, conversation_history: List[str] = None) -> Tuple[bool, float]:
        """
        Detect sarcasm with conversation context
        
        Args:
            text: Current text
            conversation_history: List of previous messages
            
        Returns:
            Tuple of (is_sarcastic: bool, confidence: float)
        """
        base_sarcastic, base_confidence = self.detect_sarcasm(text)
        
        # Add context if available
        if conversation_history and len(conversation_history) > 0:
            context_score = self.analyze_conversation_context(text, conversation_history[-1])
            enhanced_confidence = min(1.0, base_confidence + (context_score * 0.3))
            return base_sarcastic or enhanced_confidence > self.confidence_threshold, enhanced_confidence
        
        return base_sarcastic, base_confidence

# Utility function for easy integration
def get_sarcasm_detector() -> SarcasmDetector:
    """Get initialized sarcasm detector instance"""
    return EnhancedSarcasmDetector()

# Test function
def test_sarcasm_detector():
    """Test the sarcasm detector with various examples"""
    detector = EnhancedSarcasmDetector()
    
    test_cases = [
        # (text, expected_sarcastic)
        ("Oh great, another meeting. That's exactly what I wanted!", True),
        ("Wow, you're so smart!", True),
        ("I love waiting in traffic for hours!", True),
        ("This is a wonderful day.", False),
        ("Haan bilkul, tum genius ho yaar!", True),
        ("Sach mein? Tune yeh socha kaise?", True),
        ("Kya baat hai, tumhara code phir se fail ho gaya!", True),
        ("The weather is nice today.", False),
        ("Please help me with this problem.", False),
        ("Badiya! Ek aur bug mil gaya!", True),
    ]
    
    print("Testing Sarcasm Detector:")
    print("-" * 50)
    
    for text, expected in test_cases:
        is_sarcastic, confidence = detector.detect_sarcasm(text)
        status = "✓" if is_sarcastic == expected else "✗"
        print(f"{status} '{text}'")
        print(f"   Sarcastic: {is_sarcastic} (confidence: {confidence:.2f})")
        print(f"   Expected: {expected}")
        print()

if __name__ == "__main__":
    test_sarcasm_detector()