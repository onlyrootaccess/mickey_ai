# Hinglish NLP handling
"""
Mickey AI - Hinglish Processor
Handles Hindi-English mixed language processing with cultural context
"""

import logging
import re
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

class LanguageMix(Enum):
    HINDI_DOMINANT = "hindi_dominant"
    ENGLISH_DOMINANT = "english_dominant" 
    BALANCED_MIX = "balanced_mix"
    PREDOMINANTLY_ENGLISH = "predominantly_english"

class HinglishProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Hinglish to English mapping database
        self.hinglish_dict = self._load_hinglish_dictionary()
        
        # Cultural context and slang
        self.indian_slang = self._load_indian_slang()
        self.cultural_references = self._load_cultural_references()
        
        # Language patterns
        self.hindi_patterns = self._load_hindi_patterns()
        
        # Mickey's Hinglish personality
        self.hinglish_responses = self._load_hinglish_responses()
        
        self.logger.info("🕌 Hinglish Processor initialized - Desi style ready!")

    def _load_hinglish_dictionary(self) -> Dict[str, str]:
        """Load Hinglish to English dictionary"""
        return {
            # Common Hinglish words
            'yaar': 'friend',
            'bhai': 'brother',
            'behen': 'sister',
            'acha': 'okay',
            'theek': 'fine',
            'sahi': 'correct',
            'mast': 'awesome',
            'jhakaas': 'fantastic',
            'badiya': 'good',
            'kya': 'what',
            'kaise': 'how',
            'kab': 'when',
            'kyon': 'why',
            'kidhar': 'where',
            'kaun': 'who',
            'thoda': 'little',
            'bahut': 'very',
            'zyada': 'more',
            'kum': 'less',
            'abhi': 'now',
            'phir': 'then',
            'lekin': 'but',
            'magar': 'but',
            'kyuki': 'because',
            'toh': 'so',
            'fir': 'then',
            'hi': 'only',
            'bhi': 'also',
            'nahi': 'no',
            'haan': 'yes',
            'shukriya': 'thank you',
            'dhanyavaad': 'thank you',
            'chalo': 'let\'s go',
            'karo': 'do',
            'bolo': 'say',
            'dekho': 'see',
            'suno': 'listen',
            'lo': 'take',
            'de do': 'give',
            'le lo': 'take',
            'kar do': 'do it',
            'ho gaya': 'done',
            'kar raha': 'doing',
            'kar diya': 'did',
            'timepass': 'time pass',
            'jugaad': 'hack',
            'paisa': 'money',
            'tension': 'tension',
            'bas': 'enough',
            'arey': 'oh',
            'oye': 'hey',
            'chill': 'chill',
            'fun': 'fun',
            'maza': 'fun',
            'gussa': 'anger',
            'khush': 'happy',
            'udaas': 'sad',
            'thak': 'tired',
            'bhook': 'hunger',
            'pyas': 'thirst',
            'sone': 'sleep',
            'jagah': 'place',
            'ghar': 'home',
            'dost': 'friend',
            'pyaar': 'love',
            'dil': 'heart',
            'dimag': 'brain',
            'kaam': 'work',
            'naam': 'name',
            'shaam': 'evening',
            'subah': 'morning',
            'raat': 'night',
            'din': 'day'
        }

    def _load_indian_slang(self) -> Dict[str, str]:
        """Load Indian slang and colloquialisms"""
        return {
            'yaar': ['buddy', 'friend', 'mate'],
            'arre': ['oh', 'hey', 'wow'],
            'oye': ['hey', 'listen'],
            'wah': ['wow', 'great'],
            'shabaash': ['well done', 'good job'],
            'chill mar': ['chill out', 'relax'],
            'timepass': ['time pass', 'entertainment'],
            'masti': ['fun', 'enjoyment'],
            'josh': ['enthusiasm', 'energy'],
            'panga': ['trouble', 'problem'],
            'funda': ['concept', 'idea'],
            'bakwas': ['nonsense', 'rubbish']
        }

    def _load_cultural_references(self) -> Dict[str, List[str]]:
        """Load Indian cultural references"""
        return {
            'food': ['chai', 'samosa', 'biryani', 'butter chicken', 'paneer', 'gulab jamun'],
            'festivals': ['diwali', 'holi', 'eid', 'christmas', 'rakhi', 'navratri'],
            'bollywood': ['srk', 'salman', 'aamir', 'deepika', 'priyanka', 'karan johar'],
            'cities': ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad'],
            'sports': ['cricket', 'kabaddi', 'hockey', 'badminton'],
            'tv_shows': ['taarak mehta', 'shaktimaan', 'mahabharat', 'ramayan']
        }

    def _load_hindi_patterns(self) -> Dict[str, str]:
        """Load common Hindi sentence patterns"""
        return {
            r'kya.*hai': 'what is',
            r'kaise.*ho': 'how are you',
            r'kya.*kar.*rahe': 'what are you doing',
            r'kidhar.*ho': 'where are you',
            r'kab.*aoge': 'when will you come',
            r'kyon.*nahi': 'why not',
            r'mujhe.*chahiye': 'i want',
            r'main.*ja.*raha': 'i am going',
            r'tum.*kya.*kar.*rahe': 'what are you doing',
            r'bahut.*acha': 'very good'
        }

    def _load_hinglish_responses(self) -> Dict[str, List[str]]:
        """Load Mickey's Hinglish response templates"""
        return {
            'greeting': [
                "Namaste! Mickey bol raha hoon! 🐭",
                "Hello ji! Kaise ho?",
                "Hey there! Mickey here, aapko kya help chahiye?",
                "Hi friend! Mickey at your service! Kya scene hai?"
            ],
            'acknowledgment': [
                "Theek hai ji!",
                "Sahi pakde hain!",
                "Ho gaya!",
                "Kar diya!",
                "Mickey ne kar diya! 🎉"
            ],
            'confusion': [
                "Arre yaar, samjha nahi! Phir se bolo?",
                "Kya bol rahe ho bhai? Thoda clear bolo!",
                "Mickey confused hai! Can you repeat?",
                "Kya matlab? Thoda explain karo!"
            ],
            'excitement': [
                "Wah! Kya baat hai!",
                "Mast idea hai!",
                "Jhakaas! Mickey ko pasand aaya!",
                "Shabaash! Bahut badhiya!"
            ],
            'help': [
                "Chinta mat karo, Mickey hai na!",
                "Main hoon na yaar!",
                "Tension mat lo, Mickey help karega!",
                "Relax! Main sambhal leta hoon!"
            ]
        }

    def detect_language_mix(self, text: str) -> LanguageMix:
        """
        Detect the dominant language in Hinglish text
        """
        words = text.lower().split()
        hindi_word_count = 0
        english_word_count = 0
        
        for word in words:
            # Clean the word
            clean_word = re.sub(r'[^\w]', '', word)
            
            if clean_word in self.hinglish_dict:
                hindi_word_count += 1
            elif clean_word.isalpha():  # Assume it's English if it's alphabetic and not in Hindi dict
                english_word_count += 1
        
        total_words = len(words)
        if total_words == 0:
            return LanguageMix.PREDOMINANTLY_ENGLISH
        
        hindi_ratio = hindi_word_count / total_words
        english_ratio = english_word_count / total_words
        
        if hindi_ratio > 0.7:
            return LanguageMix.HINDI_DOMINANT
        elif english_ratio > 0.7:
            return LanguageMix.PREDOMINANTLY_ENGLISH
        elif hindi_ratio > english_ratio:
            return LanguageMix.BALANCED_MIX
        else:
            return LanguageMix.ENGLISH_DOMINANT

    def translate_hinglish_to_english(self, hinglish_text: str) -> str:
        """
        Translate Hinglish text to proper English
        """
        try:
            # Convert to lowercase for processing
            text = hinglish_text.lower()
            
            # Replace common Hinglish patterns
            for pattern, replacement in self.hindi_patterns.items():
                text = re.sub(pattern, replacement, text)
            
            # Replace individual words
            words = text.split()
            translated_words = []
            
            for word in words:
                clean_word = re.sub(r'[^\w]', '', word)
                
                if clean_word in self.hinglish_dict:
                    translated_word = self.hinglish_dict[clean_word]
                    # Preserve original capitalization and punctuation
                    if word[0].isupper():
                        translated_word = translated_word.capitalize()
                    translated_words.append(translated_word)
                else:
                    translated_words.append(word)
            
            # Reconstruct the sentence
            translated_text = ' '.join(translated_words)
            
            # Basic grammar fixes
            translated_text = self._fix_hinglish_grammar(translated_text)
            
            return translated_text.capitalize()
            
        except Exception as e:
            self.logger.error(f"Hinglish translation failed: {str(e)}")
            return hinglish_text  # Return original if translation fails

    def _fix_hinglish_grammar(self, text: str) -> str:
        """Fix common Hinglish grammar issues"""
        fixes = {
            'i am doing': 'i am',
            'you are doing': 'you are',
            'he is doing': 'he is',
            'she is doing': 'she is',
            'we are doing': 'we are',
            'they are doing': 'they are',
            'i want to': 'i want',
            'you want to': 'you want',
            'he want to': 'he wants',
            'she want to': 'she wants'
        }
        
        for wrong, correct in fixes.items():
            text = text.replace(wrong, correct)
        
        return text

    def extract_cultural_context(self, text: str) -> Dict[str, Any]:
        """
        Extract Indian cultural context from text
        """
        context = {
            'has_cultural_reference': False,
            'cultural_topics': [],
            'slang_used': [],
            'sentiment': 'neutral'
        }
        
        text_lower = text.lower()
        
        # Check for cultural references
        for category, items in self.cultural_references.items():
            for item in items:
                if item in text_lower:
                    context['has_cultural_reference'] = True
                    context['cultural_topics'].append({
                        'category': category,
                        'topic': item
                    })
        
        # Check for slang
        for slang in self.indian_slang.keys():
            if slang in text_lower:
                context['slang_used'].append(slang)
        
        # Basic sentiment analysis for Hinglish
        positive_words = ['mast', 'jhakaas', 'badiya', 'sahi', 'acha', 'khush', 'shabaash', 'wah']
        negative_words = ['bakwas', 'gussa', 'udaas', 'thak', 'tension', 'nahi', 'mat']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            context['sentiment'] = 'positive'
        elif negative_count > positive_count:
            context['sentiment'] = 'negative'
        
        return context

    def generate_hinglish_response(self, english_text: str, style: str = 'balanced') -> str:
        """
        Generate Hinglish response from English text
        """
        try:
            words = english_text.lower().split()
            hinglish_words = []
            
            # Convert some English words to Hinglish based on style
            conversion_rate = 0.3 if style == 'balanced' else 0.5 if style == 'desi' else 0.1
            
            for word in words:
                # Check if we should convert this word
                if random.random() < conversion_rate:
                    # Find Hindi equivalent if available
                    hindi_equivalent = self._find_hindi_equivalent(word)
                    if hindi_equivalent:
                        hinglish_words.append(hindi_equivalent)
                        continue
                
                hinglish_words.append(word)
            
            response = ' '.join(hinglish_words).capitalize()
            
            # Add Mickey's Hinglish flair
            response = self._add_mickey_hinglish_touch(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Hinglish response generation failed: {str(e)}")
            return english_text

    def _find_hindi_equivalent(self, english_word: str) -> Optional[str]:
        """Find Hindi equivalent for English word"""
        reverse_dict = {v: k for k, v in self.hinglish_dict.items()}
        return reverse_dict.get(english_word.lower())

    def _add_mickey_hinglish_touch(self, text: str) -> str:
        """Add Mickey's personality to Hinglish responses"""
        # Add common Hinglish expressions
        expressions = ['ji', 'yaar', 'bhai', 'arre', 'oye']
        
        if random.random() < 0.4:  # 40% chance to add expression
            expression = random.choice(expressions)
            if random.random() < 0.5:
                text = f"{expression} {text}"
            else:
                text = f"{text} {expression}"
        
        # Add Mickey's signature
        mickey_signatures = [
            " - Mickey",
            " 🐭",
            " | Your friend Mickey",
            " | Mickey bol raha hoon!"
        ]
        
        if random.random() < 0.3:  # 30% chance to add signature
            text += random.choice(mickey_signatures)
        
        return text

    def get_hinglish_response(self, response_type: str) -> str:
        """Get pre-defined Hinglish response"""
        responses = self.hinglish_responses.get(response_type, ['Okay', 'Theek hai'])
        return random.choice(responses)

    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Comprehensive Hinglish processing for user input
        """
        try:
            # Detect language mix
            language_mix = self.detect_language_mix(user_input)
            
            # Translate to English for processing
            english_translation = self.translate_hinglish_to_english(user_input)
            
            # Extract cultural context
            cultural_context = self.extract_cultural_context(user_input)
            
            # Generate appropriate response style
            response_style = 'balanced'
            if language_mix == LanguageMix.HINDI_DOMINANT:
                response_style = 'desi'
            elif language_mix == LanguageMix.PREDOMINANTLY_ENGLISH:
                response_style = 'light'
            
            return {
                'original_input': user_input,
                'language_mix': language_mix.value,
                'english_translation': english_translation,
                'cultural_context': cultural_context,
                'recommended_response_style': response_style,
                'processing_success': True
            }
            
        except Exception as e:
            self.logger.error(f"Hinglish processing failed: {str(e)}")
            return {
                'original_input': user_input,
                'language_mix': 'unknown',
                'english_translation': user_input,
                'cultural_context': {},
                'recommended_response_style': 'balanced',
                'processing_success': False,
                'error': str(e)
            }

# Test function
def test_hinglish_processor():
    """Test the Hinglish processor with various inputs"""
    processor = HinglishProcessor()
    
    test_inputs = [
        "Hello yaar, kaise ho?",
        "Mujhe weather ki jaankari chahiye",
        "Kya aap muje help kar sakte ho?",
        "Mast idea hai bhai!",
        "I want to know about cricket scores",
        "Arre Mickey, time kya hai?",
        "Tension mat lo, main hoon na!",
        "What is the meaning of jugaad?"
    ]
    
    for user_input in test_inputs:
        result = processor.process_user_input(user_input)
        print(f"Input: '{user_input}'")
        print(f"Language Mix: {result['language_mix']}")
        print(f"English: {result['english_translation']}")
        print(f"Cultural Context: {result['cultural_context']}")
        
        # Generate Hinglish response
        english_response = f"I understand: {result['english_translation']}"
        hinglish_response = processor.generate_hinglish_response(
            english_response, 
            result['recommended_response_style']
        )
        print(f"Mickey's Response: {hinglish_response}")
        print("---")

if __name__ == "__main__":
    test_hinglish_processor()