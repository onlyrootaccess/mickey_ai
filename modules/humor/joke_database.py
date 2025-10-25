# Curated joke collection
"""
Joke Database Module for Mickey AI
Handles curated joke collection with categories and search functionality
"""

import json
import os
import random
from typing import List, Dict, Optional, Union

class JokeDatabase:
    """
    Manages Mickey's joke collection with categories, search, and persistence
    """
    
    def __init__(self, data_file: str = "data/humor_database.json"):
        self.data_file = data_file
        self.jokes = []
        self.categories = ["puns", "hinglish", "tech", "dad_jokes", "sarcastic"]
        self._ensure_data_directory()
        self.load_jokes()
        
        # Initialize with sample jokes if database is empty
        if not self.jokes:
            self._initialize_sample_jokes()
    
    def _ensure_data_directory(self) -> None:
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
    
    def _initialize_sample_jokes(self) -> None:
        """Initialize with 20-30 diverse sample jokes reflecting Mickey's personality"""
        sample_jokes = [
            {
                "id": 1,
                "category": "tech",
                "text": "Why do Java developers wear glasses? Because they can't C#!",
                "keywords": ["programming", "java", "c#"],
                "rating": 0.8
            },
            {
                "id": 2,
                "category": "hinglish",
                "text": "Mere computer ne mujhse kaha: 'Yaar, tumhara browsing history dekh kar toh FBI waale bhi shock ho jaayenge!'",
                "keywords": ["browsing", "history", "fbi", "shock"],
                "rating": 0.9
            },
            {
                "id": 3,
                "category": "puns",
                "text": "I'm reading a book about anti-gravity. It's impossible to put down!",
                "keywords": ["book", "gravity", "reading"],
                "rating": 0.7
            },
            {
                "id": 4,
                "category": "hinglish",
                "text": "AI assistant bola: 'Sir, aapki voice command samajh nahi aayi... Shayad aapko English bolna seekhna padega!'",
                "keywords": ["voice", "command", "english", "learn"],
                "rating": 0.85
            },
            {
                "id": 5,
                "category": "dad_jokes",
                "text": "Why don't scientists trust atoms? Because they make up everything!",
                "keywords": ["science", "atoms", "trust"],
                "rating": 0.6
            },
            {
                "id": 6,
                "category": "tech",
                "text": "There are only 10 types of people in the world: those who understand binary and those who don't.",
                "keywords": ["binary", "programming", "world"],
                "rating": 0.9
            },
            {
                "id": 7,
                "category": "sarcastic",
                "text": "Oh, you're working from home? So you're basically getting paid to watch Netflix and occasionally reply to emails.",
                "keywords": ["work", "home", "netflix", "emails"],
                "rating": 0.75
            },
            {
                "id": 8,
                "category": "hinglish",
                "text": "Python programmer ne girlfriend ko propose kiya: 'Will you be the exception to my heart?' Girlfriend: 'Sorry, I only date Java guys!'",
                "keywords": ["python", "java", "programmer", "girlfriend"],
                "rating": 0.88
            },
            {
                "id": 9,
                "category": "puns",
                "text": "I told my wife she was drawing her eyebrows too high. She looked surprised.",
                "keywords": ["wife", "eyebrows", "surprised"],
                "rating": 0.65
            },
            {
                "id": 10,
                "category": "tech",
                "text": "Why do programmers prefer dark mode? Because light attracts bugs!",
                "keywords": ["programmers", "dark mode", "bugs"],
                "rating": 0.8
            },
            {
                "id": 11,
                "category": "hinglish",
                "text": "Mickey AI ka naya feature: 'Sarcasm Mode'. Ab main bhi tumhare jaise baat kar sakta hoon... AS IF!",
                "keywords": ["sarcasm", "mode", "feature", "mickey"],
                "rating": 0.92
            },
            {
                "id": 12,
                "category": "dad_jokes",
                "text": "What do you call a fake noodle? An impasta!",
                "keywords": ["noodle", "pasta", "food"],
                "rating": 0.55
            },
            {
                "id": 13,
                "category": "sarcastic", 
                "text": "Your code is so well-commented... said no one ever looking at your repository.",
                "keywords": ["code", "comments", "repository"],
                "rating": 0.78
            },
            {
                "id": 14,
                "category": "hinglish",
                "text": "User: 'Mickey, mera WiFi slow kyun hai?' Mickey: 'Sir, router ko thoda rest do, din bhar kaam karke thak gaya hoga!'",
                "keywords": ["wifi", "slow", "router", "rest"],
                "rating": 0.87
            },
            {
                "id": 15,
                "category": "tech",
                "text": "Debugging: Being the detective in a crime movie where you are also the murderer.",
                "keywords": ["debugging", "detective", "code"],
                "rating": 0.85
            },
            {
                "id": 16, 
                "category": "puns",
                "text": "I used to be a baker because I kneaded dough.",
                "keywords": ["baker", "dough", "kneaded"],
                "rating": 0.6
            },
            {
                "id": 17,
                "category": "hinglish",
                "text": "AI assistant life lesson: 'Beta, jaisa code vaisa output... Garbage in, garbage out!'",
                "keywords": ["life", "lesson", "code", "garbage"],
                "rating": 0.89
            },
            {
                "id": 18,
                "category": "sarcastic",
                "text": "I love when people use 'urgent' in email subject lines. It really makes me want to drop everything and prioritize their non-emergency.",
                "keywords": ["urgent", "email", "priority"],
                "rating": 0.82
            },
            {
                "id": 19,
                "category": "tech",
                "text": "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
                "keywords": ["programmers", "light bulb", "hardware"],
                "rating": 0.88
            },
            {
                "id": 20,
                "category": "hinglish",
                "text": "Mickey AI ka weather update: 'Bahut garmi hai, AC chalao ya coding karo... Dono se paseena aayega!'",
                "keywords": ["weather", "hot", "ac", "coding"],
                "rating": 0.84
            },
            {
                "id": 21,
                "category": "dad_jokes", 
                "text": "Why did the scarecrow win an award? Because he was outstanding in his field!",
                "keywords": ["scarecrow", "award", "field"],
                "rating": 0.7
            },
            {
                "id": 22,
                "category": "sarcastic",
                "text": "Your presentation was so engaging... I managed to plan my entire weekend during it.",
                "keywords": ["presentation", "engaging", "weekend"],
                "rating": 0.79
            },
            {
                "id": 23,
                "category": "hinglish",
                "text": "Programmer ki shaadi: 'I DO' ki jagah 'if (condition) { return true; }' bola!",
                "keywords": ["programmer", "marriage", "code", "condition"],
                "rating": 0.91
            },
            {
                "id": 24,
                "category": "tech",
                "text": "The best thing about Boolean algebra is that even if you're wrong, you're only off by a bit.",
                "keywords": ["boolean", "algebra", "binary"],
                "rating": 0.83
            },
            {
                "id": 25,
                "category": "puns",
                "text": "I'm so good at sleeping I can do it with my eyes closed.",
                "keywords": ["sleeping", "eyes", "closed"],
                "rating": 0.58
            }
        ]
        
        self.jokes = sample_jokes
        self.save_jokes()
    
    def load_jokes(self, category: Optional[str] = None) -> List[Dict]:
        """
        Load jokes from database, optionally filtered by category
        
        Args:
            category: Optional category filter
            
        Returns:
            List of joke dictionaries
        """
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.jokes = json.load(f)
            
            if category:
                return [joke for joke in self.jokes if joke.get('category') == category]
            else:
                return self.jokes
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading jokes: {e}")
            return []
    
    def save_jokes(self) -> bool:
        """
        Save jokes to JSON database
        
        Returns:
            Success status
        """
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.jokes, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving jokes: {e}")
            return False
    
    def add_joke(self, category: str, joke_text: str, keywords: Optional[List[str]] = None, rating: float = 0.5) -> bool:
        """
        Add a new joke to the database
        
        Args:
            category: Joke category
            joke_text: The joke content
            keywords: Optional search keywords
            rating: Humor rating (0.0-1.0)
            
        Returns:
            Success status
        """
        try:
            # Generate new ID
            new_id = max([joke.get('id', 0) for joke in self.jokes], default=0) + 1
            
            new_joke = {
                "id": new_id,
                "category": category,
                "text": joke_text,
                "keywords": keywords or [],
                "rating": max(0.0, min(1.0, rating))  # Clamp between 0-1
            }
            
            self.jokes.append(new_joke)
            return self.save_jokes()
            
        except Exception as e:
            print(f"Error adding joke: {e}")
            return False
    
    def search_by_keyword(self, keyword: str, category: Optional[str] = None) -> List[Dict]:
        """
        Search jokes by keyword in text or keywords list
        
        Args:
            keyword: Search term
            category: Optional category filter
            
        Returns:
            List of matching jokes
        """
        keyword_lower = keyword.lower()
        results = []
        
        for joke in self.jokes:
            # Search in joke text
            text_match = keyword_lower in joke.get('text', '').lower()
            
            # Search in keywords
            keyword_match = any(keyword_lower in kw.lower() for kw in joke.get('keywords', []))
            
            if text_match or keyword_match:
                if category is None or joke.get('category') == category:
                    results.append(joke)
        
        # Sort by rating (highest first)
        results.sort(key=lambda x: x.get('rating', 0), reverse=True)
        return results
    
    def get_random_joke(self, category: Optional[str] = None) -> Optional[Dict]:
        """
        Get a random joke, optionally filtered by category
        
        Args:
            category: Optional category filter
            
        Returns:
            Random joke or None if no jokes available
        """
        jokes_pool = self.load_jokes(category)
        return random.choice(jokes_pool) if jokes_pool else None
    
    def get_categories(self) -> List[str]:
        """
        Get list of available joke categories
        
        Returns:
            List of category names
        """
        return self.categories
    
    def get_joke_by_id(self, joke_id: int) -> Optional[Dict]:
        """
        Get specific joke by ID
        
        Args:
            joke_id: Joke identifier
            
        Returns:
            Joke dictionary or None if not found
        """
        for joke in self.jokes:
            if joke.get('id') == joke_id:
                return joke
        return None
    
    def delete_joke(self, joke_id: int) -> bool:
        """
        Delete joke by ID
        
        Args:
            joke_id: Joke identifier
            
        Returns:
            Success status
        """
        initial_count = len(self.jokes)
        self.jokes = [joke for joke in self.jokes if joke.get('id') != joke_id]
        
        if len(self.jokes) < initial_count:
            return self.save_jokes()
        return False

# Utility function for easy integration with humor_engine
def get_joke_database() -> JokeDatabase:
    """Get initialized joke database instance"""
    return JokeDatabase()

# Test function
def test_joke_database():
    """Test the joke database functionality"""
    db = JokeDatabase("test_humor_database.json")
    
    # Test loading
    jokes = db.load_jokes()
    print(f"Loaded {len(jokes)} jokes")
    
    # Test category filter
    hinglish_jokes = db.load_jokes("hinglish")
    print(f"Found {len(hinglish_jokes)} Hinglish jokes")
    
    # Test search
    search_results = db.search_by_keyword("python")
    print(f"Found {len(search_results)} jokes with 'python'")
    
    # Test random joke
    random_joke = db.get_random_joke()
    if random_joke:
        print(f"Random joke: {random_joke['text']}")
    
    # Test adding new joke
    new_joke_added = db.add_joke(
        category="tech",
        joke_text="Why do Python programmers wear glasses? Because they can't C!",
        keywords=["python", "programmers", "glasses"]
    )
    print(f"New joke added: {new_joke_added}")
    
    # Cleanup test file
    if os.path.exists("test_humor_database.json"):
        os.remove("test_humor_database.json")

if __name__ == "__main__":
    test_joke_database()