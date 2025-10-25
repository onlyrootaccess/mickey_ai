# SQLite/JSON memory operations
"""
M.I.C.K.E.Y. AI Assistant - Memory Manager
Made In Crisis, Keeping Everything Yours

NINTH FILE IN PIPELINE: Comprehensive memory system for conversation history, 
user preferences, learning, and context management. Enables persistent personality.
"""

import asyncio
import logging
import time
import json
import sqlite3
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib

# Import database and data libraries
from sqlalchemy import create_engine, Column, String, Text, Integer, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Import Mickey AI configuration
from config.settings import get_config
from config.constants import (
    SystemConstants, PersonalityConstants, LLMConstants
)

# Setup logging
logger = logging.getLogger("MickeyMemory")


class MemoryType(Enum):
    """Types of memory stored by Mickey."""
    CONVERSATION = "conversation"
    PREFERENCE = "preference"
    FACT = "fact"
    RELATIONSHIP = "relationship"
    BEHAVIOR = "behavior"
    CONTEXT = "context"


class PriorityLevel(Enum):
    """Priority levels for memory retention."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MemoryItem:
    """Individual memory item container."""
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    priority: PriorityLevel
    created_at: float
    last_accessed: float
    access_count: int
    expiration: Optional[float] = None
    tags: List[str] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['memory_type'] = self.memory_type.value
        data['priority'] = self.priority.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create from dictionary."""
        data['memory_type'] = MemoryType(data['memory_type'])
        data['priority'] = PriorityLevel(data['priority'])
        return cls(**data)


@dataclass
class ConversationTurn:
    """Single conversation turn with user and AI."""
    turn_id: str
    conversation_id: str
    user_message: str
    ai_response: str
    timestamp: float
    emotion_detected: str
    response_type: str
    tokens_used: int
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserProfile:
    """User profile and preferences container."""
    user_id: str
    name: Optional[str] = None
    preferences: Dict[str, Any] = None
    conversation_style: str = "balanced"
    humor_preference: float = 0.7
    learning_enabled: bool = True
    created_at: float = None
    last_interaction: float = None
    interaction_count: int = 0
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}
        if self.created_at is None:
            self.created_at = time.time()
        if self.last_interaction is None:
            self.last_interaction = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# SQLAlchemy Base
Base = declarative_base()


class ConversationModel(Base):
    """SQLAlchemy model for conversation history."""
    __tablename__ = 'conversations'
    
    turn_id = Column(String, primary_key=True)
    conversation_id = Column(String, index=True)
    user_message = Column(Text)
    ai_response = Column(Text)
    timestamp = Column(Float)
    emotion_detected = Column(String)
    response_type = Column(String)
    tokens_used = Column(Integer)
    processing_time = Column(Float)


class UserProfileModel(Base):
    """SQLAlchemy model for user profiles."""
    __tablename__ = 'user_profiles'
    
    user_id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    preferences = Column(JSON)
    conversation_style = Column(String)
    humor_preference = Column(Float)
    learning_enabled = Column(Boolean)
    created_at = Column(Float)
    last_interaction = Column(Float)
    interaction_count = Column(Integer)


class MemoryModel(Base):
    """SQLAlchemy model for general memory items."""
    __tablename__ = 'memory_items'
    
    memory_id = Column(String, primary_key=True)
    memory_type = Column(String)
    content = Column(JSON)
    priority = Column(Integer)
    created_at = Column(Float)
    last_accessed = Column(Float)
    access_count = Column(Integer)
    expiration = Column(Float, nullable=True)
    tags = Column(JSON)
    confidence = Column(Float)


class PreferenceLearner:
    """Learns and adapts to user preferences over time."""
    
    def __init__(self):
        self.config = get_config()
        
        # Preference tracking
        self.preference_weights = {
            'humor_level': 0.7,
            'response_length': 0.6,
            'formality': 0.5,
            'detail_level': 0.8,
            'emotion_matching': 0.9
        }
        
        # Behavior patterns
        self.behavior_patterns = {}
    
    def analyze_conversation_pattern(self, conversation_turns: List[ConversationTurn]) -> Dict[str, Any]:
        """Analyze conversation patterns to learn user preferences."""
        if not conversation_turns:
            return {}
        
        # Analyze response preferences
        response_types = [turn.response_type for turn in conversation_turns]
        emotions = [turn.emotion_detected for turn in conversation_turns]
        
        # Calculate preferences
        preferences = {
            'preferred_response_type': self._most_common(response_types) or 'direct_answer',
            'common_emotions': self._top_emotions(emotions),
            'average_conversation_length': sum(turn.tokens_used for turn in conversation_turns) / len(conversation_turns),
            'interaction_frequency': self._calculate_frequency(conversation_turns)
        }
        
        # Update preference weights based on patterns
        self._update_preference_weights(preferences)
        
        return preferences
    
    def _most_common(self, items: List) -> Any:
        """Find most common item in list."""
        if not items:
            return None
        return max(set(items), key=items.count)
    
    def _top_emotions(self, emotions: List[str], top_n: int = 3) -> List[str]:
        """Get top N most common emotions."""
        from collections import Counter
        counter = Counter(emotions)
        return [emotion for emotion, _ in counter.most_common(top_n)]
    
    def _calculate_frequency(self, turns: List[ConversationTurn]) -> str:
        """Calculate interaction frequency pattern."""
        if len(turns) < 2:
            return "unknown"
        
        timestamps = sorted([turn.timestamp for turn in turns])
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        avg_interval = sum(intervals) / len(intervals)
        
        if avg_interval < 3600:  # Less than 1 hour
            return "frequent"
        elif avg_interval < 86400:  # Less than 1 day
            return "regular"
        else:
            return "occasional"
    
    def _update_preference_weights(self, preferences: Dict[str, Any]):
        """Update preference weights based on learned patterns."""
        # Adjust humor preference based on response types
        if 'humorous' in preferences.get('preferred_response_type', ''):
            self.preference_weights['humor_level'] = min(1.0, self.preference_weights['humor_level'] + 0.1)
        
        # Adjust based on conversation length
        avg_length = preferences.get('average_conversation_length', 0)
        if avg_length > 100:
            self.preference_weights['detail_level'] = min(1.0, self.preference_weights['detail_level'] + 0.1)
        
        logger.debug(f"Updated preference weights: {self.preference_weights}")


class KnowledgeGraph:
    """Manages relationships and connections between memory items."""
    
    def __init__(self):
        self.relationships = {}
        self.entity_map = {}
    
    def add_relationship(self, source: str, relation: str, target: str, confidence: float = 1.0):
        """Add a relationship between entities."""
        if source not in self.relationships:
            self.relationships[source] = {}
        
        if relation not in self.relationships[source]:
            self.relationships[source][relation] = []
        
        self.relationships[source][relation].append({
            'target': target,
            'confidence': confidence,
            'created_at': time.time()
        })
        
        # Update entity map
        self.entity_map[source] = True
        self.entity_map[target] = True
    
    def get_related_entities(self, entity: str, relation: str = None) -> List[Dict]:
        """Get entities related to the given entity."""
        if entity not in self.relationships:
            return []
        
        if relation:
            return self.relationships[entity].get(relation, [])
        else:
            # Return all relations
            all_related = []
            for rel_type, targets in self.relationships[entity].items():
                all_related.extend(targets)
            return all_related
    
    def find_connections(self, entity1: str, entity2: str, max_depth: int = 2) -> List[List]:
        """Find connection paths between two entities."""
        if entity1 not in self.relationships or entity2 not in self.relationships:
            return []
        
        return self._bfs_connections(entity1, entity2, max_depth)
    
    def _bfs_connections(self, start: str, end: str, max_depth: int) -> List[List]:
        """Breadth-first search for connections."""
        from collections import deque
        
        queue = deque([(start, [])])
        visited = set()
        paths = []
        
        while queue:
            current, path = queue.popleft()
            
            if current == end and path:
                paths.append(path)
                continue
            
            if len(path) >= max_depth:
                continue
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current in self.relationships:
                for relation, targets in self.relationships[current].items():
                    for target_info in targets:
                        next_entity = target_info['target']
                        new_path = path + [(current, relation, next_entity)]
                        queue.append((next_entity, new_path))
        
        return paths


class MemoryManager:
    """
    Comprehensive memory management system for Mickey AI.
    Handles conversations, user profiles, preferences, and knowledge graphs.
    """
    
    def __init__(self):
        self.config = get_config()
        self.engine = None
        self.Session = None
        self.preference_learner = PreferenceLearner()
        self.knowledge_graph = KnowledgeGraph()
        self.is_initialized = False
        
        # Memory cache for performance
        self.conversation_cache = {}
        self.user_profile_cache = {}
        self.memory_cache = {}
        self.cache_size = 100
        
        # Statistics
        self.total_operations = 0
        self.cache_hits = 0
        
    async def initialize(self):
        """Initialize the memory manager and database."""
        try:
            logger.info("Initializing Memory Manager...")
            
            # Ensure data directory exists
            data_dir = Path(self.config.data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize database
            db_path = data_dir / "mickey_memory.db"
            self.engine = create_engine(f'sqlite:///{db_path}')
            
            # Create tables
            Base.metadata.create_all(self.engine)
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            # Load initial data
            await self._load_initial_data()
            
            self.is_initialized = True
            logger.info("✅ Memory Manager initialized")
            
        except Exception as e:
            logger.error(f"❌ Memory Manager initialization failed: {str(e)}")
            raise
    
    async def _load_initial_data(self):
        """Load initial data into memory caches."""
        try:
            # Load recent conversations
            recent_conversations = await self.get_recent_conversations(limit=50)
            for conv in recent_conversations:
                cache_key = f"conv_{conv.conversation_id}"
                self.conversation_cache[cache_key] = conv
            
            # Load user profiles
            with self.Session() as session:
                profiles = session.query(UserProfileModel).all()
                for profile in profiles:
                    self.user_profile_cache[profile.user_id] = UserProfile(
                        user_id=profile.user_id,
                        name=profile.name,
                        preferences=profile.preferences or {},
                        conversation_style=profile.conversation_style,
                        humor_preference=profile.humor_preference,
                        learning_enabled=profile.learning_enabled,
                        created_at=profile.created_at,
                        last_interaction=profile.last_interaction,
                        interaction_count=profile.interaction_count
                    )
            
            logger.info(f"Loaded {len(recent_conversations)} conversations and {len(profiles)} user profiles")
            
        except Exception as e:
            logger.error(f"Failed to load initial data: {str(e)}")
    
    def _generate_id(self, prefix: str = "mem") -> str:
        """Generate unique ID for memory items."""
        timestamp = int(time.time() * 1000)
        random_suffix = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_suffix}"
    
    async def create_conversation(self, user_id: str = "default") -> str:
        """Create a new conversation session."""
        conversation_id = self._generate_id("conv")
        
        # Create initial conversation entry
        initial_turn = ConversationTurn(
            turn_id=self._generate_id("turn"),
            conversation_id=conversation_id,
            user_message="[Conversation Started]",
            ai_response="[Mickey Ready]",
            timestamp=time.time(),
            emotion_detected="neutral",
            response_type="system",
            tokens_used=0,
            processing_time=0.0
        )
        
        await self.add_conversation_turn(initial_turn)
        
        logger.info(f"Created new conversation: {conversation_id} for user: {user_id}")
        return conversation_id
    
    async def add_conversation_turn(self, turn: ConversationTurn):
        """Add a conversation turn to memory."""
        try:
            with self.Session() as session:
                # Convert to model
                turn_model = ConversationModel(
                    turn_id=turn.turn_id,
                    conversation_id=turn.conversation_id,
                    user_message=turn.user_message,
                    ai_response=turn.ai_response,
                    timestamp=turn.timestamp,
                    emotion_detected=turn.emotion_detected,
                    response_type=turn.response_type,
                    tokens_used=turn.tokens_used,
                    processing_time=turn.processing_time
                )
                
                session.add(turn_model)
                session.commit()
            
            # Update cache
            cache_key = f"conv_{turn.conversation_id}"
            if cache_key in self.conversation_cache:
                # Update existing conversation in cache
                pass  # We'd need to reload the conversation
            
            self.total_operations += 1
            logger.debug(f"Added conversation turn: {turn.turn_id}")
            
        except Exception as e:
            logger.error(f"Failed to add conversation turn: {str(e)}")
            raise
    
    async def add_interaction(self, conversation_id: str, user_message: str, ai_response: str,
                            emotion_detected: str = "neutral", response_type: str = "direct_answer",
                            tokens_used: int = 0, processing_time: float = 0.0):
        """Add a complete interaction to conversation history."""
        turn = ConversationTurn(
            turn_id=self._generate_id("turn"),
            conversation_id=conversation_id,
            user_message=user_message,
            ai_response=ai_response,
            timestamp=time.time(),
            emotion_detected=emotion_detected,
            response_type=response_type,
            tokens_used=tokens_used,
            processing_time=processing_time
        )
        
        await self.add_conversation_turn(turn)
        
        # Update user profile interaction count
        await self.update_user_interaction("default")
    
    async def get_conversation_context(self, conversation_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get conversation context for LLM processing."""
        try:
            with self.Session() as session:
                turns = session.query(ConversationModel)\
                    .filter(ConversationModel.conversation_id == conversation_id)\
                    .order_by(ConversationModel.timestamp.desc())\
                    .limit(limit)\
                    .all()
                
                # Convert to dictionary format
                history = []
                for turn in reversed(turns):  # Oldest first
                    history.append({
                        "user_message": turn.user_message,
                        "ai_response": turn.ai_response,
                        "timestamp": turn.timestamp,
                        "emotion_detected": turn.emotion_detected,
                        "response_type": turn.response_type
                    })
                
                # Analyze conversation patterns for preferences
                conversation_turns = [
                    ConversationTurn(
                        turn_id=turn.turn_id,
                        conversation_id=turn.conversation_id,
                        user_message=turn.user_message,
                        ai_response=turn.ai_response,
                        timestamp=turn.timestamp,
                        emotion_detected=turn.emotion_detected,
                        response_type=turn.response_type,
                        tokens_used=turn.tokens_used,
                        processing_time=turn.processing_time
                    ) for turn in turns
                ]
                
                preferences = self.preference_learner.analyze_conversation_pattern(conversation_turns)
                
                context = {
                    "conversation_id": conversation_id,
                    "history": history,
                    "preferences": preferences,
                    "turn_count": len(turns)
                }
                
                return context
                
        except Exception as e:
            logger.error(f"Failed to get conversation context: {str(e)}")
            return {"conversation_id": conversation_id, "history": [], "preferences": {}, "turn_count": 0}
    
    async def get_recent_conversations(self, limit: int = 20) -> List[ConversationModel]:
        """Get recent conversations across all sessions."""
        try:
            with self.Session() as session:
                # Get unique conversation IDs with their latest timestamp
                subquery = session.query(
                    ConversationModel.conversation_id,
                    ConversationModel.timestamp
                ).distinct().order_by(
                    ConversationModel.timestamp.desc()
                ).limit(limit).subquery()
                
                conversations = session.query(ConversationModel).join(
                    subquery,
                    ConversationModel.conversation_id == subquery.c.conversation_id
                ).all()
                
                return conversations
                
        except Exception as e:
            logger.error(f"Failed to get recent conversations: {str(e)}")
            return []
    
    async def get_user_profile(self, user_id: str = "default") -> UserProfile:
        """Get user profile, creating if it doesn't exist."""
        # Check cache first
        if user_id in self.user_profile_cache:
            self.cache_hits += 1
            return self.user_profile_cache[user_id]
        
        try:
            with self.Session() as session:
                profile_model = session.query(UserProfileModel).filter_by(user_id=user_id).first()
                
                if profile_model:
                    profile = UserProfile(
                        user_id=profile_model.user_id,
                        name=profile_model.name,
                        preferences=profile_model.preferences or {},
                        conversation_style=profile_model.conversation_style,
                        humor_preference=profile_model.humor_preference,
                        learning_enabled=profile_model.learning_enabled,
                        created_at=profile_model.created_at,
                        last_interaction=profile_model.last_interaction,
                        interaction_count=profile_model.interaction_count
                    )
                else:
                    # Create new profile
                    profile = UserProfile(
                        user_id=user_id,
                        conversation_style="balanced",
                        humor_preference=0.7,
                        learning_enabled=True,
                        created_at=time.time(),
                        last_interaction=time.time()
                    )
                    
                    # Save new profile
                    await self.save_user_profile(profile)
                
                # Update cache
                self.user_profile_cache[user_id] = profile
                
                return profile
                
        except Exception as e:
            logger.error(f"Failed to get user profile: {str(e)}")
            # Return default profile
            return UserProfile(user_id=user_id)
    
    async def save_user_profile(self, profile: UserProfile):
        """Save user profile to database."""
        try:
            with self.Session() as session:
                profile_model = session.query(UserProfileModel).filter_by(user_id=profile.user_id).first()
                
                if profile_model:
                    # Update existing
                    profile_model.name = profile.name
                    profile_model.preferences = profile.preferences
                    profile_model.conversation_style = profile.conversation_style
                    profile_model.humor_preference = profile.humor_preference
                    profile_model.learning_enabled = profile.learning_enabled
                    profile_model.last_interaction = profile.last_interaction
                    profile_model.interaction_count = profile.interaction_count
                else:
                    # Create new
                    profile_model = UserProfileModel(
                        user_id=profile.user_id,
                        name=profile.name,
                        preferences=profile.preferences,
                        conversation_style=profile.conversation_style,
                        humor_preference=profile.humor_preference,
                        learning_enabled=profile.learning_enabled,
                        created_at=profile.created_at,
                        last_interaction=profile.last_interaction,
                        interaction_count=profile.interaction_count
                    )
                    session.add(profile_model)
                
                session.commit()
            
            # Update cache
            self.user_profile_cache[profile.user_id] = profile
            
            self.total_operations += 1
            logger.debug(f"Saved user profile: {profile.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to save user profile: {str(e)}")
            raise
    
    async def update_user_interaction(self, user_id: str = "default"):
        """Update user interaction timestamp and count."""
        try:
            profile = await self.get_user_profile(user_id)
            profile.last_interaction = time.time()
            profile.interaction_count += 1
            
            await self.save_user_profile(profile)
            
        except Exception as e:
            logger.error(f"Failed to update user interaction: {str(e)}")
    
    async def store_memory(self, memory_type: MemoryType, content: Dict[str, Any],
                         priority: PriorityLevel = PriorityLevel.MEDIUM,
                         expiration: Optional[float] = None,
                         tags: List[str] = None) -> str:
        """Store a memory item."""
        memory_id = self._generate_id()
        
        memory_item = MemoryItem(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            priority=priority,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            expiration=expiration,
            tags=tags or []
        )
        
        try:
            with self.Session() as session:
                memory_model = MemoryModel(
                    memory_id=memory_item.memory_id,
                    memory_type=memory_item.memory_type.value,
                    content=memory_item.content,
                    priority=memory_item.priority.value,
                    created_at=memory_item.created_at,
                    last_accessed=memory_item.last_accessed,
                    access_count=memory_item.access_count,
                    expiration=memory_item.expiration,
                    tags=memory_item.tags,
                    confidence=memory_item.confidence
                )
                
                session.add(memory_model)
                session.commit()
            
            # Update cache
            self.memory_cache[memory_id] = memory_item
            
            # Manage cache size
            if len(self.memory_cache) > self.cache_size:
                # Remove least recently accessed
                oldest_key = min(self.memory_cache.keys(), 
                               key=lambda k: self.memory_cache[k].last_accessed)
                del self.memory_cache[oldest_key]
            
            self.total_operations += 1
            logger.debug(f"Stored memory: {memory_id} (type: {memory_type.value})")
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {str(e)}")
            raise
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID."""
        # Check cache first
        if memory_id in self.memory_cache:
            self.cache_hits += 1
            memory = self.memory_cache[memory_id]
            memory.last_accessed = time.time()
            memory.access_count += 1
            return memory
        
        try:
            with self.Session() as session:
                memory_model = session.query(MemoryModel).filter_by(memory_id=memory_id).first()
                
                if memory_model:
                    memory = MemoryItem(
                        memory_id=memory_model.memory_id,
                        memory_type=MemoryType(memory_model.memory_type),
                        content=memory_model.content,
                        priority=PriorityLevel(memory_model.priority),
                        created_at=memory_model.created_at,
                        last_accessed=time.time(),
                        access_count=memory_model.access_count + 1,
                        expiration=memory_model.expiration,
                        tags=memory_model.tags or [],
                        confidence=memory_model.confidence
                    )
                    
                    # Update access time in database
                    memory_model.last_accessed = memory.last_accessed
                    memory_model.access_count = memory.access_count
                    session.commit()
                    
                    # Update cache
                    self.memory_cache[memory_id] = memory
                    
                    return memory
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to retrieve memory: {str(e)}")
            return None
    
    async def search_memories(self, query: str, memory_type: MemoryType = None,
                            limit: int = 10) -> List[MemoryItem]:
        """Search memories by content and tags."""
        try:
            with self.Session() as session:
                query_builder = session.query(MemoryModel)
                
                # Filter by type if specified
                if memory_type:
                    query_builder = query_builder.filter_by(memory_type=memory_type.value)
                
                # Simple text search in content (for demo - would use full-text search in production)
                memories = query_builder.all()
                
                # Filter by query in content or tags
                results = []
                for memory_model in memories:
                    content_str = json.dumps(memory_model.content).lower()
                    tags_str = json.dumps(memory_model.tags or []).lower()
                    
                    if query.lower() in content_str or query.lower() in tags_str:
                        memory = MemoryItem.from_dict({
                            'memory_id': memory_model.memory_id,
                            'memory_type': memory_model.memory_type,
                            'content': memory_model.content,
                            'priority': memory_model.priority,
                            'created_at': memory_model.created_at,
                            'last_accessed': memory_model.last_accessed,
                            'access_count': memory_model.access_count,
                            'expiration': memory_model.expiration,
                            'tags': memory_model.tags,
                            'confidence': memory_model.confidence
                        })
                        results.append(memory)
                
                # Sort by relevance (simplified - would use proper scoring)
                results.sort(key=lambda x: x.priority.value, reverse=True)
                
                return results[:limit]
                
        except Exception as e:
            logger.error(f"Failed to search memories: {str(e)}")
            return []
    
    async def cleanup_expired_memories(self):
        """Clean up expired memories from database."""
        try:
            current_time = time.time()
            
            with self.Session() as session:
                expired_memories = session.query(MemoryModel)\
                    .filter(MemoryModel.expiration.isnot(None))\
                    .filter(MemoryModel.expiration < current_time)\
                    .all()
                
                for memory in expired_memories:
                    session.delete(memory)
                    # Remove from cache if present
                    if memory.memory_id in self.memory_cache:
                        del self.memory_cache[memory.memory_id]
                
                session.commit()
                
                logger.info(f"Cleaned up {len(expired_memories)} expired memories")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired memories: {str(e)}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get memory manager performance metrics."""
        cache_hit_rate = (
            self.cache_hits / self.total_operations 
            if self.total_operations > 0 else 0
        )
        
        # Get database statistics
        try:
            with self.Session() as session:
                conversation_count = session.query(ConversationModel).count()
                user_profile_count = session.query(UserProfileModel).count()
                memory_count = session.query(MemoryModel).count()
        except:
            conversation_count = user_profile_count = memory_count = 0
        
        return {
            "initialized": self.is_initialized,
            "total_operations": self.total_operations,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "conversation_count": conversation_count,
            "user_profile_count": user_profile_count,
            "memory_count": memory_count,
            "cache_sizes": {
                "conversations": len(self.conversation_cache),
                "user_profiles": len(self.user_profile_cache),
                "memories": len(self.memory_cache)
            }
        }
    
    async def export_data(self, export_path: Path) -> bool:
        """Export all memory data to JSON file."""
        try:
            export_data = {
                "export_timestamp": time.time(),
                "conversations": [],
                "user_profiles": [],
                "memories": []
            }
            
            with self.Session() as session:
                # Export conversations
                conversations = session.query(ConversationModel).all()
                for conv in conversations:
                    export_data["conversations"].append({
                        "turn_id": conv.turn_id,
                        "conversation_id": conv.conversation_id,
                        "user_message": conv.user_message,
                        "ai_response": conv.ai_response,
                        "timestamp": conv.timestamp,
                        "emotion_detected": conv.emotion_detected,
                        "response_type": conv.response_type,
                        "tokens_used": conv.tokens_used,
                        "processing_time": conv.processing_time
                    })
                
                # Export user profiles
                profiles = session.query(UserProfileModel).all()
                for profile in profiles:
                    export_data["user_profiles"].append({
                        "user_id": profile.user_id,
                        "name": profile.name,
                        "preferences": profile.preferences,
                        "conversation_style": profile.conversation_style,
                        "humor_preference": profile.humor_preference,
                        "learning_enabled": profile.learning_enabled,
                        "created_at": profile.created_at,
                        "last_interaction": profile.last_interaction,
                        "interaction_count": profile.interaction_count
                    })
                
                # Export memories
                memories = session.query(MemoryModel).all()
                for memory in memories:
                    export_data["memories"].append({
                        "memory_id": memory.memory_id,
                        "memory_type": memory.memory_type,
                        "content": memory.content,
                        "priority": memory.priority,
                        "created_at": memory.created_at,
                        "last_accessed": memory.last_accessed,
                        "access_count": memory.access_count,
                        "expiration": memory.expiration,
                        "tags": memory.tags,
                        "confidence": memory.confidence
                    })
            
            # Write to file
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(conversations)} conversations, {len(profiles)} profiles, "
                       f"{len(memories)} memories to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export data: {str(e)}")
            return False
    
    async def shutdown(self):
        """Shutdown memory manager gracefully."""
        logger.info("Shutting down Memory Manager...")
        
        try:
            # Perform final cleanup
            await self.cleanup_expired_memories()
            
            # Clear caches
            self.conversation_cache.clear()
            self.user_profile_cache.clear()
            self.memory_cache.clear()
            
            logger.info("✅ Memory Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during memory manager shutdown: {str(e)}")


# Global memory manager instance
_memory_instance: Optional[MemoryManager] = None


async def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager instance."""
    global _memory_instance
    
    if _memory_instance is None:
        _memory_instance = MemoryManager()
        await _memory_instance.initialize()
    
    return _memory_instance


async def main():
    """Command-line testing for memory manager."""
    memory_manager = await get_memory_manager()
    
    # Test performance metrics
    metrics = await memory_manager.get_performance_metrics()
    print("Memory Manager Status:")
    print(f"Initialized: {metrics['initialized']}")
    print(f"Total Operations: {metrics['total_operations']}")
    print(f"Cache Hit Rate: {metrics['cache_hit_rate']:.1%}")
    print(f"Conversations: {metrics['conversation_count']}")
    print(f"User Profiles: {metrics['user_profile_count']}")
    print(f"Memory Items: {metrics['memory_count']}")
    
    # Test creating a conversation
    conv_id = await memory_manager.create_conversation()
    print(f"\nCreated test conversation: {conv_id}")
    
    # Test adding an interaction
    await memory_manager.add_interaction(
        conversation_id=conv_id,
        user_message="Hello Mickey!",
        ai_response="Hi there! How can I help you today?",
        emotion_detected="happy",
        response_type="friendly"
    )
    print("Added test interaction")


if __name__ == "__main__":
    asyncio.run(main())