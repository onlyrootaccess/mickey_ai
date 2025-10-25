# Tracks assistant state
"""
Mickey AI - State Manager
Manages application state, user sessions, and system status
"""

import logging
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum
import psutil
import sqlite3
from pathlib import Path

class SessionState(Enum):
    ACTIVE = "active"
    IDLE = "idle"
    SLEEPING = "sleeping"
    BUSY = "busy"

class UserState:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.session_start = datetime.now()
        self.last_activity = datetime.now()
        self.command_count = 0
        self.preferences = {}
        self.conversation_history = []
        self.state = SessionState.ACTIVE

class StateManager:
    def __init__(self, db_path: str = "data/mickey_state.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory state
        self.active_users: Dict[str, UserState] = {}
        self.system_state = {
            'start_time': datetime.now(),
            'total_commands_processed': 0,
            'active_sessions': 0,
            'system_load': 0.0,
            'memory_usage': 0.0
        }
        
        # Threading
        self._lock = threading.RLock()
        self._monitor_thread = None
        self._monitoring = False
        
        # Initialize database
        self._init_database()
        self._load_persistent_state()
        
        self.logger.info("🗂️  State Manager initialized - Tracking everything!")

    def _init_database(self):
        """Initialize SQLite database for persistent state"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # User sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        user_id TEXT PRIMARY KEY,
                        session_data TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # System metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        cpu_percent REAL,
                        memory_percent REAL,
                        active_users INTEGER,
                        commands_processed INTEGER
                    )
                ''')
                
                # User preferences table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        user_id TEXT PRIMARY KEY,
                        preferences TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")

    def _load_persistent_state(self):
        """Load persistent state from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Load user preferences
                cursor.execute("SELECT user_id, preferences FROM user_preferences")
                for user_id, prefs_json in cursor.fetchall():
                    if prefs_json:
                        preferences = json.loads(prefs_json)
                        # Initialize user state if needed
                        if user_id not in self.active_users:
                            self.active_users[user_id] = UserState(user_id)
                        self.active_users[user_id].preferences = preferences
                
                self.logger.info(f"Loaded preferences for {len(self.active_users)} users")
                
        except Exception as e:
            self.logger.error(f"Failed to load persistent state: {str(e)}")

    def start_monitoring(self):
        """Start background monitoring thread"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("📊 State monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("State monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                self._update_system_metrics()
                self._cleanup_idle_sessions()
                self._save_metrics_to_db()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(60)

    def _update_system_metrics(self):
        """Update system performance metrics"""
        with self._lock:
            self.system_state['system_load'] = psutil.cpu_percent(interval=1)
            self.system_state['memory_usage'] = psutil.virtual_memory().percent
            self.system_state['active_sessions'] = len(self.active_users)

    def _cleanup_idle_sessions(self):
        """Clean up sessions that have been idle for too long"""
        cutoff_time = datetime.now() - timedelta(hours=1)  # 1 hour timeout
        users_to_remove = []
        
        with self._lock:
            for user_id, user_state in self.active_users.items():
                if user_state.last_activity < cutoff_time:
                    users_to_remove.append(user_id)
            
            for user_id in users_to_remove:
                self._save_user_session(user_id)
                del self.active_users[user_id]
                self.logger.info(f"Removed idle session for user: {user_id}")

    def create_user_session(self, user_id: str) -> UserState:
        """Create a new user session"""
        with self._lock:
            if user_id in self.active_users:
                # Update existing session
                self.active_users[user_id].last_activity = datetime.now()
                self.active_users[user_id].state = SessionState.ACTIVE
            else:
                # Create new session
                self.active_users[user_id] = UserState(user_id)
                self.logger.info(f"🎉 New session created for user: {user_id}")
            
            return self.active_users[user_id]

    def update_user_activity(self, user_id: str):
        """Update user's last activity timestamp"""
        with self._lock:
            if user_id in self.active_users:
                self.active_users[user_id].last_activity = datetime.now()
                self.active_users[user_id].command_count += 1
                self.system_state['total_commands_processed'] += 1

    def add_conversation_message(self, user_id: str, message: str, is_user: bool = True):
        """Add message to user's conversation history"""
        with self._lock:
            if user_id not in self.active_users:
                self.create_user_session(user_id)
            
            message_entry = {
                'timestamp': datetime.now().isoformat(),
                'message': message,
                'is_user': is_user
            }
            
            user_state = self.active_users[user_id]
            user_state.conversation_history.append(message_entry)
            
            # Keep only last 50 messages to prevent memory bloat
            if len(user_state.conversation_history) > 50:
                user_state.conversation_history = user_state.conversation_history[-50:]

    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's conversation history"""
        with self._lock:
            if user_id in self.active_users:
                history = self.active_users[user_id].conversation_history
                return history[-limit:] if limit else history
            return []

    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences"""
        with self._lock:
            if user_id not in self.active_users:
                self.create_user_session(user_id)
            
            user_state = self.active_users[user_id]
            user_state.preferences.update(preferences)
            
            # Save to database
            self._save_user_preferences(user_id)
            
            self.logger.info(f"Preferences updated for user: {user_id}")

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences"""
        with self._lock:
            if user_id in self.active_users:
                return self.active_users[user_id].preferences.copy()
            return {}

    def set_user_state(self, user_id: str, state: SessionState):
        """Set user's session state"""
        with self._lock:
            if user_id in self.active_users:
                self.active_users[user_id].state = state
                self.logger.info(f"User {user_id} state changed to: {state.value}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self._lock:
            status = self.system_state.copy()
            status['uptime'] = str(datetime.now() - status['start_time'])
            status['current_time'] = datetime.now().isoformat()
            status['active_users_list'] = list(self.active_users.keys())
            
            # Add Mickey's fun status messages based on load
            if status['system_load'] < 30:
                status['mickey_mood'] = "Feeling great! Ready for adventure! 🐭"
            elif status['system_load'] < 70:
                status['mickey_mood'] = "Busy but happy! Keep those commands coming! 😊"
            else:
                status['mickey_mood'] = "Working hard! Might need a cheese break! 🧀"
            
            return status

    def _save_user_preferences(self, user_id: str):
        """Save user preferences to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                preferences = json.dumps(self.active_users[user_id].preferences)
                cursor.execute('''
                    INSERT OR REPLACE INTO user_preferences (user_id, preferences, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (user_id, preferences))
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save user preferences: {str(e)}")

    def _save_user_session(self, user_id: str):
        """Save user session data to database"""
        try:
            if user_id not in self.active_users:
                return
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                session_data = {
                    'session_start': self.active_users[user_id].session_start.isoformat(),
                    'command_count': self.active_users[user_id].command_count,
                    'conversation_history': self.active_users[user_id].conversation_history[-10:]  # Last 10 messages
                }
                
                cursor.execute('''
                    INSERT OR REPLACE INTO user_sessions (user_id, session_data)
                    VALUES (?, ?)
                ''', (user_id, json.dumps(session_data)))
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save user session: {str(e)}")

    def _save_metrics_to_db(self):
        """Save system metrics to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO system_metrics (cpu_percent, memory_percent, active_users, commands_processed)
                    VALUES (?, ?, ?, ?)
                ''', (
                    self.system_state['system_load'],
                    self.system_state['memory_usage'],
                    self.system_state['active_sessions'],
                    self.system_state['total_commands_processed']
                ))
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {str(e)}")

    def get_metrics_history(self, hours: int = 24) -> List[Dict]:
        """Get system metrics history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_time = datetime.now() - timedelta(hours=hours)
                cursor.execute('''
                    SELECT timestamp, cpu_percent, memory_percent, active_users, commands_processed
                    FROM system_metrics 
                    WHERE timestamp > ? 
                    ORDER BY timestamp
                ''', (cutoff_time.isoformat(),))
                
                metrics = []
                for row in cursor.fetchall():
                    metrics.append({
                        'timestamp': row[0],
                        'cpu_percent': row[1],
                        'memory_percent': row[2],
                        'active_users': row[3],
                        'commands_processed': row[4]
                    })
                
                return metrics
        except Exception as e:
            self.logger.error(f"Failed to get metrics history: {str(e)}")
            return []

    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        
        # Save all active sessions
        with self._lock:
            for user_id in list(self.active_users.keys()):
                self._save_user_session(user_id)

# Test function
def test_state_manager():
    """Test the state manager"""
    import time
    
    state_mgr = StateManager(":memory:")  # Use in-memory DB for testing
    state_mgr.start_monitoring()
    
    # Test user session management
    state_mgr.create_user_session("test_user_1")
    state_mgr.update_user_activity("test_user_1")
    state_mgr.add_conversation_message("test_user_1", "Hello Mickey!", True)
    state_mgr.add_conversation_message("test_user_1", "Hi there! How can I help?", False)
    
    # Test preferences
    state_mgr.update_user_preferences("test_user_1", {
        'theme': 'dark',
        'voice_speed': 1.0,
        'jokes_enabled': True
    })
    
    # Test system status
    status = state_mgr.get_system_status()
    print("System Status:", json.dumps(status, indent=2, default=str))
    
    # Test conversation history
    history = state_mgr.get_conversation_history("test_user_1")
    print("Conversation History:", history)
    
    time.sleep(2)
    state_mgr.cleanup()

if __name__ == "__main__":
    test_state_manager()