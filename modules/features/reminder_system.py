# Scheduled task manager
"""
Mickey AI - Reminder System
Intelligent reminder and scheduled task management with natural language processing
"""

import logging
import json
import time
import sqlite3
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import re
from dateutil import parser
import schedule

class ReminderType(Enum):
    ONE_TIME = "one_time"
    REPEATING = "repeating"
    BIRTHDAY = "birthday"
    ANNIVERSARY = "anniversary"
    MEDICATION = "medication"
    MEETING = "meeting"
    CUSTOM = "custom"

class ReminderPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class ReminderStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    OVERDUE = "overdue"
    SNIPPED = "snoozed"

class ReminderSystem:
    def __init__(self, db_path: str = "data/reminders.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Database connection
        self.conn = None
        self._init_database()
        
        # Scheduler for recurring reminders
        self.scheduler = schedule.Scheduler()
        self.is_scheduler_running = False
        self.scheduler_thread = None
        
        # Notification callbacks
        self.notification_callbacks = []
        
        # Natural language processing patterns
        self.time_patterns = self._init_time_patterns()
        self.recurrence_patterns = self._init_recurrence_patterns()
        
        # Mickey's reminder personalities
        self.reminder_messages = {
            ReminderPriority.LOW: [
                "Mickey set a gentle reminder! 📝",
                "Friendly reminder from Mickey! 🐭",
                "Just a little nudge from Mickey! 👆"
            ],
            ReminderPriority.MEDIUM: [
                "Reminder set! Mickey won't let you forget! ✅",
                "Mickey's got your back with this reminder! 🛡️",
                "Reminder locked in! Mickey's on watch! 👀"
            ],
            ReminderPriority.HIGH: [
                "Important reminder set! Mickey's taking this seriously! 🚨",
                "High-priority reminder! Mickey will make sure you see this! ⚡",
                "Critical reminder scheduled! Mickey's got alarms ready! 🔔"
            ],
            ReminderPriority.URGENT: [
                "URGENT REMINDER SET! Mickey's on high alert! 🚨🚨",
                "Emergency reminder! Mickey won't let this slip! 💥",
                "Critical alert scheduled! Mickey's watching the clock! ⏰"
            ]
        }
        
        # Notification messages
        self.notification_messages = {
            ReminderPriority.LOW: [
                "⏰ Gentle reminder: {message}",
                "📝 Mickey's reminder: {message}",
                "👋 Friendly nudge: {message}"
            ],
            ReminderPriority.MEDIUM: [
                "🔔 Reminder: {message}",
                "✅ Time for: {message}",
                "📋 Scheduled: {message}"
            ],
            ReminderPriority.HIGH: [
                "🚨 IMPORTANT: {message}",
                "⚡ URGENT REMINDER: {message}",
                "🔴 ATTENTION: {message}"
            ],
            ReminderPriority.URGENT: [
                "🚨🚨 CRITICAL REMINDER: {message}",
                "💥 EMERGENCY ALERT: {message}",
                "🔴🔴 URGENT: {message}"
            ]
        }
        
        self.logger.info("⏰ Reminder System initialized - Ready to keep track!")

    def _init_database(self):
        """Initialize the reminders database"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()
            
            # Create reminders table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    reminder_type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    scheduled_time DATETIME NOT NULL,
                    created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    completed_time DATETIME,
                    recurrence_pattern TEXT,
                    category TEXT,
                    tags TEXT,
                    snooze_count INTEGER DEFAULT 0,
                    max_snoozes INTEGER DEFAULT 3,
                    last_notified DATETIME,
                    next_reminder_time DATETIME
                )
            ''')
            
            # Create index for performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_scheduled_time 
                ON reminders(scheduled_time)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_status 
                ON reminders(user_id, status)
            ''')
            
            self.conn.commit()
            self.logger.info("Reminders database initialized")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise

    def _init_time_patterns(self) -> Dict[str, Any]:
        """Initialize natural language time parsing patterns"""
        return {
            'today': {
                'patterns': [
                    r'today at (\d+:\d+)',
                    r'today (\d+[ap]m)',
                    r'this afternoon',
                    r'this evening',
                    r'tonight'
                ],
                'base_date': 'today'
            },
            'tomorrow': {
                'patterns': [
                    r'tomorrow at (\d+:\d+)',
                    r'tomorrow (\d+[ap]m)',
                    r'tomorrow morning',
                    r'tomorrow afternoon',
                    r'tomorrow evening'
                ],
                'base_date': 'tomorrow'
            },
            'days_of_week': {
                'patterns': [
                    r'on monday',
                    r'on tuesday', 
                    r'on wednesday',
                    r'on thursday',
                    r'on friday',
                    r'on saturday',
                    r'on sunday',
                    r'next monday',
                    r'next tuesday'
                ]
            },
            'relative_days': {
                'patterns': [
                    r'in (\d+) days?',
                    r'(\d+) days? from now',
                    r'next week'
                ]
            },
            'times': {
                'patterns': [
                    r'at (\d+:\d+)',
                    r'at (\d+[ap]m)',
                    r'(\d+:\d+)',
                    r'(\d+[ap]m)'
                ]
            }
        }

    def _init_recurrence_patterns(self) -> Dict[str, str]:
        """Initialize recurrence pattern parsing"""
        return {
            'daily': '0 9 * * *',  # 9 AM daily
            'weekdays': '0 9 * * 1-5',  # 9 AM weekdays
            'weekly': '0 9 * * 0',  # 9 AM Sundays
            'monthly': '0 9 1 * *',  # 9 AM 1st of month
            'yearly': '0 9 1 1 *',  # 9 AM Jan 1st
            'hourly': '0 * * * *',  # Every hour
            'every morning': '0 9 * * *',
            'every evening': '0 18 * * *',
            'every night': '0 21 * * *'
        }

    def create_reminder(self, user_id: str, message: str, time_expression: str = None,
                       scheduled_time: datetime = None, reminder_type: ReminderType = ReminderType.ONE_TIME,
                       priority: ReminderPriority = ReminderPriority.MEDIUM, category: str = None,
                       tags: List[str] = None, recurrence: str = None) -> Dict[str, Any]:
        """
        Create a new reminder
        
        Args:
            user_id: User identifier
            message: Reminder message
            time_expression: Natural language time expression
            scheduled_time: Specific scheduled time
            reminder_type: Type of reminder
            priority: Reminder priority
            category: Reminder category
            tags: List of tags
            recurrence: Recurrence pattern
            
        Returns:
            Dictionary with reminder creation result
        """
        try:
            # Parse time expression if provided
            if time_expression and not scheduled_time:
                scheduled_time = self._parse_time_expression(time_expression)
                if not scheduled_time:
                    return self._create_error_response(f"Could not parse time: {time_expression}")
            
            if not scheduled_time:
                return self._create_error_response("No scheduled time provided")
            
            # Validate scheduled time is in the future
            if scheduled_time < datetime.now():
                return self._create_error_response("Scheduled time must be in the future")
            
            # Prepare reminder data
            reminder_data = {
                'user_id': user_id,
                'title': self._extract_title(message),
                'message': message,
                'reminder_type': reminder_type.value,
                'priority': priority.value,
                'status': ReminderStatus.PENDING.value,
                'scheduled_time': scheduled_time.isoformat(),
                'category': category,
                'tags': json.dumps(tags) if tags else None,
                'recurrence_pattern': recurrence,
                'next_reminder_time': scheduled_time.isoformat()
            }
            
            # Save to database
            cursor = self.conn.cursor()
            columns = ', '.join(reminder_data.keys())
            placeholders = ', '.join(['?' for _ in reminder_data])
            
            cursor.execute(
                f"INSERT INTO reminders ({columns}) VALUES ({placeholders})",
                list(reminder_data.values())
            )
            
            reminder_id = cursor.lastrowid
            self.conn.commit()
            
            # Schedule the reminder
            self._schedule_reminder(reminder_id, scheduled_time)
            
            self.logger.info(f"Reminder created: {reminder_id} for user {user_id}")
            
            return {
                'success': True,
                'reminder_id': reminder_id,
                'scheduled_time': scheduled_time.isoformat(),
                'reminder_type': reminder_type.value,
                'priority': priority.value,
                'message': f"Reminder set for {scheduled_time.strftime('%Y-%m-%d %H:%M')}",
                'mickey_response': self._get_reminder_creation_message(priority)
            }
            
        except Exception as e:
            self.logger.error(f"Reminder creation failed: {str(e)}")
            return self._create_error_response(f"Reminder creation failed: {str(e)}")

    def _parse_time_expression(self, time_expression: str) -> Optional[datetime]:
        """Parse natural language time expressions"""
        try:
            # Try dateutil parser first
            try:
                return parser.parse(time_expression, fuzzy=True)
            except:
                pass
            
            # Custom parsing for common patterns
            expression_lower = time_expression.lower()
            now = datetime.now()
            base_date = now.date()
            
            # Handle "today" patterns
            if any(pattern in expression_lower for pattern in ['today', 'this afternoon', 'this evening', 'tonight']):
                time_match = self._extract_time(expression_lower)
                if time_match:
                    return datetime.combine(base_date, time_match)
                # Default to evening if no time specified
                if 'evening' in expression_lower or 'tonight' in expression_lower:
                    return datetime.combine(base_date, datetime.strptime('19:00', '%H:%M').time())
                # Default to afternoon
                return datetime.combine(base_date, datetime.strptime('14:00', '%H:%M').time())
            
            # Handle "tomorrow" patterns
            elif any(pattern in expression_lower for pattern in ['tomorrow', 'next day']):
                base_date = now.date() + timedelta(days=1)
                time_match = self._extract_time(expression_lower)
                if time_match:
                    return datetime.combine(base_date, time_match)
                # Default to morning
                return datetime.combine(base_date, datetime.strptime('09:00', '%H:%M').time())
            
            # Handle days of week
            days_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            
            for day_name, day_num in days_map.items():
                if day_name in expression_lower:
                    days_ahead = (day_num - now.weekday() + 7) % 7
                    if days_ahead == 0:
                        days_ahead = 7  # Next week
                    base_date = now.date() + timedelta(days=days_ahead)
                    time_match = self._extract_time(expression_lower)
                    if time_match:
                        return datetime.combine(base_date, time_match)
                    return datetime.combine(base_date, datetime.strptime('09:00', '%H:%M').time())
            
            # Handle relative days
            relative_match = re.search(r'in (\d+) days?', expression_lower)
            if relative_match:
                days = int(relative_match.group(1))
                base_date = now.date() + timedelta(days=days)
                time_match = self._extract_time(expression_lower)
                if time_match:
                    return datetime.combine(base_date, time_match)
                return datetime.combine(base_date, datetime.strptime('09:00', '%H:%M').time())
            
            # If all else fails, try to extract just time and use today
            time_match = self._extract_time(expression_lower)
            if time_match:
                # If time is earlier than now, schedule for tomorrow
                proposed_time = datetime.combine(base_date, time_match)
                if proposed_time < now:
                    proposed_time += timedelta(days=1)
                return proposed_time
            
            return None
            
        except Exception as e:
            self.logger.error(f"Time expression parsing failed: {str(e)}")
            return None

    def _extract_time(self, text: str) -> Optional[datetime.time]:
        """Extract time from text"""
        try:
            # Match HH:MM format
            time_match = re.search(r'(\d{1,2}):(\d{2})', text)
            if time_match:
                hour, minute = int(time_match.group(1)), int(time_match.group(2))
                return datetime.time(hour, minute)
            
            # Match HH AM/PM format
            ampm_match = re.search(r'(\d{1,2})\s*([ap]m)', text, re.IGNORECASE)
            if ampm_match:
                hour = int(ampm_match.group(1))
                period = ampm_match.group(2).lower()
                
                if period == 'pm' and hour < 12:
                    hour += 12
                elif period == 'am' and hour == 12:
                    hour = 0
                
                return datetime.time(hour, 0)
            
            return None
            
        except:
            return None

    def _extract_title(self, message: str) -> str:
        """Extract a title from the reminder message"""
        # Take first 5 words or 30 characters as title
        words = message.split()[:5]
        title = ' '.join(words)
        if len(title) > 30:
            title = title[:27] + '...'
        return title

    def _schedule_reminder(self, reminder_id: int, scheduled_time: datetime):
        """Schedule a reminder for notification"""
        try:
            # Calculate delay in seconds
            delay = (scheduled_time - datetime.now()).total_seconds()
            
            if delay > 0:
                # Schedule using threading Timer
                timer = threading.Timer(delay, self._trigger_reminder, [reminder_id])
                timer.daemon = True
                timer.start()
                
                self.logger.info(f"Scheduled reminder {reminder_id} for {scheduled_time}")
            else:
                self.logger.warning(f"Reminder {reminder_id} is in the past")
                
        except Exception as e:
            self.logger.error(f"Reminder scheduling failed: {str(e)}")

    def _trigger_reminder(self, reminder_id: int):
        """Trigger a reminder notification"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM reminders WHERE id = ? AND status = 'pending'",
                (reminder_id,)
            )
            
            reminder = cursor.fetchone()
            if not reminder:
                return
            
            # Convert row to dict
            columns = [col[0] for col in cursor.description]
            reminder_dict = dict(zip(columns, reminder))
            
            # Update last notified time
            cursor.execute(
                "UPDATE reminders SET last_notified = ? WHERE id = ?",
                (datetime.now().isoformat(), reminder_id)
            )
            self.conn.commit()
            
            # Send notification
            self._send_notification(reminder_dict)
            
            # Handle recurring reminders
            if reminder_dict['recurrence_pattern']:
                self._schedule_next_recurrence(reminder_id, reminder_dict)
            else:
                # Mark one-time reminders as completed after notification
                cursor.execute(
                    "UPDATE reminders SET status = 'completed' WHERE id = ?",
                    (reminder_id,)
                )
                self.conn.commit()
            
            self.logger.info(f"Triggered reminder: {reminder_id}")
            
        except Exception as e:
            self.logger.error(f"Reminder trigger failed: {str(e)}")

    def _send_notification(self, reminder: Dict[str, Any]):
        """Send reminder notification to all registered callbacks"""
        notification_data = {
            'reminder_id': reminder['id'],
            'user_id': reminder['user_id'],
            'title': reminder['title'],
            'message': reminder['message'],
            'priority': reminder['priority'],
            'scheduled_time': reminder['scheduled_time'],
            'category': reminder['category'],
            'notification_time': datetime.now().isoformat()
        }
        
        # Add Mickey's notification message
        priority = ReminderPriority(reminder['priority'])
        notification_data['mickey_message'] = self._get_notification_message(
            reminder['message'], priority
        )
        
        for callback in self.notification_callbacks:
            try:
                callback(notification_data)
            except Exception as e:
                self.logger.error(f"Notification callback failed: {str(e)}")

    def _schedule_next_recurrence(self, reminder_id: int, reminder: Dict[str, Any]):
        """Schedule the next occurrence of a recurring reminder"""
        try:
            recurrence_pattern = reminder['recurrence_pattern']
            last_time = parser.parse(reminder['scheduled_time'])
            
            # Calculate next occurrence based on recurrence pattern
            if recurrence_pattern == 'daily':
                next_time = last_time + timedelta(days=1)
            elif recurrence_pattern == 'weekdays':
                next_time = last_time + timedelta(days=1)
                # Skip weekends
                while next_time.weekday() >= 5:  # 5=Saturday, 6=Sunday
                    next_time += timedelta(days=1)
            elif recurrence_pattern == 'weekly':
                next_time = last_time + timedelta(weeks=1)
            elif recurrence_pattern == 'monthly':
                next_time = last_time + timedelta(days=30)  # Approximate
            elif recurrence_pattern == 'yearly':
                next_time = last_time + timedelta(days=365)  # Approximate
            else:
                # Custom cron-like patterns would be handled here
                next_time = last_time + timedelta(days=1)  # Default fallback
            
            # Update reminder with next occurrence
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE reminders SET scheduled_time = ?, next_reminder_time = ? WHERE id = ?",
                (next_time.isoformat(), next_time.isoformat(), reminder_id)
            )
            self.conn.commit()
            
            # Schedule the next occurrence
            self._schedule_reminder(reminder_id, next_time)
            
            self.logger.info(f"Scheduled next recurrence for reminder {reminder_id} at {next_time}")
            
        except Exception as e:
            self.logger.error(f"Recurrence scheduling failed: {str(e)}")

    def get_reminders(self, user_id: str, status: ReminderStatus = None, 
                     limit: int = 50) -> Dict[str, Any]:
        """Get reminders for a user"""
        try:
            cursor = self.conn.cursor()
            
            query = "SELECT * FROM reminders WHERE user_id = ?"
            params = [user_id]
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            query += " ORDER BY scheduled_time DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            reminders = cursor.fetchall()
            
            # Convert to list of dicts
            columns = [col[0] for col in cursor.description]
            reminder_list = [dict(zip(columns, row)) for row in reminders]
            
            return {
                'success': True,
                'user_id': user_id,
                'reminders': reminder_list,
                'total_count': len(reminder_list),
                'pending_count': len([r for r in reminder_list if r['status'] == 'pending']),
                'mickey_response': f"Mickey found {len(reminder_list)} reminders for you! 📋"
            }
            
        except Exception as e:
            self.logger.error(f"Get reminders failed: {str(e)}")
            return self._create_error_response(f"Failed to get reminders: {str(e)}")

    def update_reminder(self, reminder_id: int, user_id: str, 
                       updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a reminder"""
        try:
            cursor = self.conn.cursor()
            
            # Build update query
            set_clause = ', '.join([f"{key} = ?" for key in updates.keys()])
            query = f"UPDATE reminders SET {set_clause} WHERE id = ? AND user_id = ?"
            
            params = list(updates.values()) + [reminder_id, user_id]
            
            cursor.execute(query, params)
            
            if cursor.rowcount == 0:
                return self._create_error_response("Reminder not found or access denied")
            
            self.conn.commit()
            
            # Reschedule if time was updated
            if 'scheduled_time' in updates:
                new_time = parser.parse(updates['scheduled_time'])
                self._schedule_reminder(reminder_id, new_time)
            
            self.logger.info(f"Updated reminder: {reminder_id}")
            
            return {
                'success': True,
                'reminder_id': reminder_id,
                'message': "Reminder updated successfully",
                'mickey_response': "Mickey updated your reminder! ✅"
            }
            
        except Exception as e:
            self.logger.error(f"Update reminder failed: {str(e)}")
            return self._create_error_response(f"Failed to update reminder: {str(e)}")

    def delete_reminder(self, reminder_id: int, user_id: str) -> Dict[str, Any]:
        """Delete a reminder"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "DELETE FROM reminders WHERE id = ? AND user_id = ?",
                (reminder_id, user_id)
            )
            
            if cursor.rowcount == 0:
                return self._create_error_response("Reminder not found or access denied")
            
            self.conn.commit()
            
            self.logger.info(f"Deleted reminder: {reminder_id}")
            
            return {
                'success': True,
                'reminder_id': reminder_id,
                'message': "Reminder deleted successfully",
                'mickey_response': "Mickey removed your reminder! 🗑️"
            }
            
        except Exception as e:
            self.logger.error(f"Delete reminder failed: {str(e)}")
            return self._create_error_response(f"Failed to delete reminder: {str(e)}")

    def snooze_reminder(self, reminder_id: int, user_id: str, 
                       snooze_minutes: int = 10) -> Dict[str, Any]:
        """Snooze a reminder"""
        try:
            cursor = self.conn.cursor()
            
            # Check current snooze count
            cursor.execute(
                "SELECT snooze_count, max_snoozes FROM reminders WHERE id = ? AND user_id = ?",
                (reminder_id, user_id)
            )
            result = cursor.fetchone()
            
            if not result:
                return self._create_error_response("Reminder not found")
            
            snooze_count, max_snoozes = result
            
            if snooze_count >= max_snoozes:
                return self._create_error_response("Maximum snooze limit reached")
            
            # Calculate new time
            new_time = datetime.now() + timedelta(minutes=snooze_minutes)
            
            # Update reminder
            cursor.execute(
                "UPDATE reminders SET scheduled_time = ?, status = 'snoozed', snooze_count = snooze_count + 1 WHERE id = ?",
                (new_time.isoformat(), reminder_id)
            )
            self.conn.commit()
            
            # Reschedule
            self._schedule_reminder(reminder_id, new_time)
            
            self.logger.info(f"Snoozed reminder {reminder_id} for {snooze_minutes} minutes")
            
            return {
                'success': True,
                'reminder_id': reminder_id,
                'snoozed_until': new_time.isoformat(),
                'snooze_count': snooze_count + 1,
                'max_snoozes': max_snoozes,
                'message': f"Reminder snoozed for {snooze_minutes} minutes",
                'mickey_response': f"Mickey snoozed your reminder for {snooze_minutes} minutes! ⏰"
            }
            
        except Exception as e:
            self.logger.error(f"Snooze reminder failed: {str(e)}")
            return self._create_error_response(f"Failed to snooze reminder: {str(e)}")

    def complete_reminder(self, reminder_id: int, user_id: str) -> Dict[str, Any]:
        """Mark a reminder as completed"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE reminders SET status = 'completed', completed_time = ? WHERE id = ? AND user_id = ?",
                (datetime.now().isoformat(), reminder_id, user_id)
            )
            
            if cursor.rowcount == 0:
                return self._create_error_response("Reminder not found or access denied")
            
            self.conn.commit()
            
            self.logger.info(f"Completed reminder: {reminder_id}")
            
            return {
                'success': True,
                'reminder_id': reminder_id,
                'message': "Reminder marked as completed",
                'mickey_response': "Mickey marked that reminder as done! ✅"
            }
            
        except Exception as e:
            self.logger.error(f"Complete reminder failed: {str(e)}")
            return self._create_error_response(f"Failed to complete reminder: {str(e)}")

    def add_notification_callback(self, callback: Callable):
        """Add a notification callback"""
        self.notification_callbacks.append(callback)

    def _get_reminder_creation_message(self, priority: ReminderPriority) -> str:
        """Get Mickey's reminder creation message"""
        import random
        messages = self.reminder_messages.get(priority, ["Reminder set! ✅"])
        return random.choice(messages)

    def _get_notification_message(self, message: str, priority: ReminderPriority) -> str:
        """Get Mickey's notification message"""
        import random
        messages = self.notification_messages.get(priority, ["Reminder: {message}"])
        template = random.choice(messages)
        return template.format(message=message)

    def get_reminder_stats(self, user_id: str) -> Dict[str, Any]:
        """Get reminder statistics for a user"""
        try:
            cursor = self.conn.cursor()
            
            # Get counts by status
            cursor.execute(
                "SELECT status, COUNT(*) FROM reminders WHERE user_id = ? GROUP BY status",
                (user_id,)
            )
            status_counts = dict(cursor.fetchall())
            
            # Get counts by priority
            cursor.execute(
                "SELECT priority, COUNT(*) FROM reminders WHERE user_id = ? GROUP BY priority",
                (user_id,)
            )
            priority_counts = dict(cursor.fetchall())
            
            # Get upcoming reminders
            cursor.execute(
                "SELECT COUNT(*) FROM reminders WHERE user_id = ? AND status = 'pending' AND scheduled_time > ?",
                (user_id, datetime.now().isoformat())
            )
            upcoming_count = cursor.fetchone()[0]
            
            # Get overdue reminders
            cursor.execute(
                "SELECT COUNT(*) FROM reminders WHERE user_id = ? AND status = 'pending' AND scheduled_time <= ?",
                (user_id, datetime.now().isoformat())
            )
            overdue_count = cursor.fetchone()[0]
            
            return {
                'success': True,
                'user_id': user_id,
                'statistics': {
                    'total_reminders': sum(status_counts.values()),
                    'status_breakdown': status_counts,
                    'priority_breakdown': priority_counts,
                    'upcoming_reminders': upcoming_count,
                    'overdue_reminders': overdue_count
                },
                'mickey_response': f"Mickey's tracking {sum(status_counts.values())} reminders for you! 📊"
            }
            
        except Exception as e:
            self.logger.error(f"Get stats failed: {str(e)}")
            return self._create_error_response(f"Failed to get statistics: {str(e)}")

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': error_message,
            'mickey_response': random.choice([
                "Oops! Mickey had trouble with that reminder! 😅",
                "Reminder magic failed! Let's try again! ✨",
                "Mickey's reminder system is having a moment! 🐭"
            ])
        }

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.conn:
                self.conn.close()
        except:
            pass

# Test function
def test_reminder_system():
    """Test the reminder system"""
    reminder_system = ReminderSystem(":memory:")  # In-memory DB for testing
    
    print("Testing Reminder System...")
    
    # Test creating a reminder
    result = reminder_system.create_reminder(
        user_id="test_user",
        message="Test meeting with team",
        time_expression="in 1 minute",
        priority=ReminderPriority.MEDIUM
    )
    print("Create Reminder:", result)
    
    # Test getting reminders
    reminders = reminder_system.get_reminders("test_user")
    print("Get Reminders:", reminders['total_count'])
    
    # Test stats
    stats = reminder_system.get_reminder_stats("test_user")
    print("Reminder Stats:", stats['statistics'])
    
    # Wait a bit for reminder to trigger
    print("Waiting for reminder notification...")
    time.sleep(65)
    
    reminder_system.cleanup()
    print("Reminder system test completed!")

if __name__ == "__main__":
    test_reminder_system()