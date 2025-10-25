# System health checks
"""
Mickey AI - Health Monitor
Monitors system health, module status, and performs automatic recovery
"""

import logging
import time
import threading
import psutil
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import sqlite3
from pathlib import Path

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"

class ModuleStatus:
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.status = HealthStatus.HEALTHY
        self.last_check = datetime.now()
        self.response_time = 0.0
        self.error_count = 0
        self.last_error = None
        self.recovery_attempts = 0

class HealthMonitor:
    def __init__(self, db_path: str = "data/health_metrics.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Monitoring state
        self.modules: Dict[str, ModuleStatus] = {}
        self.system_metrics = {
            'cpu_threshold': 85.0,
            'memory_threshold': 80.0,
            'disk_threshold': 90.0,
            'response_time_threshold': 5.0
        }
        
        # Callbacks for health events
        self.health_callbacks: List[Callable] = []
        
        # Threading
        self._lock = threading.RLock()
        self._monitor_thread = None
        self._monitoring = False
        self._check_interval = 60  # seconds
        
        # Alert state
        self.alerts_sent = {}
        
        # Initialize
        self._init_database()
        self._register_core_modules()
        
        self.logger.info("❤️  Health Monitor initialized - Keeping Mickey healthy!")

    def _init_database(self):
        """Initialize health metrics database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Health metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS health_metrics (
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        module_name TEXT,
                        status TEXT,
                        response_time REAL,
                        cpu_percent REAL,
                        memory_percent REAL,
                        disk_percent REAL
                    )
                ''')
                
                # Alerts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS health_alerts (
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        module_name TEXT,
                        alert_type TEXT,
                        message TEXT,
                        resolved BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                conn.commit()
                self.logger.info("Health database initialized")
                
        except Exception as e:
            self.logger.error(f"Health database init failed: {str(e)}")

    def _register_core_modules(self):
        """Register core Mickey AI modules for monitoring"""
        core_modules = [
            'security_orchestrator',
            'voice_stt_engine', 
            'voice_tts_synthesizer',
            'reasoning_engine',
            'memory_manager',
            'command_dispatcher',
            'state_manager',
            'api_server',
            'gui_main'
        ]
        
        for module in core_modules:
            self.register_module(module)

    def register_module(self, module_name: str):
        """Register a module for health monitoring"""
        with self._lock:
            if module_name not in self.modules:
                self.modules[module_name] = ModuleStatus(module_name)
                self.logger.info(f"📋 Registered module for health monitoring: {module_name}")

    def start_monitoring(self):
        """Start the health monitoring loop"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("🔍 Health monitoring started")

    def stop_monitoring(self):
        """Stop health monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        self.logger.info("Health monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                self._check_system_health()
                self._check_module_health()
                self._cleanup_old_alerts()
                self._save_health_metrics()
                time.sleep(self._check_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(30)  # Wait before retrying

    def _check_system_health(self):
        """Check overall system health"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # System health status
            system_health = HealthStatus.HEALTHY
            
            if (cpu_percent > self.system_metrics['cpu_threshold'] or 
                memory.percent > self.system_metrics['memory_threshold']):
                system_health = HealthStatus.CRITICAL
            elif (disk.percent > self.system_metrics['disk_threshold']):
                system_health = HealthStatus.DEGRADED
            
            # Trigger alerts if needed
            if system_health != HealthStatus.HEALTHY:
                alert_msg = f"System health: {system_health.value}. CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%"
                self._trigger_alert('system', 'resource_usage', alert_msg)
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'status': system_health
            }
            
        except Exception as e:
            self.logger.error(f"System health check failed: {str(e)}")
            return {'status': HealthStatus.CRITICAL, 'error': str(e)}

    def _check_module_health(self):
        """Check health of registered modules"""
        with self._lock:
            for module_name, module_status in self.modules.items():
                try:
                    start_time = time.time()
                    
                    # Simulate module health check - in real implementation, 
                    # this would call actual module health endpoints
                    health_status = self._perform_module_health_check(module_name)
                    
                    response_time = (time.time() - start_time) * 1000  # ms
                    
                    # Update module status
                    module_status.response_time = response_time
                    module_status.last_check = datetime.now()
                    
                    if health_status:
                        module_status.status = HealthStatus.HEALTHY
                        module_status.error_count = 0
                        module_status.last_error = None
                    else:
                        module_status.status = HealthStatus.DEGRADED
                        module_status.error_count += 1
                        module_status.last_error = "Health check failed"
                        
                        # Escalate to critical after multiple failures
                        if module_status.error_count >= 3:
                            module_status.status = HealthStatus.CRITICAL
                            self._trigger_alert(module_name, 'module_failure', 
                                              f"Module {module_name} is critical after {module_status.error_count} failures")
                    
                    # Check response time
                    if response_time > self.system_metrics['response_time_threshold'] * 1000:
                        self._trigger_alert(module_name, 'slow_response', 
                                          f"Module {module_name} response time: {response_time:.2f}ms")
                        
                except Exception as e:
                    self.logger.error(f"Health check failed for {module_name}: {str(e)}")
                    module_status.status = HealthStatus.CRITICAL
                    module_status.error_count += 1
                    module_status.last_error = str(e)

    def _perform_module_health_check(self, module_name: str) -> bool:
        """
        Perform actual health check for a module.
        This is a simplified version - in practice, each module would have its own health endpoint.
        """
        try:
            # For API server, check if it's responding
            if module_name == 'api_server':
                response = requests.get('http://localhost:8000/api/system/status', timeout=5)
                return response.status_code == 200
            
            # For GUI, check if process is running
            elif module_name == 'gui_main':
                for proc in psutil.process_iter(['name']):
                    if 'python' in proc.info['name'].lower():
                        cmdline = proc.cmdline()
                        if any('gui_main' in str(arg) for arg in cmdline):
                            return True
                return False
            
            # For other modules, simulate checks
            else:
                # In real implementation, these would have proper health checks
                return True  # Simulate success for now
                
        except Exception as e:
            self.logger.debug(f"Health check for {module_name} failed: {str(e)}")
            return False

    def _trigger_alert(self, module_name: str, alert_type: str, message: str):
        """Trigger a health alert"""
        alert_key = f"{module_name}_{alert_type}"
        
        # Prevent duplicate alerts within 5 minutes
        if alert_key in self.alerts_sent:
            last_alert_time = self.alerts_sent[alert_key]
            if datetime.now() - last_alert_time < timedelta(minutes=5):
                return
        
        self.alerts_sent[alert_key] = datetime.now()
        
        # Log alert
        self.logger.warning(f"🚨 HEALTH ALERT: {message}")
        
        # Save to database
        self._save_alert(module_name, alert_type, message)
        
        # Call registered callbacks
        for callback in self.health_callbacks:
            try:
                callback(module_name, alert_type, message)
            except Exception as e:
                self.logger.error(f"Health callback failed: {str(e)}")

    def register_health_callback(self, callback: Callable):
        """Register callback for health events"""
        self.health_callbacks.append(callback)

    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        with self._lock:
            system_health = self._check_system_health()
            
            # Calculate module health
            healthy_modules = 0
            total_modules = len(self.modules)
            
            for module_status in self.modules.values():
                if module_status.status == HealthStatus.HEALTHY:
                    healthy_modules += 1
            
            module_health_percent = (healthy_modules / total_modules * 100) if total_modules > 0 else 0
            
            # Determine overall status
            overall_status = HealthStatus.HEALTHY
            if system_health['status'] == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
            elif system_health['status'] == HealthStatus.DEGRADED or module_health_percent < 80:
                overall_status = HealthStatus.DEGRADED
            
            # Mickey's health message
            health_messages = {
                HealthStatus.HEALTHY: [
                    "Mickey is feeling fantastic! Ready for action! 🐭",
                    "Everything's working perfectly! Let's have some fun! 🎉",
                    "System health: Excellent! Mickey's in top form! 💪"
                ],
                HealthStatus.DEGRADED: [
                    "Mickey's feeling a bit under the weather, but still working! 🤒",
                    "Minor issues detected, but I can still help! 🛠️",
                    "System needs a little attention, but operational! ⚠️"
                ],
                HealthStatus.CRITICAL: [
                    "Critical issues! Mickey needs immediate help! 🚑",
                    "System health critical! Some features may not work! 🔴",
                    "Urgent attention required! Mickey's struggling! 💔"
                ]
            }
            
            import random
            mickey_message = random.choice(health_messages[overall_status])
            
            return {
                'overall_status': overall_status.value,
                'system_health': system_health,
                'module_health': {
                    'healthy_modules': healthy_modules,
                    'total_modules': total_modules,
                    'health_percent': module_health_percent
                },
                'timestamp': datetime.now().isoformat(),
                'mickey_message': mickey_message,
                'details': {
                    module: {
                        'status': status.status.value,
                        'response_time': status.response_time,
                        'last_check': status.last_check.isoformat(),
                        'error_count': status.error_count
                    }
                    for module, status in self.modules.items()
                }
            }

    def get_health_history(self, hours: int = 24) -> List[Dict]:
        """Get health metrics history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_time = datetime.now() - timedelta(hours=hours)
                cursor.execute('''
                    SELECT timestamp, module_name, status, response_time, cpu_percent, memory_percent, disk_percent
                    FROM health_metrics 
                    WHERE timestamp > ? 
                    ORDER BY timestamp DESC
                    LIMIT 100
                ''', (cutoff_time.isoformat(),))
                
                metrics = []
                for row in cursor.fetchall():
                    metrics.append({
                        'timestamp': row[0],
                        'module_name': row[1],
                        'status': row[2],
                        'response_time': row[3],
                        'cpu_percent': row[4],
                        'memory_percent': row[5],
                        'disk_percent': row[6]
                    })
                
                return metrics
        except Exception as e:
            self.logger.error(f"Failed to get health history: {str(e)}")
            return []

    def _save_health_metrics(self):
        """Save current health metrics to database"""
        try:
            system_health = self._check_system_health()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for module_name, module_status in self.modules.items():
                    cursor.execute('''
                        INSERT INTO health_metrics 
                        (module_name, status, response_time, cpu_percent, memory_percent, disk_percent)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        module_name,
                        module_status.status.value,
                        module_status.response_time,
                        system_health.get('cpu_percent', 0),
                        system_health.get('memory_percent', 0),
                        system_health.get('disk_percent', 0)
                    ))
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save health metrics: {str(e)}")

    def _save_alert(self, module_name: str, alert_type: str, message: str):
        """Save alert to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO health_alerts (module_name, alert_type, message)
                    VALUES (?, ?, ?)
                ''', (module_name, alert_type, message))
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save alert: {str(e)}")

    def _cleanup_old_alerts(self):
        """Clean up old alerts from database"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM health_alerts WHERE timestamp < ?', 
                             (cutoff_time.isoformat(),))
                
                cursor.execute('DELETE FROM health_metrics WHERE timestamp < ?',
                             (cutoff_time.isoformat(),))
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to cleanup old alerts: {str(e)}")

    def attempt_recovery(self, module_name: str) -> bool:
        """Attempt to recover a failing module"""
        with self._lock:
            if module_name not in self.modules:
                return False
            
            module_status = self.modules[module_name]
            module_status.recovery_attempts += 1
            
            self.logger.info(f"Attempting recovery for {module_name} (attempt {module_status.recovery_attempts})")
            
            # Simulate recovery logic
            # In real implementation, this would restart services, clear caches, etc.
            time.sleep(2)  # Simulate recovery time
            
            # For demo purposes, assume recovery succeeds 80% of the time
            import random
            success = random.random() < 0.8
            
            if success:
                module_status.status = HealthStatus.HEALTHY
                module_status.error_count = 0
                module_status.last_error = None
                self.logger.info(f"✅ Recovery successful for {module_name}")
            else:
                self.logger.warning(f"❌ Recovery failed for {module_name}")
            
            return success

    def set_check_interval(self, interval: int):
        """Set health check interval in seconds"""
        self._check_interval = interval
        self.logger.info(f"Health check interval set to {interval} seconds")

    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()

# Test function
def test_health_monitor():
    """Test the health monitor"""
    import json
    
    health_monitor = HealthMonitor(":memory:")
    health_monitor.start_monitoring()
    
    # Test overall health
    time.sleep(2)
    health_status = health_monitor.get_overall_health()
    print("Health Status:", json.dumps(health_status, indent=2, default=str))
    
    # Test health history
    history = health_monitor.get_health_history()
    print(f"Health History: {len(history)} records")
    
    # Test recovery
    success = health_monitor.attempt_recovery('api_server')
    print(f"Recovery attempt: {'Success' if success else 'Failed'}")
    
    time.sleep(2)
    health_monitor.cleanup()

if __name__ == "__main__":
    test_health_monitor()