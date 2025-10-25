# Action confirmation prompts
"""
Mickey AI - Safety Confirmation System
Handles user confirmation for critical operations with intelligent risk assessment
"""

import logging
import time
import random
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class OperationType(Enum):
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_RESTART = "system_restart"
    FILE_DELETION = "file_deletion"
    ADMIN_OPERATION = "admin_operation"
    NETWORK_ACCESS = "network_access"
    EXTERNAL_COMMAND = "external_command"
    PRIVILEGED_ACTION = "privileged_action"

class SafetyConfirm:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Risk assessment rules
        self.risk_rules = self._initialize_risk_rules()
        
        # Confirmation history for learning
        self.confirmation_history = []
        self.max_history_size = 1000
        
        # User preferences and trust levels
        self.user_trust_levels = {}  # user_id -> trust_score (0-100)
        self.auto_confirm_threshold = 80  # Trust score for auto-confirmation
        
        # Safety parameters
        self.safety_timeout = 30  # seconds to wait for confirmation
        self.max_retries = 3
        
        # Mickey's safety messages
        self.safety_messages = {
            RiskLevel.LOW: [
                "Just double-checking: Do you want me to proceed with this? ✅",
                "Mickey wants to confirm: Should I go ahead? 🐭",
                "Quick confirmation needed: Proceed with this action? 👀"
            ],
            RiskLevel.MEDIUM: [
                "This action requires confirmation. Are you sure? 🤔",
                "Mickey needs your OK for this one. Proceed? 📝",
                "Safety check: Confirm you want to do this? 🔍"
            ],
            RiskLevel.HIGH: [
                "⚠️ Important: This action can't be undone. Please confirm carefully!",
                "🚨 Safety alert: This is a high-risk operation. Confirm to proceed?",
                "🔒 Critical action requiring explicit confirmation. Continue?"
            ],
            RiskLevel.CRITICAL: [
                "🚨🚨 CRITICAL SAFETY WARNING: This action is irreversible and high-risk!",
                "🛑 EMERGENCY CONFIRMATION REQUIRED: This will affect system stability!",
                "💥 EXTREME CAUTION: This operation can cause data loss or system damage!"
            ]
        }
        
        # Confirmation responses
        self.confirmation_responses = {
            'confirmed': [
                "Confirmed! Mickey's proceeding with the action! 🚀",
                "Got it! Executing now! ⚡",
                "Confirmation received! Action underway! ✅",
                "Roger that! Mickey's on it! 🐭"
            ],
            'cancelled': [
                "Action cancelled! Mickey's standing by! 🛑",
                "Operation aborted! Your safety comes first! ✅",
                "Cancelled as requested! Ready for next command! 👌",
                "Mickey cancelled the operation! Safety first! 🛡️"
            ],
            'timeout': [
                "Confirmation timeout! Mickey cancelled the operation for safety! ⏰",
                "No response received - operation cancelled automatically! 🚫",
                "Safety timeout! Action cancelled to protect your system! 🔒"
            ]
        }
        
        self.logger.info("🛡️ Safety Confirmation system initialized - User protection active!")

    def _initialize_risk_rules(self) -> Dict[OperationType, RiskLevel]:
        """Initialize risk assessment rules for different operation types"""
        return {
            OperationType.SYSTEM_SHUTDOWN: RiskLevel.HIGH,
            OperationType.SYSTEM_RESTART: RiskLevel.HIGH,
            OperationType.FILE_DELETION: RiskLevel.MEDIUM,
            OperationType.ADMIN_OPERATION: RiskLevel.HIGH,
            OperationType.NETWORK_ACCESS: RiskLevel.MEDIUM,
            OperationType.EXTERNAL_COMMAND: RiskLevel.CRITICAL,
            OperationType.PRIVILEGED_ACTION: RiskLevel.HIGH
        }

    def assess_risk(self, operation: OperationType, parameters: Dict[str, Any] = None, 
                   user_id: str = "default") -> Dict[str, Any]:
        """
        Assess risk level for an operation
        
        Args:
            operation: Type of operation
            parameters: Operation parameters for context
            user_id: User identifier for trust assessment
            
        Returns:
            Dictionary with risk assessment
        """
        try:
            # Base risk from operation type
            base_risk = self.risk_rules.get(operation, RiskLevel.MEDIUM)
            
            # Adjust risk based on parameters
            adjusted_risk = self._adjust_risk_with_context(base_risk, parameters, user_id)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(adjusted_risk, user_id)
            
            # Determine if confirmation is required
            requires_confirmation = self._requires_confirmation(adjusted_risk, confidence, user_id)
            
            assessment = {
                'operation': operation.value,
                'base_risk': base_risk.value,
                'adjusted_risk': adjusted_risk.value,
                'confidence_score': confidence,
                'requires_confirmation': requires_confirmation,
                'user_trust_level': self.user_trust_levels.get(user_id, 50),
                'assessment_timestamp': time.time(),
                'mickey_response': self._get_safety_message(adjusted_risk)
            }
            
            self.logger.info(f"Risk assessment: {operation.value} -> {adjusted_risk.value} (confidence: {confidence})")
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {str(e)}")
            return {
                'operation': operation.value,
                'adjusted_risk': RiskLevel.HIGH.value,
                'requires_confirmation': True,
                'error': str(e),
                'mickey_response': "Mickey detected a problem with risk assessment! Defaulting to safety mode! 🛡️"
            }

    def _adjust_risk_with_context(self, base_risk: RiskLevel, parameters: Dict[str, Any], 
                                user_id: str) -> RiskLevel:
        """Adjust risk level based on operation context and parameters"""
        adjusted_risk = base_risk
        
        if not parameters:
            return adjusted_risk
        
        # Increase risk for dangerous parameters
        dangerous_patterns = [
            'format', 'delete', 'remove', 'uninstall', 'shutdown', 'restart',
            'sudo', 'admin', 'root', 'chmod 777', 'rm -rf'
        ]
        
        param_str = str(parameters).lower()
        for pattern in dangerous_patterns:
            if pattern in param_str:
                if adjusted_risk == RiskLevel.LOW:
                    adjusted_risk = RiskLevel.MEDIUM
                elif adjusted_risk == RiskLevel.MEDIUM:
                    adjusted_risk = RiskLevel.HIGH
                elif adjusted_risk == RiskLevel.HIGH:
                    adjusted_risk = RiskLevel.CRITICAL
                break
        
        # Adjust based on user trust
        user_trust = self.user_trust_levels.get(user_id, 50)
        if user_trust > 80 and adjusted_risk != RiskLevel.CRITICAL:
            # Trusted users get slightly lower risk assessment
            if adjusted_risk == RiskLevel.HIGH:
                adjusted_risk = RiskLevel.MEDIUM
            elif adjusted_risk == RiskLevel.MEDIUM:
                adjusted_risk = RiskLevel.LOW
        
        return adjusted_risk

    def _calculate_confidence(self, risk_level: RiskLevel, user_id: str) -> float:
        """Calculate confidence score for automatic decision making"""
        user_trust = self.user_trust_levels.get(user_id, 50) / 100.0
        
        # Base confidence decreases with higher risk
        risk_confidence = {
            RiskLevel.LOW: 0.9,
            RiskLevel.MEDIUM: 0.7,
            RiskLevel.HIGH: 0.4,
            RiskLevel.CRITICAL: 0.1
        }
        
        base_confidence = risk_confidence.get(risk_level, 0.5)
        
        # Adjust with user trust
        adjusted_confidence = base_confidence * (0.3 + 0.7 * user_trust)
        
        return min(adjusted_confidence, 0.95)  # Cap at 95%

    def _requires_confirmation(self, risk_level: RiskLevel, confidence: float, 
                             user_id: str) -> bool:
        """Determine if confirmation is required"""
        user_trust = self.user_trust_levels.get(user_id, 50)
        
        # Always require confirmation for critical operations
        if risk_level == RiskLevel.CRITICAL:
            return True
        
        # High trust users might not need confirmation for low-risk operations
        if risk_level == RiskLevel.LOW and user_trust > self.auto_confirm_threshold:
            return confidence < 0.8
        
        # Medium and high risk usually require confirmation
        return risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH] or confidence < 0.7

    def request_confirmation(self, operation: OperationType, parameters: Dict[str, Any] = None,
                           user_id: str = "default", callback: Callable = None) -> Dict[str, Any]:
        """
        Request user confirmation for an operation
        
        Args:
            operation: Operation requiring confirmation
            parameters: Operation parameters
            user_id: User identifier
            callback: Callback function for confirmation result
            
        Returns:
            Dictionary with confirmation request details
        """
        try:
            # Assess risk first
            assessment = self.assess_risk(operation, parameters, user_id)
            
            # If no confirmation needed, return immediately
            if not assessment['requires_confirmation']:
                return {
                    'confirmation_required': False,
                    'auto_confirmed': True,
                    'assessment': assessment,
                    'mickey_response': "Mickey's proceeding with confidence! 🚀"
                }
            
            # Create confirmation request
            confirmation_id = self._generate_confirmation_id()
            confirmation_request = {
                'confirmation_id': confirmation_id,
                'operation': operation.value,
                'parameters': parameters,
                'user_id': user_id,
                'risk_level': assessment['adjusted_risk'],
                'timestamp': time.time(),
                'status': 'pending',
                'callback': callback
            }
            
            # Store the request (in production, this would be in a database)
            self.confirmation_history.append(confirmation_request)
            if len(self.confirmation_history) > self.max_history_size:
                self.confirmation_history.pop(0)
            
            self.logger.info(f"Confirmation requested: {confirmation_id} for {operation.value}")
            
            return {
                'confirmation_required': True,
                'confirmation_id': confirmation_id,
                'assessment': assessment,
                'timeout_seconds': self.safety_timeout,
                'mickey_response': assessment['mickey_response']
            }
            
        except Exception as e:
            self.logger.error(f"Confirmation request failed: {str(e)}")
            return {
                'confirmation_required': True,  # Default to requiring confirmation on error
                'error': str(e),
                'mickey_response': "Mickey encountered an error! Confirmation required for safety! 🛡️"
            }

    def process_confirmation(self, confirmation_id: str, user_response: bool, 
                           user_id: str = "default") -> Dict[str, Any]:
        """
        Process user confirmation response
        
        Args:
            confirmation_id: Confirmation request ID
            user_response: User's confirmation (True/False)
            user_id: User identifier
            
        Returns:
            Dictionary with confirmation result
        """
        try:
            # Find the confirmation request
            confirmation_request = None
            for req in self.confirmation_history:
                if req['confirmation_id'] == confirmation_id and req['user_id'] == user_id:
                    confirmation_request = req
                    break
            
            if not confirmation_request:
                return {
                    'success': False,
                    'error': 'Confirmation request not found',
                    'mickey_response': "Mickey couldn't find that confirmation request! 🤔"
                }
            
            # Update request status
            confirmation_request['status'] = 'confirmed' if user_response else 'cancelled'
            confirmation_request['response_timestamp'] = time.time()
            confirmation_request['user_response'] = user_response
            
            # Update user trust level based on response
            self._update_user_trust(user_id, user_response, confirmation_request)
            
            # Execute callback if provided
            if confirmation_request['callback']:
                try:
                    confirmation_request['callback'](user_response, confirmation_request)
                except Exception as e:
                    self.logger.error(f"Confirmation callback failed: {str(e)}")
            
            response_type = 'confirmed' if user_response else 'cancelled'
            
            self.logger.info(f"Confirmation processed: {confirmation_id} -> {response_type}")
            
            return {
                'success': True,
                'confirmation_id': confirmation_id,
                'user_response': user_response,
                'response_type': response_type,
                'trust_level_updated': self.user_trust_levels.get(user_id, 50),
                'mickey_response': self._get_confirmation_response(response_type)
            }
            
        except Exception as e:
            self.logger.error(f"Confirmation processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'mickey_response': "Mickey had trouble processing your confirmation! 😅"
            }

    def check_timeout(self) -> List[Dict[str, Any]]:
        """
        Check for timed out confirmation requests
        
        Returns:
            List of timed out requests
        """
        current_time = time.time()
        timed_out_requests = []
        
        for request in self.confirmation_history:
            if (request['status'] == 'pending' and 
                current_time - request['timestamp'] > self.safety_timeout):
                
                request['status'] = 'timeout'
                request['response_timestamp'] = current_time
                timed_out_requests.append(request)
                
                # Execute callback with timeout
                if request['callback']:
                    try:
                        request['callback'](False, request)  # False for cancelled due to timeout
                    except Exception as e:
                        self.logger.error(f"Timeout callback failed: {str(e)}")
                
                self.logger.warning(f"Confirmation timeout: {request['confirmation_id']}")
        
        return timed_out_requests

    def _generate_confirmation_id(self) -> str:
        """Generate unique confirmation ID"""
        timestamp = int(time.time() * 1000)
        random_suffix = random.randint(1000, 9999)
        return f"confirm_{timestamp}_{random_suffix}"

    def _update_user_trust(self, user_id: str, user_response: bool, request: Dict[str, Any]):
        """Update user trust level based on confirmation response"""
        current_trust = self.user_trust_levels.get(user_id, 50)
        risk_level = request['risk_level']
        
        # Trust increases when users make safe decisions
        if user_response:
            # User confirmed an action - trust increases for low-risk, decreases for high-risk
            if risk_level in ['low', 'medium']:
                trust_change = 5
            else:
                trust_change = -10  # Confirming high-risk actions decreases trust
        else:
            # User cancelled - trust increases for high-risk cancellations
            if risk_level in ['high', 'critical']:
                trust_change = 8  # Good decision to cancel high-risk
            else:
                trust_change = 2  # Minor increase for cancelling low-risk
        
        new_trust = max(0, min(100, current_trust + trust_change))
        self.user_trust_levels[user_id] = new_trust
        
        self.logger.info(f"User trust updated: {user_id} -> {new_trust} (change: {trust_change})")

    def _get_safety_message(self, risk_level: RiskLevel) -> str:
        """Get Mickey's safety message for risk level"""
        import random
        messages = self.safety_messages.get(risk_level, ["Safety check required! 👀"])
        return random.choice(messages)

    def _get_confirmation_response(self, response_type: str) -> str:
        """Get Mickey's response for confirmation result"""
        import random
        messages = self.confirmation_responses.get(response_type, ["Action processed! ✅"])
        return random.choice(messages)

    def get_user_trust_report(self, user_id: str = "default") -> Dict[str, Any]:
        """Get trust report for a user"""
        trust_level = self.user_trust_levels.get(user_id, 50)
        
        if trust_level >= 80:
            trust_category = "Highly Trusted"
            mickey_message = "Mickey trusts you completely! You're a safety pro! 🌟"
        elif trust_level >= 60:
            trust_category = "Trusted"
            mickey_message = "Mickey considers you a trusted user! 😊"
        elif trust_level >= 40:
            trust_category = "Standard"
            mickey_message = "You have standard trust level with Mickey! 👍"
        else:
            trust_category = "Restricted"
            mickey_message = "Mickey's being extra careful with your requests! 🛡️"
        
        return {
            'user_id': user_id,
            'trust_level': trust_level,
            'trust_category': trust_category,
            'auto_confirm_threshold': self.auto_confirm_threshold,
            'confirmation_requests': len([r for r in self.confirmation_history if r['user_id'] == user_id]),
            'mickey_response': mickey_message
        }

    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety system statistics"""
        total_requests = len(self.confirmation_history)
        confirmed_requests = len([r for r in self.confirmation_history if r.get('user_response') is True])
        cancelled_requests = len([r for r in self.confirmation_history if r.get('user_response') is False])
        timeout_requests = len([r for r in self.confirmation_history if r.get('status') == 'timeout'])
        pending_requests = len([r for r in self.confirmation_history if r.get('status') == 'pending'])
        
        return {
            'total_confirmation_requests': total_requests,
            'confirmed_operations': confirmed_requests,
            'cancelled_operations': cancelled_requests,
            'timeout_operations': timeout_requests,
            'pending_requests': pending_requests,
            'unique_users': len(set(r['user_id'] for r in self.confirmation_history)),
            'safety_timeout_seconds': self.safety_timeout,
            'auto_confirm_threshold': self.auto_confirm_threshold
        }

    def reset_user_trust(self, user_id: str):
        """Reset user trust level to default"""
        self.user_trust_levels[user_id] = 50
        self.logger.info(f"Trust level reset for user: {user_id}")

    def set_auto_confirm_threshold(self, threshold: int):
        """Set auto-confirmation trust threshold"""
        self.auto_confirm_threshold = max(0, min(100, threshold))
        self.logger.info(f"Auto-confirm threshold set to: {threshold}")

    def cleanup(self):
        """Cleanup resources"""
        # Clear old history entries
        current_time = time.time()
        self.confirmation_history = [
            req for req in self.confirmation_history 
            if current_time - req['timestamp'] < 86400  # Keep only last 24 hours
        ]

# Test function
def test_safety_confirm():
    """Test the safety confirmation system"""
    safety = SafetyConfirm()
    
    print("Testing Safety Confirmation System...")
    
    # Test risk assessment
    assessment = safety.assess_risk(OperationType.SYSTEM_SHUTDOWN, {"force": True})
    print("Risk Assessment:", assessment)
    
    # Test confirmation request
    confirmation = safety.request_confirmation(
        OperationType.FILE_DELETION, 
        {"files": ["important_document.txt"]},
        "test_user"
    )
    print("Confirmation Request:", confirmation)
    
    # Process confirmation
    if confirmation.get('confirmation_required'):
        result = safety.process_confirmation(
            confirmation['confirmation_id'], 
            True,  # User confirmed
            "test_user"
        )
        print("Confirmation Result:", result)
    
    # Test trust report
    trust_report = safety.get_user_trust_report("test_user")
    print("Trust Report:", trust_report)
    
    # Test safety stats
    stats = safety.get_safety_stats()
    print("Safety Stats:", stats)
    
    # Check for timeouts
    timeouts = safety.check_timeout()
    print("Timeouts:", len(timeouts))
    
    safety.cleanup()
    print("Safety confirmation test completed!")

if __name__ == "__main__":
    test_safety_confirm()