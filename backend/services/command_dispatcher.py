# Routes validated commands to modules
"""
Mickey AI - Command Dispatcher
Routes commands to appropriate modules and manages execution flow
"""

import logging
import json
from typing import Dict, Any, List
from enum import Enum
from modules.control.input_controller import InputController
from modules.control.browser_automator import BrowserAutomator
from backend.intelligence.reasoning_engine import ReasoningEngine
from modules.features.web_search import WebSearch
from modules.features.media_controller import MediaController
from backend.services.state_manager import StateManager

class CommandType(Enum):
    SYSTEM = "system"
    BROWSER = "browser"
    MEDIA = "media"
    SEARCH = "search"
    CONTROL = "control"
    CHAT = "chat"

class CommandDispatcher:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.state_manager = StateManager()
        self.input_controller = InputController()
        self.browser_automator = BrowserAutomator()
        self.reasoning_engine = ReasoningEngine()
        self.web_search = WebSearch()
        self.media_controller = MediaController()
        
        # Command routing table
        self.command_routes = {
            CommandType.SYSTEM: self._handle_system_command,
            CommandType.BROWSER: self._handle_browser_command,
            CommandType.MEDIA: self._handle_media_command,
            CommandType.SEARCH: self._handle_search_command,
            CommandType.CONTROL: self._handle_control_command,
            CommandType.CHAT: self._handle_chat_command
        }
        
        # Mickey's fun responses
        self.fun_responses = [
            "Mickey on the case! 🐭",
            "Aye aye, captain! Processing your command!",
            "Haha, let me handle that for you!",
            "Mickey magic coming right up! ✨",
            "One command, coming with extra cheese! 🧀"
        ]
        
        self.logger.info("🎯 Command Dispatcher initialized - Ready to route!")

    async def dispatch_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main dispatch method - routes commands to appropriate handlers
        
        Args:
            command_data: {
                'type': 'browser|media|search|control|chat',
                'action': 'open|close|play|search|click|type',
                'parameters': {...},
                'user_id': 'user123'
            }
        """
        try:
            # Validate command
            if not self._validate_command(command_data):
                return self._create_error_response("Invalid command format")
            
            command_type = CommandType(command_data.get('type', 'chat'))
            user_id = command_data.get('user_id', 'default')
            
            # Update user activity state
            self.state_manager.update_user_activity(user_id)
            
            # Route to appropriate handler
            handler = self.command_routes.get(command_type, self._handle_chat_command)
            result = await handler(command_data)
            
            # Add Mickey's fun personality
            if result.get('success', False):
                result['mickey_response'] = self._get_fun_response()
            
            # Log command execution
            self._log_command_execution(command_type, user_id, result.get('success', False))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Command dispatch failed: {str(e)}")
            return self._create_error_response(f"Command failed: {str(e)}")

    def _validate_command(self, command_data: Dict[str, Any]) -> bool:
        """Validate command structure"""
        required_fields = ['type', 'action']
        return all(field in command_data for field in required_fields)

    async def _handle_system_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system-related commands"""
        action = command_data['action']
        params = command_data.get('parameters', {})
        
        try:
            if action == 'shutdown':
                # Graceful shutdown
                return {'success': True, 'message': 'System shutting down...'}
                
            elif action == 'restart':
                return {'success': True, 'message': 'System restarting...'}
                
            elif action == 'status':
                system_status = self.state_manager.get_system_status()
                return {'success': True, 'status': system_status}
                
            else:
                return self._create_error_response(f"Unknown system action: {action}")
                
        except Exception as e:
            self.logger.error(f"System command failed: {str(e)}")
            return self._create_error_response(f"System command failed: {str(e)}")

    async def _handle_browser_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle browser automation commands"""
        action = command_data['action']
        params = command_data.get('parameters', {})
        
        try:
            if action == 'open_url':
                url = params.get('url', 'https://google.com')
                result = self.browser_automator.open_url(url)
                return {'success': True, 'result': result}
                
            elif action == 'search':
                query = params.get('query', '')
                result = self.browser_automator.search(query)
                return {'success': True, 'result': result}
                
            elif action == 'close':
                result = self.browser_automator.close_browser()
                return {'success': True, 'result': result}
                
            else:
                return self._create_error_response(f"Unknown browser action: {action}")
                
        except Exception as e:
            self.logger.error(f"Browser command failed: {str(e)}")
            return self._create_error_response(f"Browser command failed: {str(e)}")

    async def _handle_media_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle media control commands"""
        action = command_data['action']
        params = command_data.get('parameters', {})
        
        try:
            if action == 'play':
                result = self.media_controller.play()
                return {'success': True, 'result': result}
                
            elif action == 'pause':
                result = self.media_controller.pause()
                return {'success': True, 'result': result}
                
            elif action == 'volume_up':
                result = self.media_controller.volume_up()
                return {'success': True, 'result': result}
                
            elif action == 'volume_down':
                result = self.media_controller.volume_down()
                return {'success': True, 'result': result}
                
            else:
                return self._create_error_response(f"Unknown media action: {action}")
                
        except Exception as e:
            self.logger.error(f"Media command failed: {str(e)}")
            return self._create_error_response(f"Media command failed: {str(e)}")

    async def _handle_search_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle web search commands"""
        action = command_data['action']
        params = command_data.get('parameters', {})
        
        try:
            if action == 'web_search':
                query = params.get('query', '')
                result = await self.web_search.search(query)
                return {'success': True, 'result': result}
                
            elif action == 'news_search':
                query = params.get('query', '')
                result = await self.web_search.search_news(query)
                return {'success': True, 'result': result}
                
            else:
                return self._create_error_response(f"Unknown search action: {action}")
                
        except Exception as e:
            self.logger.error(f"Search command failed: {str(e)}")
            return self._create_error_response(f"Search command failed: {str(e)}")

    async def _handle_control_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system control commands (mouse, keyboard)"""
        action = command_data['action']
        params = command_data.get('parameters', {})
        
        try:
            if action == 'mouse_move':
                x = params.get('x', 0)
                y = params.get('y', 0)
                result = self.input_controller.mouse_move(x, y)
                return {'success': True, 'result': result}
                
            elif action == 'mouse_click':
                result = self.input_controller.mouse_click()
                return {'success': True, 'result': result}
                
            elif action == 'type_text':
                text = params.get('text', '')
                result = self.input_controller.type_text(text)
                return {'success': True, 'result': result}
                
            else:
                return self._create_error_response(f"Unknown control action: {action}")
                
        except Exception as e:
            self.logger.error(f"Control command failed: {str(e)}")
            return self._create_error_response(f"Control command failed: {str(e)}")

    async def _handle_chat_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chat/conversation commands"""
        action = command_data['action']
        params = command_data.get('parameters', {})
        
        try:
            if action == 'converse':
                message = params.get('message', '')
                # Use reasoning engine for intelligent responses
                response = await self.reasoning_engine.process_message(message)
                return {'success': True, 'response': response}
                
            else:
                return self._create_error_response(f"Unknown chat action: {action}")
                
        except Exception as e:
            self.logger.error(f"Chat command failed: {str(e)}")
            return self._create_error_response(f"Chat command failed: {str(e)}")

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': error_message,
            'mickey_response': "Oops! Mickey slipped on a banana peel! 🍌 Let's try that again!"
        }

    def _get_fun_response(self) -> str:
        """Get a random fun response from Mickey"""
        import random
        return random.choice(self.fun_responses)

    def _log_command_execution(self, command_type: CommandType, user_id: str, success: bool):
        """Log command execution for analytics"""
        self.logger.info(f"Command executed - Type: {command_type.value}, User: {user_id}, Success: {success}")

# Test function
async def test_command_dispatcher():
    """Test the command dispatcher with sample commands"""
    dispatcher = CommandDispatcher()
    
    test_commands = [
        {
            'type': 'browser',
            'action': 'open_url',
            'parameters': {'url': 'https://github.com'},
            'user_id': 'test_user'
        },
        {
            'type': 'chat',
            'action': 'converse',
            'parameters': {'message': 'Hello Mickey!'},
            'user_id': 'test_user'
        }
    ]
    
    for command in test_commands:
        result = await dispatcher.dispatch_command(command)
        print(f"Command: {command['action']}")
        print(f"Result: {result}")
        print("---")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_command_dispatcher())