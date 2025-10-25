# Local music/video playback
"""
Mickey AI - Media Player
Controls media playback, volume, and manages playlists
"""

import logging
import os
import time
import random
import threading
from typing import Dict, List, Any, Optional
from enum import Enum
import subprocess
import platform

class PlaybackState(Enum):
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"
    BUFFERING = "buffering"

class MediaType(Enum):
    MUSIC = "music"
    VIDEO = "video"
    PODCAST = "podcast"
    AUDIOBOOK = "audiobook"

class MediaPlayer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Player state
        self.playback_state = PlaybackState.STOPPED
        self.current_media = None
        self.current_position = 0
        self.volume_level = 80
        self.is_muted = False
        
        # Playlist management
        self.playlist: List[Dict[str, Any]] = []
        self.current_track_index = -1
        self.playlist_name = "Mickey's Playlist"
        
        # Platform-specific commands
        self.system_commands = self._get_system_commands()
        
        # Threading for background playback monitoring
        self._monitor_thread = None
        self._monitoring = False
        
        # Mickey's media personalities
        self.media_responses = {
            'play': [
                "Mickey's got the music going! 🎵",
                "Let the good times roll! Music activated! 🎶",
                "Hot dog! Time for some tunes! 🌭",
                "Mickey's DJing now! Let's dance! 💃"
            ],
            'pause': [
                "Music paused! Mickey's waiting for your signal! ⏸️",
                "Taking a little break! Say when to continue!",
                "Paused! Ready when you are! 🐭",
                "Music on hold! Mickey's listening! 👂"
            ],
            'stop': [
                "Music stopped! Mickey's ready for the next request! ⏹️",
                "All quiet now! What's next?",
                "Stopped the music! Ready for your command! 🎯",
                "Mickey's turned off the tunes! What shall we do now?"
            ],
            'volume': [
                "Volume adjusted! Mickey's got the perfect level! 🔊",
                "Sound levels modified! How's that?",
                "Mickey's tweaking the knobs! Volume changed! 🎛️",
                "Audio levels adjusted! Crystal clear! ✨"
            ]
        }
        
        self.logger.info("🎵 Media Player initialized - Ready to rock!")

    def _get_system_commands(self) -> Dict[str, Any]:
        """Get platform-specific media control commands"""
        system = platform.system().lower()
        
        if system == "windows":
            return {
                'volume_up': r'nircmd.exe changesysvolume 2000',
                'volume_down': r'nircmd.exe changesysvolume -2000',
                'mute': r'nircmd.exe mutesysvolume 1',
                'unmute': r'nircmd.exe mutesysvolume 0',
                'media_play': r'nircmd.exe mediaplay',
                'media_pause': r'nircmd.exe mediapause',
                'media_stop': r'nircmd.exe mediastop',
                'media_next': r'nircmd.exe medianext',
                'media_prev': r'nircmd.exe mediaprev'
            }
        elif system == "darwin":  # macOS
            return {
                'volume_up': 'osascript -e "set volume output volume (output volume of (get volume settings) + 10)"',
                'volume_down': 'osascript -e "set volume output volume (output volume of (get volume settings) - 10)"',
                'mute': 'osascript -e "set volume output muted true"',
                'unmute': 'osascript -e "set volume output muted false"',
                'media_play': 'osascript -e "tell application \"Music\" to play"',
                'media_pause': 'osascript -e "tell application \"Music\" to pause"',
                'media_next': 'osascript -e "tell application \"Music\" to next track"',
                'media_prev': 'osascript -e "tell application \"Music\" to previous track"'
            }
        else:  # Linux
            return {
                'volume_up': 'pactl set-sink-volume @DEFAULT_SINK@ +10%',
                'volume_down': 'pactl set-sink-volume @DEFAULT_SINK@ -10%',
                'mute': 'pactl set-sink-mute @DEFAULT_SINK@ 1',
                'unmute': 'pactl set-sink-mute @DEFAULT_SINK@ 0',
                'media_play': 'playerctl play',
                'media_pause': 'playerctl pause',
                'media_next': 'playerctl next',
                'media_prev': 'playerctl previous'
            }

    def play(self, media_path: str = None, media_type: MediaType = MediaType.MUSIC) -> Dict[str, Any]:
        """
        Start media playback
        
        Args:
            media_path: Path to media file or URL
            media_type: Type of media content
            
        Returns:
            Dictionary with playback result
        """
        try:
            if media_path:
                # Add to playlist and play
                self.add_to_playlist(media_path, media_type)
                self.current_track_index = len(self.playlist) - 1
            
            if self.current_track_index >= 0:
                self.current_media = self.playlist[self.current_track_index]
                
                # Simulate playback start (in real implementation, this would use a media library)
                self.playback_state = PlaybackState.PLAYING
                self.current_position = 0
                
                # Start monitoring thread if not already running
                if not self._monitoring:
                    self._start_monitoring()
                
                self.logger.info(f"Started playback: {self.current_media.get('title', 'Unknown')}")
                
                return {
                    'success': True,
                    'action': 'play',
                    'media': self.current_media,
                    'position': self.current_position,
                    'playback_state': self.playback_state.value,
                    'message': f"Now playing: {self.current_media.get('title', 'Unknown')}",
                    'mickey_response': random.choice(self.media_responses['play'])
                }
            else:
                # No media in playlist - try system media play
                return self._system_media_control('media_play')
                
        except Exception as e:
            self.logger.error(f"Play failed: {str(e)}")
            return self._create_error_response(f"Play failed: {str(e)}")

    def pause(self) -> Dict[str, Any]:
        """Pause current playback"""
        try:
            if self.playback_state == PlaybackState.PLAYING:
                self.playback_state = PlaybackState.PAUSED
                
                self.logger.info("Playback paused")
                
                return {
                    'success': True,
                    'action': 'pause',
                    'playback_state': self.playback_state.value,
                    'message': "Playback paused",
                    'mickey_response': random.choice(self.media_responses['pause'])
                }
            else:
                # Try system media pause
                return self._system_media_control('media_pause')
                
        except Exception as e:
            self.logger.error(f"Pause failed: {str(e)}")
            return self._create_error_response(f"Pause failed: {str(e)}")

    def stop(self) -> Dict[str, Any]:
        """Stop playback"""
        try:
            self.playback_state = PlaybackState.STOPPED
            self.current_position = 0
            
            self.logger.info("Playback stopped")
            
            return {
                'success': True,
                'action': 'stop',
                'playback_state': self.playback_state.value,
                'message': "Playback stopped",
                'mickey_response': random.choice(self.media_responses['stop'])
            }
            
        except Exception as e:
            self.logger.error(f"Stop failed: {str(e)}")
            return self._create_error_response(f"Stop failed: {str(e)}")

    def next_track(self) -> Dict[str, Any]:
        """Play next track in playlist"""
        try:
            if self.playlist and self.current_track_index < len(self.playlist) - 1:
                self.current_track_index += 1
                self.current_media = self.playlist[self.current_track_index]
                self.current_position = 0
                self.playback_state = PlaybackState.PLAYING
                
                self.logger.info(f"Next track: {self.current_media.get('title', 'Unknown')}")
                
                return {
                    'success': True,
                    'action': 'next',
                    'media': self.current_media,
                    'playback_state': self.playback_state.value,
                    'message': f"Playing next: {self.current_media.get('title', 'Unknown')}",
                    'mickey_response': "Skipping to next track! 🎵"
                }
            else:
                # Try system next track
                return self._system_media_control('media_next')
                
        except Exception as e:
            self.logger.error(f"Next track failed: {str(e)}")
            return self._create_error_response(f"Next track failed: {str(e)}")

    def previous_track(self) -> Dict[str, Any]:
        """Play previous track in playlist"""
        try:
            if self.playlist and self.current_track_index > 0:
                self.current_track_index -= 1
                self.current_media = self.playlist[self.current_track_index]
                self.current_position = 0
                self.playback_state = PlaybackState.PLAYING
                
                self.logger.info(f"Previous track: {self.current_media.get('title', 'Unknown')}")
                
                return {
                    'success': True,
                    'action': 'previous',
                    'media': self.current_media,
                    'playback_state': self.playback_state.value,
                    'message': f"Playing previous: {self.current_media.get('title', 'Unknown')}",
                    'mickey_response': "Going back one track! 🔄"
                }
            else:
                # Try system previous track
                return self._system_media_control('media_prev')
                
        except Exception as e:
            self.logger.error(f"Previous track failed: {str(e)}")
            return self._create_error_response(f"Previous track failed: {str(e)}")

    def volume_up(self, increment: int = 10) -> Dict[str, Any]:
        """Increase volume"""
        try:
            if not self.is_muted:
                self.volume_level = min(100, self.volume_level + increment)
            
            # System volume control
            result = self._system_media_control('volume_up')
            
            if result.get('success', False):
                self.logger.info(f"Volume increased to {self.volume_level}%")
                
                return {
                    'success': True,
                    'action': 'volume_up',
                    'volume_level': self.volume_level,
                    'is_muted': self.is_muted,
                    'message': f"Volume increased to {self.volume_level}%",
                    'mickey_response': random.choice(self.media_responses['volume'])
                }
            else:
                return result
                
        except Exception as e:
            self.logger.error(f"Volume up failed: {str(e)}")
            return self._create_error_response(f"Volume up failed: {str(e)}")

    def volume_down(self, decrement: int = 10) -> Dict[str, Any]:
        """Decrease volume"""
        try:
            if not self.is_muted:
                self.volume_level = max(0, self.volume_level - decrement)
            
            # System volume control
            result = self._system_media_control('volume_down')
            
            if result.get('success', False):
                self.logger.info(f"Volume decreased to {self.volume_level}%")
                
                return {
                    'success': True,
                    'action': 'volume_down',
                    'volume_level': self.volume_level,
                    'is_muted': self.is_muted,
                    'message': f"Volume decreased to {self.volume_level}%",
                    'mickey_response': random.choice(self.media_responses['volume'])
                }
            else:
                return result
                
        except Exception as e:
            self.logger.error(f"Volume down failed: {str(e)}")
            return self._create_error_response(f"Volume down failed: {str(e)}")

    def mute(self) -> Dict[str, Any]:
        """Toggle mute"""
        try:
            self.is_muted = not self.is_muted
            
            command = 'mute' if self.is_muted else 'unmute'
            result = self._system_media_control(command)
            
            if result.get('success', False):
                action = "muted" if self.is_muted else "unmuted"
                self.logger.info(f"Volume {action}")
                
                return {
                    'success': True,
                    'action': 'mute_toggle',
                    'is_muted': self.is_muted,
                    'message': f"Volume {action}",
                    'mickey_response': f"Volume {action}! {"🔇" if self.is_muted else "🔈"}"
                }
            else:
                return result
                
        except Exception as e:
            self.logger.error(f"Mute failed: {str(e)}")
            return self._create_error_response(f"Mute failed: {str(e)}")

    def _system_media_control(self, command: str) -> Dict[str, Any]:
        """Execute system media control command"""
        try:
            if command in self.system_commands:
                cmd = self.system_commands[command]
                
                # Execute command based on platform
                if platform.system().lower() == "windows":
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                else:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    return {'success': True, 'command': command}
                else:
                    return {'success': False, 'error': result.stderr}
            else:
                return {'success': False, 'error': f"Unknown command: {command}"}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def add_to_playlist(self, media_path: str, media_type: MediaType = MediaType.MUSIC, 
                       title: str = None, artist: str = None) -> Dict[str, Any]:
        """
        Add media to playlist
        
        Args:
            media_path: Path or URL to media
            media_type: Type of media
            title: Optional title
            artist: Optional artist name
            
        Returns:
            Dictionary with add result
        """
        try:
            # Extract filename if no title provided
            if not title:
                title = os.path.basename(media_path)
            
            media_item = {
                'path': media_path,
                'type': media_type.value,
                'title': title,
                'artist': artist or 'Unknown',
                'duration': 0,  # Would be extracted in real implementation
                'added_at': time.time()
            }
            
            self.playlist.append(media_item)
            
            self.logger.info(f"Added to playlist: {title}")
            
            return {
                'success': True,
                'action': 'add_to_playlist',
                'media': media_item,
                'playlist_size': len(self.playlist),
                'message': f"Added '{title}' to playlist",
                'mickey_response': f"Mickey added '{title}' to the playlist! 🎶"
            }
            
        except Exception as e:
            self.logger.error(f"Add to playlist failed: {str(e)}")
            return self._create_error_response(f"Add to playlist failed: {str(e)}")

    def remove_from_playlist(self, index: int) -> Dict[str, Any]:
        """Remove item from playlist by index"""
        try:
            if 0 <= index < len(self.playlist):
                removed_item = self.playlist.pop(index)
                
                # Adjust current track index if needed
                if index == self.current_track_index:
                    self.current_track_index = -1
                    self.current_media = None
                    self.playback_state = PlaybackState.STOPPED
                elif index < self.current_track_index:
                    self.current_track_index -= 1
                
                self.logger.info(f"Removed from playlist: {removed_item.get('title', 'Unknown')}")
                
                return {
                    'success': True,
                    'action': 'remove_from_playlist',
                    'removed_media': removed_item,
                    'playlist_size': len(self.playlist),
                    'message': f"Removed '{removed_item.get('title', 'Unknown')}' from playlist",
                    'mickey_response': f"Mickey removed that track from the playlist! 🗑️"
                }
            else:
                return self._create_error_response("Invalid playlist index")
                
        except Exception as e:
            self.logger.error(f"Remove from playlist failed: {str(e)}")
            return self._create_error_response(f"Remove from playlist failed: {str(e)}")

    def get_playlist(self) -> Dict[str, Any]:
        """Get current playlist"""
        return {
            'success': True,
            'playlist_name': self.playlist_name,
            'playlist': self.playlist,
            'current_track_index': self.current_track_index,
            'total_tracks': len(self.playlist)
        }

    def clear_playlist(self) -> Dict[str, Any]:
        """Clear entire playlist"""
        try:
            self.playlist.clear()
            self.current_track_index = -1
            self.current_media = None
            self.playback_state = PlaybackState.STOPPED
            
            self.logger.info("Playlist cleared")
            
            return {
                'success': True,
                'action': 'clear_playlist',
                'message': "Playlist cleared",
                'mickey_response': "Mickey cleared the playlist! Fresh start! 🎵"
            }
            
        except Exception as e:
            self.logger.error(f"Clear playlist failed: {str(e)}")
            return self._create_error_response(f"Clear playlist failed: {str(e)}")

    def get_player_status(self) -> Dict[str, Any]:
        """Get current player status"""
        return {
            'success': True,
            'playback_state': self.playback_state.value,
            'current_media': self.current_media,
            'current_position': self.current_position,
            'volume_level': self.volume_level,
            'is_muted': self.is_muted,
            'current_track_index': self.current_track_index,
            'playlist_size': len(self.playlist),
            'mickey_response': self._get_status_message()
        }

    def _get_status_message(self) -> str:
        """Get Mickey's status message based on player state"""
        if self.playback_state == PlaybackState.PLAYING and self.current_media:
            return f"Mickey's playing '{self.current_media.get('title', 'Unknown')}'! 🎵"
        elif self.playback_state == PlaybackState.PAUSED:
            return "Music is paused! Mickey's waiting! ⏸️"
        else:
            return "Mickey's ready to play some music! 🐭"

    def _start_monitoring(self):
        """Start background playback monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_playback, daemon=True)
        self._monitor_thread.start()

    def _monitor_playback(self):
        """Background thread to monitor playback progress"""
        while self._monitoring and self.playback_state == PlaybackState.PLAYING:
            try:
                # Simulate playback progress
                if self.current_media and self.current_position < 300:  # Assume 5min tracks
                    self.current_position += 1
                else:
                    # Track ended, play next
                    if self.playlist and self.current_track_index < len(self.playlist) - 1:
                        self.next_track()
                    else:
                        self.stop()
                        self._monitoring = False
                
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Playback monitoring error: {str(e)}")
                time.sleep(5)

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': error_message,
            'mickey_response': random.choice([
                "Oops! Mickey's having trouble with the media player! 😅",
                "Hot dog! The music magic isn't working right now! 🌭",
                "Mickey's speakers are feeling shy! Try again?",
                "Uh oh! Media control failed! Mickey's checking the wires! 🔌"
            ])
        }

    def shutdown(self):
        """Cleanup resources"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Media player shutdown")

# Test function
def test_media_player():
    """Test the media player"""
    player = MediaPlayer()
    
    try:
        # Test adding to playlist
        result = player.add_to_playlist("/music/song1.mp3", MediaType.MUSIC, "Test Song", "Test Artist")
        print("Add to playlist:", result)
        
        # Test play
        result = player.play()
        print("Play:", result)
        
        # Test volume control
        result = player.volume_up()
        print("Volume up:", result)
        
        # Test player status
        status = player.get_player_status()
        print("Player status:", status)
        
        # Test playlist
        playlist = player.get_playlist()
        print("Playlist:", playlist)
        
        time.sleep(2)
        
        # Test pause
        result = player.pause()
        print("Pause:", result)
        
    finally:
        player.shutdown()

if __name__ == "__main__":
    test_media_player()