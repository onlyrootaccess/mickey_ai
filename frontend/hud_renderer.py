# Dynamic UI elements
"""
Mickey AI - HUD Renderer
Heads-Up Display for Mickey AI with real-time status, animations, and interactive elements
"""

import logging
import tkinter as tk
from tkinter import ttk
import time
import threading
from typing import Dict, List, Any, Optional
from enum import Enum
import math
from PIL import Image, ImageTk
import requests
from io import BytesIO

class HUDTheme(Enum):
    MODERN = "modern"
    CLASSIC = "classic"
    DARK = "dark"
    MICKEY = "mickey"

class HUDRenderer:
    def __init__(self, master, theme: HUDTheme = HUDTheme.MICKEY):
        self.logger = logging.getLogger(__name__)
        self.master = master
        self.theme = theme
        
        # HUD State
        self.is_visible = True
        self.animation_state = "idle"
        self.current_expression = "happy"
        
        # Theme colors
        self.colors = self._get_theme_colors()
        
        # HUD Elements
        self.canvas = None
        self.status_text = None
        self.voice_indicator = None
        self.animation_elements = {}
        
        # Animation properties
        self.animation_frame = 0
        self.animation_direction = 1
        self.is_animating = False
        
        # Real-time data
        self.system_status = {}
        self.voice_level = 0
        self.current_action = "Ready"
        
        # Mickey expressions
        self.expressions = {
            "happy": "😊",
            "listening": "👂",
            "thinking": "🤔",
            "speaking": "🗣️",
            "working": "🔧",
            "error": "😅"
        }
        
        self.logger.info("🖥️ HUD Renderer initialized - Ready to display!")

    def _get_theme_colors(self) -> Dict[str, str]:
        """Get color scheme based on theme"""
        themes = {
            HUDTheme.MODERN: {
                'bg': '#1a1a1a',
                'fg': '#ffffff',
                'accent': '#007acc',
                'secondary': '#2d2d30',
                'text': '#cccccc'
            },
            HUDTheme.CLASSIC: {
                'bg': '#2c3e50',
                'fg': '#ecf0f1',
                'accent': '#e74c3c',
                'secondary': '#34495e',
                'text': '#bdc3c7'
            },
            HUDTheme.DARK: {
                'bg': '#000000',
                'fg': '#00ff00',
                'accent': '#ff00ff',
                'secondary': '#222222',
                'text': '#00ff00'
            },
            HUDTheme.MICKEY: {
                'bg': '#ff0000',
                'fg': '#000000',
                'accent': '#ffff00',
                'secondary': '#000000',
                'text': '#ffffff'
            }
        }
        return themes.get(self.theme, themes[HUDTheme.MICKEY])

    def create_hud(self) -> tk.Canvas:
        """Create the main HUD canvas"""
        try:
            # Create canvas
            self.canvas = tk.Canvas(
                self.master,
                bg=self.colors['bg'],
                highlightthickness=0,
                width=400,
                height=300
            )
            self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create HUD elements
            self._create_mickey_avatar()
            self._create_status_display()
            self._create_voice_indicator()
            self._create_system_monitor()
            self._create_quick_actions()
            
            # Start animation loop
            self._start_animation_loop()
            
            self.logger.info("HUD created successfully")
            return self.canvas
            
        except Exception as e:
            self.logger.error(f"HUD creation failed: {str(e)}")
            raise

    def _create_mickey_avatar(self):
        """Create Mickey Mouse avatar with animations"""
        # Mickey's head (circle)
        self.animation_elements['head'] = self.canvas.create_oval(
            50, 50, 150, 150,
            fill=self.colors['accent'],
            outline=self.colors['fg'],
            width=3
        )
        
        # Mickey's ears
        self.animation_elements['ear_left'] = self.canvas.create_oval(
            30, 30, 70, 70,
            fill=self.colors['accent'],
            outline=self.colors['fg'],
            width=2
        )
        
        self.animation_elements['ear_right'] = self.canvas.create_oval(
            130, 30, 170, 70,
            fill=self.colors['accent'],
            outline=self.colors['fg'],
            width=2
        )
        
        # Expression text
        self.animation_elements['expression'] = self.canvas.create_text(
            100, 100,
            text=self.expressions[self.current_expression],
            font=('Arial', 20, 'bold'),
            fill=self.colors['fg']
        )

    def _create_status_display(self):
        """Create status text display"""
        self.animation_elements['status_bg'] = self.canvas.create_rectangle(
            180, 50, 380, 100,
            fill=self.colors['secondary'],
            outline=self.colors['accent'],
            width=2
        )
        
        self.animation_elements['status_text'] = self.canvas.create_text(
            280, 75,
            text="Mickey AI Ready!",
            font=('Arial', 12, 'bold'),
            fill=self.colors['text'],
            width=180
        )

    def _create_voice_indicator(self):
        """Create voice activity indicator"""
        # Voice level background
        self.animation_elements['voice_bg'] = self.canvas.create_rectangle(
            180, 110, 380, 130,
            fill=self.colors['secondary'],
            outline=self.colors['accent'],
            width=1
        )
        
        # Voice level indicator
        self.animation_elements['voice_level'] = self.canvas.create_rectangle(
            182, 112, 182, 128,
            fill=self.colors['accent'],
            outline=''
        )
        
        # Voice label
        self.animation_elements['voice_label'] = self.canvas.create_text(
            280, 140,
            text="Voice Activity",
            font=('Arial', 10),
            fill=self.colors['text']
        )

    def _create_system_monitor(self):
        """Create system monitoring display"""
        # CPU usage
        self.animation_elements['cpu_bg'] = self.canvas.create_rectangle(
            180, 160, 380, 180,
            fill=self.colors['secondary'],
            outline=self.colors['accent'],
            width=1
        )
        
        self.animation_elements['cpu_level'] = self.canvas.create_rectangle(
            182, 162, 182, 178,
            fill='#00ff00',
            outline=''
        )
        
        self.animation_elements['cpu_label'] = self.canvas.create_text(
            280, 190,
            text="CPU: 0%",
            font=('Arial', 10),
            fill=self.colors['text']
        )
        
        # Memory usage
        self.animation_elements['memory_bg'] = self.canvas.create_rectangle(
            180, 200, 380, 220,
            fill=self.colors['secondary'],
            outline=self.colors['accent'],
            width=1
        )
        
        self.animation_elements['memory_level'] = self.canvas.create_rectangle(
            182, 202, 182, 218,
            fill='#ffff00',
            outline=''
        )
        
        self.animation_elements['memory_label'] = self.canvas.create_text(
            280, 230,
            text="Memory: 0%",
            font=('Arial', 10),
            fill=self.colors['text']
        )

    def _create_quick_actions(self):
        """Create quick action buttons"""
        actions = [
            ("🎤", "Listen", 60, 260),
            ("🔍", "Search", 120, 260),
            ("🎵", "Music", 180, 260),
            ("⚙️", "Settings", 240, 260)
        ]
        
        for i, (icon, tooltip, x, y) in enumerate(actions):
            btn_bg = self.canvas.create_oval(
                x-20, y-20, x+20, y+20,
                fill=self.colors['secondary'],
                outline=self.colors['accent'],
                width=2
            )
            
            btn_icon = self.canvas.create_text(
                x, y,
                text=icon,
                font=('Arial', 12),
                fill=self.colors['text']
            )
            
            # Store button elements
            self.animation_elements[f'btn_{i}'] = (btn_bg, btn_icon)

    def _start_animation_loop(self):
        """Start the HUD animation loop"""
        self.is_animating = True
        self._animate_hud()

    def _animate_hud(self):
        """Main HUD animation loop"""
        if not self.is_animating:
            return
            
        try:
            # Update animation frame
            self.animation_frame += self.animation_direction
            
            # Mickey breathing animation
            breath_scale = 1 + 0.05 * math.sin(self.animation_frame * 0.1)
            self._animate_breathing(breath_scale)
            
            # Update voice indicator
            self._update_voice_indicator()
            
            # Update system monitors
            self._update_system_monitors()
            
            # Schedule next frame
            self.master.after(50, self._animate_hud)
            
        except Exception as e:
            self.logger.error(f"Animation error: {str(e)}")
            self.master.after(100, self._animate_hud)

    def _animate_breathing(self, scale: float):
        """Animate Mickey's breathing effect"""
        if 'head' not in self.animation_elements:
            return
            
        # Scale head size
        base_size = 50
        scaled_size = base_size * scale
        
        self.canvas.coords(
            self.animation_elements['head'],
            100 - scaled_size, 100 - scaled_size,
            100 + scaled_size, 100 + scaled_size
        )

    def _update_voice_indicator(self):
        """Update voice activity indicator"""
        if 'voice_level' not in self.animation_elements:
            return
            
        # Simulate voice activity (in real implementation, this would use actual voice data)
        voice_width = 182 + (self.voice_level * 196)
        self.canvas.coords(
            self.animation_elements['voice_level'],
            182, 112, voice_width, 128
        )

    def _update_system_monitors(self):
        """Update system monitoring displays"""
        # CPU usage
        cpu_percent = self.system_status.get('cpu_percent', 0)
        cpu_width = 182 + (cpu_percent * 1.96)  # Scale to 200px width
        
        if 'cpu_level' in self.animation_elements:
            self.canvas.coords(
                self.animation_elements['cpu_level'],
                182, 162, cpu_width, 178
            )
            
            # Update color based on usage
            cpu_color = '#00ff00' if cpu_percent < 50 else '#ffff00' if cpu_percent < 80 else '#ff0000'
            self.canvas.itemconfig(self.animation_elements['cpu_level'], fill=cpu_color)
            
            # Update label
            self.canvas.itemconfig(
                self.animation_elements['cpu_label'],
                text=f"CPU: {cpu_percent:.1f}%"
            )
        
        # Memory usage
        memory_percent = self.system_status.get('memory_percent', 0)
        memory_width = 182 + (memory_percent * 1.96)
        
        if 'memory_level' in self.animation_elements:
            self.canvas.coords(
                self.animation_elements['memory_level'],
                182, 202, memory_width, 218
            )
            
            # Update label
            self.canvas.itemconfig(
                self.animation_elements['memory_label'],
                text=f"Memory: {memory_percent:.1f}%"
            )

    def update_status(self, status: str, expression: str = None):
        """Update HUD status message"""
        try:
            if expression and expression in self.expressions:
                self.current_expression = expression
                self.canvas.itemconfig(
                    self.animation_elements['expression'],
                    text=self.expressions[expression]
                )
            
            if 'status_text' in self.animation_elements:
                self.canvas.itemconfig(
                    self.animation_elements['status_text'],
                    text=status
                )
            
            self.current_action = status
            self.logger.info(f"HUD Status updated: {status}")
            
        except Exception as e:
            self.logger.error(f"Status update failed: {str(e)}")

    def update_voice_level(self, level: float):
        """Update voice activity level (0.0 to 1.0)"""
        self.voice_level = max(0.0, min(1.0, level))

    def update_system_status(self, status_data: Dict[str, Any]):
        """Update system status data"""
        self.system_status.update(status_data)

    def set_theme(self, theme: HUDTheme):
        """Change HUD theme"""
        self.theme = theme
        self.colors = self._get_theme_colors()
        self._refresh_theme()

    def _refresh_theme(self):
        """Refresh all elements with new theme colors"""
        if not self.canvas:
            return
            
        try:
            # Update canvas background
            self.canvas.config(bg=self.colors['bg'])
            
            # Update all elements (simplified implementation)
            # In a full implementation, we would update each element's colors
            
        except Exception as e:
            self.logger.error(f"Theme refresh failed: {str(e)}")

    def show_hud(self):
        """Show the HUD"""
        self.is_visible = True
        if self.canvas:
            self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def hide_hud(self):
        """Hide the HUD"""
        self.is_visible = False
        if self.canvas:
            self.canvas.pack_forget()

    def toggle_hud(self):
        """Toggle HUD visibility"""
        if self.is_visible:
            self.hide_hud()
        else:
            self.show_hud()

    def cleanup(self):
        """Cleanup resources"""
        self.is_animating = False
        if self.canvas:
            self.canvas.destroy()

# Test function
def test_hud_renderer():
    """Test the HUD renderer"""
    root = tk.Tk()
    root.title("Mickey AI HUD Test")
    root.geometry("400x300")
    
    try:
        hud = HUDRenderer(root)
        hud_canvas = hud.create_hud()
        
        # Test status updates
        hud.update_status("Listening...", "listening")
        hud.update_voice_level(0.7)
        
        # Test system status
        hud.update_system_status({
            'cpu_percent': 45.5,
            'memory_percent': 67.8
        })
        
        root.mainloop()
        
    finally:
        hud.cleanup()

if __name__ == "__main__":
    test_hud_renderer()