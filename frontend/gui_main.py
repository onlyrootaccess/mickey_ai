# Primary transparent window
"""
M.I.C.K.E.Y. AI Assistant - Main GUI Interface
Made In Crisis, Keeping Everything Yours

THIRTEENTH FILE IN PIPELINE: Transparent HUD interface with Jarvis-inspired 
"Wire Gucci" aesthetic. Features pulse animations, voice visualizer, and system status.
"""

import asyncio
import logging
import time
import threading
import json
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import GUI libraries
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
import pygame

# Import Mickey AI configuration
from config.settings import get_config
from config.constants import (
    GUIConstants, SystemConstants, PersonalityConstants,
    AudioConstants, ErrorCodes
)

# Setup logging
logger = logging.getLogger("MickeyGUI")


class GUIState(Enum):
    """GUI state machine states."""
    BOOTING = "booting"
    SECURITY_CHECK = "security_check"
    READY = "ready"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class AnimationFrame:
    """Animation frame data container."""
    frame_id: str
    elements: List[Dict]
    duration: int
    callback: Optional[Callable] = None


class PulseAnimation:
    """Jarvis-inspired pulse animation system."""
    
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.config = get_config()
        self.pulses = []
        self.is_animating = False
        
    def create_pulse(self, x: int, y: int, color: str = None, max_radius: int = 100):
        """Create a new pulse animation."""
        if color is None:
            color = GUIConstants.COLORS["primary"]
        
        pulse_id = f"pulse_{len(self.pulses)}"
        pulse = {
            'id': pulse_id,
            'x': x,
            'y': y,
            'radius': 5,
            'max_radius': max_radius,
            'color': color,
            'alpha': 1.0,
            'speed': 2,
            'canvas_id': None
        }
        
        # Create initial circle
        pulse['canvas_id'] = self.canvas.create_oval(
            x - 5, y - 5, x + 5, y + 5,
            outline=color,
            width=2,
            tags=pulse_id
        )
        
        self.pulses.append(pulse)
        
        if not self.is_animating:
            self.start_animation()
    
    def start_animation(self):
        """Start the pulse animation loop."""
        self.is_animating = True
        self._animate_frame()
    
    def _animate_frame(self):
        """Animate a single frame of all pulses."""
        pulses_to_remove = []
        
        for pulse in self.pulses:
            # Update pulse properties
            pulse['radius'] += pulse['speed']
            pulse['alpha'] = 1.0 - (pulse['radius'] / pulse['max_radius'])
            
            # Remove if pulse has expanded too much
            if pulse['radius'] >= pulse['max_radius']:
                pulses_to_remove.append(pulse)
                continue
            
            # Update canvas element
            x1 = pulse['x'] - pulse['radius']
            y1 = pulse['y'] - pulse['radius']
            x2 = pulse['x'] + pulse['radius']
            y2 = pulse['y'] + pulse['radius']
            
            self.canvas.coords(pulse['canvas_id'], x1, y1, x2, y2)
            
            # Update color with alpha
            color = self._apply_alpha(pulse['color'], pulse['alpha'])
            self.canvas.itemconfig(pulse['canvas_id'], outline=color)
        
        # Remove finished pulses
        for pulse in pulses_to_remove:
            self.canvas.delete(pulse['canvas_id'])
            self.pulses.remove(pulse)
        
        # Continue animation if there are active pulses
        if self.pulses:
            self.canvas.after(16, self._animate_frame)  # ~60 FPS
        else:
            self.is_animating = False
    
    def _apply_alpha(self, color: str, alpha: float) -> str:
        """Apply alpha to hex color (simplified)."""
        # This is a simplified version - in a real implementation,
        # we'd use a more sophisticated approach for transparency
        return color
    
    def stop_all_pulses(self):
        """Stop all pulse animations."""
        for pulse in self.pulses:
            self.canvas.delete(pulse['canvas_id'])
        self.pulses.clear()
        self.is_animating = False


class VoiceVisualizer:
    """Real-time voice activity visualizer."""
    
    def __init__(self, canvas: tk.Canvas, width: int, height: int):
        self.canvas = canvas
        self.width = width
        self.height = height
        self.bars = []
        self.is_active = False
        
        # Create visualizer bars
        bar_count = 20
        bar_width = (width - 20) // bar_count
        
        for i in range(bar_count):
            x = 10 + i * (bar_width + 2)
            bar_id = canvas.create_rectangle(
                x, height // 2, x + bar_width, height // 2,
                fill=GUIConstants.COLORS["primary"],
                outline=""
            )
            self.bars.append({
                'id': bar_id,
                'base_height': height // 2,
                'x': x,
                'width': bar_width
            })
    
    def update_visualizer(self, audio_level: float):
        """Update visualizer with new audio level."""
        if not self.is_active:
            return
        
        bar_heights = self._calculate_bar_heights(audio_level)
        
        for i, bar in enumerate(self.bars):
            height = bar_heights[i] if i < len(bar_heights) else 0
            y1 = bar['base_height'] - height
            y2 = bar['base_height'] + height
            
            self.canvas.coords(bar['id'], bar['x'], y1, bar['x'] + bar['width'], y2)
            
            # Update color based on intensity
            intensity = height / (self.height // 2)
            color = self._get_intensity_color(intensity)
            self.canvas.itemconfig(bar['id'], fill=color)
    
    def _calculate_bar_heights(self, audio_level: float) -> List[int]:
        """Calculate bar heights based on audio level."""
        # Create a wave pattern based on audio level
        bar_count = len(self.bars)
        heights = []
        
        for i in range(bar_count):
            # Sine wave pattern with audio level modulation
            wave = (audio_level * 0.8 + 0.2) * math.sin(
                (i / bar_count) * math.pi * 2 + time.time() * 10
            )
            height = int(abs(wave) * (self.height // 2 - 10))
            heights.append(height)
        
        return heights
    
    def _get_intensity_color(self, intensity: float) -> str:
        """Get color based on intensity level."""
        colors = GUIConstants.COLORS
        
        if intensity > 0.8:
            return colors["error"]
        elif intensity > 0.6:
            return colors["warning"]
        elif intensity > 0.4:
            return colors["accent"]
        else:
            return colors["primary"]
    
    def set_active(self, active: bool):
        """Set visualizer active state."""
        self.is_active = active
        
        if not active:
            # Reset all bars to zero
            for bar in self.bars:
                self.canvas.coords(
                    bar['id'], 
                    bar['x'], bar['base_height'],
                    bar['x'] + bar['width'], bar['base_height']
                )
                self.canvas.itemconfig(bar['id'], fill=GUIConstants.COLORS["primary"])


class ResponseDisplay:
    """Animated text display for Mickey's responses."""
    
    def __init__(self, parent, width: int, height: int):
        self.parent = parent
        self.width = width
        self.height = height
        
        # Create text widget with custom styling
        self.text_widget = tk.Text(
            parent,
            width=width // 8,  # Approximate character width
            height=height // 20,  # Approximate line height
            bg=GUIConstants.COLORS["background"],
            fg=GUIConstants.COLORS["text_primary"],
            font=(GUIConstants.FONTS["primary"], GUIConstants.FONTS["sizes"]["normal"]),
            relief="flat",
            borderwidth=0,
            wrap="word",
            state="disabled"
        )
        
        # Configure tags for different text styles
        self.text_widget.tag_configure("normal", foreground=GUIConstants.COLORS["text_primary"])
        self.text_widget.tag_configure("highlight", foreground=GUIConstants.COLORS["primary"])
        self.text_widget.tag_configure("warning", foreground=GUIConstants.COLORS["warning"])
        self.text_widget.tag_configure("error", foreground=GUIConstants.COLORS["error"])
        
        self.current_text = ""
        self.is_animating = False
    
    def pack(self, **kwargs):
        """Pack the text widget."""
        self.text_widget.pack(**kwargs)
    
    def display_text(self, text: str, animate: bool = True):
        """Display text with optional typewriter animation."""
        self.current_text = text
        
        if not animate:
            self._set_text_immediate(text)
            return
        
        self.is_animating = True
        self._animate_text(text, 0)
    
    def _animate_text(self, text: str, position: int):
        """Animate text display character by character."""
        if position <= len(text):
            displayed_text = text[:position]
            self._set_text_immediate(displayed_text)
            
            if position < len(text):
                # Schedule next character
                delay = 30  # milliseconds between characters
                self.parent.after(delay, lambda: self._animate_text(text, position + 1))
            else:
                self.is_animating = False
        else:
            self.is_animating = False
    
    def _set_text_immediate(self, text: str):
        """Set text immediately without animation."""
        self.text_widget.config(state="normal")
        self.text_widget.delete(1.0, "end")
        
        # Simple text formatting (could be enhanced with more sophisticated parsing)
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line.strip():
                # Apply basic formatting
                if line.startswith('!') or line.startswith('⚠️'):
                    self.text_widget.insert("end", line + '\n', "warning")
                elif line.startswith('❌') or line.startswith('ERROR'):
                    self.text_widget.insert("end", line + '\n', "error")
                elif any(word in line.lower() for word in ['mickey', 'ai', 'assistant']):
                    self.text_widget.insert("end", line + '\n', "highlight")
                else:
                    self.text_widget.insert("end", line + '\n', "normal")
            else:
                self.text_widget.insert("end", '\n')
        
        self.text_widget.config(state="disabled")
        self.text_widget.see("end")
    
    def clear(self):
        """Clear the display."""
        self.text_widget.config(state="normal")
        self.text_widget.delete(1.0, "end")
        self.text_widget.config(state="disabled")
        self.current_text = ""
        self.is_animating = False


class StatusIndicator:
    """System status indicator with animated icons."""
    
    def __init__(self, parent, title: str, size: int = 24):
        self.parent = parent
        self.title = title
        self.size = size
        self.status = "unknown"
        
        # Create container frame
        self.frame = ttk.Frame(parent, style="Status.TFrame")
        
        # Create status icon (canvas-based)
        self.canvas = tk.Canvas(
            self.frame,
            width=size,
            height=size,
            bg=GUIConstants.COLORS["background"],
            highlightthickness=0
        )
        self.canvas.pack(side="left", padx=(0, 5))
        
        # Create status label
        self.label = ttk.Label(
            self.frame,
            text=title,
            style="Status.TLabel"
        )
        self.label.pack(side="left")
        
        # Create value label
        self.value_label = ttk.Label(
            self.frame,
            text="",
            style="StatusValue.TLabel"
        )
        self.value_label.pack(side="right")
        
        self._draw_icon()
    
    def pack(self, **kwargs):
        """Pack the status indicator."""
        self.frame.pack(**kwargs)
    
    def set_status(self, status: str, value: str = ""):
        """Set status and optional value."""
        self.status = status
        self.value_label.config(text=value)
        self._draw_icon()
    
    def _draw_icon(self):
        """Draw status icon based on current status."""
        self.canvas.delete("all")
        
        center = self.size // 2
        radius = self.size // 3
        
        colors = {
            "online": GUIConstants.COLORS["success"],
            "offline": GUIConstants.COLORS["error"],
            "warning": GUIConstants.COLORS["warning"],
            "processing": GUIConstants.COLORS["accent"],
            "unknown": GUIConstants.COLORS["text_secondary"]
        }
        
        color = colors.get(self.status, colors["unknown"])
        
        if self.status == "processing":
            # Animated processing icon (spinning)
            self.canvas.create_arc(
                center - radius, center - radius,
                center + radius, center + radius,
                start=time.time() * 50 % 360,
                extent=120,
                outline=color,
                width=2,
                style="arc",
                tags="icon"
            )
        else:
            # Static icon based on status
            if self.status == "online":
                # Green circle
                self.canvas.create_oval(
                    center - radius, center - radius,
                    center + radius, center + radius,
                    fill=color,
                    outline="",
                    tags="icon"
                )
            elif self.status == "offline":
                # Red X
                self.canvas.create_line(
                    center - radius, center - radius,
                    center + radius, center + radius,
                    fill=color,
                    width=2,
                    tags="icon"
                )
                self.canvas.create_line(
                    center - radius, center + radius,
                    center + radius, center - radius,
                    fill=color,
                    width=2,
                    tags="icon"
                )
            elif self.status == "warning":
                # Yellow triangle
                points = [
                    center, center - radius,
                    center - radius, center + radius,
                    center + radius, center + radius
                ]
                self.canvas.create_polygon(
                    points,
                    fill=color,
                    outline="",
                    tags="icon"
                )
        
        if self.status == "processing":
            # Schedule redraw for animation
            self.parent.after(50, self._draw_icon)


class MickeyGUI:
    """
    Main GUI class for M.I.C.K.E.Y. AI Assistant.
    Features transparent HUD with Jarvis-inspired design.
    """
    
    def __init__(self):
        self.config = get_config()
        self.root = None
        self.is_initialized = False
        self.current_state = GUIState.BOOTING
        
        # GUI components
        self.pulse_animation = None
        self.voice_visualizer = None
        self.response_display = None
        self.status_indicators = {}
        
        # Backend API client
        self.api_client = None
        
        # Animation and state tracking
        self.animation_queue = []
        self.current_animation = None
        
    async def initialize(self):
        """Initialize the GUI system."""
        try:
            logger.info("Initializing Mickey GUI...")
            
            # Initialize pygame for audio (needed for some systems)
            pygame.mixer.init()
            
            # Create main window in main thread
            await self._create_main_window()
            
            # Initialize backend API connection
            await self._initialize_api_client()
            
            self.is_initialized = True
            logger.info("✅ Mickey GUI initialized")
            
        except Exception as e:
            logger.error(f"❌ GUI initialization failed: {str(e)}")
            raise
    
    async def _create_main_window(self):
        """Create the main transparent HUD window."""
        # This needs to run in the main thread
        def create_window():
            # Set customtkinter theme
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")
            
            # Create main window
            self.root = ctk.CTk()
            
            # Configure window properties
            self.root.title(f"{SystemConstants.APP_NAME} - {SystemConstants.APP_FULL_NAME}")
            self.root.geometry(f"{self.config.gui.window_width}x{self.config.gui.window_height}")
            
            # Make window transparent
            self.root.attributes('-alpha', self.config.gui.transparency)
            self.root.attributes('-topmost', self.config.gui.always_on_top)
            
            # Remove window decorations
            self.root.overrideredirect(True)
            
            # Center window on screen
            self._center_window()
            
            # Create main content
            self._create_main_content()
            
            # Start boot sequence
            self._start_boot_sequence()
        
        # Run in main thread
        if threading.current_thread() == threading.main_thread():
            create_window()
        else:
            # Schedule in main thread
            self.root.after(0, create_window)
    
    def _center_window(self):
        """Center the window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_main_content(self):
        """Create main GUI content."""
        # Create main frame with wireframe style
        self.main_frame = ctk.CTkFrame(
            self.root,
            fg_color=GUIConstants.COLORS["background"],
            border_color=GUIConstants.COLORS["primary"],
            border_width=2
        )
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create header with pulse animation canvas
        self._create_header()
        
        # Create content area
        self._create_content_area()
        
        # Create footer with status indicators
        self._create_footer()
    
    def _create_header(self):
        """Create header with logo and pulse animation."""
        header_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="transparent",
            height=100
        )
        header_frame.pack(fill="x", padx=20, pady=10)
        header_frame.pack_propagate(False)
        
        # Create pulse animation canvas
        self.pulse_canvas = tk.Canvas(
            header_frame,
            bg=GUIConstants.COLORS["background"],
            highlightthickness=0,
            height=80
        )
        self.pulse_canvas.pack(fill="x", padx=20)
        
        # Initialize pulse animation
        self.pulse_animation = PulseAnimation(self.pulse_canvas)
        
        # Add title
        title_label = ctk.CTkLabel(
            header_frame,
            text="M.I.C.K.E.Y. AI",
            text_color=GUIConstants.COLORS["primary"],
            font=(GUIConstants.FONTS["primary"], GUIConstants.FONTS["sizes"]["title"], "bold")
        )
        title_label.place(relx=0.5, rely=0.5, anchor="center")
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Made In Crisis, Keeping Everything Yours",
            text_color=GUIConstants.COLORS["text_secondary"],
            font=(GUIConstants.FONTS["primary"], GUIConstants.FONTS["sizes"]["small"])
        )
        subtitle_label.place(relx=0.5, rely=0.8, anchor="center")
    
    def _create_content_area(self):
        """Create main content area with response display and visualizer."""
        content_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="transparent"
        )
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Create response display
        self.response_display = ResponseDisplay(content_frame, 600, 200)
        self.response_display.pack(fill="both", expand=True, pady=(0, 10))
        
        # Create voice visualizer
        visualizer_frame = ctk.CTkFrame(
            content_frame,
            fg_color="transparent",
            height=60
        )
        visualizer_frame.pack(fill="x", pady=5)
        visualizer_frame.pack_propagate(False)
        
        self.visualizer_canvas = tk.Canvas(
            visualizer_frame,
            bg=GUIConstants.COLORS["background"],
            highlightthickness=0,
            height=50
        )
        self.visualizer_canvas.pack(fill="x", padx=10)
        
        self.voice_visualizer = VoiceVisualizer(self.visualizer_canvas, 600, 50)
    
    def _create_footer(self):
        """Create footer with status indicators."""
        footer_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="transparent",
            height=60
        )
        footer_frame.pack(fill="x", padx=20, pady=10)
        footer_frame.pack_propagate(False)
        
        # Create status indicators grid
        status_grid = ttk.Frame(footer_frame, style="StatusGrid.TFrame")
        status_grid.pack(fill="both", expand=True)
        
        # System status
        self.status_indicators["system"] = StatusIndicator(status_grid, "System")
        self.status_indicators["system"].pack(side="left", padx=10)
        
        # Voice status
        self.status_indicators["voice"] = StatusIndicator(status_grid, "Voice")
        self.status_indicators["voice"].pack(side="left", padx=10)
        
        # Security status
        self.status_indicators["security"] = StatusIndicator(status_grid, "Security")
        self.status_indicators["security"].pack(side="left", padx=10)
        
        # AI status
        self.status_indicators["ai"] = StatusIndicator(status_grid, "AI")
        self.status_indicators["ai"].pack(side="left", padx=10)
    
    def _start_boot_sequence(self):
        """Start the boot animation sequence."""
        logger.info("Starting GUI boot sequence...")
        
        # Initial boot message
        self.response_display.display_text(
            "🚀 Initializing M.I.C.K.E.Y. AI Assistant...\n"
            "Made In Crisis, Keeping Everything Yours\n\n"
            "System boot sequence initiated...",
            animate=True
        )
        
        # Start pulse animation
        self.pulse_animation.create_pulse(
            self.pulse_canvas.winfo_width() // 2,
            self.pulse_canvas.winfo_height() // 2,
            GUIConstants.COLORS["primary"],
            150
        )
        
        # Update status indicators
        self.status_indicators["system"].set_status("processing")
        self.status_indicators["voice"].set_status("offline")
        self.status_indicators["security"].set_status("unknown")
        self.status_indicators["ai"].set_status("processing")
        
        # Schedule state transitions
        self.root.after(2000, self._update_boot_state, "security_check")
    
    def _update_boot_state(self, state: str):
        """Update boot sequence state."""
        if state == "security_check":
            self.response_display.display_text(
                "🔒 Starting security systems...\n"
                "Face recognition: Initializing\n"
                "Voice biometrics: Calibrating",
                animate=True
            )
            
            self.status_indicators["security"].set_status("processing")
            self.root.after(3000, self._update_boot_state, "ai_init")
            
        elif state == "ai_init":
            self.response_display.display_text(
                "🧠 Initializing AI core...\n"
                "Neural networks: Online\n"
                "Reasoning engine: Active\n"
                "Personality matrix: Loaded",
                animate=True
            )
            
            self.status_indicators["ai"].set_status("online")
            self.root.after(2000, self._update_boot_state, "voice_init")
            
        elif state == "voice_init":
            self.response_display.display_text(
                "🎤 Audio systems coming online...\n"
                "Speech recognition: Ready\n"
                "Voice synthesis: Operational",
                animate=True
            )
            
            self.status_indicators["voice"].set_status("online")
            self.root.after(2000, self._update_boot_state, "ready")
            
        elif state == "ready":
            self.response_display.display_text(
                "✅ M.I.C.K.E.Y. AI Assistant ready!\n\n"
                "Say 'Hey Mickey' to begin...",
                animate=True
            )
            
            self.status_indicators["system"].set_status("online")
            self.current_state = GUIState.READY
            
            # Final pulse animation
            self.pulse_animation.create_pulse(
                self.pulse_canvas.winfo_width() // 2,
                self.pulse_canvas.winfo_height() // 2,
                GUIConstants.COLORS["success"],
                200
            )
    
    async def _initialize_api_client(self):
        """Initialize backend API client."""
        # This would typically create an HTTP client to communicate with the backend API
        # For now, we'll simulate the connection
        logger.info("Initializing API client...")
        
        # Simulate API connection
        await asyncio.sleep(1)
        
        # Update GUI status
        if self.root:
            self.root.after(0, lambda: self.status_indicators["system"].set_status("online", "Connected"))
    
    def update_voice_level(self, level: float):
        """Update voice visualizer with audio level."""
        if self.voice_visualizer and self.root:
            self.root.after(0, lambda: self.voice_visualizer.update_visualizer(level))
    
    def set_listening_state(self, listening: bool):
        """Set listening state and update visualizer."""
        if self.voice_visualizer and self.root:
            self.root.after(0, lambda: self.voice_visualizer.set_active(listening))
            
            if listening:
                self.current_state = GUIState.LISTENING
                self.response_display.display_text("🎤 Listening...", animate=False)
            else:
                self.current_state = GUIState.READY
    
    def display_response(self, text: str, animate: bool = True):
        """Display Mickey's response text."""
        if self.response_display and self.root:
            self.root.after(0, lambda: self.response_display.display_text(text, animate))
            self.current_state = GUIState.SPEAKING
    
    def set_processing_state(self, processing: bool):
        """Set processing state."""
        if processing:
            self.current_state = GUIState.PROCESSING
            if self.response_display and self.root:
                self.root.after(0, lambda: self.response_display.display_text("🤔 Processing...", animate=False))
        else:
            self.current_state = GUIState.READY
    
    def show_error(self, error_message: str):
        """Display error message."""
        if self.response_display and self.root:
            self.root.after(0, lambda: self.response_display.display_text(
                f"❌ Error: {error_message}\n\nPlease check system logs for details.",
                animate=True
            ))
            self.current_state = GUIState.ERROR
    
    def update_status(self, component: str, status: str, value: str = ""):
        """Update status indicator."""
        if component in self.status_indicators and self.root:
            self.root.after(0, lambda: self.status_indicators[component].set_status(status, value))
    
    async def run(self):
        """Run the GUI main loop."""
        if not self.is_initialized:
            raise RuntimeError("GUI not initialized")
        
        logger.info("Starting GUI main loop...")
        
        # Start the Tkinter main loop in a separate thread
        def start_main_loop():
            try:
                self.root.mainloop()
            except Exception as e:
                logger.error(f"GUI main loop error: {str(e)}")
        
        # Run Tkinter in main thread
        if threading.current_thread() == threading.main_thread():
            start_main_loop()
        else:
            # Start in separate thread
            gui_thread = threading.Thread(target=start_main_loop, daemon=True)
            gui_thread.start()
        
        # Keep the async context alive
        while self.root and self.root.winfo_exists():
            await asyncio.sleep(0.1)
    
    async def shutdown(self):
        """Shutdown the GUI gracefully."""
        logger.info("Shutting down Mickey GUI...")
        
        try:
            if self.root:
                # Display shutdown message
                self.display_response(
                    "🛑 Shutting down M.I.C.K.E.Y. AI Assistant...\n\n"
                    "Thank you for using Mickey AI!\n"
                    "Made In Crisis, Keeping Everything Yours",
                    animate=True
                )
                
                # Update status indicators
                for indicator in self.status_indicators.values():
                    indicator.set_status("offline")
                
                # Wait for message to display
                await asyncio.sleep(3)
                
                # Close window
                self.root.after(0, self.root.quit)
                self.root = None
            
            logger.info("✅ Mickey GUI shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during GUI shutdown: {str(e)}")


# Global GUI instance
_gui_instance: Optional[MickeyGUI] = None


async def get_gui() -> MickeyGUI:
    """Get or create global GUI instance."""
    global _gui_instance
    
    if _gui_instance is None:
        _gui_instance = MickeyGUI()
        await _gui_instance.initialize()
    
    return _gui_instance


async def main():
    """Command-line testing for GUI."""
    try:
        gui = await get_gui()
        
        print("Mickey GUI Status:")
        print(f"Initialized: {gui.is_initialized}")
        print(f"Current State: {gui.current_state}")
        print("\nGUI window should be visible with boot sequence.")
        print("Close the window to exit.")
        
        # Run GUI
        await gui.run()
        
    except Exception as e:
        print(f"GUI test failed: {str(e)}")


if __name__ == "__main__":
    # Import math for visualizer
    import math
    
    asyncio.run(main())