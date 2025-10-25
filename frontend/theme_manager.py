# "Wire-Gucci" aesthetic controller
"""
Mickey AI - Theme Manager
Manages "Wire-Gucci" aesthetic with dynamic themes, animations, and visual effects
"""

import logging
import json
import os
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import colorsys
from PIL import Image, ImageTk, ImageOps

class ThemeStyle(Enum):
    WIRE_GUCCI = "wire_gucci"
    MODERN_MINIMAL = "modern_minimal"
    DARK_FUTURE = "dark_future"
    MICKEY_CLASSIC = "mickey_classic"
    NEON_DREAM = "neon_dream"

class ColorPalette:
    def __init__(self, primary: str, secondary: str, accent: str, 
                 background: str, text: str, success: str, warning: str, error: str):
        self.primary = primary
        self.secondary = secondary
        self.accent = accent
        self.background = background
        self.text = text
        self.success = success
        self.warning = warning
        self.error = error

class AnimationPreset:
    def __init__(self, name: str, duration: float, easing: str, effects: List[str]):
        self.name = name
        self.duration = duration
        self.easing = easing
        self.effects = effects

class ThemeManager:
    def __init__(self, root: tk.Tk = None):
        self.logger = logging.getLogger(__name__)
        self.root = root
        
        # Current theme state
        self.current_theme = ThemeStyle.WIRE_GUCCI
        self.current_palette = None
        self.animation_enabled = True
        self.dark_mode = False
        
        # Theme configurations
        self.themes = self._initialize_themes()
        self.animation_presets = self._initialize_animation_presets()
        
        # Dynamic theme properties
        self.theme_variants = {}
        self.gradient_cache = {}
        
        # UI element references for dynamic updates
        self.themed_widgets = {}
        self.style = None
        
        # Theme transition state
        self.is_transitioning = False
        self.transition_progress = 0.0
        
        # Mickey's theme messages
        self.theme_messages = {
            ThemeStyle.WIRE_GUCCI: [
                "Switching to Wire-Gucci aesthetic! So stylish! 💫",
                "Wire-Gucci mode activated! Fashion meets function! 👔",
                "Mickey's going high-fashion with Wire-Gucci! 🎩"
            ],
            ThemeStyle.MODERN_MINIMAL: [
                "Clean and modern! Mickey's keeping it simple! 🎯",
                "Minimalist mode! Less is more! ✨",
                "Modern aesthetics activated! Sleek and clean! 🔷"
            ],
            ThemeStyle.DARK_FUTURE: [
                "Dark future mode! Cyberpunk vibes! 🌃",
                "Future dark theme! Ready for the next century! 🚀",
                "Cyber aesthetic activated! Mickey's futuristic! 🔮"
            ],
            ThemeStyle.MICKEY_CLASSIC: [
                "Classic Mickey style! Retro and fun! 🐭",
                "Back to classics! Mickey's original vibe! 🎪",
                "Classic theme! Just like the good old days! 📺"
            ],
            ThemeStyle.NEON_DREAM: [
                "Neon dream activated! So vibrant! 🌈",
                "Bright and colorful! Mickey's neon mode! 💖",
                "Neon theme! Let's light it up! 💡"
            ]
        }
        
        # Initialize ttk style if root is provided
        if self.root:
            self.style = ttk.Style(self.root)
            self._setup_ttk_styles()
        
        self.logger.info("🎨 Theme Manager initialized - Ready to style!")

    def _initialize_themes(self) -> Dict[ThemeStyle, ColorPalette]:
        """Initialize all theme color palettes"""
        return {
            ThemeStyle.WIRE_GUCCI: ColorPalette(
                primary="#2D2D2D",      # Dark charcoal
                secondary="#8B4513",    # Saddle brown (Gucci brown)
                accent="#D4AF37",       # Metallic gold
                background="#1A1A1A",   # Near black
                text="#F5F5F5",         # Off white
                success="#228B22",      # Forest green
                warning="#FFD700",      # Gold warning
                error="#B22222"         # Firebrick red
            ),
            ThemeStyle.MODERN_MINIMAL: ColorPalette(
                primary="#2C3E50",      # Dark blue gray
                secondary="#34495E",    # Lighter blue gray
                accent="#3498DB",       # Bright blue
                background="#ECF0F1",   # Light gray
                text="#2C3E50",         # Dark text
                success="#27AE60",      # Green
                warning="#F39C12",      # Orange
                error="#E74C3C"         # Red
            ),
            ThemeStyle.DARK_FUTURE: ColorPalette(
                primary="#00FF41",      # Matrix green
                secondary="#FF0099",    # Cyber pink
                accent="#00D8FF",       # Electric blue
                background="#0A0A0A",   # Pure black
                text="#00FF41",         # Green text
                success="#00FF41",      # Green
                warning="#FF9900",      # Orange
                error="#FF0066"         # Red pink
            ),
            ThemeStyle.MICKEY_CLASSIC: ColorPalette(
                primary="#FF0000",    # Mickey red
                secondary="#000000",    # Black
                accent="#FFFF00",       # Yellow
                background="#FFFFFF",   # White
                text="#000000",         # Black text
                success="#00AA00",      # Green
                warning="#FFA500",      # Orange
                error="#FF0000"         # Red
            ),
            ThemeStyle.NEON_DREAM: ColorPalette(
                primary="#FF6BFF",      # Pink
                secondary="#4D4DFF",    # Blue
                accent="#00FFFF",       # Cyan
                background="#1A0033",   # Dark purple
                text="#FFFFFF",         # White
                success="#00FF00",      # Green
                warning="#FFFF00",      # Yellow
                error="#FF0000"         # Red
            )
        }

    def _initialize_animation_presets(self) -> Dict[str, AnimationPreset]:
        """Initialize animation presets for theme transitions"""
        return {
            "smooth_fade": AnimationPreset(
                name="smooth_fade",
                duration=0.5,
                easing="ease_in_out",
                effects=["fade", "color_shift"]
            ),
            "slide_transition": AnimationPreset(
                name="slide_transition", 
                duration=0.7,
                easing="ease_out",
                effects=["slide", "fade"]
            ),
            "color_morph": AnimationPreset(
                name="color_morph",
                duration=1.0,
                easing="ease_in_out_cubic", 
                effects=["color_morph", "glow"]
            ),
            "instant": AnimationPreset(
                name="instant",
                duration=0.1,
                easing="linear",
                effects=["instant"]
            )
        }

    def _setup_ttk_styles(self):
        """Setup ttk widget styles for the current theme"""
        if not self.style:
            return
            
        palette = self.themes[self.current_theme]
        
        # Configure ttk styles
        self.style.configure("TFrame", background=palette.background)
        self.style.configure("TLabel", background=palette.background, foreground=palette.text)
        self.style.configure("TButton", 
                           background=palette.primary,
                           foreground=palette.text,
                           focuscolor=palette.accent)
        self.style.configure("TEntry", 
                           fieldbackground=palette.background,
                           foreground=palette.text,
                           insertcolor=palette.accent)
        self.style.configure("TScrollbar", 
                           background=palette.secondary,
                           troughcolor=palette.background)

    def set_theme(self, theme: ThemeStyle, animation: str = "smooth_fade"):
        """
        Set the current theme with optional animation
        
        Args:
            theme: Theme to apply
            animation: Animation preset name
        """
        try:
            if theme not in self.themes:
                self.logger.error(f"Unknown theme: {theme}")
                return False
            
            if self.is_transitioning:
                self.logger.warning("Theme transition already in progress")
                return False
            
            old_theme = self.current_theme
            self.current_theme = theme
            self.current_palette = self.themes[theme]
            
            self.logger.info(f"Theme changed to: {theme.value}")
            
            # Apply theme with animation
            if self.animation_enabled and animation in self.animation_presets:
                self._animate_theme_transition(old_theme, theme, animation)
            else:
                self._apply_theme_immediately()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Theme change failed: {str(e)}")
            return False

    def _animate_theme_transition(self, old_theme: ThemeStyle, new_theme: ThemeStyle, animation: str):
        """Animate theme transition"""
        preset = self.animation_presets[animation]
        self.is_transitioning = True
        self.transition_progress = 0.0
        
        old_palette = self.themes[old_theme]
        new_palette = self.themes[new_theme]
        
        def update_transition():
            if self.transition_progress >= 1.0:
                self.is_transitioning = False
                self._apply_theme_immediately()
                return
            
            # Calculate intermediate colors
            progress = self._apply_easing(self.transition_progress, preset.easing)
            intermediate_palette = self._interpolate_palettes(old_palette, new_palette, progress)
            
            # Apply intermediate theme
            self._apply_palette(intermediate_palette)
            
            # Update progress
            self.transition_progress += 1.0 / (preset.duration * 60)  # 60 FPS
            self.root.after(16, update_transition)  # ~60 FPS
        
        update_transition()

    def _apply_easing(self, progress: float, easing: str) -> float:
        """Apply easing function to progress"""
        if easing == "ease_in_out":
            return progress * progress * (3 - 2 * progress)
        elif easing == "ease_out":
            return 1 - (1 - progress) ** 2
        elif easing == "ease_in_out_cubic":
            return 4 * progress * progress * progress if progress < 0.5 else 1 - ((-2 * progress + 2) ** 3) / 2
        else:  # linear
            return progress

    def _interpolate_palettes(self, palette1: ColorPalette, palette2: ColorPalette, progress: float) -> ColorPalette:
        """Interpolate between two color palettes"""
        def interpolate_color(color1: str, color2: str, progress: float) -> str:
            # Convert hex to RGB
            r1, g1, b1 = self.hex_to_rgb(color1)
            r2, g2, b2 = self.hex_to_rgb(color2)
            
            # Interpolate
            r = int(r1 + (r2 - r1) * progress)
            g = int(g1 + (g2 - g1) * progress)
            b = int(b1 + (b2 - b1) * progress)
            
            return self.rgb_to_hex(r, g, b)
        
        return ColorPalette(
            primary=interpolate_color(palette1.primary, palette2.primary, progress),
            secondary=interpolate_color(palette1.secondary, palette2.secondary, progress),
            accent=interpolate_color(palette1.accent, palette2.accent, progress),
            background=interpolate_color(palette1.background, palette2.background, progress),
            text=interpolate_color(palette1.text, palette2.text, progress),
            success=interpolate_color(palette1.success, palette2.success, progress),
            warning=interpolate_color(palette1.warning, palette2.warning, progress),
            error=interpolate_color(palette1.error, palette2.error, progress)
        )

    def _apply_theme_immediately(self):
        """Apply current theme immediately without animation"""
        palette = self.themes[self.current_theme]
        self._apply_palette(palette)
        
        # Update ttk styles
        if self.style:
            self._setup_ttk_styles()

    def _apply_palette(self, palette: ColorPalette):
        """Apply color palette to all registered widgets"""
        for widget_id, widget_info in self.themed_widgets.items():
            try:
                widget = widget_info['widget']
                widget_type = widget_info['type']
                
                if widget_type == "frame":
                    widget.config(bg=palette.background)
                elif widget_type == "label":
                    widget.config(bg=palette.background, fg=palette.text)
                elif widget_type == "button":
                    widget.config(bg=palette.primary, fg=palette.text,
                                activebackground=palette.accent, 
                                activeforeground=palette.text)
                elif widget_type == "entry":
                    widget.config(bg=palette.background, fg=palette.text,
                                insertbackground=palette.accent)
                elif widget_type == "canvas":
                    widget.config(bg=palette.background)
                
            except Exception as e:
                self.logger.warning(f"Failed to update widget {widget_id}: {str(e)}")

    def register_widget(self, widget_id: str, widget: tk.Widget, widget_type: str):
        """Register a widget for automatic theme updates"""
        self.themed_widgets[widget_id] = {
            'widget': widget,
            'type': widget_type
        }
        
        # Apply current theme immediately
        palette = self.themes[self.current_theme]
        self._update_widget_theme(widget_id, palette)

    def unregister_widget(self, widget_id: str):
        """Unregister a widget from theme updates"""
        if widget_id in self.themed_widgets:
            del self.themed_widgets[widget_id]

    def _update_widget_theme(self, widget_id: str, palette: ColorPalette):
        """Update specific widget with theme colors"""
        if widget_id not in self.themed_widgets:
            return
            
        widget_info = self.themed_widgets[widget_id]
        widget = widget_info['widget']
        widget_type = widget_info['type']
        
        try:
            if widget_type == "frame":
                widget.config(bg=palette.background)
            elif widget_type == "label":
                widget.config(bg=palette.background, fg=palette.text)
            elif widget_type == "button":
                widget.config(bg=palette.primary, fg=palette.text)
            elif widget_type == "entry":
                widget.config(bg=palette.background, fg=palette.text)
            elif widget_type == "canvas":
                widget.config(bg=palette.background)
        except Exception as e:
            self.logger.warning(f"Failed to update widget {widget_id}: {str(e)}")

    def create_gradient(self, color1: str, color2: str, steps: int = 10) -> List[str]:
        """Create a gradient between two colors"""
        cache_key = f"{color1}-{color2}-{steps}"
        if cache_key in self.gradient_cache:
            return self.gradient_cache[cache_key]
        
        r1, g1, b1 = self.hex_to_rgb(color1)
        r2, g2, b2 = self.hex_to_rgb(color2)
        
        gradient = []
        for i in range(steps):
            ratio = i / (steps - 1)
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            gradient.append(self.rgb_to_hex(r, g, b))
        
        self.gradient_cache[cache_key] = gradient
        return gradient

    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(self, r: int, g: int, b: int) -> str:
        """Convert RGB tuple to hex color"""
        return f"#{r:02x}{g:02x}{b:02x}"

    def adjust_brightness(self, hex_color: str, factor: float) -> str:
        """Adjust color brightness by factor (0.0 to 2.0)"""
        r, g, b = self.hex_to_rgb(hex_color)
        
        # Convert to HSL, adjust lightness, convert back to RGB
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        l = max(0, min(1, l * factor))
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        
        return self.rgb_to_hex(int(r*255), int(g*255), int(b*255))

    def get_theme_info(self) -> Dict[str, Any]:
        """Get current theme information"""
        palette = self.themes[self.current_theme]
        return {
            'current_theme': self.current_theme.value,
            'dark_mode': self.dark_mode,
            'animation_enabled': self.animation_enabled,
            'colors': {
                'primary': palette.primary,
                'secondary': palette.secondary,
                'accent': palette.accent,
                'background': palette.background,
                'text': palette.text
            },
            'registered_widgets': len(self.themed_widgets),
            'is_transitioning': self.is_transitioning,
            'mickey_response': self._get_theme_message()
        }

    def _get_theme_message(self) -> str:
        """Get Mickey's theme message"""
        import random
        messages = self.theme_messages.get(self.current_theme, ["Theme updated!"])
        return random.choice(messages)

    def toggle_dark_mode(self):
        """Toggle dark mode (inverts brightness for some themes)"""
        self.dark_mode = not self.dark_mode
        self.logger.info(f"Dark mode {'enabled' if self.dark_mode else 'disabled'}")
        
        # Adjust current theme for dark mode
        if self.dark_mode:
            self._apply_dark_mode()
        else:
            self._apply_theme_immediately()

    def _apply_dark_mode(self):
        """Apply dark mode adjustments to current theme"""
        palette = self.themes[self.current_theme]
        
        # Create darkened version of the palette
        dark_palette = ColorPalette(
            primary=self.adjust_brightness(palette.primary, 0.7),
            secondary=self.adjust_brightness(palette.secondary, 0.7),
            accent=palette.accent,  # Keep accent bright
            background=self.adjust_brightness(palette.background, 0.3),
            text=self.adjust_brightness(palette.text, 1.2),  # Brighter text
            success=palette.success,
            warning=palette.warning,
            error=palette.error
        )
        
        self._apply_palette(dark_palette)

    def save_theme_preferences(self, filepath: str = "data/theme_preferences.json"):
        """Save current theme preferences to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            preferences = {
                'current_theme': self.current_theme.value,
                'dark_mode': self.dark_mode,
                'animation_enabled': self.animation_enabled
            }
            
            with open(filepath, 'w') as f:
                json.dump(preferences, f, indent=2)
            
            self.logger.info(f"Theme preferences saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save theme preferences: {str(e)}")

    def load_theme_preferences(self, filepath: str = "data/theme_preferences.json"):
        """Load theme preferences from file"""
        try:
            if not os.path.exists(filepath):
                self.logger.warning(f"Theme preferences file not found: {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                preferences = json.load(f)
            
            # Apply preferences
            theme_name = preferences.get('current_theme', 'wire_gucci')
            theme = ThemeStyle(theme_name)
            self.set_theme(theme, animation="instant")
            
            self.dark_mode = preferences.get('dark_mode', False)
            self.animation_enabled = preferences.get('animation_enabled', True)
            
            self.logger.info(f"Theme preferences loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load theme preferences: {str(e)}")
            return False

    def create_custom_theme(self, name: str, palette: ColorPalette):
        """Create a custom theme"""
        # Convert string to ThemeStyle if it's a new custom theme
        theme_style = ThemeStyle(name.upper())
        self.themes[theme_style] = palette
        self.logger.info(f"Custom theme created: {name}")

    def get_available_themes(self) -> List[Dict[str, Any]]:
        """Get list of all available themes"""
        themes_info = []
        for theme_style, palette in self.themes.items():
            themes_info.append({
                'name': theme_style.value,
                'display_name': theme_style.value.replace('_', ' ').title(),
                'colors': {
                    'primary': palette.primary,
                    'secondary': palette.secondary,
                    'accent': palette.accent
                }
            })
        
        return themes_info

# Test function
def test_theme_manager():
    """Test the theme manager"""
    root = tk.Tk()
    root.title("Theme Manager Test")
    root.geometry("400x300")
    
    theme_manager = ThemeManager(root)
    
    # Create test widgets
    frame = tk.Frame(root)
    frame.pack(pady=20)
    
    label = tk.Label(frame, text="Theme Manager Test", font=('Arial', 16))
    label.pack(pady=10)
    
    button_frame = tk.Frame(frame)
    button_frame.pack(pady=10)
    
    # Register widgets for theme updates
    theme_manager.register_widget("main_frame", frame, "frame")
    theme_manager.register_widget("main_label", label, "label")
    
    # Theme buttons
    def change_theme(theme_name):
        theme = ThemeStyle(theme_name)
        theme_manager.set_theme(theme)
    
    themes = [
        ("Wire Gucci", "wire_gucci"),
        ("Modern Minimal", "modern_minimal"), 
        ("Dark Future", "dark_future"),
        ("Mickey Classic", "mickey_classic"),
        ("Neon Dream", "neon_dream")
    ]
    
    for display_name, theme_name in themes:
        btn = tk.Button(button_frame, text=display_name, 
                       command=lambda tn=theme_name: change_theme(tn))
        btn.pack(side=tk.LEFT, padx=5)
        theme_manager.register_widget(f"btn_{theme_name}", btn, "button")
    
    # Test theme info
    info = theme_manager.get_theme_info()
    print("Theme Info:", info)
    
    root.mainloop()

if __name__ == "__main__":
    test_theme_manager()