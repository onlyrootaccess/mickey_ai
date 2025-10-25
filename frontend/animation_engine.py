# Pulse effects, wireframe animations
"""
M.I.C.K.E.Y. AI Assistant - Advanced Animation Engine
Made In Crisis, Keeping Everything Yours

FOURTEENTH FILE IN PIPELINE: Advanced animation system with particle effects,
smooth transitions, and dynamic visual feedback for Mickey's HUD interface.
"""

import asyncio
import logging
import time
import math
import random
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

# Import GUI and graphics libraries
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw

# Import Mickey AI configuration
from config.settings import get_config
from config.constants import GUIConstants, SystemConstants

# Setup logging
logger = logging.getLogger("MickeyAnimation")


class AnimationType(Enum):
    """Types of animations supported."""
    PULSE = "pulse"
    FADE = "fade"
    SLIDE = "slide"
    PARTICLE = "particle"
    WAVE = "wave"
    GLOW = "glow"
    MORPH = "morph"
    SPARKLE = "sparkle"


class EasingFunction(Enum):
    """Easing functions for smooth animations."""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    BOUNCE = "bounce"
    ELASTIC = "elastic"
    SPRING = "spring"


@dataclass
class AnimationConfig:
    """Configuration for individual animations."""
    animation_type: AnimationType
    duration: float  # in seconds
    easing: EasingFunction
    delay: float = 0.0
    repeat: int = 1
    reverse: bool = False
    callback: Optional[Callable] = None


@dataclass
class Particle:
    """Individual particle for particle effects."""
    x: float
    y: float
    vx: float  # velocity x
    vy: float  # velocity y
    life: float  # remaining life (0.0 to 1.0)
    max_life: float
    size: float
    color: str
    decay: float = 0.02


class EasingEngine:
    """Handles mathematical easing functions for smooth animations."""
    
    @staticmethod
    def apply_easing(progress: float, easing: EasingFunction) -> float:
        """Apply easing function to progress (0.0 to 1.0)."""
        if easing == EasingFunction.LINEAR:
            return progress
        
        elif easing == EasingFunction.EASE_IN:
            return progress * progress
        
        elif easing == EasingFunction.EASE_OUT:
            return 1 - (1 - progress) * (1 - progress)
        
        elif easing == EasingFunction.EASE_IN_OUT:
            if progress < 0.5:
                return 2 * progress * progress
            else:
                return 1 - pow(-2 * progress + 2, 2) / 2
        
        elif easing == EasingFunction.BOUNCE:
            if progress < 4 / 11:
                return (121 * progress * progress) / 16
            elif progress < 8 / 11:
                return (363 / 40 * progress * progress) - (99 / 10 * progress) + 17 / 5
            elif progress < 9 / 10:
                return (4356 / 361 * progress * progress) - (35442 / 1805 * progress) + 16061 / 1805
            else:
                return (54 / 5 * progress * progress) - (513 / 25 * progress) + 268 / 25
        
        elif easing == EasingFunction.ELASTIC:
            if progress == 0 or progress == 1:
                return progress
            return -pow(2, 10 * progress - 10) * math.sin((progress * 10 - 10.75) * (2 * math.pi) / 3)
        
        elif easing == EasingFunction.SPRING:
            return 1 - math.cos(progress * math.pi * 2) * (1 - progress)
        
        return progress


class ParticleSystem:
    """Advanced particle system for visual effects."""
    
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.particles: List[Particle] = []
        self.is_running = False
        self.last_update = time.time()
        
    def create_explosion(self, x: int, y: int, color: str = None, count: int = 20):
        """Create an explosion particle effect."""
        if color is None:
            color = GUIConstants.COLORS["primary"]
        
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            life = random.uniform(0.8, 1.5)
            size = random.uniform(2, 6)
            
            particle = Particle(
                x=x, y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=1.0,
                max_life=life,
                size=size,
                color=color,
                decay=random.uniform(0.01, 0.03)
            )
            self.particles.append(particle)
        
        if not self.is_running:
            self.start()
    
    def create_fountain(self, x: int, y: int, color: str = None, count: int = 15):
        """Create a fountain particle effect."""
        if color is None:
            color = GUIConstants.COLORS["secondary"]
        
        for _ in range(count):
            angle = random.uniform(math.pi * 0.7, math.pi * 1.3)  # Upward arc
            speed = random.uniform(3, 7)
            life = random.uniform(1.0, 2.0)
            size = random.uniform(3, 8)
            
            particle = Particle(
                x=x, y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=1.0,
                max_life=life,
                size=size,
                color=color,
                decay=random.uniform(0.015, 0.025)
            )
            self.particles.append(particle)
        
        if not self.is_running:
            self.start()
    
    def create_sparkle(self, x: int, y: int, color: str = None, count: int = 8):
        """Create a sparkle particle effect."""
        if color is None:
            color = GUIConstants.COLORS["accent"]
        
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            life = random.uniform(0.5, 1.0)
            size = random.uniform(1, 4)
            
            particle = Particle(
                x=x, y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=1.0,
                max_life=life,
                size=size,
                color=color,
                decay=random.uniform(0.02, 0.04)
            )
            self.particles.append(particle)
        
        if not self.is_running:
            self.start()
    
    def start(self):
        """Start the particle system."""
        self.is_running = True
        self.last_update = time.time()
        self._update()
    
    def stop(self):
        """Stop the particle system."""
        self.is_running = False
    
    def _update(self):
        """Update all particles."""
        if not self.is_running:
            return
        
        current_time = time.time()
        delta_time = current_time - self.last_update
        self.last_update = current_time
        
        particles_to_remove = []
        
        for particle in self.particles:
            # Update position
            particle.x += particle.vx
            particle.y += particle.vy
            
            # Apply gravity
            particle.vy += 0.2
            
            # Update life
            particle.life -= particle.decay
            
            # Remove dead particles
            if particle.life <= 0:
                particles_to_remove.append(particle)
                continue
            
            # Draw particle (simplified - in practice, we'd manage canvas items)
            self._draw_particle(particle)
        
        # Remove dead particles
        for particle in particles_to_remove:
            self.particles.remove(particle)
        
        # Continue animation if there are active particles
        if self.particles:
            self.canvas.after(16, self._update)  # ~60 FPS
        else:
            self.is_running = False
    
    def _draw_particle(self, particle: Particle):
        """Draw a single particle on canvas."""
        alpha = particle.life
        size = particle.size * alpha
        
        # Calculate color with alpha (simplified)
        color = particle.color
        
        # Create oval for particle
        x1 = particle.x - size
        y1 = particle.y - size
        x2 = particle.x + size
        y2 = particle.y + size
        
        self.canvas.create_oval(
            x1, y1, x2, y2,
            fill=color,
            outline="",
            tags="particle"
        )
    
    def clear_all(self):
        """Clear all particles."""
        self.particles.clear()
        self.canvas.delete("particle")


class WaveAnimation:
    """Wave propagation animation for audio visualization."""
    
    def __init__(self, canvas: tk.Canvas, width: int, height: int):
        self.canvas = canvas
        self.width = width
        self.height = height
        self.waves = []
        self.is_active = False
        
    def create_wave(self, x: int, y: int, color: str = None, max_radius: int = 200):
        """Create a new wave animation."""
        if color is None:
            color = GUIConstants.COLORS["primary"]
        
        wave = {
            'x': x,
            'y': y,
            'radius': 10,
            'max_radius': max_radius,
            'color': color,
            'width': 3,
            'alpha': 1.0,
            'speed': 4,
            'canvas_id': None
        }
        
        wave['canvas_id'] = self.canvas.create_oval(
            x - 10, y - 10, x + 10, y + 10,
            outline=color,
            width=3,
            tags="wave"
        )
        
        self.waves.append(wave)
        
        if not self.is_active:
            self.start()
    
    def start(self):
        """Start wave animation."""
        self.is_active = True
        self._animate()
    
    def stop(self):
        """Stop wave animation."""
        self.is_active = False
    
    def _animate(self):
        """Animate all waves."""
        waves_to_remove = []
        
        for wave in self.waves:
            # Update wave properties
            wave['radius'] += wave['speed']
            wave['alpha'] = 1.0 - (wave['radius'] / wave['max_radius'])
            wave['width'] = max(1, 3 * wave['alpha'])
            
            # Remove if wave has expanded too much
            if wave['radius'] >= wave['max_radius']:
                waves_to_remove.append(wave)
                continue
            
            # Update canvas element
            x1 = wave['x'] - wave['radius']
            y1 = wave['y'] - wave['radius']
            x2 = wave['x'] + wave['radius']
            y2 = wave['y'] + wave['radius']
            
            self.canvas.coords(wave['canvas_id'], x1, y1, x2, y2)
            
            # Update appearance
            color = self._apply_alpha(wave['color'], wave['alpha'])
            self.canvas.itemconfig(wave['canvas_id'], outline=color, width=wave['width'])
        
        # Remove finished waves
        for wave in waves_to_remove:
            self.canvas.delete(wave['canvas_id'])
            self.waves.remove(wave)
        
        # Continue animation if active
        if self.is_active and self.waves:
            self.canvas.after(16, self._animate)
        else:
            self.is_active = False
    
    def _apply_alpha(self, color: str, alpha: float) -> str:
        """Apply alpha to color (simplified)."""
        return color
    
    def clear_all(self):
        """Clear all waves."""
        self.waves.clear()
        self.canvas.delete("wave")


class GlowEffect:
    """Glow and highlight effects for UI elements."""
    
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.glow_elements = {}
        
    def add_glow(self, element_id: str, target_id: str, color: str = None, intensity: float = 1.0):
        """Add glow effect to a canvas element."""
        if color is None:
            color = GUIConstants.COLORS["primary"]
        
        # Get target element coordinates
        coords = self.canvas.coords(target_id)
        if not coords:
            return
        
        # Create glow element (larger version of target)
        if element_id in self.glow_elements:
            self.canvas.delete(self.glow_elements[element_id])
        
        # Determine element type and create appropriate glow
        element_type = self.canvas.type(target_id)
        
        if element_type == "oval":
            glow_id = self._create_oval_glow(coords, color, intensity)
        elif element_type == "rectangle":
            glow_id = self._create_rect_glow(coords, color, intensity)
        else:
            return
        
        self.glow_elements[element_id] = glow_id
        self.canvas.tag_lower(glow_id)  # Send glow behind target
    
    def _create_oval_glow(self, coords: List[float], color: str, intensity: float) -> str:
        """Create glow for oval element."""
        # Expand coordinates for glow
        expansion = 10 * intensity
        x1, y1, x2, y2 = coords
        glow_coords = [x1 - expansion, y1 - expansion, x2 + expansion, y2 + expansion]
        
        return self.canvas.create_oval(
            *glow_coords,
            fill=color,
            outline="",
            stipple="gray50",  # Semi-transparent pattern
            tags="glow"
        )
    
    def _create_rect_glow(self, coords: List[float], color: str, intensity: float) -> str:
        """Create glow for rectangle element."""
        # Expand coordinates for glow
        expansion = 8 * intensity
        x1, y1, x2, y2 = coords
        glow_coords = [x1 - expansion, y1 - expansion, x2 + expansion, y2 + expansion]
        
        return self.canvas.create_rectangle(
            *glow_coords,
            fill=color,
            outline="",
            stipple="gray50",
            tags="glow"
        )
    
    def remove_glow(self, element_id: str):
        """Remove glow effect."""
        if element_id in self.glow_elements:
            self.canvas.delete(self.glow_elements[element_id])
            del self.glow_elements[element_id]
    
    def clear_all(self):
        """Clear all glow effects."""
        for glow_id in self.glow_elements.values():
            self.canvas.delete(glow_id)
        self.glow_elements.clear()


class MorphAnimation:
    """Shape morphing and transformation animations."""
    
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.morph_animations = {}
        
    def morph_element(self, element_id: str, target_coords: List[float], 
                     duration: float = 1.0, easing: EasingFunction = EasingFunction.EASE_IN_OUT):
        """Morph a canvas element to new coordinates."""
        if element_id not in self.morph_animations:
            self.morph_animations[element_id] = {
                'start_coords': self.canvas.coords(element_id),
                'target_coords': target_coords,
                'start_time': time.time(),
                'duration': duration,
                'easing': easing
            }
            
            if element_id not in self.morph_animations:
                self._animate_morph(element_id)
    
    def _animate_morph(self, element_id: str):
        """Animate the morphing process."""
        if element_id not in self.morph_animations:
            return
        
        anim_data = self.morph_animations[element_id]
        elapsed = time.time() - anim_data['start_time']
        progress = min(elapsed / anim_data['duration'], 1.0)
        
        # Apply easing
        eased_progress = EasingEngine.apply_easing(progress, anim_data['easing'])
        
        # Interpolate coordinates
        current_coords = []
        start_coords = anim_data['start_coords']
        target_coords = anim_data['target_coords']
        
        for i in range(len(start_coords)):
            current_val = start_coords[i] + (target_coords[i] - start_coords[i]) * eased_progress
            current_coords.append(current_val)
        
        # Update element
        self.canvas.coords(element_id, *current_coords)
        
        # Continue animation or clean up
        if progress < 1.0:
            self.canvas.after(16, lambda: self._animate_morph(element_id))
        else:
            del self.morph_animations[element_id]


class AnimationOrchestrator:
    """Main animation orchestrator that coordinates all animation systems."""
    
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.config = get_config()
        
        # Initialize animation systems
        self.particle_system = ParticleSystem(canvas)
        self.wave_animation = WaveAnimation(canvas, 800, 600)
        self.glow_effect = GlowEffect(canvas)
        self.morph_animation = MorphAnimation(canvas)
        
        # Active animations tracking
        self.active_animations = {}
        self.animation_queue = []
        
        # Performance monitoring
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0
        
    def create_security_pulse(self, x: int, y: int):
        """Create security authentication pulse animation."""
        # Main pulse
        self.wave_animation.create_wave(x, y, GUIConstants.COLORS["primary"], 150)
        
        # Secondary particles
        self.particle_system.create_sparkle(x, y, GUIConstants.COLORS["accent"], 12)
        
        # Delayed tertiary effect
        self.canvas.after(200, lambda: self.particle_system.create_explosion(
            x, y, GUIConstants.COLORS["secondary"], 8
        ))
    
    def create_voice_activity_wave(self, x: int, y: int, intensity: float = 1.0):
        """Create voice activity wave animation."""
        color = GUIConstants.COLORS["primary"]
        
        # Adjust wave properties based on intensity
        max_radius = 100 + (intensity * 50)
        wave_count = 1 + int(intensity * 2)
        
        for i in range(wave_count):
            delay = i * 100  # Stagger waves
            self.canvas.after(delay, lambda: self.wave_animation.create_wave(
                x, y, color, max_radius
            ))
    
    def create_success_celebration(self, x: int, y: int):
        """Create success celebration animation sequence."""
        # Fountain effect
        self.particle_system.create_fountain(x, y, GUIConstants.COLORS["success"], 25)
        
        # Multiple waves
        for i in range(3):
            delay = i * 300
            self.canvas.after(delay, lambda: self.wave_animation.create_wave(
                x, y, GUIConstants.COLORS["accent"], 200
            ))
        
        # Final sparkle burst
        self.canvas.after(900, lambda: self.particle_system.create_sparkle(
            x, y, GUIConstants.COLORS["primary"], 15
        ))
    
    def create_error_effect(self, x: int, y: int):
        """Create error notification animation."""
        # Rapid red pulses
        for i in range(3):
            delay = i * 150
            self.canvas.after(delay, lambda: self.wave_animation.create_wave(
                x, y, GUIConstants.COLORS["error"], 120
            ))
        
        # Particle explosion
        self.canvas.after(450, lambda: self.particle_system.create_explosion(
            x, y, GUIConstants.COLORS["error"], 12
        ))
    
    def create_processing_animation(self, x: int, y: int):
        """Create processing/loading animation."""
        # Continuous sparkles
        def continuous_sparkles():
            self.particle_system.create_sparkle(
                x + random.randint(-20, 20),
                y + random.randint(-20, 20),
                GUIConstants.COLORS["accent"],
                3
            )
            # Continue until stopped
            if hasattr(self, 'processing_active') and self.processing_active:
                self.canvas.after(200, continuous_sparkles)
        
        self.processing_active = True
        continuous_sparkles()
    
    def stop_processing_animation(self):
        """Stop processing animation."""
        self.processing_active = False
    
    def create_boot_sequence(self, center_x: int, center_y: int):
        """Create boot sequence animation."""
        # Initial system pulse
        self.wave_animation.create_wave(center_x, center_y, GUIConstants.COLORS["primary"], 300)
        
        # Progressive particle rings
        for ring in range(3):
            delay = ring * 500
            self.canvas.after(delay, lambda: self._create_particle_ring(
                center_x, center_y, 50 + ring * 40, 12
            ))
    
    def _create_particle_ring(self, x: int, y: int, radius: int, count: int):
        """Create a ring of particles."""
        for i in range(count):
            angle = (2 * math.pi * i) / count
            px = x + math.cos(angle) * radius
            py = y + math.sin(angle) * radius
            
            self.particle_system.create_sparkle(int(px), int(py), count=1)
    
    def animate_element_highlight(self, element_id: str, duration: float = 2.0):
        """Animate element highlight with glow and pulse."""
        # Add glow effect
        self.glow_effect.add_glow(f"highlight_{element_id}", element_id)
        
        # Create pulsing effect
        def pulse_glow(cycle: int):
            if cycle >= 3:  # 3 pulses
                self.glow_effect.remove_glow(f"highlight_{element_id}")
                return
            
            # Intensify and reduce glow
            self.canvas.after(300, lambda: self.glow_effect.add_glow(
                f"highlight_{element_id}", element_id, intensity=1.5
            ))
            self.canvas.after(600, lambda: self.glow_effect.add_glow(
                f"highlight_{element_id}", element_id, intensity=1.0
            ))
            
            self.canvas.after(700, lambda: pulse_glow(cycle + 1))
        
        pulse_glow(0)
    
    def create_data_flow_animation(self, start_x: int, start_y: int, end_x: int, end_y: int):
        """Create data flow animation between points."""
        particle_count = 8
        
        for i in range(particle_count):
            delay = i * 100
            
            self.canvas.after(delay, lambda: self._create_flow_particle(
                start_x, start_y, end_x, end_y
            ))
    
    def _create_flow_particle(self, start_x: int, start_y: int, end_x: int, end_y: int):
        """Create a single flow particle."""
        # Calculate direction vector
        dx = end_x - start_x
        dy = end_y - start_y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance == 0:
            return
        
        # Normalize direction
        dx /= distance
        dy /= distance
        
        # Create particle moving along the path
        travel_time = 1.0  # seconds
        steps = int(travel_time * 60)  # 60 FPS
        
        def animate_particle(step: int):
            if step >= steps:
                return
            
            progress = step / steps
            x = start_x + dx * distance * progress
            y = start_y + dy * distance * progress
            
            # Draw particle at current position
            size = 3
            self.canvas.create_oval(
                x - size, y - size, x + size, y + size,
                fill=GUIConstants.COLORS["primary"],
                outline="",
                tags="flow_particle"
            )
            
            # Schedule next step
            if step < steps - 1:
                self.canvas.after(16, lambda: animate_particle(step + 1))
            else:
                # Clean up after animation completes
                self.canvas.after(100, lambda: self.canvas.delete("flow_particle"))
        
        animate_particle(0)
    
    def update_performance(self):
        """Update performance monitoring."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_update >= 1.0:
            self.current_fps = self.frame_count
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get animation performance metrics."""
        return {
            "fps": self.current_fps,
            "active_particles": len(self.particle_system.particles),
            "active_waves": len(self.wave_animation.waves),
            "active_glows": len(self.glow_effect.glow_elements),
            "active_morphs": len(self.morph_animation.morph_animations)
        }
    
    def clear_all_animations(self):
        """Clear all active animations."""
        self.particle_system.clear_all()
        self.wave_animation.clear_all()
        self.glow_effect.clear_all()
        self.morph_animation.morph_animations.clear()
        self.active_animations.clear()
        self.animation_queue.clear()
        
        # Clear any remaining animation elements
        self.canvas.delete("particle")
        self.canvas.delete("wave")
        self.canvas.delete("glow")
        self.canvas.delete("flow_particle")


# Global animation orchestrator instance
_animation_instance: Optional[AnimationOrchestrator] = None


def get_animation_orchestrator(canvas: tk.Canvas) -> AnimationOrchestrator:
    """Get or create global animation orchestrator instance."""
    global _animation_instance
    
    if _animation_instance is None:
        _animation_instance = AnimationOrchestrator(canvas)
    
    return _animation_instance


async def main():
    """Command-line testing for animation engine."""
    import asyncio
    
    # Create test window
    root = tk.Tk()
    root.title("Animation Engine Test")
    root.geometry("800x600")
    
    # Create canvas
    canvas = tk.Canvas(root, bg="black", width=800, height=600)
    canvas.pack(fill="both", expand=True)
    
    # Initialize animation engine
    animator = get_animation_orchestrator(canvas)
    
    # Test animations
    def test_animations():
        # Security pulse
        animator.create_security_pulse(400, 300)
        
        # Success celebration after delay
        root.after(2000, lambda: animator.create_success_celebration(400, 300))
        
        # Data flow animation
        root.after(4000, lambda: animator.create_data_flow_animation(100, 100, 700, 500))
        
        # Error effect
        root.after(6000, lambda: animator.create_error_effect(200, 200))
    
    # Start test animations
    root.after(1000, test_animations)
    
    # Performance display
    def update_performance_display():
        metrics = animator.get_performance_metrics()
        perf_text = f"FPS: {metrics['fps']} | Particles: {metrics['active_particles']}"
        canvas.delete("performance")
        canvas.create_text(
            10, 10, text=perf_text, anchor="nw", 
            fill="white", tags="performance", font=("Arial", 10)
        )
        animator.update_performance()
        root.after(1000, update_performance_display)
    
    update_performance_display()
    
    print("Animation Engine Test Running...")
    print("Close the window to exit.")
    
    # Run main loop
    root.mainloop()


if __name__ == "__main__":
    asyncio.run(main())