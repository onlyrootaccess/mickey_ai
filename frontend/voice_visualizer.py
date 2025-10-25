# Audio waveform display
"""
Mickey AI - Voice Visualizer
Real-time audio waveform and frequency spectrum visualization with Mickey-themed animations
"""

import logging
import tkinter as tk
from tkinter import ttk
import numpy as np
import math
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import colorsys

class VisualizationMode(Enum):
    WAVEFORM = "waveform"
    SPECTRUM = "spectrum"
    BARS = "bars"
    CIRCULAR = "circular"
    MICKEY = "mickey"

class VoiceVisualizer:
    def __init__(self, master, width: int = 400, height: int = 200, 
                 mode: VisualizationMode = VisualizationMode.MICKEY):
        self.logger = logging.getLogger(__name__)
        self.master = master
        self.width = width
        self.height = height
        self.mode = mode
        
        # Audio data buffers
        self.audio_buffer = np.zeros(1024, dtype=np.float32)
        self.spectrum_buffer = np.zeros(256, dtype=np.float32)
        self.volume_history = []
        self.max_history = 100
        
        # Visualization state
        self.is_visualizing = False
        self.animation_id = None
        self.current_volume = 0.0
        self.peak_volume = 0.0
        self.silence_threshold = 0.01
        
        # Visual properties
        self.colors = self._get_default_colors()
        self.amplitude_scale = 1.0
        self.smoothing_factor = 0.8
        self.animation_speed = 16  # ms per frame (~60 FPS)
        
        # Mickey-specific visualization
        self.mickey_animation_state = 0
        self.mickey_expression = "listening"
        self.expression_changed = False
        
        # Create canvas
        self.canvas = tk.Canvas(
            master,
            width=width,
            height=height,
            bg=self.colors['background'],
            highlightthickness=0
        )
        
        # Visualization elements
        self.visual_elements = {
            'waveform': [],
            'spectrum_bars': [],
            'volume_circle': None,
            'mickey_face': [],
            'peak_indicator': None
        }
        
        # Initialize visualization
        self._setup_visualization()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        self.logger.info("🎤 Voice Visualizer initialized - Ready to show audio magic!")

    def _get_default_colors(self) -> Dict[str, str]:
        """Get default color scheme for visualizations"""
        return {
            'background': '#1a1a1a',
            'waveform_primary': '#ff6b6b',
            'waveform_secondary': '#4ecdc4',
            'spectrum_low': '#00ff00',
            'spectrum_mid': '#ffff00',
            'spectrum_high': '#ff0000',
            'mickey_primary': '#ff0000',
            'mickey_secondary': '#000000',
            'mickey_accent': '#ffff00',
            'volume_indicator': '#3498db',
            'peak_indicator': '#e74c3c'
        }

    def _setup_visualization(self):
        """Setup initial visualization elements based on mode"""
        self.canvas.delete("all")
        self.visual_elements = {key: [] for key in self.visual_elements.keys()}
        
        if self.mode == VisualizationMode.WAVEFORM:
            self._setup_waveform_visualization()
        elif self.mode == VisualizationMode.SPECTRUM:
            self._setup_spectrum_visualization()
        elif self.mode == VisualizationMode.BARS:
            self._setup_bars_visualization()
        elif self.mode == VisualizationMode.CIRCULAR:
            self._setup_circular_visualization()
        elif self.mode == VisualizationMode.MICKEY:
            self._setup_mickey_visualization()

    def _setup_waveform_visualization(self):
        """Setup waveform visualization elements"""
        # Create center line
        center_y = self.height // 2
        self.canvas.create_line(
            0, center_y, self.width, center_y,
            fill=self.colors['waveform_secondary'],
            width=1,
            dash=(2, 2)
        )

    def _setup_spectrum_visualization(self):
        """Setup frequency spectrum visualization"""
        # Create grid lines
        for i in range(1, 4):
            y = self.height * i // 4
            self.canvas.create_line(
                0, y, self.width, y,
                fill='#333333',
                width=1
            )

    def _setup_bars_visualization(self):
        """Setup bar graph visualization"""
        # Create baseline
        baseline_y = self.height - 20
        self.canvas.create_line(
            0, baseline_y, self.width, baseline_y,
            fill='#444444',
            width=2
        )

    def _setup_circular_visualization(self):
        """Setup circular visualization"""
        center_x, center_y = self.width // 2, self.height // 2
        radius = min(center_x, center_y) - 10
        
        # Create outer circle
        self.canvas.create_oval(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            outline='#333333',
            width=2
        )

    def _setup_mickey_visualization(self):
        """Setup Mickey-themed visualization"""
        center_x, center_y = self.width // 2, self.height // 2
        head_radius = min(center_x, center_y) - 20
        
        # Mickey's head
        head = self.canvas.create_oval(
            center_x - head_radius, center_y - head_radius,
            center_x + head_radius, center_y + head_radius,
            fill=self.colors['mickey_primary'],
            outline=self.colors['mickey_secondary'],
            width=3
        )
        
        # Mickey's ears
        ear_radius = head_radius // 2
        left_ear = self.canvas.create_oval(
            center_x - head_radius - ear_radius//2, center_y - head_radius,
            center_x - head_radius + ear_radius//2, center_y - head_radius + ear_radius,
            fill=self.colors['mickey_primary'],
            outline=self.colors['mickey_secondary'],
            width=2
        )
        right_ear = self.canvas.create_oval(
            center_x + head_radius - ear_radius//2, center_y - head_radius,
            center_x + head_radius + ear_radius//2, center_y - head_radius + ear_radius,
            fill=self.colors['mickey_primary'],
            outline=self.colors['mickey_secondary'],
            width=2
        )
        
        # Expression area (will be animated)
        self.visual_elements['mickey_face'].extend([head, left_ear, right_ear])

    def start_visualization(self):
        """Start the visualization animation loop"""
        if self.is_visualizing:
            return
            
        self.is_visualizing = True
        self.frame_count = 0
        self.start_time = time.time()
        self._animation_loop()
        self.logger.info("Voice visualization started")

    def stop_visualization(self):
        """Stop the visualization"""
        self.is_visualizing = False
        if self.animation_id:
            self.master.after_cancel(self.animation_id)
        self.logger.info("Voice visualization stopped")

    def _animation_loop(self):
        """Main animation loop"""
        if not self.is_visualizing:
            return
            
        try:
            start_frame_time = time.time()
            
            # Update visualization based on mode
            if self.mode == VisualizationMode.WAVEFORM:
                self._update_waveform()
            elif self.mode == VisualizationMode.SPECTRUM:
                self._update_spectrum()
            elif self.mode == VisualizationMode.BARS:
                self._update_bars()
            elif self.mode == VisualizationMode.CIRCULAR:
                self._update_circular()
            elif self.mode == VisualizationMode.MICKEY:
                self._update_mickey()
            
            # Update performance metrics
            self.frame_count += 1
            frame_time = (time.time() - start_frame_time) * 1000
            
            # Adjust animation speed if needed
            if frame_time > self.animation_speed * 1.5:
                self.logger.warning(f"Frame time {frame_time:.1f}ms exceeds target {self.animation_speed}ms")
            
            # Schedule next frame
            self.animation_id = self.master.after(self.animation_speed, self._animation_loop)
            
        except Exception as e:
            self.logger.error(f"Animation loop error: {str(e)}")
            self.animation_id = self.master.after(self.animation_speed, self._animation_loop)

    def update_audio_data(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """
        Update visualizer with new audio data
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of audio data
        """
        try:
            # Update audio buffer
            self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
            self.audio_buffer[-len(audio_data):] = audio_data
            
            # Calculate current volume (RMS)
            self.current_volume = np.sqrt(np.mean(audio_data**2))
            
            # Update volume history
            self.volume_history.append(self.current_volume)
            if len(self.volume_history) > self.max_history:
                self.volume_history.pop(0)
            
            # Update peak volume
            if self.current_volume > self.peak_volume:
                self.peak_volume = self.current_volume
            else:
                # Gradually decay peak volume
                self.peak_volume *= 0.995
            
            # Calculate spectrum if needed
            if self.mode in [VisualizationMode.SPECTRUM, VisualizationMode.BARS, VisualizationMode.MICKEY]:
                self._calculate_spectrum(audio_data, sample_rate)
                
        except Exception as e:
            self.logger.error(f"Audio data update failed: {str(e)}")

    def _calculate_spectrum(self, audio_data: np.ndarray, sample_rate: int):
        """Calculate frequency spectrum using FFT"""
        try:
            # Apply window function to reduce spectral leakage
            window = np.hanning(len(audio_data))
            windowed_data = audio_data * window
            
            # Perform FFT
            fft_data = np.fft.rfft(windowed_data)
            magnitudes = np.abs(fft_data)
            
            # Convert to dB scale and normalize
            db_spectrum = 20 * np.log10(magnitudes + 1e-10)
            normalized_spectrum = (db_spectrum - np.min(db_spectrum)) / (np.max(db_spectrum) - np.min(db_spectrum) + 1e-10)
            
            # Smooth the spectrum
            self.spectrum_buffer = self.smoothing_factor * self.spectrum_buffer + (1 - self.smoothing_factor) * normalized_spectrum[:len(self.spectrum_buffer)]
            
        except Exception as e:
            self.logger.error(f"Spectrum calculation failed: {str(e)}")

    def _update_waveform(self):
        """Update waveform visualization"""
        try:
            # Clear previous waveform
            for element in self.visual_elements['waveform']:
                self.canvas.delete(element)
            self.visual_elements['waveform'].clear()
            
            # Draw new waveform
            center_y = self.height // 2
            points = []
            
            for i in range(self.width):
                buffer_index = int(i * len(self.audio_buffer) / self.width)
                if buffer_index < len(self.audio_buffer):
                    amplitude = self.audio_buffer[buffer_index] * self.amplitude_scale * self.height / 2
                    points.extend([i, center_y + amplitude])
            
            if len(points) >= 4:
                waveform = self.canvas.create_line(
                    points,
                    fill=self.colors['waveform_primary'],
                    width=2,
                    smooth=True
                )
                self.visual_elements['waveform'].append(waveform)
            
            # Draw volume indicator
            self._draw_volume_indicator()
            
        except Exception as e:
            self.logger.error(f"Waveform update failed: {str(e)}")

    def _update_spectrum(self):
        """Update frequency spectrum visualization"""
        try:
            # Clear previous spectrum
            for element in self.visual_elements['spectrum_bars']:
                self.canvas.delete(element)
            self.visual_elements['spectrum_bars'].clear()
            
            # Draw spectrum bars
            num_bars = min(64, len(self.spectrum_buffer))
            bar_width = self.width / num_bars
            
            for i in range(num_bars):
                value = self.spectrum_buffer[i]
                bar_height = value * (self.height - 20)
                x1 = i * bar_width
                x2 = (i + 1) * bar_width - 1
                y1 = self.height - bar_height
                y2 = self.height
                
                # Color based on frequency band
                if i < num_bars // 3:
                    color = self.colors['spectrum_low']
                elif i < 2 * num_bars // 3:
                    color = self.colors['spectrum_mid']
                else:
                    color = self.colors['spectrum_high']
                
                bar = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline=''
                )
                self.visual_elements['spectrum_bars'].append(bar)
            
            # Draw volume indicator
            self._draw_volume_indicator()
            
        except Exception as e:
            self.logger.error(f"Spectrum update failed: {str(e)}")

    def _update_bars(self):
        """Update bar graph visualization"""
        try:
            # Clear previous bars
            for element in self.visual_elements['spectrum_bars']:
                self.canvas.delete(element)
            self.visual_elements['spectrum_bars'].clear()
            
            # Draw volume history as bars
            num_bars = min(50, len(self.volume_history))
            bar_width = self.width / num_bars
            
            for i in range(num_bars):
                value = self.volume_history[-(num_bars - i)]
                bar_height = value * (self.height - 40)
                x1 = i * bar_width
                x2 = (i + 1) * bar_width - 2
                y1 = self.height - 20 - bar_height
                y2 = self.height - 20
                
                # Gradient color based on height
                hue = 0.3 * (1 - value)  # Green to red
                color = self._hsv_to_hex(hue, 0.8, 0.8)
                
                bar = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline=''
                )
                self.visual_elements['spectrum_bars'].append(bar)
            
        except Exception as e:
            self.logger.error(f"Bars update failed: {str(e)}")

    def _update_circular(self):
        """Update circular visualization"""
        try:
            center_x, center_y = self.width // 2, self.height // 2
            max_radius = min(center_x, center_y) - 10
            
            # Clear previous elements
            if self.visual_elements['volume_circle']:
                self.canvas.delete(self.visual_elements['volume_circle'])
            
            # Draw volume circle
            radius = max_radius * self.current_volume * 2
            radius = min(radius, max_radius)
            
            circle = self.canvas.create_oval(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                fill=self.colors['volume_indicator'],
                outline='',
                stipple='gray50'
            )
            self.visual_elements['volume_circle'] = circle
            
            # Draw spectrum rings
            num_rings = min(8, len(self.spectrum_buffer) // 4)
            for i in range(num_rings):
                ring_value = np.mean(self.spectrum_buffer[i*4:(i+1)*4])
                ring_radius = (i + 1) * (max_radius / (num_rings + 1)) * (0.5 + ring_value)
                
                ring = self.canvas.create_oval(
                    center_x - ring_radius, center_y - ring_radius,
                    center_x + ring_radius, center_y + ring_radius,
                    outline=self._hsv_to_hex(i/num_rings, 0.8, 0.8),
                    width=2
                )
                self.visual_elements['spectrum_bars'].append(ring)
            
        except Exception as e:
            self.logger.error(f"Circular update failed: {str(e)}")

    def _update_mickey(self):
        """Update Mickey-themed visualization"""
        try:
            center_x, center_y = self.width // 2, self.height // 2
            head_radius = min(center_x, center_y) - 20
            
            # Update Mickey's expression based on audio activity
            self._update_mickey_expression()
            
            # Animate Mickey's face based on volume
            mouth_height = 5 + self.current_volume * 20
            eye_size = 8 + self.peak_volume * 15
            
            # Clear previous face elements (except head and ears)
            for element in self.visual_elements['mickey_face'][3:]:
                self.canvas.delete(element)
            self.visual_elements['mickey_face'] = self.visual_elements['mickey_face'][:3]
            
            # Draw eyes
            left_eye = self.canvas.create_oval(
                center_x - head_radius//2 - eye_size//2, center_y - head_radius//3 - eye_size//2,
                center_x - head_radius//2 + eye_size//2, center_y - head_radius//3 + eye_size//2,
                fill=self.colors['mickey_secondary'],
                outline=''
            )
            right_eye = self.canvas.create_oval(
                center_x + head_radius//2 - eye_size//2, center_y - head_radius//3 - eye_size//2,
                center_x + head_radius//2 + eye_size//2, center_y - head_radius//3 + eye_size//2,
                fill=self.colors['mickey_secondary'],
                outline=''
            )
            
            # Draw mouth based on expression
            if self.mickey_expression == "speaking":
                # Open mouth (oval)
                mouth = self.canvas.create_oval(
                    center_x - mouth_height, center_y + head_radius//4,
                    center_x + mouth_height, center_y + head_radius//4 + mouth_height,
                    fill=self.colors['mickey_secondary'],
                    outline=''
                )
            elif self.mickey_expression == "surprised":
                # Round mouth
                mouth = self.canvas.create_oval(
                    center_x - mouth_height//2, center_y + head_radius//4,
                    center_x + mouth_height//2, center_y + head_radius//4 + mouth_height,
                    fill=self.colors['mickey_secondary'],
                    outline=''
                )
            else:
                # Smiling mouth (arc)
                mouth = self.canvas.create_arc(
                    center_x - mouth_height, center_y + head_radius//4,
                    center_x + mouth_height, center_y + head_radius//4 + mouth_height,
                    start=180, extent=180,
                    fill=self.colors['mickey_secondary'],
                    outline=''
                )
            
            self.visual_elements['mickey_face'].extend([left_eye, right_eye, mouth])
            
            # Add audio visualization around Mickey
            self._draw_mickey_audio_effects()
            
        except Exception as e:
            self.logger.error(f"Mickey visualization update failed: {str(e)}")

    def _update_mickey_expression(self):
        """Update Mickey's facial expression based on audio activity"""
        if self.current_volume > self.silence_threshold * 3:
            new_expression = "speaking"
        elif self.peak_volume > self.silence_threshold * 5:
            new_expression = "surprised"
        else:
            new_expression = "listening"
        
        if new_expression != self.mickey_expression:
            self.mickey_expression = new_expression
            self.expression_changed = True

    def _draw_mickey_audio_effects(self):
        """Draw audio visualization effects around Mickey"""
        center_x, center_y = self.width // 2, self.height // 2
        head_radius = min(center_x, center_y) - 20
        
        # Draw audio waves emanating from Mickey
        num_waves = 3
        for i in range(num_waves):
            wave_radius = head_radius + 10 + i * 15
            wave_amplitude = self.spectrum_buffer[i * 8] * 10
            
            # Create wavy circle
            points = []
            for angle in range(0, 360, 10):
                rad = math.radians(angle)
                radius_var = wave_amplitude * math.sin(rad * 3 + self.mickey_animation_state)
                current_radius = wave_radius + radius_var
                x = center_x + current_radius * math.cos(rad)
                y = center_y + current_radius * math.sin(rad)
                points.extend([x, y])
            
            if len(points) >= 4:
                wave = self.canvas.create_line(
                    points,
                    fill=self.colors['mickey_accent'],
                    width=1,
                    smooth=True
                )
                self.visual_elements['mickey_face'].append(wave)
        
        self.mickey_animation_state += 0.1

    def _draw_volume_indicator(self):
        """Draw volume level indicator"""
        try:
            # Clear previous indicator
            if self.visual_elements['volume_circle']:
                self.canvas.delete(self.visual_elements['volume_circle'])
            if self.visual_elements['peak_indicator']:
                self.canvas.delete(self.visual_elements['peak_indicator'])
            
            # Draw current volume indicator
            indicator_height = self.current_volume * (self.height - 40)
            volume_indicator = self.canvas.create_rectangle(
                self.width - 20, self.height - 20 - indicator_height,
                self.width - 10, self.height - 20,
                fill=self.colors['volume_indicator'],
                outline=''
            )
            self.visual_elements['volume_circle'] = volume_indicator
            
            # Draw peak indicator
            peak_height = self.peak_volume * (self.height - 40)
            peak_indicator = self.canvas.create_line(
                self.width - 25, self.height - 20 - peak_height,
                self.width - 5, self.height - 20 - peak_height,
                fill=self.colors['peak_indicator'],
                width=2
            )
            self.visual_elements['peak_indicator'] = peak_indicator
            
        except Exception as e:
            self.logger.error(f"Volume indicator drawing failed: {str(e)}")

    def _hsv_to_hex(self, h: float, s: float, v: float) -> str:
        """Convert HSV color to hex"""
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def set_mode(self, mode: VisualizationMode):
        """Change visualization mode"""
        self.mode = mode
        self._setup_visualization()
        self.logger.info(f"Visualization mode changed to: {mode.value}")

    def set_colors(self, colors: Dict[str, str]):
        """Update color scheme"""
        self.colors.update(colors)
        self.canvas.config(bg=self.colors['background'])
        self._setup_visualization()

    def set_amplitude_scale(self, scale: float):
        """Set amplitude scaling factor"""
        self.amplitude_scale = max(0.1, min(5.0, scale))

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'frames_rendered': self.frame_count,
            'elapsed_time': elapsed_time,
            'fps': fps,
            'current_volume': self.current_volume,
            'peak_volume': self.peak_volume,
            'is_visualizing': self.is_visualizing,
            'visualization_mode': self.mode.value
        }

    def get_canvas(self) -> tk.Canvas:
        """Get the visualization canvas"""
        return self.canvas

    def cleanup(self):
        """Cleanup resources"""
        self.stop_visualization()
        if self.canvas:
            self.canvas.destroy()

# Test function
def test_voice_visualizer():
    """Test the voice visualizer"""
    root = tk.Tk()
    root.title("Voice Visualizer Test")
    root.geometry("600x400")
    
    # Create visualizer
    visualizer = VoiceVisualizer(root, width=600, height=300, mode=VisualizationMode.MICKEY)
    visualizer.get_canvas().pack(pady=20)
    
    # Control frame
    control_frame = tk.Frame(root)
    control_frame.pack(pady=10)
    
    # Mode selector
    mode_var = tk.StringVar(value="mickey")
    modes = [
        ("Waveform", "waveform"),
        ("Spectrum", "spectrum"),
        ("Bars", "bars"),
        ("Circular", "circular"),
        ("Mickey", "mickey")
    ]
    
    for text, mode in modes:
        btn = tk.Radiobutton(control_frame, text=text, variable=mode_var, value=mode,
                           command=lambda: visualizer.set_mode(VisualizationMode(mode_var.get())))
        btn.pack(side=tk.LEFT, padx=5)
    
    # Start/stop buttons
    def start_test():
        visualizer.start_visualization()
        # Simulate audio data
        simulate_audio_data(visualizer)
    
    def stop_test():
        visualizer.stop_visualization()
    
    start_btn = tk.Button(control_frame, text="Start", command=start_test)
    start_btn.pack(side=tk.LEFT, padx=5)
    
    stop_btn = tk.Button(control_frame, text="Stop", command=stop_test)
    stop_btn.pack(side=tk.LEFT, padx=5)
    
    def simulate_audio_data(viz):
        """Simulate audio data for testing"""
        if viz.is_visualizing:
            # Generate test audio signal
            t = time.time()
            frequency = 440 + 220 * math.sin(t * 0.5)  # Varying frequency
            samples = 1024
            audio_data = 0.3 * np.sin(2 * np.pi * frequency * np.arange(samples) / 16000 + t)
            
            # Add some noise
            audio_data += 0.1 * np.random.randn(samples)
            
            # Update visualizer
            viz.update_audio_data(audio_data, 16000)
            
            # Schedule next update
            root.after(50, lambda: simulate_audio_data(viz))
    
    root.mainloop()

if __name__ == "__main__":
    test_voice_visualizer()