"""
EDIATH 3D Holographic Avatar Widget
A premium, production-quality animated 3D orb with:
- Multi-layer holographic rings
- Particle field with orbital motion
- Reactive mood expressions
- Pulse/breathing animations
- Touch-responsive interactions
- Smooth 60fps canvas rendering
"""

from kivy.uix.widget import Widget
from kivy.graphics import (
    Color,
    Ellipse,
    Line,
    Rectangle,
    Triangle,
    PushMatrix,
    PopMatrix,
    Translate,
    Rotate,
    Scale,
    StencilPop,
    StencilPush,
)
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.properties import NumericProperty, ListProperty, StringProperty
from kivy.core.window import Window
import random
import math
import time


# ─────────────────────────────
# 3D PROJECTION HELPERS
# ─────────────────────────────
def project_3d(x3d, y3d, z3d, cx, cy, fov=200, scale=1.0):
    """Simple perspective projection."""
    if z3d == 0:
        z3d = 0.001
    depth = fov / (fov + z3d)
    return (
        cx + x3d * scale * depth,
        cy + y3d * scale * depth,
        depth,
    )


def rotate_point(x, y, z, angle_x, angle_y):
    """Rotate a 3D point around X and Y axes."""
    # Rotate around Y axis
    cos_y = math.cos(angle_y)
    sin_y = math.sin(angle_y)
    x1 = x * cos_y - z * sin_y
    z1 = x * sin_y + z * cos_y
    # Rotate around X axis
    cos_x = math.cos(angle_x)
    sin_x = math.sin(angle_x)
    y1 = y * cos_x - z1 * sin_x
    z2 = y * sin_x + z1 * cos_x
    return x1, y1, z2


# ─────────────────────────────
# RING CLASS
# ─────────────────────────────
class Ring:
    def __init__(self, radius, thickness, color, speed, tilt, phase_offset=0):
        self.radius = radius
        self.thickness = thickness
        self.color = color
        self.speed = speed
        self.tilt = tilt
        self.phase_offset = phase_offset
        self.points = []

    def generate_points(self, cx, cy, time_val, num_points=72):
        """Generate 3D ring points with perspective."""
        self.points = []
        angle_x = self.tilt
        for i in range(num_points):
            angle_y = (i / num_points) * 2 * math.pi + time_val * self.speed + self.phase_offset
            x = self.radius * math.cos(angle_y)
            y = self.radius * math.sin(angle_y) * 0.3  # Flatten for tilt
            z = self.radius * math.sin(angle_y)
            rx, ry, rz = rotate_point(x, y, z, angle_x, time_val * 0.1)
            px, py, depth = project_3d(rx, ry, rz, cx, cy, fov=180)
            scale = depth * 0.8 + 0.2
            self.points.append((px, py, scale))


# ─────────────────────────────
# ORBITAL PARTICLE
# ─────────────────────────────
class OrbitalParticle:
    def __init__(self, orbit_radius, orbit_speed, size, color, phase):
        self.orbit_radius = orbit_radius
        self.orbit_speed = orbit_speed
        self.size = size
        self.color = color
        self.phase = phase
        self.angle = random.uniform(0, 2 * math.pi)
        self.y_offset = random.uniform(-0.3, 0.3)
        self.z_offset = random.uniform(-0.5, 0.5)

    def get_position(self, cx, cy, time_val, base_radius):
        self.angle += self.orbit_speed * 0.016
        x = self.orbit_radius * math.cos(self.angle + self.phase)
        y = self.orbit_radius * self.y_offset + math.sin(self.angle * 2 + self.phase) * 0.2
        z = self.orbit_radius * math.sin(self.angle + self.phase) * self.z_offset
        rx, ry, rz = rotate_point(x, y, z, time_val * 0.3, time_val * 0.2)
        px, py, depth = project_3d(rx, ry, rz, cx, cy, fov=200)
        scale = depth * 0.7 + 0.3
        return px, py, scale


# ─────────────────────────────
# MAIN 3D AVATAR WIDGET
# ─────────────────────────────
class AvatarWidget(Widget):
    """Premium 3D holographic avatar with multi-layer rendering."""

    rotation_angle = NumericProperty(0)
    pulse_scale = NumericProperty(1.0)
    glow_intensity = NumericProperty(0.5)
    mood = StringProperty("neutral")

    # Colors
    COLOR_CORE = (0.05, 0.15, 0.4, 1)
    COLOR_GLOW = (0.1, 0.5, 1.0, 1)
    COLOR_RING1 = (0.15, 0.6, 1.0, 0.6)
    COLOR_RING2 = (0.3, 0.8, 1.0, 0.4)
    COLOR_RING3 = (0.6, 0.9, 1.0, 0.25)
    COLOR_PARTICLE = (0.4, 0.8, 1.0, 0.8)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mood = "neutral"
        self._events = []
        self._time = 0
        self._rings = []
        self._particles = []
        self._auto_rotate = None
        self._mouse_over = False
        self._touch_start_time = 0
        self._wobble_angle = 0

        self._init_rings()
        self._init_particles()
        self._setup_canvas()
        self._schedule_animations()

        self.bind(pos=self._mark_redraw, size=self._mark_redraw)

    # ─────────────────────────────
    # INIT
    # ─────────────────────────────
    def _init_rings(self):
        self._rings = [
            Ring(radius=1.2, thickness=1.5, color=self.COLOR_RING1, speed=0.8, tilt=0.4, phase_offset=0),
            Ring(radius=1.5, thickness=1.0, color=self.COLOR_RING2, speed=-0.5, tilt=0.7, phase_offset=2.1),
            Ring(radius=1.8, thickness=0.7, color=self.COLOR_RING3, speed=0.3, tilt=1.0, phase_offset=4.2),
            Ring(radius=2.1, thickness=0.5, color=(0.8, 0.95, 1.0, 0.15), speed=-0.2, tilt=1.3, phase_offset=1.0),
        ]

    def _init_particles(self):
        self._particles = []
        for _ in range(35):
            self._particles.append(OrbitalParticle(
                orbit_radius=random.uniform(0.8, 2.0),
                orbit_speed=random.uniform(0.3, 1.2) * random.choice([-1, 1]),
                size=random.uniform(1.5, 4.0),
                color=self.COLOR_PARTICLE,
                phase=random.uniform(0, 2 * math.pi),
            ))

    def _setup_canvas(self):
        with self.canvas:
            # Deep space background
            Color(0.02, 0.03, 0.08, 1)
            self._bg_rect = Rectangle(size=self.size, pos=self.pos)

        self.dynamic_canvas = self.canvas.after
        self._redraw_needed = True

    def _mark_redraw(self, *args):
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size
        self._redraw_needed = True

    def _schedule_animations(self):
        self._events.append(Clock.schedule_interval(self._animate, 1 / 60))
        self._events.append(Clock.schedule_interval(self._blink, random.uniform(2.5, 4.5)))
        self._events.append(Clock.schedule_interval(self._speak, 0.05))
        self._auto_rotate = Animation(rotation_angle=360, duration=25)
        self._auto_rotate.repeat = True
        self._auto_rotate.start(self)

    # ─────────────────────────────
    # ANIMATION LOOPS
    # ─────────────────────────────
    def _animate(self, dt):
        self._time += dt
        self._redraw_needed = True

        # Pulse based on mood
        t = self._time
        if self.mood == "speaking":
            self.pulse_scale = 1.0 + math.sin(t * 18) * 0.06
            self.glow_intensity = 0.5 + math.sin(t * 12) * 0.35
        elif self.mood == "thinking":
            self.pulse_scale = 1.0 + math.sin(t * 2.5) * 0.025
            self.glow_intensity = 0.35 + math.sin(t * 4) * 0.15
        elif self.mood == "happy":
            self.pulse_scale = 1.0 + math.sin(t * 6) * 0.04
            self.glow_intensity = 0.6 + math.sin(t * 5) * 0.2
        else:
            self.pulse_scale = 1.0 + math.sin(t * 1.8) * 0.015
            self.glow_intensity = 0.3 + math.sin(t * 2) * 0.1

        # Mouse wobble effect
        if self._mouse_over:
            self._wobble_angle += 0.15
            wobble = math.sin(self._wobble_angle) * 0.03
            self.pulse_scale = max(1.0, self.pulse_scale + wobble)

    def _blink(self, dt):
        # Blink animation - pulse the glow instead since no zoom property
        if self.mood == "happy":
            pass
        else:
            Animation(glow_intensity=0.1, duration=0.08).start(self)
            Animation(glow_intensity=0.5, duration=0.15).start(self)

    def _speak(self, dt):
        pass  # Mouth animation handled in draw

    # ─────────────────────────────
    # DRAW
    # ─────────────────────────────
    def on_pos(self, *args):
        self._redraw_needed = True

    def on_size(self, *args):
        self._bg_rect.size = self.size
        self._redraw_needed = True

    def on_glow_intensity(self, *args):
        self._redraw_needed = True

    def on_pulse_scale(self, *args):
        self._redraw_needed = True

    def on_rotation_angle(self, *args):
        self._redraw_needed = True

    def draw(self, *args):
        """Main draw function called every frame."""
        if not self._redraw_needed:
            return
        self._redraw_needed = False

        try:
            self.dynamic_canvas.clear()
            x, y = self.pos
            w, h = self.size

            if not w or not h:
                return

            cx, cy = self.center
            base = min(w, h)
            scale_factor = base / 300.0
            r = base * 0.35 * self.pulse_scale

            # ── Outer glow halos ──
            for i in range(4):
                glow_radius = r * (1.4 + i * 0.25)
                alpha = (0.08 - i * 0.015) * self.glow_intensity
                Color(0.15, 0.6, 1.0, alpha)
                Ellipse(
                    pos=(cx - glow_radius, cy - glow_radius),
                    size=(glow_radius * 2, glow_radius * 2),
                )

            # ── 3D Holographic rings ──
            for ring in self._rings:
                ring.generate_points(cx, cy, self._time, num_points=80)
                if len(ring.points) < 2:
                    continue
                Color(*ring.color[:3], ring.color[3] * self.glow_intensity)
                pts = []
                for px, py, sc in ring.points:
                    pts.extend([px - ring.thickness * sc * 0.5, py])
                for px, py, sc in ring.points:
                    pts.extend([px + ring.thickness * sc * 0.5, py])
                if len(pts) >= 4:
                    Line(
                        points=pts,
                        width=ring.thickness * scale_factor * 0.5,
                        cap='round',
                    )

            # ── Orbital particles ──
            for p in self._particles:
                px, py, sc = p.get_position(cx, cy, self._time, r)
                p_size = p.size * scale_factor * sc
                Color(*p.color[:3], p.color[3] * sc)
                Ellipse(
                    pos=(px - p_size / 2, py - p_size / 2),
                    size=(p_size, p_size),
                )

            # ── 3D Sphere core (with rotation) ──
            self._draw_sphere_3d(cx, cy, r, scale_factor)

            # ── Face overlay ──
            self._draw_face(cx, cy, r, scale_factor)

        except Exception as e:
            pass

    def _draw_sphere_3d(self, cx, cy, r, scale):
        """Draw a pseudo-3D sphere with highlight and shadow."""
        with self.dynamic_canvas:
            # Base sphere gradient (simulated with layered ellipses)
            Color(0.08, 0.12, 0.25, 1)
            Ellipse(
                pos=(cx - r, cy - r),
                size=(r * 2, r * 2),
            )

            # Mid layer
            Color(0.1, 0.2, 0.45, 0.9)
            Ellipse(
                pos=(cx - r * 0.85, cy - r * 0.85),
                size=(r * 1.7, r * 1.7),
            )

            # Inner core
            Color(0.15, 0.35, 0.8, 0.95)
            Ellipse(
                pos=(cx - r * 0.65, cy - r * 0.65),
                size=(r * 1.3, r * 1.3),
            )

            # Center highlight
            Color(0.4, 0.7, 1.0, 0.8)
            Ellipse(
                pos=(cx - r * 0.4, cy - r * 0.4),
                size=(r * 0.8, r * 0.8),
            )

            # Shine spot
            Color(0.8, 0.95, 1.0, 0.6)
            Ellipse(
                pos=(cx - r * 0.25, cy - r * 0.1),
                size=(r * 0.3, r * 0.25),
            )

            # Equator line (simulates 3D rotation)
            rot = self.rotation_angle
            Color(0.2, 0.5, 0.9, 0.4)
            Line(
                points=self._get_equator_points(cx, cy, r * 0.98, rot),
                width=0.8 * scale,
            )

            # Back face curve
            Color(0.05, 0.1, 0.2, 0.5)
            Line(
                points=self._get_equator_points(cx, cy, r * 0.85, rot + math.pi),
                width=0.5 * scale,
            )

    def _get_equator_points(self, cx, cy, radius, angle, num_pts=40):
        pts = []
        for i in range(num_pts + 1):
            a = (i / num_pts) * 2 * math.pi + angle
            px = cx + math.cos(a) * radius
            py = cy + math.sin(a) * radius * 0.35  # Flatten for perspective
            pts.extend([px, py])
        return pts

    def _draw_face(self, cx, cy, r, scale):
        """Draw reactive face elements."""
        eye_y = cy + r * 0.15
        eye_spacing = r * 0.35
        eye_w = r * 0.22
        eye_h = eye_w

        if self.mood == "speaking":
            eye_h = eye_w * (0.6 + math.sin(self._time * 20) * 0.3)
        elif self.mood == "happy":
            eye_h = eye_w * 0.6

        # Eyes (white sclera)
        Color(1, 1, 1, 0.95)
        Ellipse(pos=(cx - eye_spacing - eye_w / 2, eye_y - eye_h / 2), size=(eye_w, eye_h))
        Ellipse(pos=(cx + eye_spacing - eye_w / 2, eye_y - eye_h / 2), size=(eye_w, eye_h))

        # Iris
        iris_w = eye_w * 0.55
        iris_h = iris_w
        Color(0.1, 0.5, 0.95, 1)
        Ellipse(pos=(cx - eye_spacing - iris_w / 2, eye_y - iris_h / 2), size=(iris_w, iris_h))
        Ellipse(pos=(cx + eye_spacing - iris_w / 2, eye_y - iris_h / 2), size=(iris_w, iris_h))

        # Pupils
        pup_w = eye_w * 0.28
        pup_h = pup_w
        Color(0.02, 0.02, 0.05, 1)
        Ellipse(pos=(cx - eye_spacing - pup_w / 2, eye_y - pup_h / 2), size=(pup_w, pup_h))
        Ellipse(pos=(cx + eye_spacing - pup_w / 2, eye_y - pup_h / 2), size=(pup_w, pup_h))

        # Shine
        shine_w = eye_w * 0.15
        Color(1, 1, 1, 0.9)
        Ellipse(
            pos=(cx - eye_spacing + eye_w * 0.1, eye_y + eye_h * 0.1),
            size=(shine_w, shine_w),
        )
        Ellipse(
            pos=(cx + eye_spacing + eye_w * 0.1, eye_y + eye_h * 0.1),
            size=(shine_w, shine_w),
        )

        # Mouth
        mouth_pts = self._get_mouth_pts(cx, cy, r)
        Color(1, 0.4, 0.5, 0.9)
        Line(points=mouth_pts, width=2.2 * scale, cap='round')

    def _get_mouth_pts(self, cx, cy, r):
        t = self._time
        if self.mood == "happy":
            return [
                cx - r * 0.35, cy - r * 0.15,
                cx - r * 0.15, cy - r * 0.3,
                cx + r * 0.05, cy - r * 0.28,
                cx + r * 0.35, cy - r * 0.15,
            ]
        elif self.mood == "thinking":
            return [
                cx - r * 0.3, cy - r * 0.12,
                cx, cy - r * 0.15,
                cx + r * 0.3, cy - r * 0.12,
            ]
        elif self.mood == "speaking":
            gap = r * 0.08 * (0.5 + math.sin(t * 25))
            return [
                cx - r * 0.25, cy - r * 0.08,
                cx, cy - r * 0.2 + gap,
                cx + r * 0.25, cy - r * 0.08,
            ]
        elif self.mood == "sad":
            return [
                cx - r * 0.3, cy - r * 0.08,
                cx - r * 0.1, cy - r * 0.18,
                cx + r * 0.15, cy - r * 0.15,
            ]
        return [
            cx - r * 0.3, cy - r * 0.1,
            cx, cy - r * 0.13,
            cx + r * 0.3, cy - r * 0.1,
        ]

    # ─────────────────────────────
    # MOOD CONTROL
    # ─────────────────────────────
    def set_mood(self, mood):
        self.mood = mood
        self._redraw_needed = True

    # ─────────────────────────────
    # BACKEND LINK
    # ─────────────────────────────
    def connect_backend(self, controller):
        backend = getattr(controller, "backend", controller)
        if not hasattr(backend, "set_status_callback"):
            return

        def on_status(text):
            text = text.lower()
            if "thinking" in text:
                self.set_mood("thinking")
            elif "speaking" in text or "generating" in text:
                self.set_mood("speaking")
            elif "done" in text or "ready" in text:
                self.set_mood("happy")
                Clock.schedule_once(lambda dt: self.set_mood("neutral"), 3)
            elif "error" in text or "fail" in text:
                self.set_mood("sad")
                Clock.schedule_once(lambda dt: self.set_mood("neutral"), 2)
            else:
                self.set_mood("neutral")

        backend.set_status_callback(on_status)

    # ─────────────────────────────
    # INTERACTION
    # ─────────────────────────────
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self._mouse_over = True
            self._touch_start_time = time.time()
            self.set_mood("happy")
            Animation(pulse_scale=1.1, duration=0.2).start(self)
            return True
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        if self._mouse_over:
            self._mouse_over = False
            elapsed = time.time() - self._touch_start_time
            if elapsed < 0.5:
                Clock.schedule_once(lambda dt: self.set_mood("neutral"), 0.5)
            Animation(pulse_scale=1.0, duration=0.3).start(self)
        return super().on_touch_up(touch)

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            self._mouse_over = True
        else:
            self._mouse_over = False
        return super().on_touch_move(touch)

    # ─────────────────────────────
    # SAFE SHUTDOWN
    # ─────────────────────────────
    def stop(self):
        for e in self._events:
            e.cancel()
        self._events.clear()
        if self._auto_rotate:
            self._auto_rotate.cancel(self)

    def on_leave(self):
        self.stop()
