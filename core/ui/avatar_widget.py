from kivy.uix.widget import Widget
from kivy.graphics import (
    Color,
    Ellipse,
    Line,
    RoundedRectangle,
    PushMatrix,
    PopMatrix,
    Translate,
    Rotate,
)
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.properties import NumericProperty
import random
import math


# ─────────────────────────────
# PARTICLE CLASS
# ─────────────────────────────
class Particle:
    def __init__(self, x, y, radius, speed_x, speed_y, color, alpha=1.0):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.color = color
        self.alpha = alpha


# ─────────────────────────────
# AVATAR WIDGET (FIXED)
# ─────────────────────────────
class AvatarWidget(Widget):

    rotation_angle = NumericProperty(0)
    pulse_scale = NumericProperty(1.0)
    glow_intensity = NumericProperty(0.5)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mood = "neutral"
        self.eye_open = 1.0
        self.particles = []
        self._particle_count = 40
        self._rotation_speed = 0.5

        # SAFE CLOCK STORAGE (fix shutdown crashes)
        self._events = []

        self._setup_canvas()
        self._init_particles()

        self.bind(pos=self.update_graphics, size=self.update_graphics)

        # SAFE scheduling
        self._events.append(Clock.schedule_interval(self._blink, 3))
        self._events.append(Clock.schedule_interval(self._update_particles, 1 / 60))
        self._events.append(Clock.schedule_interval(self._rotate_sphere, 1 / 30))
        self._events.append(Clock.schedule_interval(self._pulse_effect, 1 / 20))

        # Rotation animation
        self._auto_rotate = Animation(rotation_angle=360, duration=20)
        self._auto_rotate.repeat = True
        self._auto_rotate.start(self)

    # ─────────────────────────────
    # SAFE SHUTDOWN (CRITICAL FIX)
    # ─────────────────────────────
    def stop(self):
        for e in self._events:
            e.cancel()
        self._events.clear()

        if self._auto_rotate:
            self._auto_rotate.cancel(self)

    # ─────────────────────────────
    # PARTICLES
    # ─────────────────────────────
    def _init_particles(self):
        self.particles.clear()

        for _ in range(self._particle_count):
            self.particles.append(
                Particle(
                    0,
                    0,
                    random.uniform(2, 4),
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    (0.3, 0.6, 1, 1),
                    random.uniform(0.3, 0.8),
                )
            )

    def _update_particles(self, dt):
        if not self.size[0] or not self.size[1]:
            return

        cx = self.center_x
        cy = self.center_y
        radius = min(self.width, self.height) * 0.4

        for p in self.particles:
            p.x += p.speed_x * dt * 50
            p.y += p.speed_y * dt * 50

            dx = p.x - cx
            dy = p.y - cy
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > radius * 1.2:
                angle = random.uniform(0, 2 * math.pi)
                p.x = cx + math.cos(angle) * radius * 0.8
                p.y = cy + math.sin(angle) * radius * 0.8
                p.speed_x = random.uniform(-3, 3)
                p.speed_y = random.uniform(-3, 3)

    # ─────────────────────────────
    # CANVAS
    # ─────────────────────────────
    def _setup_canvas(self):
        with self.canvas:
            Color(0.05, 0.05, 0.1, 1)
            self.bg = RoundedRectangle(radius=[15])

        # SAFE dynamic layer (fixes Kivy remove crash)
        self.dynamic_canvas = self.canvas.after

    # ─────────────────────────────
    def update_graphics(self, *args):
        try:
            self.dynamic_canvas.clear()

            x, y = self.pos
            w, h = self.size

            if not w or not h:
                return

            cx, cy = self.center
            radius = min(w, h) * 0.4

            self.bg.pos = self.pos
            self.bg.size = self.size

            with self.dynamic_canvas:
                # glow
                Color(0, 0.5, 1, 0.2 + self.glow_intensity * 0.3)
                Ellipse(
                    pos=(cx - radius * 1.1, cy - radius * 1.1),
                    size=(radius * 2.2, radius * 2.2),
                )

                # sphere with rotation transform
                PushMatrix()
                Translate(cx, cy)
                rot = Rotate(angle=self.rotation_angle, origin=(0, 0))
                Color(*self._get_color())
                Ellipse(
                    pos=(-radius * self.pulse_scale, -radius * self.pulse_scale),
                    size=(radius * 2 * self.pulse_scale, radius * 2 * self.pulse_scale),
                )
                PopMatrix()

                # eyes
                eye_w = radius * 0.25
                eye_h = eye_w * self.eye_open

                Color(1, 1, 1, 1)
                Ellipse(pos=(cx - radius * 0.4, cy + radius * 0.2), size=(eye_w, eye_h))
                Ellipse(pos=(cx + radius * 0.2, cy + radius * 0.2), size=(eye_w, eye_h))

                # pupils
                Color(0, 0, 0, 1)
                Ellipse(
                    pos=(cx - radius * 0.35, cy + radius * 0.2),
                    size=(eye_w * 0.5, eye_w * 0.5),
                )
                Ellipse(
                    pos=(cx + radius * 0.25, cy + radius * 0.2),
                    size=(eye_w * 0.5, eye_w * 0.5),
                )

                # mouth
                Color(1, 0.6, 0.3, 1)
                Line(points=self._get_mouth(cx, cy, radius), width=2)

                # particles (initialize positions if still at origin)
                for p in self.particles:
                    if p.x == 0 and p.y == 0:
                        angle = random.uniform(0, 2 * math.pi)
                        p.x = cx + math.cos(angle) * radius * random.uniform(0.2, 1.0)
                        p.y = cy + math.sin(angle) * radius * random.uniform(0.2, 1.0)

                    Color(*p.color[:3], p.alpha)
                    Ellipse(
                        pos=(p.x - p.radius / 2, p.y - p.radius / 2),
                        size=(p.radius, p.radius),
                    )

        except Exception as e:
            print("Avatar draw error:", e)

    # ─────────────────────────────
    def _get_color(self):
        if self.mood == "happy":
            return (0.3, 0.7, 0.5, 1)
        elif self.mood == "thinking":
            return (0.2, 0.4, 0.8, 1)
        elif self.mood == "speaking":
            return (0.8, 0.3, 0.5, 1)
        return (0.2, 0.3, 0.6, 1)

    def _get_mouth(self, cx, cy, r):
        if self.mood == "happy":
            return [
                cx - r * 0.3,
                cy - r * 0.1,
                cx,
                cy - r * 0.25,
                cx + r * 0.3,
                cy - r * 0.1,
            ]
        elif self.mood == "thinking":
            return [
                cx - r * 0.3,
                cy - r * 0.1,
                cx,
                cy - r * 0.12,
                cx + r * 0.3,
                cy - r * 0.1,
            ]
        elif self.mood == "speaking":
            return [
                cx - r * 0.25,
                cy - r * 0.05,
                cx,
                cy - r * 0.2,
                cx + r * 0.25,
                cy - r * 0.05,
            ]
        return [
            cx - r * 0.3,
            cy - r * 0.08,
            cx,
            cy - r * 0.1,
            cx + r * 0.3,
            cy - r * 0.08,
        ]

    # ─────────────────────────────
    # ANIMATIONS
    # ─────────────────────────────
    def _rotate_sphere(self, dt):
        self.rotation_angle += self._rotation_speed

    def _pulse_effect(self, dt):
        t = Clock.get_time()

        if self.mood == "speaking":
            self.pulse_scale = 1 + math.sin(t * 20) * 0.05
            self.glow_intensity = 0.5 + math.sin(t * 15) * 0.3
        else:
            self.pulse_scale = 1 + math.sin(t * 3) * 0.02
            self.glow_intensity = 0.3 + math.sin(t * 3) * 0.1

    def _blink(self, dt):
        Animation(eye_open=0, duration=0.1).start(self)
        Animation(eye_open=1, duration=0.2).start(self)

    # ─────────────────────────────
    # MOOD CONTROL
    # ─────────────────────────────
    def set_mood(self, mood):
        self.mood = mood
        self.update_graphics()

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

            elif "speaking" in text:
                self.set_mood("speaking")

            elif "done" in text or "ready" in text:
                self.set_mood("happy")

            elif "error" in text:
                self.set_mood("neutral")

        backend.set_status_callback(on_status)

    # ─────────────────────────────
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.set_mood("happy")
            Clock.schedule_once(lambda dt: self.set_mood("neutral"), 1)
            return True
        return super().on_touch_down(touch)
