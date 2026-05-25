"""
EDIATH Dashboard Screen - Production Quality UI
Premium glassmorphism layout with:
- Deep space dark theme
- 3D holographic avatar panel with scan lines
- Glassmorphism info panels
- Animated status bar with pulse dots
- Responsive 3-column layout
- Smooth loading overlay
"""

from kivy.uix.screenmanager import Screen
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Line, RoundedRectangle, Ellipse
from kivy.metrics import dp
from kivy.animation import Animation
import math

from core.ui.avatar_widget import AvatarWidget
from core.ui.system_stats_panel import SystemStatsPanel
from core.ui.memory_graph_widget import MemoryGraphWidget
from core.ui.planning_steps_panel import PlanningStepsPanel
from core.ui.task_display import TaskDisplay
from core.ui.status_bar import StatusBar
from core.ui.time_widget import TimeWidget
from core.ui.chat_screen import ChatScreen
from core.ui.ai_backend import AIBackend
from core.ui.components import GlassPanel, StatusDot, ScanLine


# ─────────────────────────────
# GRID BACKGROUND
# ─────────────────────────────
class GridBackground(Widget):
    """Subtle grid background for depth."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *args):
        self.canvas.clear()
        if not self.size[0] or not self.size[1]:
            return
        with self.canvas:
            # Vertical lines
            Color(0.15, 0.4, 0.8, 0.04)
            spacing = 40
            x = self.pos[0]
            while x < self.pos[0] + self.size[0]:
                Line(points=[x, self.pos[1], x, self.pos[1] + self.size[1]], width=0.5)
                x += spacing
            # Horizontal lines
            y = self.pos[1]
            while y < self.pos[1] + self.size[1]:
                Line(points=[self.pos[0], y, self.pos[0] + self.size[0], y], width=0.5)
                y += spacing


# ─────────────────────────────
# PANEL WITH GLASS EFFECT
# ─────────────────────────────
class GlassBox(BoxLayout):
    """BoxLayout with glassmorphism styling."""

    def __init__(self, bg_alpha=0.88, border_alpha=0.5, radius=18, **kwargs):
        super().__init__(**kwargs)
        self._bg_alpha = bg_alpha
        self._border_alpha = border_alpha
        self._radius = radius
        with self.canvas.before:
            Color(0.06, 0.06, 0.12, bg_alpha)
            self._bg = RoundedRectangle(pos=self.pos, size=self.size, radius=[radius])
            Color(0.15, 0.55, 1.0, border_alpha)
            self._border = Line(
                rounded_rectangle=(self.pos[0], self.pos[1], self.size[0], self.size[1], radius),
                width=1.0,
            )
        self.bind(pos=self._update, size=self._update)

    def _update(self, *args):
        if not self.size[0] or not self.size[1]:
            return
        self._bg.pos = self.pos
        self._bg.size = self.size
        self._bg.radius = [self._radius]
        self._border.rounded_rectangle = (
            self.pos[0], self.pos[1], self.size[0], self.size[1], self._radius
        )


# ─────────────────────────────
# HEADER LABEL
# ─────────────────────────────
class SectionLabel(Label):
    """Stylized section header label."""

    def __init__(self, text="", size_scale=1.0, **kwargs):
        super().__init__(
            text=text,
            font_size=f"{int(16 * size_scale)}sp",
            bold=True,
            color=(0.2, 0.8, 1.0, 1.0),
            halign="left",
            valign="middle",
            **kwargs,
        )


# ─────────────────────────────
# TYPING INDICATOR
# ─────────────────────────────
class TypingIndicator(BoxLayout):
    """Animated 3-dot typing indicator."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.size_hint = (None, None)
        self.size = (60, 30)
        self._dots = []
        self._phase = 0.0
        self._events = []
        for i in range(3):
            dot = Widget(size_hint=(None, None), size=(dp(8), dp(8)))
            dot.canvas.clear()
            with dot.canvas:
                Color(0.15, 0.65, 1.0, 1.0)
                dot._circle = Rectangle(size=dot.size, pos=dot.pos)
            dot.bind(pos=self._upd, size=self._upd)
            self._dots.append(dot)
            self.add_widget(dot)
        self._events.append(Clock.schedule_interval(self._animate, 1 / 30))
        self.bind(pos=self._upd_all, size=self._upd_all)

    def _upd(self, w, *args):
        w.canvas.clear()
        with w.canvas:
            Color(0.15, 0.65, 1.0, 1.0)
            cx = w.center_x
            cy = w.center_y
            r = min(w.width, w.height) / 2
            Ellipse(pos=(cx - r, cy - r), size=(r * 2, r * 2))

    def _upd_all(self, *args):
        for d in self._dots:
            d.pos = (d.pos[0], d.pos[1])
            self._upd(d)

    def _animate(self, dt):
        self._phase += 0.15
        for i, dot in enumerate(self._dots):
            alpha = max(0.1, math.sin(self._phase + i * 0.8) * 0.8 + 0.2)
            dot.canvas.clear()
            with dot.canvas:
                Color(0.15, 0.65, 1.0, alpha)
                cx = dot.center_x
                cy = dot.center_y
                r = min(dot.width, dot.height) / 2
                Ellipse(pos=(cx - r, cy - r), size=(r * 2, r * 2))

    def show(self):
        self.opacity = 1

    def hide(self):
        self.opacity = 0

    def stop(self):
        for e in self._events:
            e.cancel()


# ─────────────────────────────
# LOADING OVERLAY
# ─────────────────────────────
class LoadingOverlay(BoxLayout):
    """Full-screen loading overlay with animated orb."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.opacity = 1
        self.size_hint = (1, 1)
        self.pos_hint = {"center_x": 0.5, "center_y": 0.5}
        with self.canvas.before:
            Color(0.02, 0.02, 0.06, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._upd_rect, pos=self._upd_rect)

        # Content
        content = BoxLayout(orientation="vertical", size_hint=(None, None), size=(320, 240))
        content.pos_hint = {"center_x": 0.5, "center_y": 0.5}
        content.spacing = 10

        # Orb animation widget
        orb_container = BoxLayout(size_hint=(1, None), size=(100, 100))
        self.orb = _LoadingOrb()
        orb_container.add_widget(self.orb)

        # Title
        title = Label(
            text="EDIATH AI",
            font_size="32sp",
            halign="center",
            bold=True,
            color=(0.3, 0.9, 0.5, 1),
        )

        # Status
        self.status_label = Label(
            text="Initializing...",
            font_size="15sp",
            halign="center",
            color=(0.6, 0.8, 1.0, 1),
        )

        # Dots
        self.dots_label = Label(
            text="●",
            font_size="20sp",
            color=(0.3, 0.9, 0.5, 1),
        )

        content.add_widget(orb_container)
        content.add_widget(title)
        content.add_widget(self.status_label)
        content.add_widget(self.dots_label)
        self.add_widget(content)

        self._dot_index = 0
        Clock.schedule_interval(self._pulse, 0.4)
        self._events = []
        self._events.append(Clock.schedule_interval(self._animate_orb, 1 / 60))

    def _upd_rect(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos

    def _pulse(self, dt):
        dots = ["●", "●●", "●●●"]
        self._dot_index = (self._dot_index + 1) % len(dots)
        self.dots_label.text = dots[self._dot_index]

    def _animate_orb(self, dt):
        if hasattr(self.orb, "animate"):
            self.orb.animate(dt)

    def set_status(self, text):
        self.status_label.text = text

    def dismiss(self, *args):
        def fade_out(dt):
            self.opacity -= 0.08
            if self.opacity <= 0:
                Clock.unschedule(fade_out)
                if self.parent:
                    self.parent.remove_widget(self)
        Clock.schedule_interval(fade_out, 0.03)

    def stop(self):
        for e in self._events:
            e.cancel()
        if hasattr(self.orb, "stop"):
            self.orb.stop()


class _LoadingOrb(Widget):
    """Animated loading orb for the splash screen."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.size = (100, 100)
        self._phase = 0.0
        self._pulse_phase = 0.0
        self.bind(pos=self._redraw, size=self._redraw)

    def animate(self, dt):
        self._phase += 0.03
        self._pulse_phase += 0.08
        self._redraw()

    def _redraw(self, *args):
        self.canvas.clear()
        cx = self.center_x
        cy = self.center_y
        r = min(self.width, self.height) / 2 * 0.8
        if r <= 0:
            return
        # Outer glow
        for i in range(3):
            glow_r = r * (1.2 + i * 0.3)
            alpha = (0.15 - i * 0.04) * (0.5 + math.sin(self._pulse_phase) * 0.5)
            Color(0.15, 0.6, 1.0, alpha)
            Ellipse(pos=(cx - glow_r, cy - glow_r), size=(glow_r * 2, glow_r * 2))
        # Core
        Color(0.1, 0.3, 0.7, 0.9)
        Ellipse(pos=(cx - r, cy - r), size=(r * 2, r * 2))
        # Inner
        Color(0.3, 0.7, 1.0, 0.8)
        Ellipse(pos=(cx - r * 0.6, cy - r * 0.6), size=(r * 1.2, r * 1.2))
        # Equator line
        rot = self._phase
        Color(0.2, 0.6, 1.0, 0.5)
        pts = []
        for i in range(73):
            a = (i / 72) * 2 * math.pi + rot
            pts.extend([cx + math.cos(a) * r, cy + math.sin(a) * r * 0.35])
        Line(points=pts, width=1.5, cap='round')


# ─────────────────────────────
# MAIN DASHBOARD SCREEN
# ─────────────────────────────
class DashboardScreen(Screen):
    """Production-quality EDIATH dashboard with 3D avatar and glassmorphism UI."""

    def __init__(self, controller=None, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller or AIBackend()
        self._loading_overlay = None
        self._typing_indicator = None
        self._scan_line = None
        self._events = []

        self.build_ui()
        self._show_loading_overlay()

    # ─────────────────────────────
    # UI BUILD
    # ─────────────────────────────
    def build_ui(self):
        # Background grid
        self._bg_grid = GridBackground()
        self.add_widget(self._bg_grid)

        # Root layout
        root = BoxLayout(orientation="vertical", spacing=dp(8), padding=dp(8))

        # ── TOP ROW ──
        top = BoxLayout(orientation="horizontal", spacing=dp(8), size_hint_y=0.88)

        # ─── LEFT: Avatar + Planning ───
        left_col = BoxLayout(orientation="vertical", spacing=dp(8), size_hint_x=0.22)

        # Avatar panel
        avatar_panel = GlassBox(radius=20, bg_alpha=0.9, border_alpha=0.4)
        avatar_panel.size_hint_y = 0.48

        # Avatar title
        avatar_header = BoxLayout(size_hint_y=None, size=(0, 36), padding=(dp(12), 0))
        avatar_header.add_widget(SectionLabel(text="AVATAR", size_scale=0.9))
        avatar_panel.add_widget(avatar_header)

        # Avatar widget with scan line
        avatar_content = BoxLayout(size_hint_y=1)
        self.avatar = AvatarWidget()
        avatar_content.add_widget(self.avatar)

        # Scan line overlay
        self._scan_line = ScanLine()
        avatar_content.add_widget(self._scan_line)

        avatar_panel.add_widget(avatar_content)
        left_col.add_widget(avatar_panel)

        # Planning panel
        planning_panel = GlassBox(radius=20, bg_alpha=0.9, border_alpha=0.4)
        planning_panel.size_hint_y = 0.52

        planning_header = BoxLayout(size_hint_y=None, size=(0, 36), padding=(dp(12), 0))
        planning_header.add_widget(SectionLabel(text="PLANNING", size_scale=0.9))
        planning_panel.add_widget(planning_header)

        self.planning_panel = PlanningStepsPanel()
        planning_panel.add_widget(self.planning_panel)

        left_col.add_widget(planning_panel)

        # ─── CENTER: Chat ───
        center_col = BoxLayout(orientation="vertical", spacing=dp(8), size_hint_x=0.56)

        # Title bar
        title_bar = GlassBox(radius=14, bg_alpha=0.85, border_alpha=0.35)
        title_bar.size_hint_y = 0.08

        title_label = Label(
            text="EDIATH AI  •  Intelligent Assistant",
            font_size="20sp",
            bold=True,
            color=(0.2, 0.85, 1.0, 1.0),
            halign="left",
            valign="middle",
            padding=(dp(16), 0),
        )

        time_w = TimeWidget(size_hint_x=0.3)

        title_bar.add_widget(title_label)
        title_bar.add_widget(time_w)

        center_col.add_widget(title_bar)

        # Chat container
        chat_panel = GlassBox(radius=18, bg_alpha=0.88, border_alpha=0.35)
        chat_panel.size_hint_y = 0.92

        self.chat_screen = ChatScreen(controller=self.controller)
        chat_panel.add_widget(self.chat_screen)

        # Typing indicator
        self._typing_indicator = TypingIndicator()
        self._typing_indicator.opacity = 0
        # Position it at bottom-right of chat
        chat_panel.add_widget(self._typing_indicator)

        center_col.add_widget(chat_panel)

        # ─── RIGHT: Stats + Memory ───
        right_col = BoxLayout(orientation="vertical", spacing=dp(8), size_hint_x=0.22)

        # Memory panel
        memory_panel = GlassBox(radius=20, bg_alpha=0.9, border_alpha=0.4)
        memory_panel.size_hint_y = 0.38

        memory_header = BoxLayout(size_hint_y=None, size=(0, 36), padding=(dp(12), 0))
        memory_header.add_widget(SectionLabel(text="MEMORY", size_scale=0.9))
        memory_panel.add_widget(memory_header)

        self.memory_graph = MemoryGraphWidget(controller=self.controller)
        memory_panel.add_widget(self.memory_graph)

        right_col.add_widget(memory_panel)

        # Stats panel
        stats_panel = GlassBox(radius=20, bg_alpha=0.9, border_alpha=0.4)
        stats_panel.size_hint_y = 0.62

        stats_header = BoxLayout(size_hint_y=None, size=(0, 36), padding=(dp(12), 0))
        stats_header.add_widget(SectionLabel(text="SYSTEM", size_scale=0.9))
        stats_panel.add_widget(stats_header)

        self.stats_panel = SystemStatsPanel()
        stats_panel.add_widget(self.stats_panel)

        right_col.add_widget(stats_panel)

        # Add columns to top row
        top.add_widget(left_col)
        top.add_widget(center_col)
        top.add_widget(right_col)

        # ── BOTTOM BAR ──
        bottom_bar = GlassBox(radius=12, bg_alpha=0.85, border_alpha=0.35)
        bottom_bar.size_hint_y = 0.10
        bottom_bar.padding = (dp(12), dp(4))

        bottom_layout = BoxLayout()

        # Status with dot
        status_container = BoxLayout(size_hint_x=0.3, spacing=dp(8))
        self._status_dot = StatusDot(color=[0.2, 1.0, 0.4, 1.0])
        self._status_dot.size_hint = (None, None)
        self._status_dot.size = (dp(10), dp(10))
        status_label = Label(
            text="●  Online",
            font_size="14sp",
            color=(0.4, 1.0, 0.5, 1.0),
            halign="left",
            valign="middle",
        )
        status_container.add_widget(self._status_dot)
        status_container.add_widget(status_label)

        self.status_bar = StatusBar()

        time_w2 = TimeWidget(size_hint_x=0.2)
        self.task_display = TaskDisplay()

        bottom_layout.add_widget(status_container)
        bottom_layout.add_widget(self.status_bar)
        bottom_layout.add_widget(time_w2)
        bottom_layout.add_widget(self.task_display)

        bottom_bar.add_widget(bottom_layout)

        # ── ASSEMBLE ──
        root.add_widget(top)
        root.add_widget(bottom_bar)

        self.add_widget(root)

        # ── SCHEDULE ──
        self._events.append(Clock.schedule_once(self.connect_backend, 0.3))
        self._events.append(Clock.schedule_interval(self.update_stats, 2))
        self._events.append(Clock.schedule_interval(self._update_background, 1 / 30))

    def _update_background(self, dt):
        """Keep background grid sized correctly."""
        if self._bg_grid:
            self._bg_grid.pos = self.pos
            self._bg_grid.size = self.size

    # ─────────────────────────────
    # LOADING OVERLAY
    # ─────────────────────────────
    def _show_loading_overlay(self):
        self._loading_overlay = LoadingOverlay()
        self.add_widget(self._loading_overlay)

    def hide_loading_overlay(self, *args):
        if self._loading_overlay:
            self._loading_overlay.dismiss()
            self._loading_overlay = None

    # ─────────────────────────────
    # CONNECT BACKEND
    # ─────────────────────────────
    def connect_backend(self, dt):
        if not self.controller:
            return

        if hasattr(self.controller, "bind_chat_screen"):
            self.controller.bind_chat_screen(self.chat_screen)

        if hasattr(self.controller, "bind_status_bar"):
            self.controller.bind_status_bar(self.status_bar)

        if hasattr(self.controller, "bind_task_display"):
            self.controller.bind_task_display(self.task_display)

        if hasattr(self.controller, "bind_planning_panel"):
            self.controller.bind_planning_panel(self.planning_panel)

        if hasattr(self.avatar, "connect_backend"):
            self.avatar.connect_backend(self.controller)

        # Register response callback to hide loading overlay
        if hasattr(self.controller, "set_response_callback"):
            self.controller.set_response_callback(self._on_ai_ready)

        # Also schedule a fallback dismiss after 8 seconds
        Clock.schedule_once(self.hide_loading_overlay, 8)

    def _on_ai_ready(self, text):
        """Called when first AI response arrives."""
        self.hide_loading_overlay()
        if self._status_dot:
            self._status_dot.dot_color = [0.2, 1.0, 0.4, 1.0]

    # ─────────────────────────────
    # SYSTEM STATS UPDATE
    # ─────────────────────────────
    def update_stats(self, dt):
        try:
            if hasattr(self.controller, "get_system_stats"):
                stats = self.controller.get_system_stats()
                if hasattr(self.stats_panel, "update_stats"):
                    self.stats_panel.update_stats(stats)
        except Exception:
            pass

    # ─────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────
    def stop(self):
        for e in self._events:
            e.cancel()
        if self._scan_line:
            self._scan_line.stop()
        if self._loading_overlay:
            self._loading_overlay.stop()
        if hasattr(self.avatar, "stop"):
            self.avatar.stop()
        if self._typing_indicator:
            self._typing_indicator.stop()
