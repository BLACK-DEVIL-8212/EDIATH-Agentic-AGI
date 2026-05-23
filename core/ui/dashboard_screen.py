from kivy.uix.screenmanager import Screen
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle

from core.ui.avatar_widget import AvatarWidget
from core.ui.system_stats_panel import SystemStatsPanel
from core.ui.memory_graph_widget import MemoryGraphWidget
from core.ui.planning_steps_panel import PlanningStepsPanel
from core.ui.task_display import TaskDisplay
from core.ui.status_bar import StatusBar
from core.ui.time_widget import TimeWidget
from core.ui.chat_screen import ChatScreen
from core.ui.ai_backend import AIBackend


# ───────────── PANEL ─────────────
class Panel(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(0.1, 0.1, 0.1, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self.update_rect, pos=self.update_rect)

    def update_rect(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos


# ───────────── LOADING OVERLAY ─────────────
class LoadingOverlay(BoxLayout):
    """Full-screen loading overlay shown until backend is ready."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.opacity = 1
        self.size_hint = (1, 1)
        self.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
        with self.canvas.before:
            Color(0.05, 0.05, 0.1, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)

        # Center content in a vertical box
        content = BoxLayout(orientation='vertical', size_hint=(None, None), size=(300, 200))
        content.pos_hint = {'center_x': 0.5, 'center_y': 0.5}

        # Title
        title = Label(
            text="EDIATH AI",
            font_size='28sp',
            halign='center',
            valign='middle',
            color=(0.3, 0.9, 0.5, 1),
            size_hint_y=0.4
        )
        content.add_widget(title)

        # Status message
        self.status_label = Label(
            text="Initializing Systems...",
            font_size='16sp',
            halign='center',
            valign='middle',
            color=(0.7, 0.85, 1, 1),
            size_hint_y=0.3
        )
        content.add_widget(self.status_label)

        # Pulsing dots
        self.dots = Label(
            text="●",
            font_size='18sp',
            color=(0.3, 0.9, 0.5, 1),
            size_hint_y=0.3
        )
        content.add_widget(self.dots)

        self.add_widget(content)

        # Schedule pulse animation
        Clock.schedule_interval(self._pulse, 0.5)

    def _update_rect(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos

    def _pulse(self, dt):
        dots = ["●", "●●", "●●●"]
        current = getattr(self, '_dot_index', 0)
        self._dot_index = (current + 1) % len(dots)
        self.dots.text = dots[self._dot_index]

    def dismiss(self, *args):
        """Fade out and remove the overlay."""
        def fade_out(dt):
            self.opacity -= 0.1
            if self.opacity <= 0:
                Clock.unschedule(fade_out)
                if self.parent:
                    self.parent.remove_widget(self)
        Clock.schedule_interval(fade_out, 0.03)


# ───────────── MAIN SCREEN ─────────────
class DashboardScreen(Screen):

    def __init__(self, controller=None, **kwargs):
        super().__init__(**kwargs)

        # 🔥 Ensure controller exists
        self.controller = controller or AIBackend()
        self._loading_overlay = None

        self.build_ui()
        self._show_loading_overlay()

    # ─────────────────────────────
    # UI BUILD
    # ─────────────────────────────
    def build_ui(self):

        root = GridLayout(cols=1, padding=10, spacing=10)

        # ───────── TOP AREA ─────────
        top = GridLayout(cols=3, spacing=10)

        # ─── LEFT PANEL ───
        left = BoxLayout(orientation="vertical", spacing=10, size_hint_x=0.2)

        avatar_box = Panel(size_hint_y=0.4)
        self.avatar = AvatarWidget()
        avatar_box.add_widget(self.avatar)

        planning_box = Panel(size_hint_y=0.6)
        self.planning_panel = PlanningStepsPanel()
        planning_box.add_widget(self.planning_panel)

        left.add_widget(avatar_box)
        left.add_widget(planning_box)

        # ─── CENTER PANEL ───
        center = BoxLayout(orientation="vertical", spacing=10, size_hint_x=0.6)

        title_row = BoxLayout(size_hint_y=0.15)
        title_row.add_widget(Label(text="EDIATH AI\nIntelligent Assistant"))
        title_row.add_widget(TimeWidget(size_hint_x=0.3))

        chat_container = Panel()
        self.chat_screen = ChatScreen(controller=self.controller)
        chat_container.add_widget(self.chat_screen)

        center.add_widget(title_row)
        center.add_widget(chat_container)

        # ─── RIGHT PANEL ───
        right = BoxLayout(orientation="vertical", spacing=10, size_hint_x=0.2)

        self.memory_graph = MemoryGraphWidget(
            controller=self.controller, size_hint_y=0.4
        )

        stats_box = Panel(size_hint_y=0.6)
        self.stats_panel = SystemStatsPanel()
        stats_box.add_widget(self.stats_panel)

        right.add_widget(self.memory_graph)
        right.add_widget(stats_box)

        # ADD ALL
        top.add_widget(left)
        top.add_widget(center)
        top.add_widget(right)

        # ───────── BOTTOM BAR ─────────
        bottom = Panel(size_hint_y=0.1)

        bottom_layout = BoxLayout()

        self.status_bar = StatusBar()
        self.task_display = TaskDisplay()

        bottom_layout.add_widget(self.status_bar)
        bottom_layout.add_widget(TimeWidget())
        bottom_layout.add_widget(self.task_display)

        bottom.add_widget(bottom_layout)

        # FINAL
        root.add_widget(top)
        root.add_widget(bottom)
        self.add_widget(root)

        # 🔥 Start system updates
        Clock.schedule_once(self.connect_backend, 0.3)
        Clock.schedule_interval(self.update_stats, 1)

    # ─────────────────────────────
    # LOADING OVERLAY MANAGEMENT
    # ─────────────────────────────
    def _show_loading_overlay(self):
        """Show loading overlay on top of the UI."""
        self._loading_overlay = LoadingOverlay()
        self.add_widget(self._loading_overlay)

    def hide_loading_overlay(self, *args):
        """Dismiss loading overlay when backend is ready."""
        if self._loading_overlay:
            self._loading_overlay.dismiss()
            self._loading_overlay = None

    # ─────────────────────────────
    # CONNECT BACKEND
    # ─────────────────────────────
    def connect_backend(self, dt):

        if not self.controller:
            return

        # Chat
        if hasattr(self.controller, "bind_chat_screen"):
            self.controller.bind_chat_screen(self.chat_screen)

        # Status bar
        if hasattr(self.controller, "bind_status_bar"):
            self.controller.bind_status_bar(self.status_bar)

        # Task display
        if hasattr(self.controller, "bind_task_display"):
            self.controller.bind_task_display(self.task_display)

        # Planning
        if hasattr(self.controller, "bind_planning_panel"):
            self.controller.bind_planning_panel(self.planning_panel)

        # Avatar reactions
        if hasattr(self.avatar, "connect_backend"):
            self.avatar.connect_backend(self.controller)

    # ─────────────────────────────
    # SYSTEM STATS LOOP (FIXED)
    # ─────────────────────────────
    def update_stats(self, dt):
        try:
            if hasattr(self.controller, "get_system_stats"):
                stats = self.controller.get_system_stats()

                if hasattr(self.stats_panel, "update_stats"):
                    self.stats_panel.update_stats(stats)

        except Exception as e:
            print("Stats update error:", e)
