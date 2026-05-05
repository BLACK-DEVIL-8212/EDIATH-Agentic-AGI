from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from datetime import datetime
from kivy.app import App


class StatusBar(BoxLayout):
    """
    Bottom status bar (system state, task, connection, time)
    """

    def __init__(self, controller=None, **kwargs):
        super().__init__(orientation="horizontal", **kwargs)

        self.controller = controller

        self.size_hint_y = 0.06
        self.padding = [10, 5]
        self.spacing = 20

        self._build_ui()
        Clock.schedule_interval(self.update_time, 1)

    # ─────────────────────────────
    # UI SETUP
    # ─────────────────────────────
    def _build_ui(self):
        # Get app reference safely
        app = App.get_running_app()

        # Default colors (fallback if app.theme doesn't exist)
        primary_color = (0.2, 0.6, 0.8, 1)  # Default blue
        text_color = (0.9, 0.9, 0.9, 1)  # Default light gray

        # Try to get theme colors if available
        if app and hasattr(app, "theme"):
            try:
                primary_color = app.theme.get_color("primary_color")
                text_color = app.theme.get_color("text_color")
            except Exception:
                pass
        elif app and hasattr(app, "theme_config"):
            try:
                primary_color = app.theme_config.get("primary_color", primary_color)
                text_color = app.theme_config.get("text_color", text_color)
            except Exception:
                pass

        # ─── STATUS ───
        self.status_label = Label(
            text="🟢 Ready", halign="left", valign="middle", color=text_color
        )
        self.status_label.bind(size=self._update_text_size)
        self.add_widget(self.status_label)

        # ─── TASK ───
        self.task_label = Label(
            text="Task: Idle", halign="center", valign="middle", color=text_color
        )
        self.task_label.bind(size=self._update_text_size)
        self.add_widget(self.task_label)

        # ─── CONNECTION ───
        self.connection_label = Label(
            text="📶 Connected",
            halign="center",
            valign="middle",
            color=(0.5, 0.8, 0.5, 1),
        )
        self.connection_label.bind(size=self._update_text_size)
        self.add_widget(self.connection_label)

        # ─── TIME ───
        self.time_label = Label(
            text="--:--:--", halign="right", valign="middle", color=text_color
        )
        self.time_label.bind(size=self._update_text_size)
        self.add_widget(self.time_label)

    def _update_text_size(self, instance, value):
        instance.text_size = instance.size

    # ─────────────────────────────
    # TIME UPDATE
    # ─────────────────────────────
    def update_time(self, dt):
        self.time_label.text = datetime.now().strftime("%H:%M:%S")

    # ─────────────────────────────
    # STATUS UPDATE (FROM BACKEND)
    # ─────────────────────────────
    def update_status(self, text):
        text_lower = text.lower()

        if "error" in text_lower:
            self.status_label.text = "🔴 Error"
            self.status_label.color = (1, 0.4, 0.4, 1)

        elif "thinking" in text_lower:
            self.status_label.text = "🧠 Thinking..."
            self.status_label.color = (1, 0.8, 0.3, 1)

        elif "ready" in text_lower or "done" in text_lower:
            self.status_label.text = "🟢 Ready"
            self.status_label.color = (0.5, 0.8, 0.5, 1)

        else:
            self.status_label.text = text
            self.status_label.color = (0.9, 0.9, 0.9, 1)

    # ─────────────────────────────
    # TASK UPDATE
    # ─────────────────────────────
    def set_task(self, task_name):
        self.task_label.text = f"Task: {task_name}"
        self.task_label.color = (0.9, 0.9, 0.9, 1)

    # backward compatibility
    set_current_task = set_task

    # ─────────────────────────────
    # CONNECTION STATE
    # ─────────────────────────────
    def set_connection(self, connected=True):
        if connected:
            self.connection_label.text = "📶 Connected"
            self.connection_label.color = (0.5, 0.8, 0.5, 1)
        else:
            self.connection_label.text = "⚠ Disconnected"
            self.connection_label.color = (1, 0.4, 0.4, 1)

    # ─────────────────────────────
    # CONNECT TO BACKEND
    # ─────────────────────────────
    def connect_backend(self, controller):
        self.controller = controller

        def on_status(text):
            self.update_status(text)

        if controller and controller.backend:
            controller.backend.set_status_callback(on_status)
