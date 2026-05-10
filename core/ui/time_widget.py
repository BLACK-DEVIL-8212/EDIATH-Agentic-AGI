from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from datetime import datetime
from kivy.app import App


class TimeWidget(BoxLayout):
    """
    Displays current time, date, and AM/PM
    """

    def __init__(self, use_24h=False, **kwargs):
        super().__init__(orientation="vertical", **kwargs)

        self.use_24h = use_24h

        self.padding = [10, 5]
        self.spacing = 2

        # Default colors (fallback if app.theme doesn't exist)
        self._primary_color = (0.2, 0.6, 0.8, 1)  # Default blue
        self._subtext_color = (0.7, 0.7, 0.7, 1)  # Default gray
        self._accent_color = (0.8, 0.6, 0.2, 1)  # Default amber

        self._build_ui()

        Clock.schedule_interval(self.update_time, 1)

    # ─────────────────────────────
    # UI SETUP
    # ─────────────────────────────
    def _build_ui(self):
        app = App.get_running_app()

        # Try to get theme colors if available
        if app and hasattr(app, "theme"):
            try:
                self._primary_color = app.theme.get_color("primary_color")
                self._subtext_color = app.theme.get_color("subtext_color")
                self._accent_color = app.theme.get_color("accent_color")
            except Exception:
                pass
        elif app and hasattr(app, "theme_config"):
            try:
                self._primary_color = app.theme_config.get(
                    "primary_color", self._primary_color
                )
                self._subtext_color = app.theme_config.get(
                    "subtext_color", self._subtext_color
                )
                self._accent_color = app.theme_config.get(
                    "accent_color", self._accent_color
                )
            except Exception:
                pass

        # TIME
        self.time_label = Label(
            text="--:--",
            font_size="28sp",
            bold=True,
            halign="center",
            valign="middle",
            color=self._primary_color,
        )
        self.time_label.bind(size=self._update_text_size)
        self.add_widget(self.time_label)

        # DATE
        self.date_label = Label(
            text="",
            font_size="12sp",
            halign="center",
            valign="middle",
            color=self._subtext_color,
        )
        self.date_label.bind(size=self._update_text_size)
        self.add_widget(self.date_label)

        # AM/PM
        self.ampm_label = Label(
            text="",
            font_size="14sp",
            halign="center",
            valign="middle",
            color=self._accent_color,
        )
        self.ampm_label.bind(size=self._update_text_size)
        self.add_widget(self.ampm_label)

    def _update_text_size(self, instance, value):
        instance.text_size = instance.size

    # ─────────────────────────────
    # TIME UPDATE
    # ─────────────────────────────
    def update_time(self, dt):
        now = datetime.now()

        if self.use_24h:
            self.time_label.text = now.strftime("%H:%M")
            self.ampm_label.text = ""
        else:
            self.time_label.text = now.strftime("%I:%M")
            self.ampm_label.text = now.strftime("%p")

        self.date_label.text = now.strftime("%B %d, %Y")

    # ─────────────────────────────
    # THEME SUPPORT
    # ─────────────────────────────
    def apply_theme(self):
        app = App.get_running_app()

        # Get colors from app theme if available, otherwise use stored defaults
        primary_color = self._primary_color
        subtext_color = self._subtext_color
        accent_color = self._accent_color

        if app and hasattr(app, "theme"):
            try:
                primary_color = app.theme.get_color("primary_color")
                subtext_color = app.theme.get_color("subtext_color")
                accent_color = app.theme.get_color("accent_color")
            except Exception:
                pass
        elif app and hasattr(app, "theme_config"):
            try:
                primary_color = app.theme_config.get("primary_color", primary_color)
                subtext_color = app.theme_config.get("subtext_color", subtext_color)
                accent_color = app.theme_config.get("accent_color", accent_color)
            except Exception:
                pass

        self.time_label.color = primary_color
        self.date_label.color = subtext_color
        self.ampm_label.color = accent_color

    def connect_theme(self, theme):
        """
        Optional: auto-update on theme change
        """

        def on_theme_change(name, data):
            self.apply_theme()

        theme.add_observer(on_theme_change)
