import json
import os
from kivy.core.window import Window


class ThemeConfig:
    """
    Central theme manager (reactive + persistent)
    """

    _instance = None  # singleton safety

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True

        # Safe path
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.settings_file = os.path.join(self.base_dir, "theme_settings.json")

        self.current_theme = "dark"
        self._observers = []

        # ─────────────────────────────
        # THEMES
        # ─────────────────────────────
        self.themes = {
            "dark": {
                "bg_color": [0.05, 0.05, 0.1, 1],
                "primary_color": [0.3, 0.6, 1, 1],
                "secondary_color": [0.2, 0.3, 0.5, 1],
                "text_color": [0.9, 0.9, 0.9, 1],
                "subtext_color": [0.7, 0.7, 0.7, 1],
                "accent_color": [0.5, 0.8, 0.5, 1],
                "card_bg": [0.12, 0.12, 0.18, 1],
                "input_bg": [0.1, 0.1, 0.15, 1],
                "error_color": [0.8, 0.3, 0.3, 1],
                "warning_color": [0.9, 0.7, 0.2, 1],
                "success_color": [0.3, 0.8, 0.3, 1],
            },
            "light": {
                "bg_color": [0.95, 0.95, 0.98, 1],
                "primary_color": [0.2, 0.5, 0.9, 1],
                "secondary_color": [0.8, 0.85, 0.95, 1],
                "text_color": [0.1, 0.1, 0.15, 1],
                "subtext_color": [0.4, 0.4, 0.4, 1],
                "accent_color": [0.3, 0.7, 0.3, 1],
                "card_bg": [1, 1, 1, 1],
                "input_bg": [0.9, 0.9, 0.95, 1],
                "error_color": [0.8, 0.3, 0.3, 1],
                "warning_color": [0.9, 0.6, 0.1, 1],
                "success_color": [0.2, 0.7, 0.2, 1],
            },
        }

        self.load_settings()
        self.apply_theme()

    # ─────────────────────────────
    # APPLY THEME
    # ─────────────────────────────
    def apply_theme(self):
        try:
            theme = self.themes.get(self.current_theme, self.themes["dark"])

            # Apply window background
            Window.clearcolor = theme["bg_color"]

            # Notify observers safely
            for observer in self._observers[:]:
                try:
                    observer(self.current_theme, theme)
                except Exception as e:
                    print(f"[Theme Observer Error] {e}")

        except Exception as e:
            print(f"[Theme ERROR] apply_theme: {e}")

    # ─────────────────────────────
    # GET COLOR (SAFE)
    # ─────────────────────────────
    def get_color(self, name):
        theme = self.themes.get(self.current_theme, {})
        return theme.get(name, [0.5, 0.5, 0.5, 1])

    # ─────────────────────────────
    # SETTINGS
    # ─────────────────────────────
    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r") as f:
                    data = json.load(f)
                    self.current_theme = data.get("theme", "dark")
        except Exception as e:
            print(f"[Theme ERROR] load_settings: {e}")

    def save_settings(self):
        try:
            with open(self.settings_file, "w") as f:
                json.dump({"theme": self.current_theme}, f)
        except Exception as e:
            print(f"[Theme ERROR] save_settings: {e}")

    # ─────────────────────────────
    # TOGGLE
    # ─────────────────────────────
    def toggle_theme(self):
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.apply_theme()
        self.save_settings()

    # ─────────────────────────────
    # OBSERVER SYSTEM
    # ─────────────────────────────
    def add_observer(self, callback):
        if callable(callback) and callback not in self._observers:
            self._observers.append(callback)

    def remove_observer(self, callback):
        if callback in self._observers:
            self._observers.remove(callback)

    # ─────────────────────────────
    # HELPERS
    # ─────────────────────────────
    def get_current_theme(self):
        return self.current_theme

    def is_dark(self):
        return self.current_theme == "dark"

    def is_light(self):
        return self.current_theme == "light"


# ─────────────────────────────
# GLOBAL INSTANCE
# ─────────────────────────────
theme = ThemeConfig()
