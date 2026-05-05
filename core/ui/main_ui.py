from kivy.config import Config

Config.set("graphics", "width", "1400")
Config.set("graphics", "height", "900")
Config.set("graphics", "resizable", True)
Config.set("graphics", "minimum_width", "1000")
Config.set("graphics", "minimum_height", "700")

from kivy.app import App
from kivy.core.window import Window
import traceback

from core.ui.theme_config import ThemeConfig


class MainUI(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.theme = ThemeConfig()
        self.ui_manager = None
        self.backend = None
        self.controller = None

    # ─────────────────────────────
    # BUILD APP
    # ─────────────────────────────
    def build(self):
        try:
            Window.clearcolor = (0.05, 0.05, 0.1, 1)

            # ─── BACKEND ───
            try:
                from core.ui.ai_backend import AIBackend

                self.backend = AIBackend()
                self.backend.start()  # 🔥 THIS WAS MISSING
                print("✅ AI Backend started")
            except Exception as e:
                print(f"⚠ Backend failed: {e}")
                self.backend = None

            # ─── CONTROLLER ───
            try:
                from core.ui.app import AppController

                self.controller = AppController()

                # inject backend into controller
                if self.backend:
                    self.controller.backend = self.backend

                self.controller.start()

                print("✅ Controller initialized")
            except Exception as e:
                print(f"⚠ Controller error: {e}")
                traceback.print_exc()

            # ─── UI MANAGER ───
            try:
                from core.ui.ui_manager import UIManager

                self.ui_manager = UIManager(controller=self.controller)

                print("✅ UI Manager ready")
            except Exception as e:
                print(f"❌ UI Manager failed: {e}")
                traceback.print_exc()
                raise

            # ─── APPLY THEME ───
            self.theme.apply_theme()

            return self.ui_manager

        except Exception as e:
            print(f"❌ UI Initialization failed: {e}")
            traceback.print_exc()

            from kivy.uix.label import Label

            return Label(text=f"UI Failed:\n{str(e)}")

    # ─────────────────────────────
    # CLEAN SHUTDOWN
    # ─────────────────────────────
    def on_stop(self):
        print("🛑 Shutting down UI...")

        try:
            if self.controller:
                self.controller.shutdown()
        except Exception as e:
            print(f"⚠ Controller shutdown error: {e}")

        try:
            if self.backend:
                self.backend.shutdown()
        except Exception as e:
            print(f"⚠ Backend shutdown error: {e}")

        try:
            self.theme.save_settings()
        except Exception as e:
            print(f"⚠ Theme save error: {e}")

        print("✅ Shutdown complete")


# ─────────────────────────────
# ENTRY POINT
# ─────────────────────────────
if __name__ == "__main__":
    MainUI().run()
