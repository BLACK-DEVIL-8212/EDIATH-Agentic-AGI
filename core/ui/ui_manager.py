from kivy.uix.screenmanager import ScreenManager
from kivy.clock import Clock


class UIManager(ScreenManager):
    """
    Central UI controller (handles screens + navigation)
    """

    def __init__(self, controller=None, **kwargs):
        super().__init__(**kwargs)

        self.controller = controller

        # delay setup until UI is ready
        Clock.schedule_once(self._setup_screens, 0)

    # ─────────────────────────────
    # SETUP SCREENS
    # ─────────────────────────────
    def _setup_screens(self, dt):
        try:
            # 🔥 FIXED IMPORTS (IMPORTANT)
            from core.ui.dashboard_screen import DashboardScreen
            from core.ui.chat_screen import ChatScreen

            # ─── DASHBOARD ───
            self.dashboard_screen = DashboardScreen(
                name="dashboard", controller=self.controller
            )
            self.add_widget(self.dashboard_screen)

            # ─── CHAT ───
            self.chat_screen = ChatScreen(name="chat", controller=self.controller)
            self.add_widget(self.chat_screen)

            # default screen
            self.current = "dashboard"

            print("✅ Screens initialized")

        except Exception:
            import traceback

            print("❌ UIManager setup error:")
            traceback.print_exc()

    # ─────────────────────────────
    # NAVIGATION API
    # ─────────────────────────────
    def go_to(self, screen_name):
        if screen_name in self.screen_names:
            self.current = screen_name
        else:
            print(f"⚠ Screen '{screen_name}' not found")

    # ─────────────────────────────
    # SAFE SHORTCUTS
    # ─────────────────────────────
    def switch_to_chat(self):
        self.go_to("chat")

    def switch_to_dashboard(self):
        self.go_to("dashboard")

    # ─────────────────────────────
    # DEBUG HELP
    # ─────────────────────────────
    def list_screens(self):
        print("Available screens:", self.screen_names)
