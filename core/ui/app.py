from core.ui.ai_backend import AIBackend


class AppController:
    """
    Connects UI components with AIBackend
    """

    def __init__(self):
        self.backend = AIBackend()

        # UI references (set later)
        self.chat_box = None
        self.status_bar = None
        self.task_display = None

    # ─────────────────────────────────────────────
    # INIT SYSTEM
    # ─────────────────────────────────────────────
    def start(self):
        """Start backend + bind callbacks"""
        self.backend.start()

        self.backend.set_response_callback(self.on_ai_response)
        self.backend.set_status_callback(self.on_status_update)

    # ─────────────────────────────────────────────
    # UI → BACKEND
    # ─────────────────────────────────────────────
    def send_user_message(self, text: str):
        """Called from chat input"""
        if not text.strip():
            return

        # show user message in UI
        if self.chat_box:
            self.chat_box.add_message("YOU", text)

        # send to backend
<<<<<<< HEAD
        # AIBackend uses send_user_message as the stable entrypoint
        if hasattr(self.backend, "send_user_message"):
            self.backend.send_user_message(text)
        else:
            # fallback for older backend implementations
            self.backend.send_message(text)
=======
        self.backend.send_message(text)
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7

    # ─────────────────────────────────────────────
    # BACKEND → UI
    # ─────────────────────────────────────────────
    def on_ai_response(self, text: str):
        """Receive AI response"""
        if self.chat_box:
            self.chat_box.add_message("AI", text)

    def on_status_update(self, text: str):
        """Update status bar"""
        if self.status_bar:
            self.status_bar.update_status(text)

        if self.task_display:
            self.task_display.update_task(text)

    # ─────────────────────────────────────────────
    # DASHBOARD DATA
    # ─────────────────────────────────────────────
    def get_system_stats(self):
        return self.backend.get_status()

    # ─────────────────────────────────────────────
    # BIND UI COMPONENTS
    # ─────────────────────────────────────────────
    def bind_chat_screen(self, chat_screen):
        self.chat_screen = chat_screen
        self.chat_box = (
            chat_screen.ids.chat_box if hasattr(chat_screen, "ids") else None
        )

    def bind_status_bar(self, status_bar):
        self.status_bar = status_bar

    def bind_task_display(self, task_display):
        self.task_display = task_display

    # ─────────────────────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────────────────────
    def shutdown(self):
        self.backend.shutdown()
