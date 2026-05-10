from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar


class TaskDisplay(BoxLayout):
    """
    Displays current system task + progress
    """

    def __init__(self, controller=None, **kwargs):
        super().__init__(orientation="vertical", **kwargs)

        self.controller = controller

        self.padding = [10, 5]
        self.spacing = 5
        self.size_hint_y = 0.1

        self.current_task = None

        self._build_ui()

    # ─────────────────────────────
    # UI SETUP
    # ─────────────────────────────
    def _build_ui(self):
        # Title
        self.title_label = Label(
            text="[b]Current Task[/b]", markup=True, size_hint_y=0.25
        )
        self.add_widget(self.title_label)

        # Task name
        self.task_label = Label(text="No active task", size_hint_y=0.25)
        self.add_widget(self.task_label)

        # Progress bar
        self.progress_bar = ProgressBar(max=100, value=0, size_hint_y=0.25)
        self.add_widget(self.progress_bar)

        # Status
        self.status_label = Label(text="Status: Idle", size_hint_y=0.25)
        self.add_widget(self.status_label)

    # ─────────────────────────────
    # MAIN UPDATE METHOD
    # ─────────────────────────────
    def update_task(self, task_name=None, progress=None, status=None):
        if task_name is not None:
            self.current_task = task_name
            self.task_label.text = task_name

        if progress is not None:
            self.progress_bar.value = max(0, min(100, progress))

        if status:
            self._set_status(status)

    # ─────────────────────────────
    # STATUS HANDLING
    # ─────────────────────────────
    def _set_status(self, status):
        status_lower = status.lower()

        if status_lower == "complete":
            self.status_label.text = "✅ Complete"
            self.status_label.color = (0.5, 0.8, 0.5, 1)
            self.progress_bar.value = 100

        elif status_lower == "processing":
            self.status_label.text = "⚙ Processing..."
            self.status_label.color = (0.3, 0.6, 1, 1)

        elif status_lower == "error":
            self.status_label.text = "❌ Error"
            self.status_label.color = (1, 0.4, 0.4, 1)

        else:
            self.status_label.text = f"Status: {status}"
            self.status_label.color = (0.7, 0.7, 0.7, 1)

    # ─────────────────────────────
    # SIMPLE API (COMPATIBILITY)
    # ─────────────────────────────
    def set_current_task(self, task_name):
        self.update_task(task_name=task_name)

    # ─────────────────────────────
    # CONNECT TO BACKEND
    # ─────────────────────────────
    def connect_backend(self, controller):
        self.controller = controller

        def on_status(text):
            text_lower = text.lower()

            if "thinking" in text_lower:
                self.update_task(
                    task_name="Processing request", progress=50, status="processing"
                )

            elif "done" in text_lower or "ready" in text_lower:
                self.update_task(progress=100, status="complete")

            elif "error" in text_lower:
                self.update_task(status="error")

        if controller and controller.backend:
            controller.backend.set_status_callback(on_status)
