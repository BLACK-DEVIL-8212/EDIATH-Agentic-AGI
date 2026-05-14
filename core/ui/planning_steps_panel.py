from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.properties import ListProperty


class PlanningStepsPanel(BoxLayout):
    """
    Dynamic planning steps panel
    """

    steps = ListProperty([])

    def __init__(self, controller=None, **kwargs):
        super().__init__(orientation="vertical", **kwargs)

        self.controller = controller

        self._build_ui()

        # default steps
        self.set_steps(
            [
                "Initialize system",
                "Load configurations",
                "Establish connections",
                "Ready for input",
            ]
        )

    # ─────────────────────────────
    # UI
    # ─────────────────────────────
    def _build_ui(self):
        # Title
        self.title = Label(text="[b]Planning Steps[/b]", markup=True, size_hint_y=0.12)
        self.add_widget(self.title)

        # Scroll container
        self.scroll = ScrollView(size_hint=(1, 0.88))

        self.steps_layout = BoxLayout(
            orientation="vertical", size_hint_y=None, spacing=6, padding=5
        )
        self.steps_layout.bind(minimum_height=self.steps_layout.setter("height"))

        self.scroll.add_widget(self.steps_layout)
        self.add_widget(self.scroll)

    # ─────────────────────────────
    # SET FULL LIST
    # ─────────────────────────────
    def set_steps(self, steps_list):
        self.steps = [{"text": s, "done": False} for s in steps_list]
        self._refresh_ui()

    # ─────────────────────────────
    # ADD STEP
    # ─────────────────────────────
    def add_step(self, text):
        self.steps.append({"text": text, "done": False})
        self._add_step_widget(text, False)

    # ─────────────────────────────
    # COMPLETE STEP
    # ─────────────────────────────
    def complete_step(self, index):
        if 0 <= index < len(self.steps):
            self.steps[index]["done"] = True
            self._refresh_ui()

    # ─────────────────────────────
    # UI RENDER
    # ─────────────────────────────
    def _refresh_ui(self):
        self.steps_layout.clear_widgets()

        for step in self.steps:
            self._add_step_widget(step["text"], step["done"])

    def _add_step_widget(self, text, done):
        icon = "✓" if done else "○"

        color = (0.5, 0.8, 0.5, 1) if done else (0.6, 0.6, 0.7, 1)

        label = Label(
            text=f"{icon} {text}",
            size_hint_y=None,
            halign="left",
            valign="middle",
            color=color,
        )

        # dynamic sizing
        label.bind(texture_size=lambda inst, val: setattr(inst, "height", val[1] + 10))

        label.text_size = (self.width * 0.9, None)

        self.steps_layout.add_widget(label)

    # ─────────────────────────────
    # OPTIONAL: CONNECT TO BACKEND
    # ─────────────────────────────
    def connect_backend(self, controller):
        """
        Example: react to system events
        """
        self.controller = controller
        backend = getattr(controller, "backend", controller)
        if not hasattr(backend, "set_status_callback"):
            return

        def on_status(text):
            text = text.lower()

            if "ready" in text:
                self.complete_step(3)

            elif "loading" in text:
                self.complete_step(1)

            elif "connection" in text:
                self.complete_step(2)

        backend.set_status_callback(on_status)
