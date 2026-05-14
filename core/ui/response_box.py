from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.core.clipboard import Clipboard


class ResponseBox(BoxLayout):
    """
    Displays AI responses (read-only, scrollable, dynamic)
    """

    def __init__(self, controller=None, **kwargs):
        super().__init__(orientation="vertical", spacing=10, **kwargs)

        self.controller = controller
        self._build_ui()

    # ─────────────────────────────
    # UI SETUP
    # ─────────────────────────────
    def _build_ui(self):
        # Title
        self.title = Label(text="[b]AI Response[/b]", markup=True, size_hint_y=0.12)
        self.add_widget(self.title)

        # Scrollable response area
        self.scroll = ScrollView(size_hint=(1, 0.75))

        self.response_label = Label(
            text="Ready to assist...", size_hint_y=None, halign="left", valign="top"
        )

        # dynamic height
        self.response_label.bind(
            texture_size=lambda inst, val: setattr(inst, "height", val[1] + 20)
        )

        self.response_label.text_size = (self.width * 0.95, None)

        self.scroll.add_widget(self.response_label)
        self.add_widget(self.scroll)

        # Buttons
        btn_layout = BoxLayout(size_hint_y=0.13, spacing=10)

        self.copy_btn = Button(text="Copy")
        self.clear_btn = Button(text="Clear")

        self.copy_btn.bind(on_press=self.copy_response)
        self.clear_btn.bind(on_press=self.clear_response)

        btn_layout.add_widget(self.copy_btn)
        btn_layout.add_widget(self.clear_btn)

        self.add_widget(btn_layout)

    # ─────────────────────────────
    # SET RESPONSE
    # ─────────────────────────────
    def set_response(self, text):
        self.response_label.text = text
        Clock.schedule_once(lambda dt: self.scroll_to_bottom(), 0.05)

    # ─────────────────────────────
    # STREAMING SUPPORT (OPTIONAL)
    # ─────────────────────────────
    def append_response(self, chunk):
        self.response_label.text += chunk
        Clock.schedule_once(lambda dt: self.scroll_to_bottom(), 0.01)

    # ─────────────────────────────
    # AUTO SCROLL
    # ─────────────────────────────
    def scroll_to_bottom(self):
        self.scroll.scroll_y = 0

    # ─────────────────────────────
    # COPY
    # ─────────────────────────────
    def copy_response(self, instance):
        Clipboard.copy(self.response_label.text)
        self.copy_btn.text = "Copied!"
        Clock.schedule_once(lambda dt: self._reset_copy_text(), 1.5)

    def _reset_copy_text(self):
        self.copy_btn.text = "Copy"

    # ─────────────────────────────
    # CLEAR
    # ─────────────────────────────
    def clear_response(self, instance):
        self.response_label.text = ""

    # ─────────────────────────────
    # CONNECT TO BACKEND
    # ─────────────────────────────
    def connect_backend(self, controller):
        self.controller = controller
        backend = getattr(controller, "backend", controller)
        if not hasattr(backend, "set_response_callback"):
            return

        def on_response(text):
            self.set_response(text)

        backend.set_response_callback(on_response)
