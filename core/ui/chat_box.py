from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import Color, RoundedRectangle
from kivy.clock import Clock


class ChatBox(BoxLayout):
    """
    PRODUCTION CHAT PANEL
    Matches your UI sketch:
    - Header
    - Chat bubbles
    - Proper alignment
    - Clean spacing
    """

    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", spacing=10, padding=10, **kwargs)

        self.controller = None

        # 🔥 HEADER (matches "my chat")
        self.header = Label(text="My Chat", size_hint_y=None, height=40, bold=True)

        # 🔥 MESSAGE AREA
        self.messages = BoxLayout(orientation="vertical", size_hint_y=None, spacing=8)
        self.messages.bind(minimum_height=self.messages.setter("height"))

        self.add_widget(self.header)
        self.add_widget(self.messages)

    # ─────────────────────────────
    # ADD MESSAGE
    # ─────────────────────────────
    def add_message(self, role, text):
        bubble = self.create_bubble(role, text)
        self.messages.add_widget(bubble)

        Clock.schedule_once(lambda dt: self._scroll_parent(), 0.05)

    def _scroll_parent(self):
        parent = self.parent
        if parent and hasattr(parent, "scroll_y"):
            parent.scroll_y = 0

    # ─────────────────────────────
    # CREATE CHAT BUBBLE
    # ─────────────────────────────
    def create_bubble(self, role, text):
        container = BoxLayout(size_hint_y=None, height=1, padding=(5, 2))
        container.bind(minimum_height=container.setter("height"))

        is_user = role == "YOU"

        bubble = Label(
            text=text,
            size_hint=(None, None),
            halign="left",
            valign="middle",
            padding=(12, 10),
            color=(1, 1, 1, 1),
        )

        max_width = self.width * 0.65
        bubble.text_size = (max_width, None)

        def update_size(instance, value):
            instance.size = (value[0] + 25, value[1] + 20)

        bubble.bind(texture_size=update_size)

        # 🔥 MODERN CHAT BUBBLE
        with bubble.canvas.before:
            if is_user:
                Color(0.2, 0.6, 1, 1)  # blue (user)
            else:
                Color(0.15, 0.15, 0.2, 1)  # dark (AI)

            bubble.bg = RoundedRectangle(radius=[14])

        def update_bg(*args):
            bubble.bg.pos = bubble.pos
            bubble.bg.size = bubble.size

        bubble.bind(pos=update_bg, size=update_bg)

        # 🔥 ALIGNMENT FIX
        if is_user:
            container.add_widget(Widget())
            container.add_widget(bubble)
        else:
            container.add_widget(bubble)
            container.add_widget(Widget())

        return container
