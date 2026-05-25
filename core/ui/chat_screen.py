"""
EDIATH Chat Screen - Production Quality UI
Premium chat interface with:
- Animated chat bubbles with sender-specific styling
- Typing indicator with bouncing dots
- Auto-scroll to bottom
- Message timestamps
- Smooth entrance animations
"""

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.properties import BooleanProperty, StringProperty
from kivy.graphics import Color, RoundedRectangle, Ellipse, Rectangle
from kivy.animation import Animation
from kivy.metrics import dp
import time
import sys


def safe_print(*args, **kwargs):
    """Write to stdout using UTF-8 with replacement for unencodable chars."""
    try:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        s = sep.join(str(a) for a in args) + end
        try:
            sys.stdout.buffer.write(s.encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()
            return
        except Exception:
            pass
        try:
            sys.stdout.write(s)
            sys.stdout.flush()
        except Exception:
            pass
    except Exception:
        pass


# ─────────────────────────────
# CHAT BUBBLE
# ─────────────────────────────
class ChatBubble(Label):
    """Premium styled chat bubble with animation support."""

    def __init__(self, text, sender="AI", timestamp=None, **kwargs):
        super().__init__(**kwargs)

        self.text = text
        self.sender = sender
        self.timestamp = timestamp or time.time()
        self.size_hint_y = None
        self.halign = "left"
        self.valign = "middle"
        self.padding = (dp(14), dp(10))
        self.line_height = 1.4
        self.markup = False

        # Bubble styling
        if sender == "YOU":
            self.color = (0.95, 0.95, 1.0, 1.0)  # Near-white text
            self.bg_color = (0.12, 0.35, 0.75, 1.0)  # Blue bubble
        elif sender == "AI":
            self.color = (0.92, 0.92, 0.95, 1.0)  # Light gray text
            self.bg_color = (0.1, 0.1, 0.18, 1.0)  # Dark glass bubble
        else:
            self.color = (0.6, 0.6, 0.7, 1.0)
            self.bg_color = (0.08, 0.08, 0.12, 0.8)

        # Setup bubble background
        self._setup_bubble()

        self.bind(texture_size=self.update_height)

    def _setup_bubble(self):
        """Draw bubble background."""
        with self.canvas:
            Color(*self.bg_color)
            self._bubble_bg = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[dp(16), dp(16), dp(16), dp(4)],
            )
        self.bind(pos=self._update_bubble, size=self._update_bubble)

    def _update_bubble(self, *args):
        self._bubble_bg.pos = self.pos
        self._bubble_bg.size = self.size

    def update_height(self, *args):
        self.height = max(self.texture_size[1] + dp(20), dp(40))
        self.text_size = (self.width - dp(28), None)

    def on_touch_down(self, touch):
        """Allow touch passthrough to parent."""
        return False


# ─────────────────────────────
# TYPING DOTS WIDGET
# ─────────────────────────────
class TypingDots(BoxLayout):
    """Animated 3-dot typing indicator for chat."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.size_hint = (None, None)
        self.size = (dp(60), dp(30))
        self.spacing = dp(4)
        self._dots = []
        self._phase = 0.0
        self._events = []

        for _ in range(3):
            dot = Widget(size_hint=(None, None), size=(dp(8), dp(8)))
            self._dots.append(dot)
            self.add_widget(dot)

        self.bind(pos=self._redraw, size=self._redraw)
        self._events.append(Clock.schedule_interval(self._animate, 1 / 30))

    def _animate(self, dt):
        self._phase += 0.12
        for i, dot in enumerate(self._dots):
            alpha = max(0.15, 0.5 + math.sin(self._phase + i * 0.9) * 0.5)
            dot.canvas.clear()
            with dot.canvas:
                Color(0.15, 0.65, 1.0, alpha)
                cx = dot.center_x
                cy = dot.center_y
                r = min(dot.width, dot.height) / 2
                if r > 0:
                    Ellipse(pos=(cx - r, cy - r), size=(r * 2, r * 2))

    def _redraw(self, *args):
        for dot in self._dots:
            dot.pos = (dot.pos[0], dot.pos[1])

    def stop(self):
        for e in self._events:
            e.cancel()


import math


# ─────────────────────────────
# CHAT CONTAINER ITEM
# ─────────────────────────────
class ChatItem(BoxLayout):
    """Single chat message container with alignment."""

    def __init__(self, sender="AI", **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.size_hint_y = None
        self.padding = (dp(8), dp(4))
        self.spacing = dp(6)

        # Build bubble + spacer
        self._spacer = Label(size_hint_x=None, width=dp(0))
        self._bubble_ref = None

        if sender == "YOU":
            self.add_widget(self._spacer)
        elif sender == "AI":
            self.add_widget(self._spacer)
        else:
            self.add_widget(Widget())

        self.bind(minimum_height=self.setter("height"))

    def set_bubble(self, bubble):
        """Replace the spacer with an actual bubble after animation."""
        if self._bubble_ref:
            self.remove_widget(self._bubble_ref)
        self._bubble_ref = bubble
        self.add_widget(bubble)


# ─────────────────────────────
# MAIN CHAT SCREEN
# ─────────────────────────────
class ChatScreen(Screen):

    preview_mode = BooleanProperty(False)

    def __init__(self, controller=None, preview_mode=False, **kwargs):
        super().__init__(**kwargs)

        self.controller = controller
        self.preview_mode = preview_mode
        self._typing_widget = None

        self.build_ui()

        # Register response callback so AI replies reach the chat
        self.set_controller(controller)

    # ─────────────────────────────
    # SET CONTROLLER
    # ─────────────────────────────
    def set_controller(self, controller):
        """Attach backend controller to ChatScreen safely."""
        self.controller = controller

        if not controller:
            safe_print("ChatScreen: No controller provided")
            return

        if not hasattr(controller, "send_user_message"):
            safe_print("ChatScreen: Invalid controller (missing send_user_message)")
            return

        if hasattr(controller, "set_response_callback"):
            controller.set_response_callback(self.receive_ai_message)

        safe_print("ChatScreen connected to backend")

    # ─────────────────────────────
    # BUILD UI
    # ─────────────────────────────
    def build_ui(self):
        root = BoxLayout(orientation="vertical", spacing=dp(6), padding=(dp(8), dp(6)))

        # Chat display area
        self.scroll = ScrollView()
        self.scroll.do_scroll_x = False
        self.scroll.scroll_y = 0

        self.chat_layout = BoxLayout(
            orientation="vertical",
            size_hint_y=None,
            spacing=dp(8),
            padding=(dp(4), dp(8)),
        )
        self.chat_layout.bind(
            minimum_height=self.chat_layout.setter("height")
        )

        self.scroll.add_widget(self.chat_layout)
        root.add_widget(self.scroll)

        # Input bar
        input_bar = BoxLayout(size_hint_y=None, size=(0, dp(52)), spacing=dp(8))

        self.input_box = TextInput(
            multiline=False,
            hint_text="Type your message...",
            size_hint_x=0.78,
            padding=(dp(14), dp(12)),
            font_size="16sp",
            background_color=(0.08, 0.08, 0.14, 1),
            foreground_color=(0.92, 0.92, 0.95, 1),
            cursor_color=(0.2, 0.75, 1.0, 1),
            hint_text_color=(0.4, 0.4, 0.5, 0.8),
            write_tab=False,
        )
        self.input_box.bind(on_text_validate=self.send_message)

        # Send button with glass style
        self.send_btn = Button(
            text="Send",
            size_hint_x=0.22,
            font_size="15sp",
            bold=True,
            background_color=(0.1, 0.35, 0.75, 1.0),
            color=(1, 1, 1, 1),
        )
        self.send_btn.bind(on_press=self.send_message)

        input_bar.add_widget(self.input_box)
        input_bar.add_widget(self.send_btn)

        root.add_widget(input_bar)

        self.add_widget(root)

        # Welcome message
        Clock.schedule_once(
            lambda dt: self.add_message(
                "AI",
                "Hello! I'm EDIATH, your intelligent AI assistant. How can I help you today?"
            ),
            0.5,
        )

    # ─────────────────────────────
    # ADD MESSAGE
    # ─────────────────────────────
    def add_message(self, sender, text):
        """Add a chat bubble with entrance animation."""
        bubble = ChatBubble(text=text, sender=sender)

        container = BoxLayout(size_hint_y=None, padding=(dp(8), dp(4)))
        container.bind(minimum_height=container.setter("height"))

        if sender == "YOU":
            # Right-aligned: spacer on left
            spacer = Label(size_hint_x=0.15)
            container.add_widget(spacer)
            container.add_widget(bubble)
        elif sender == "AI":
            # Left-aligned: bubble first, spacer after
            container.add_widget(bubble)
            spacer = Label(size_hint_x=0.15)
            container.add_widget(spacer)
        else:
            # System message: centered
            c_spacer = Label(size_hint_x=0.1)
            container.add_widget(c_spacer)
            container.add_widget(bubble)
            container.add_widget(Label(size_hint_x=0.1))

        self.chat_layout.add_widget(container)

        # Scroll to bottom
        Clock.schedule_once(lambda dt: self._scroll_bottom(), 0.05)

        return container

    def _scroll_bottom(self):
        self.scroll.scroll_y = 0

    # ─────────────────────────────
    # SEND MESSAGE
    # ─────────────────────────────
    def send_message(self, *args):
        text = self.input_box.text.strip()

        if not text:
            return

        # Clear input immediately and show user message
        self.input_box.text = ""
        self.add_message("YOU", text)

        if self.preview_mode:
            Clock.schedule_once(
                lambda dt: self.add_message("AI", "Preview response..."), 1
            )
            return

        if not self.controller:
            self.add_message("AI", "Backend not connected.")
            return

        if not hasattr(self.controller, "send_user_message"):
            self.add_message("AI", "Backend method missing.")
            return

        try:
            safe_print("Sending to backend:", text)
            self.input_box.disabled = True
            self._show_typing()

            try:
                self.controller.send_user_message(text)
            except TypeError:
                try:
                    self.controller.send_user_message(text, {})
                except Exception:
                    raise

            Clock.schedule_once(lambda dt: self._enable_input(), 3)

        except Exception as e:
            safe_print("Send error:", e)
            self.add_message("AI", f"Error: {str(e)}")
            self._enable_input()

    def _enable_input(self):
        self.input_box.disabled = False
        self._hide_typing()

    def _show_typing(self):
        """Show typing indicator."""
        if self._typing_widget:
            self._typing_widget.opacity = 1
            return

        typing_container = BoxLayout(
            size_hint_y=None,
            size=(0, dp(36)),
            padding=(dp(8), dp(4)),
        )
        spacer = Label(size_hint_x=0.15)
        typing_container.add_widget(spacer)

        self._typing_widget = BoxLayout(
            size_hint_x=None,
            width=dp(70),
            padding=(dp(4), dp(4)),
        )

        dots = TypingDots()
        self._typing_widget.add_widget(dots)
        typing_container.add_widget(self._typing_widget)
        typing_container.add_widget(Label(size_hint_x=0.15))

        self.chat_layout.add_widget(typing_container)
        self._typing_widget = typing_container
        self._typing_widget.opacity = 1

        Clock.schedule_once(lambda dt: self._scroll_bottom(), 0.02)

    def _hide_typing(self):
        """Hide typing indicator."""
        if self._typing_widget:
            self._typing_widget.opacity = 0
            Clock.schedule_once(
                lambda dt: self._remove_typing(), 0.3
            )

    def _remove_typing(self):
        if self._typing_widget and self._typing_widget.parent:
            self.chat_layout.remove_widget(self._typing_widget)
        self._typing_widget = None

    # ─────────────────────────────
    # RECEIVE FROM BACKEND
    # ─────────────────────────────
    def receive_ai_message(self, text):
        """Handle AI response received from backend."""
        safe_print("Received from backend:", text)
        self._hide_typing()
        if text:
            Clock.schedule_once(lambda dt: self.add_message("AI", text), 0)
        else:
            Clock.schedule_once(
                lambda dt: self.add_message(
                    "AI", "I received your message but have no response."
                ),
                0,
            )