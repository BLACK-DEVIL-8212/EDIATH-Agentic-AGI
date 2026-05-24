from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.properties import BooleanProperty
import sys


def safe_print(*args, **kwargs):
    """Write to stdout using UTF-8 with replacement for unencodable chars.

    This avoids crashes on Windows consoles using cp1252 when printing emojis.
    """
    try:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        s = sep.join(str(a) for a in args) + end
        # write bytes to stdout buffer with utf-8 and replacement for errors
        try:
            sys.stdout.buffer.write(s.encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()
            return
        except Exception:
            pass
        # fallback to text write
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
    def __init__(self, text, sender="AI", **kwargs):
        super().__init__(**kwargs)

        self.text = text
        self.size_hint_y = None
        self.text_size = (400, None)
        self.padding = (10, 10)
        self.halign = "left"

        self.bind(texture_size=self.update_height)

        if sender == "YOU":
            self.color = (0.4, 0.8, 1, 1)
        else:
            self.color = (1, 1, 1, 1)

    def update_height(self, *args):
        self.height = self.texture_size[1] + 20


# ─────────────────────────────
# MAIN CHAT SCREEN
# ─────────────────────────────
class ChatScreen(Screen):

    preview_mode = BooleanProperty(False)

    def __init__(self, controller=None, preview_mode=False, **kwargs):
        super().__init__(**kwargs)

        self.controller = controller
        self.preview_mode = preview_mode

        self.build_ui()

        # Register response callback so AI replies reach the chat
        self.set_controller(controller)

    # 🔥 ADD THIS (CRITICAL FIX)
    def set_controller(self, controller):
        """Attach backend controller to ChatScreen safely."""

        self.controller = controller

        if not controller:
            safe_print("❌ ChatScreen: No controller provided")
            return

        # 🔥 Verify backend capability
        if not hasattr(controller, "send_user_message"):
            safe_print("❌ ChatScreen: Invalid controller (missing send_user_message)")
            return

        # 🔥 Bind response callback (CRITICAL)
        if hasattr(controller, "set_response_callback"):
            controller.set_response_callback(self.receive_ai_message)

        safe_print("✅ ChatScreen connected to backend")

    # ─────────────────────────────
    def build_ui(self):

        root = BoxLayout(orientation="vertical", spacing=10, padding=10)

        # CHAT DISPLAY
        self.scroll = ScrollView()

        self.chat_layout = BoxLayout(
            orientation="vertical", size_hint_y=None, spacing=10
        )
        self.chat_layout.bind(minimum_height=self.chat_layout.setter("height"))

        self.scroll.add_widget(self.chat_layout)

        # INPUT BAR
        input_bar = BoxLayout(size_hint_y=0.15, spacing=10)

        self.input_box = TextInput(
            multiline=False, hint_text="Type your message...", size_hint_x=0.8
        )
        self.input_box.bind(on_text_validate=self.send_message)

        self.send_btn = Button(text="Send", size_hint_x=0.2)
        self.send_btn.bind(on_press=self.send_message)

        input_bar.add_widget(self.input_box)
        input_bar.add_widget(self.send_btn)

        root.add_widget(self.scroll)
        root.add_widget(input_bar)

        self.add_widget(root)

        Clock.schedule_once(
            lambda dt: self.add_message("AI", "👋 Hello! I'm EDIATH."), 0.5
        )

    # ─────────────────────────────
    def add_message(self, sender, text):
        bubble = ChatBubble(text=text, sender=sender)

        container = BoxLayout(size_hint_y=None)
        container.bind(minimum_height=container.setter("height"))

        if sender == "YOU":
            container.add_widget(Label(size_hint_x=0.3))
            container.add_widget(bubble)
        else:
            container.add_widget(bubble)
            container.add_widget(Label(size_hint_x=0.3))

        self.chat_layout.add_widget(container)

        Clock.schedule_once(lambda dt: self.scroll_to_bottom(), 0.1)

    def scroll_to_bottom(self):
        self.scroll.scroll_y = 0

    # ─────────────────────────────
    # SEND MESSAGE (FIXED)
    # ─────────────────────────────
    def send_message(self, *args):
        text = self.input_box.text.strip()

        # Empty check
        if not text:
            return

        # Clear input immediately and show user message
        self.input_box.text = ""
        self.add_message("YOU", text)

        # Preview mode: do not call backend
        if self.preview_mode:
            Clock.schedule_once(
                lambda dt: self.add_message("AI", "Preview response..."), 1
            )
            return

        # Backend connection check
        if not self.controller:
            safe_print("❌ No controller connected")
            self.add_message("AI", "⚠️ Backend not connected.")
            return

        if not hasattr(self.controller, "send_user_message"):
            safe_print("❌ Backend missing send_user_message")
            self.add_message("AI", "⚠️ Backend method missing.")
            return

        # Send to backend safely (off main thread via controller)
        try:
            safe_print("➡ Sending to backend:", text)

            # disable input while processing
            self.input_box.disabled = True

            # submit to controller (non-blocking)
            try:
                self.controller.send_user_message(text)
            except TypeError:
                # older controllers may expect different signature
                try:
                    self.controller.send_user_message(text, {})
                except Exception:
                    raise

            # re-enable input after short delay as a failsafe
            Clock.schedule_once(lambda dt: self._enable_input(), 2)

        except Exception as e:
            safe_print("❌ Send error:", e)
            self.add_message("AI", f"❌ Error: {str(e)}")
            self._enable_input()

    # 🔥 HELPER FUNCTION (ADD THIS ALSO)
    def _enable_input(self):
        self.input_box.disabled = False

    # ─────────────────────────────
    # RECEIVE FROM BACKEND
    # ─────────────────────────────
    def receive_ai_message(self, text):
        safe_print("⬅ Received from backend:", text)
        if text:
            Clock.schedule_once(lambda dt: self.add_message("AI", text), 0)
