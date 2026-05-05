from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, RoundedRectangle, Line
from kivy.properties import ListProperty
from kivy.animation import Animation


# ─────────────────────────────
# ROUNDED BUTTON (MODERN UI)
# ─────────────────────────────
class RoundedButton(Button):
    bg_color = ListProperty([0.25, 0.6, 1, 1])
    press_color = ListProperty([0.18, 0.5, 0.9, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)

        with self.canvas.before:
            self.bg_instr = Color(*self.bg_color)
            self.bg_rect = RoundedRectangle(radius=[14])

        self.bind(pos=self.update_canvas, size=self.update_canvas)

    def update_canvas(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size

    def on_press(self):
        Animation(rgba=self.press_color, duration=0.1).start(self.bg_instr)

    def on_release(self):
        Animation(rgba=self.bg_color, duration=0.1).start(self.bg_instr)


# ─────────────────────────────
# HEADER LABEL (CLEAN)
# ─────────────────────────────
class GradientLabel(Label):
    bg_color = ListProperty([0.12, 0.12, 0.18, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with self.canvas.before:
            self.bg_instr = Color(*self.bg_color)
            self.bg_rect = RoundedRectangle(radius=[12])

        self.bind(pos=self.update_canvas, size=self.update_canvas)

    def update_canvas(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size


# ─────────────────────────────
# INPUT FIELD (BOTTOM BAR STYLE)
# ─────────────────────────────
class AnimatedTextInput(TextInput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.background_normal = ""
        self.background_active = ""

        self.background_color = (0, 0, 0, 0)
        self.foreground_color = (1, 1, 1, 1)
        self.cursor_color = (0.3, 0.6, 1, 1)

        with self.canvas.before:
            self.bg_instr = Color(0.1, 0.1, 0.15, 1)
            self.bg_rect = RoundedRectangle(radius=[12])

            # 🔥 bottom glow line (like modern UI)
            self.line_instr = Color(0.3, 0.6, 1, 0.6)
            self.line = Line(width=1.5)

        self.bind(pos=self.update_canvas, size=self.update_canvas)
        self.bind(focus=self.on_focus)

    def update_canvas(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size

        self.line.points = [self.x, self.y, self.right, self.y]

    def on_focus(self, instance, value):
        if value:
            Animation(rgba=(0.2, 0.2, 0.25, 1), duration=0.15).start(self.bg_instr)
        else:
            Animation(rgba=(0.1, 0.1, 0.15, 1), duration=0.15).start(self.bg_instr)


# ─────────────────────────────
# CARD (MAIN PANEL CONTAINER)
# ─────────────────────────────
class Card(BoxLayout):
    bg_color = ListProperty([0.1, 0.1, 0.15, 1])
    border_color = ListProperty([0.25, 0.25, 0.35, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.orientation = "vertical"
        self.padding = 12
        self.spacing = 8

        with self.canvas.before:
            self.bg_instr = Color(*self.bg_color)
            self.bg_rect = RoundedRectangle(radius=[14])

            self.border_instr = Color(*self.border_color)
            self.border = Line(width=1.2, rounded_rectangle=[0, 0, 0, 0, 14])

        self.bind(pos=self.update_canvas, size=self.update_canvas)

    def update_canvas(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size

        self.border.rounded_rectangle = (self.x, self.y, self.width, self.height, 14)
