from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
from kivy.properties import StringProperty
from kivy.uix.filechooser import FileChooserListView


class Panel(BoxLayout):
    title = StringProperty("")

    def __init__(self, title="", **kwargs):
        super().__init__(orientation="vertical", spacing=6, padding=6, **kwargs)
        self.title = title
        self.header = Label(text=title, size_hint_y=None, height=28)
        self.add_widget(self.header)


class ChatBubble(Label):
    def __init__(self, text, who="ai", **kwargs):
        super().__init__(
            text=text, size_hint_y=None, halign="left", valign="middle", **kwargs
        )
        self.who = who
        self.text_size = (self.width, None)
        self.bind(width=self._update_text_size)

    def _update_text_size(self, *a):
        self.text_size = (self.width - 10, None)


class LogsView(ScrollView):
    def __init__(self, **kwargs):
        content = GridLayout(cols=1, size_hint_y=None, spacing=4, padding=4)
        content.bind(minimum_height=content.setter("height"))
        super().__init__(do_scroll_x=False, **kwargs)
        self.content = content
        self.add_widget(content)

    def add_line(self, text: str):
        lbl = Label(
            text=text, size_hint_y=None, height=22, halign="left", valign="middle"
        )
        lbl.text_size = (self.width - 20, None)
        self.content.add_widget(lbl)


class TaskRow(BoxLayout):
    def __init__(self, task_id: str, title: str, status: str = "idle", **kwargs):
        super().__init__(
            orientation="horizontal", size_hint_y=None, height=28, **kwargs
        )
        self.add_widget(Label(text=task_id, size_hint_x=0.2))
        self.add_widget(Label(text=title, size_hint_x=0.6))
        self.status_label = Label(text=status, size_hint_x=0.2)
        self.add_widget(self.status_label)

    def update_status(self, status: str):
        self.status_label.text = status


class StatusItem(BoxLayout):
    def __init__(self, label: str, value: str = "", **kwargs):
        super().__init__(
            orientation="horizontal", size_hint_y=None, height=24, **kwargs
        )
        self.add_widget(Label(text=label, size_hint_x=0.6))
        self.value_label = Label(text=value, size_hint_x=0.4)
        self.add_widget(self.value_label)

    def set(self, v: str):
        self.value_label.text = v


class CommandPanel(BoxLayout):
    def __init__(self, send_callback, **kwargs):
        super().__init__(
            orientation="vertical",
            spacing=6,
            padding=6,
            size_hint_y=None,
            height=140,
            **kwargs,
        )
        self.input = TextInput(size_hint_y=None, height=80, multiline=True)
        btn = Button(text="Send", size_hint_y=None, height=36)
        btn.bind(on_release=lambda *_: send_callback(self.input.text))
        self.add_widget(self.input)
        self.add_widget(btn)


class FileUploader(Button):
    def __init__(self, upload_callback, **kwargs):
        super().__init__(text="Upload File", size_hint_y=None, height=36, **kwargs)
        self.upload_callback = upload_callback
        self.bind(on_release=self.open_chooser)

    def open_chooser(self, *a):
        chooser = FileChooserListView(size_hint=(1, 1))
        popup = Popup(title="Select file", content=chooser, size_hint=(0.9, 0.9))

        def on_selection(*_):
            if chooser.selection:
                path = chooser.selection[0]
                popup.dismiss()
                try:
                    self.upload_callback(path)
                except Exception:
                    pass

        chooser.bind(on_submit=lambda *x: on_selection())
        popup.open()


# ─────────────────────────────
# GLASSMORPHISM COMPONENTS
# ─────────────────────────────

from kivy.uix.widget import Widget
from kivy.graphics import Color, RoundedRectangle, Ellipse, Line
from kivy.clock import Clock
from kivy.properties import NumericProperty, ListProperty, StringProperty
from kivy.metrics import dp
import math


class GlassPanel(BoxLayout):
    """Premium glassmorphism panel with gradient bg and animated glow border."""

    border_color = ListProperty([0.15, 0.55, 1.0, 0.5])
    bg_color = ListProperty([0.06, 0.06, 0.12, 0.88])
    glow_strength = NumericProperty(0.5)
    corner_radius = NumericProperty(18)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._anim_events = []
        self._hovered = False
        self._glow_phase = 0
        self._setup_graphics()
        self.bind(pos=self._update_graphics, size=self._update_graphics)
        self._schedule_glow()

    def _setup_graphics(self):
        with self.canvas.before:
            Color(*self.bg_color)
            self._bg = RoundedRectangle(pos=self.pos, size=self.size, radius=[self.corner_radius])

        self._border_canvas = self.canvas
        self._update_graphics()

    def _update_graphics(self, *args):
        if not self.size[0] or not self.size[1]:
            return
        self._bg.pos = self.pos
        self._bg.size = self.size
        self._bg.radius = [self.corner_radius]
        self._border_canvas.clear()
        with self._border_canvas:
            alpha = self.border_color[3] * self.glow_strength
            Color(self.border_color[0], self.border_color[1], self.border_color[2], alpha)
            Line(
                rounded_rectangle=(
                    self.pos[0], self.pos[1],
                    self.size[0], self.size[1],
                    self.corner_radius,
                ),
                width=1.2,
            )

    def _schedule_glow(self):
        self._anim_events.append(Clock.schedule_interval(self._animate_glow, 1 / 30))

    def _animate_glow(self, dt):
        self._glow_phase += 0.05
        if self._hovered:
            self.glow_strength = min(1.0, self.glow_strength + 0.06)
        else:
            self.glow_strength = max(0.35, self.glow_strength - 0.03)
        self._update_graphics()

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self._hovered = True
            return True
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        self._hovered = False
        return super().on_touch_up(touch)

    def stop(self):
        for e in self._anim_events:
            e.cancel()


class StatusDot(Widget):
    """Animated status indicator dot with pulse."""

    dot_color = ListProperty([0.2, 1.0, 0.4, 1.0])

    def __init__(self, color=[0.2, 1.0, 0.4, 1.0], **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.size = (dp(10), dp(10))
        self.dot_color = color
        self._phase = 0
        self._events = []
        self._events.append(Clock.schedule_interval(self._pulse, 1 / 30))
        self.bind(pos=self._redraw, size=self._redraw)

    def _pulse(self, dt):
        self._phase += 0.1
        alpha = 0.5 + math.sin(self._phase) * 0.5
        self.dot_color = [self.dot_color[0], self.dot_color[1], self.dot_color[2], alpha]
        self._redraw()

    def _redraw(self, *args):
        self.canvas.clear()
        with self.canvas:
            Color(*self.dot_color)
            Ellipse(pos=self.pos, size=self.size)

    def stop(self):
        for e in self._events:
            e.cancel()


class ScanLine(Widget):
    """Horizontal scan line overlay for tech aesthetic."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._phase = 0.0
        self._events = []
        self._events.append(Clock.schedule_interval(self._animate, 1 / 60))
        self.bind(pos=self._redraw, size=self._redraw)

    def _animate(self, dt):
        self._phase = (self._phase + 0.015) % 1.0
        self._redraw()

    def _redraw(self, *args):
        self.canvas.clear()
        if not self.size[0] or not self.size[1]:
            return
        scan_y = self.pos[1] + self.size[1] * self._phase
        with self.canvas:
            Color(0.3, 0.8, 1.0, 0.07)
            Line(points=[self.pos[0], scan_y, self.pos[0] + self.size[0], scan_y], width=1)

    def stop(self):
        for e in self._events:
            e.cancel()


class ProgressRing(Widget):
    """Circular progress ring."""

    value = NumericProperty(0)
    max_val = NumericProperty(100)
    ring_color = ListProperty([0.15, 0.6, 1.0, 1.0])
    track_color = ListProperty([0.1, 0.1, 0.18, 1.0])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.size = (dp(70), dp(70))
        self.bind(pos=self._redraw, size=self._redraw, value=self._redraw, ring_color=self._redraw)

    def _redraw(self, *args):
        self.canvas.clear()
        cx = self.center_x
        cy = self.center_y
        r = min(self.size[0], self.size[1]) / 2 - dp(5)
        if r <= 0:
            return
        with self.canvas:
            Color(*self.track_color)
            Ellipse(pos=(cx - r, cy - r), size=(r * 2, r * 2))
        progress = min(self.value / max(self.max_val, 1), 1.0)
        pts = []
        for i in range(int(progress * 72) + 1):
            angle = (i / 72) * 2 * math.pi - math.pi / 2
            pts.extend([cx + math.cos(angle) * r, cy + math.sin(angle) * r])
        if len(pts) >= 4:
            with self.canvas:
                Color(*self.ring_color)
                Line(points=pts, width=dp(5), cap='round')

    def set_value(self, value):
        self.value = value
