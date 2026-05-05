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
