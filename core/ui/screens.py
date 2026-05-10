from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.clock import Clock

from .components import (
    Panel,
    ChatBubble,
    LogsView,
    TaskRow,
    StatusItem,
    CommandPanel,
    FileUploader,
)
from .controller import controller


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        root = BoxLayout(orientation="horizontal", spacing=8, padding=8)

        # Left column: Chat + Logs
        left = BoxLayout(orientation="vertical", size_hint_x=0.6, spacing=8)
        self.chat_panel = Panel(title="Chat")
        self.chat_scroll = ScrollView()
        self.chat_list = BoxLayout(orientation="vertical", size_hint_y=None)
        self.chat_list.bind(minimum_height=self.chat_list.setter("height"))
        self.chat_scroll.add_widget(self.chat_list)
        # chat input area
        chat_input_container = BoxLayout(
            orientation="horizontal", size_hint_y=None, height=40, spacing=6
        )
        self.chat_input = Label(text="")
        from kivy.uix.textinput import TextInput
        from kivy.uix.button import Button

        self.chat_input = TextInput(size_hint_x=0.8, multiline=False)
        send_btn = Button(text="Send", size_hint_x=0.2)
        send_btn.bind(on_release=lambda *_: self._on_send_chat(self.chat_input.text))
        chat_input_container.add_widget(self.chat_input)
        chat_input_container.add_widget(send_btn)
        self.chat_panel.add_widget(self.chat_scroll)
        self.chat_panel.add_widget(chat_input_container)

        self.logs_panel = Panel(title="Logs")
        self.logs_view = LogsView(size_hint_y=0.4)
        self.logs_panel.add_widget(self.logs_view)

        left.add_widget(self.chat_panel)
        left.add_widget(self.logs_panel)

        # Right column: Status + Tasks + Command
        right = BoxLayout(orientation="vertical", size_hint_x=0.4, spacing=8)
        self.status_panel = Panel(title="System Status")
        self.cpu_item = StatusItem("CPU", "0%")
        self.mem_item = StatusItem("Memory", "0%")
        self.state_item = StatusItem("State", "stopped")
        self.cb_item = StatusItem("Circuit", "ok")
        self.voice_item = StatusItem("Voice", "unknown")
        self.vision_item = StatusItem("Vision", "unknown")
        for it in (self.cpu_item, self.mem_item, self.state_item, self.cb_item):
            self.status_panel.add_widget(it)
        # add voice/vision at end
        self.status_panel.add_widget(self.voice_item)
        self.status_panel.add_widget(self.vision_item)

        self.tasks_panel = Panel(title="Tasks")
        self.tasks_list = BoxLayout(orientation="vertical", size_hint_y=None)
        self.tasks_list.bind(minimum_height=self.tasks_list.setter("height"))
        tasks_scroll = ScrollView()
        tasks_scroll.add_widget(self.tasks_list)
        self.tasks_panel.add_widget(tasks_scroll)

        self.command_panel = Panel(title="Command")
        self.command = CommandPanel(self._on_send_command)
        self.file_uploader = FileUploader(self._on_upload_file)
        self.command_panel.add_widget(self.command)
        self.command_panel.add_widget(self.file_uploader)

        right.add_widget(self.status_panel)
        right.add_widget(self.tasks_panel)
        right.add_widget(self.command_panel)

        root.add_widget(left)
        root.add_widget(right)
        self.add_widget(root)

        # periodic poll
        Clock.schedule_interval(self._poll_backend, 1.0)

    def _on_send_command(self, text: str):
        if not text or not text.strip():
            return
        controller.send_command(text, callback=self._on_command_result)

    def _on_send_chat(self, text: str):
        if not text or not text.strip():
            return
        # show user message immediately

        self.chat_list.add_widget(
            ChatBubble(text=f"You: {text}", who="user", size_hint_y=None, height=28)
        )
        self.chat_input.text = ""
        controller.send_chat(text, callback=self._on_chat_response)

    def _on_chat_response(self, res):
        def _cb(dt):
            try:
                msg = res.get("message") if isinstance(res, dict) else str(res)
            except Exception:
                msg = str(res)

            self.chat_list.add_widget(
                ChatBubble(text=f"AI: {msg}", who="ai", size_hint_y=None, height=28)
            )

        Clock.schedule_once(_cb)

    def _on_command_result(self, result):
        def _cb(dt):
            line = str(result)
            self.logs_view.add_line(line)

        Clock.schedule_once(_cb)

    def _on_upload_file(self, path: str):
        controller.upload_file(path, callback=self._on_upload_result)

    def _on_upload_result(self, r):
        def _cb(dt):
            self.logs_view.add_line(f"Upload result: {r}")

        Clock.schedule_once(_cb)

    def _poll_backend(self, dt):
        # poll aggregated state and update UI non-blocking
        def on_poll(res):
            def _update_ui(_):
                status = res.get("status") if isinstance(res, dict) else None
                if status and isinstance(status, dict):
                    cpu = status.get("cpu", "N/A")
                    mem = status.get("memory", "N/A")
                    state = status.get("state", "unknown")
                    circuit = status.get("circuit_breaker", "ok")
                    self.cpu_item.set(str(cpu))
                    self.mem_item.set(str(mem))
                    self.state_item.set(str(state))
                    self.cb_item.set(str(circuit))

                logs = res.get("logs")
                if isinstance(logs, (list, tuple)):
                    for l in logs[-20:]:
                        self.logs_view.add_line(str(l))

                tasks = res.get("tasks")
                if isinstance(tasks, (list, tuple)):
                    self.tasks_list.clear_widgets()
                    for t in tasks:
                        try:
                            tr = TaskRow(str(t.get("id", "")), t.get("title", str(t)))
                        except Exception:
                            tr = TaskRow(str(t), str(t))
                        self.tasks_list.add_widget(tr)

            Clock.schedule_once(_update_ui)

        try:
            controller.poll(callback=on_poll)
        except Exception:
            pass


class RootScreenManager(ScreenManager):
    pass


def build_screen_manager():
    sm = RootScreenManager()
    sm.add_widget(MainScreen(name="main"))
    return sm
