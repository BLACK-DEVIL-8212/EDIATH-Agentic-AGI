from kivy.app import App
from kivy.core.window import Window
from kivy.utils import get_color_from_hex

from .screens import build_screen_manager
from .controller import controller


class EDIATHUIApp(App):
    def build(self):
        # dark theme
        Window.clearcolor = get_color_from_hex("#0f1720")
        self.sm = build_screen_manager()
        return self.sm

    def on_start(self):
        # warm up controller: check availability
        def cb(res):
            # simply log status to UI if available
            pass

        try:
            controller.get_status(callback=cb)
        except Exception:
            pass

    def on_stop(self):
        try:
            controller.shutdown()
        except Exception:
            pass


def run_app():
    EDIATHUIApp().run()


if __name__ == "__main__":
    run_app()
