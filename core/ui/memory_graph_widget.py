from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line, Rectangle
from kivy.clock import Clock
from collections import deque
import psutil


# ─────────────────────────────
# MEMORY GRAPH WIDGET
# ─────────────────────────────
class MemoryGraphWidget(BoxLayout):

    def __init__(self, controller=None, **kwargs):
        super().__init__(orientation="vertical", **kwargs)

        self.controller = controller
        self.memory_history = deque(maxlen=60)

        self._event = None
        self._build_ui()

        # safer scheduling
        self._event = Clock.schedule_interval(self.update_data, 1)

    def _build_ui(self):
        self.title = Label(text="[b]Memory Usage[/b]", markup=True, size_hint_y=0.15)
        self.add_widget(self.title)

        self.graph = GraphWidget()
        self.add_widget(self.graph)

        self.value_label = Label(text="Memory: 0%", size_hint_y=0.15)
        self.add_widget(self.value_label)

    # ─────────────────────────────
    def update_data(self, dt):
        try:
            memory_percent = 0

            # Backend stats (safe)
            if self.controller:
                try:
                    stats = self.controller.get_system_stats()
                    if isinstance(stats, dict):
                        memory_percent = stats.get("memory", {}).get(
                            "vector", 0
                        ) or stats.get("memory_percent", 0)
                except Exception:
                    pass

            # Fallback
            if not memory_percent:
                memory_percent = psutil.virtual_memory().percent

            self.memory_history.append(memory_percent)

            self.graph.update_graph(list(self.memory_history))
            self.value_label.text = f"Memory: {memory_percent:.1f}%"

        except Exception as e:
            print("MemoryGraph error:", e)

    # ─────────────────────────────
    def stop(self):
        """Prevent background crash on shutdown"""
        if self._event:
            self._event.cancel()
            self._event = None


# ─────────────────────────────
# GRAPH WIDGET (ULTRA STABLE)
# ─────────────────────────────
class GraphWidget(Widget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data_points = []

        with self.canvas:
            Color(0.1, 0.1, 0.1, 1)
            self.bg = Rectangle(pos=self.pos, size=self.size)

        # dedicated safe drawing layer
        self.dynamic_canvas = self.canvas.after

        self.bind(pos=self.redraw, size=self.redraw)

    # ─────────────────────────────
    def update_graph(self, data):
        self.data_points = data
        self.redraw()

    # ─────────────────────────────
    def redraw(self, *args):
        try:
            self.bg.pos = self.pos
            self.bg.size = self.size

            # 🔥 CRITICAL FIX: no manual remove
            self.dynamic_canvas.clear()

            if not self.data_points or self.width <= 0 or self.height <= 0:
                return

            # ─── GRID ───
            with self.dynamic_canvas:
                Color(0.3, 0.3, 0.3, 0.3)

                for i in range(5):
                    y = self.y + (i * self.height / 4)
                    Line(points=[self.x, y, self.x + self.width, y], width=1)

                # ─── GRAPH ───
                Color(0.2, 0.7, 1, 1)

                points = []
                step = self.width / max(len(self.data_points) - 1, 1)

                for i, value in enumerate(self.data_points):
                    value = max(0, min(100, value))

                    x = self.x + (i * step)
                    y = self.y + (value / 100 * self.height)

                    points.extend([x, y])

                if len(points) >= 4:
                    Line(points=points, width=2)

        except Exception as e:
            print("Graph redraw error:", e)
