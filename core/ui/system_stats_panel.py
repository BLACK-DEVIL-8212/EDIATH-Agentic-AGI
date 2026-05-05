from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock


class SystemStatsPanel(BoxLayout):
    """
    Real-time system stats panel
    """

    def __init__(self, controller=None, **kwargs):
        super().__init__(orientation="vertical", **kwargs)

        self.controller = controller
        self.padding = [10, 10]
        self.spacing = 10

        self.rows = {}

        self._build_ui()
        Clock.schedule_interval(self.update_stats, 1)

    # ─────────────────────────────
    # UI BUILD
    # ─────────────────────────────
    def _build_ui(self):
        # Title
        title = Label(text="[b]System Statistics[/b]", markup=True, size_hint_y=0.15)
        self.add_widget(title)

        # Stats container
        self.stats_container = BoxLayout(orientation="vertical", spacing=8)
        self.add_widget(self.stats_container)

        # Create rows
        self._create_row("cpu", "CPU Usage")
        self._create_row("memory", "Memory")
        self._create_row("gpu", "GPU Usage")
        self._create_row("gpu_mem", "GPU Memory")

    # ─────────────────────────────
    # ROW FACTORY
    # ─────────────────────────────
    def _create_row(self, key, label_text):
        row = BoxLayout(size_hint_y=None, height=30, spacing=10)

        label = Label(
            text=f"{label_text}:", size_hint_x=0.3, halign="right", valign="middle"
        )
        label.bind(size=lambda inst, val: setattr(inst, "text_size", val))

        progress = ProgressBar(max=100, value=0, size_hint_x=0.5)

        value_label = Label(text="0%", size_hint_x=0.2, halign="left")
        value_label.bind(size=lambda inst, val: setattr(inst, "text_size", val))

        row.add_widget(label)
        row.add_widget(progress)
        row.add_widget(value_label)

        self.stats_container.add_widget(row)

        self.rows[key] = {"progress": progress, "label": value_label}

    # ─────────────────────────────
    # UPDATE STATS
    # ─────────────────────────────
    def update_stats(self, dt):
        try:
            if not self.controller:
                return

            stats = self.controller.get_system_stats()

            # 🔥 FIX: Parse REAL stats from system.get_system_status()
            ai_ready = 100 if stats.get("llm_ready", False) else 0
            running = 100 if stats.get("is_running", False) else 0
            queue_len = (
                stats.get("task_queue", {}).get("queued", 0)
                if stats.get("task_queue")
                else 0
            )
            mem_total = stats.get("memory", {}).get("episodic", 0) + stats.get(
                "memory", {}
            ).get("semantic", 0)

            # Map to rows (reuse existing)
            self._update_row("cpu", ai_ready, "AI")
            self._update_row("memory", running, "SYS")
            self._update_row("gpu", queue_len / 10, f"Q:{queue_len}")  # scale queue
            self._update_row("gpu_mem", min(mem_total / 10, 100), f"M:{mem_total}")

        except Exception as e:
            print("SystemStats error:", e)

    def _update_row(self, key, value, label_suffix=""):
        row = self.rows.get(key)
        if not row:
            return

        row["progress"].value = value
        row["label"].text = f"{value:.0f}%{label_suffix}"

    # ─────────────────────────────
    def _update_row(self, key, value):
        row = self.rows.get(key)
        if not row:
            return

        row["progress"].value = value
        row["label"].text = f"{value:.1f}%"
