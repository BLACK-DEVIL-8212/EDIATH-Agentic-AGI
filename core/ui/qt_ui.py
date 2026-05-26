"""
EDIATH Desktop UI — PyQt6
A full-featured desktop AI assistant UI.
Communicates with the backend via the existing WebUIServer HTTP/WS API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Optional

from PyQt6.QtCore import (
    QEventLoop,
    QObject,
    QSize,
    Qt,
    QThread,
    QTimer,
    QUrl,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QPalette, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QGroupBox,
    QSpinBox,
)

logger = logging.getLogger("ediath.qt_ui")


# ─────────────────────────────────────────────────────────────────
# CHAT BUBBLE WIDGET
# ─────────────────────────────────────────────────────────────────
class ChatBubble(QWidget):
    """A single chat message bubble."""

    def __init__(self, role: str, text: str, timestamp: float, parent=None):
        super().__init__(parent)
        self.role = role
        is_you = role.upper() == "YOU"
        is_system = role.upper() == "SYSTEM"

        # Bubble color
        if is_you:
            bg = "#1a5276"
            fg = "#ffffff"
            align = Qt.AlignmentFlag.AlignRight
        elif is_system:
            bg = "#4a4a4a"
            fg = "#f0f0f0"
            align = Qt.AlignmentFlag.AlignCenter
        else:
            bg = "#2c2c2c"
            fg = "#e8e8e8"
            align = Qt.AlignmentFlag.AlignLeft

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(0)

        # Header: role + time
        header_label = QLabel(
            f"<b>{role.upper()}</b> · {datetime.fromtimestamp(timestamp).strftime('%H:%M')}"
        )
        header_label.setStyleSheet(
            f"color: {'#aaa' if is_system else '#ccc'}; background: transparent; border: none;"
        )
        header_label.setAlignment(align)
        layout.addWidget(header_label)

        # Message body
        body = QTextEdit(text)
        body.setReadOnly(True)
        body.setFrameShape(QTextEdit.Shape.NoFrame)
        body.setStyleSheet(
            f"""
            QTextEdit {{
                color: {fg};
                background: {bg};
                border: none;
                border-radius: 10px;
                padding: 8px 12px;
                font-size: 13px;
            }}
            """
        )
        body.document().setDefaultTextStyleSheet(f"p {{ color: {fg}; }}")
        body.setAlignment(align)
        body.document().setDocumentMargin(0)
        layout.addWidget(body)

        self.setLayout(layout)


# ─────────────────────────────────────────────────────────────────
# CHAT PANEL
# ─────────────────────────────────────────────────────────────────
class ChatPanel(QWidget):
    """Chat screen: message list + input bar."""

    sendRequested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet("background: #1e1e1e; border-bottom: 1px solid #333;")
        hl = QVBoxLayout(header)
        hl.setContentsMargins(12, 0, 12, 0)
        title = QLabel("Chat")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0; background: transparent; border: none;")
        hl.addWidget(title)

        btn_clear = QPushButton("Clear")
        btn_clear.setFixedSize(60, 28)
        btn_clear.setStyleSheet(
            "QPushButton { background: #333; color: #aaa; border: none; border-radius: 4px; }"
            "QPushButton:hover { background: #444; color: #fff; }"
        )
        btn_clear.clicked.connect(self._on_clear)
        hl.addWidget(btn_clear, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(header)

        # Message list (scroll area)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { background: #141414; border: none; }"
            "QScrollBar:vertical { background: #222; width: 8px; }"
            "QScrollBar::handle { background: #444; border-radius: 4px; }"
        )
        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setContentsMargins(12, 12, 12, 12)
        self.messages_layout.setSpacing(8)
        self.messages_layout.addStretch()
        scroll.setWidget(self.messages_widget)
        layout.addWidget(scroll)

        # Input bar
        input_bar = QWidget()
        input_bar.setFixedHeight(52)
        input_bar.setStyleSheet("background: #1e1e1e; border-top: 1px solid #333;")
        il = QVBoxLayout(input_bar)
        il.setContentsMargins(12, 6, 12, 6)
        il.setSpacing(4)

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Type your message…")
        self.input_line.setStyleSheet(
            """
            QLineEdit {
                background: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
            }
            QLineEdit:focus { border: 1px solid #555; }
            """
        )
        self.input_line.returnPressed.connect(self._on_send)
        il.addWidget(self.input_line)

        layout.addWidget(input_bar)

    def _on_send(self):
        text = self.input_line.text().strip()
        if not text:
            return
        self.input_line.clear()
        self.sendRequested.emit(text)

    def _on_clear(self):
        # Remove all bubble widgets except the stretch
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item and item.widget() and item.widget() != self.messages_layout.itemAt(0).widget():
                w = item.widget()
                if w:
                    w.deleteLater()

    def add_message(self, role: str, content: str, ts: float = 0):
        """Insert a new chat bubble at the end of the message list."""
        if ts == 0:
            ts = time.time()

        # Remove stretch
        self.messages_layout.removeItem(self.messages_layout.itemAt(self.messages_layout.count() - 1))

        bubble = ChatBubble(role, content, ts)
        bubble.setMinimumWidth(self.messages_widget.width() - 30)
        self.messages_layout.addWidget(bubble)
        self.messages_layout.addStretch()

        # Scroll to bottom
        scroll = self.messages_widget.parent()
        while scroll and not isinstance(scroll, QScrollArea):
            scroll = scroll.parent()
        if scroll:
            QTimer.singleShot(10, lambda: scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum()))

    def set_messages(self, messages: list):
        """Replace all messages with a history list."""
        # Clear all bubbles
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        # Remove stretch then re-add at end
        self.messages_layout.removeItem(self.messages_layout.itemAt(self.messages_layout.count() - 1))
        for m in messages:
            self.add_message(m.get("role", "AI"), m.get("content", ""), m.get("ts", 0))
        self.messages_layout.addStretch()


# ─────────────────────────────────────────────────────────────────
# DASHBOARD PANEL
# ─────────────────────────────────────────────────────────────────
class DashboardPanel(QWidget):
    """System dashboard with stats and quick actions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._start_time = time.time()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Dashboard")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(title)

        # Stats grid
        grid = QSplitter(Qt.Orientation.Horizontal)
        grid.setSizes([150, 150, 150, 150])

        self.stat_cards = {}
        for label, key in [
            ("Status", "status"),
            ("Active Tasks", "tasks"),
            ("Memory Entries", "memory"),
            ("Uptime", "uptime"),
        ]:
            card = self._make_stat_card(label, "—")
            self.stat_cards[key] = card
            grid.addWidget(card)
        layout.addWidget(grid)

        # Activity feed
        act_label = QLabel("Activity")
        act_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        act_label.setStyleSheet("color: #aaa;")
        layout.addWidget(act_label)

        self.activity_list = QListWidget()
        self.activity_list.setStyleSheet(
            """
            QListWidget { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; color: #ccc; }
            QListWidget::item { padding: 6px; }
            """
        )
        layout.addWidget(self.activity_list, stretch=1)

        # Quick actions
        act_label2 = QLabel("Quick Actions")
        act_label2.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        act_label2.setStyleSheet("color: #aaa;")
        layout.addWidget(act_label2)

        btn_row = QWidget()
        btn_layout = QVBoxLayout(btn_row)
        btn_layout.setSpacing(8)
        self.btn_new_chat = QPushButton("New Chat")
        self.btn_explore = QPushButton("Explore Memories")
        self.btn_schedule = QPushButton("Schedule Task")
        for btn in [self.btn_new_chat, self.btn_explore, self.btn_schedule]:
            btn.setFixedHeight(32)
            btn.setStyleSheet(
                "QPushButton { background: #2a2a2a; color: #ccc; border: 1px solid #3a3a3a; "
                "border-radius: 6px; padding: 4px 12px; }"
                "QPushButton:hover { background: #383838; color: #fff; }"
            )
            btn_layout.addWidget(btn)
        layout.addWidget(btn_row)

        layout.addStretch()

        # Uptime timer
        self._uptime_timer = QTimer()
        self._uptime_timer.timeout.connect(self._tick_uptime)
        self._uptime_timer.start(1000)

    def _make_stat_card(self, label: str, value: str) -> QWidget:
        card = QWidget()
        card.setStyleSheet(
            "background: #1e1e1e; border: 1px solid #2a2a2a; border-radius: 10px; padding: 12px;"
        )
        vl = QVBoxLayout(card)
        vl.setContentsMargins(12, 8, 12, 8)
        vl.setSpacing(4)
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #777; font-size: 11px;")
        vl.addWidget(lbl)
        val = QLabel(value)
        val.setStyleSheet("color: #e0e0e0; font-size: 20px; font-weight: bold;")
        vl.addWidget(val)
        return card

    def _tick_uptime(self):
        try:
            elapsed = int(time.time() - self._start_time)
            h, rem = divmod(elapsed, 3600)
            m, _ = divmod(rem, 60)
            uptime = f"{h}h {m}m" if h > 0 else f"{m}m"
            self.stat_cards["uptime"].findChild(QLabel, options=Qt.FindChildOption.FindChildrenRecursively).setText(uptime)
        except Exception:
            pass

    def add_activity(self, text: str):
        item = QListWidgetItem(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
        self.activity_list.insertItem(0, item)
        if self.activity_list.count() > 50:
            self.activity_list.takeRow(50)

    def set_stat(self, key: str, value: str):
        if key in self.stat_cards:
            card = self.stat_cards[key]
            # Find the value label (second QLabel)
            labels = card.findChildren(QLabel)
            if len(labels) >= 2:
                labels[1].setText(value)


# ─────────────────────────────────────────────────────────────────
# MEMORY PANEL
# ─────────────────────────────────────────────────────────────────
class MemoryPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("Memory")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(title)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search memory…")
        self.search_box.setStyleSheet(
            "QLineEdit { background: #2a2a2a; color: #ccc; border: 1px solid #333; "
            "border-radius: 6px; padding: 6px 10px; }"
        )
        layout.addWidget(self.search_box)

        self.memory_list = QListWidget()
        self.memory_list.addItem("Connect a memory backend to view entries.")
        self.memory_list.setStyleSheet(
            "QListWidget { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; color: #aaa; }"
        )
        layout.addWidget(self.memory_list, stretch=1)


# ─────────────────────────────────────────────────────────────────
# TASKS PANEL
# ─────────────────────────────────────────────────────────────────
class TasksPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("Task Queue")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(title)

        self.task_list = QListWidget()
        self.task_list.addItem("No active tasks.")
        self.task_list.setStyleSheet(
            "QListWidget { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; color: #aaa; }"
        )
        layout.addWidget(self.task_list, stretch=1)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setFixedHeight(30)
        btn_refresh.setStyleSheet(
            "QPushButton { background: #2a2a2a; color: #ccc; border: 1px solid #3a3a3a; border-radius: 6px; }"
            "QPushButton:hover { background: #383838; color: #fff; }"
        )
        layout.addWidget(btn_refresh)


# ─────────────────────────────────────────────────────────────────
# SETTINGS PANEL
# ─────────────────────────────────────────────────────────────────
class SettingsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Settings")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(title)

        def make_group(title_str: str) -> QGroupBox:
            g = QGroupBox(title_str)
            g.setStyleSheet(
                "QGroupBox { color: #888; border: 1px solid #2a2a2a; border-radius: 8px; "
                "margin-top: 8px; font-weight: bold; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
            )
            return g

        # LLM Provider
        g_provider = make_group("LLM Provider")
        vl = QVBoxLayout(g_provider)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Local (llama-cpp)", "OpenAI", "Anthropic"])
        self.provider_combo.setStyleSheet(
            "QComboBox { background: #2a2a2a; color: #ccc; border: 1px solid #3a3a3a; border-radius: 6px; padding: 6px; }"
        )
        vl.addWidget(self.provider_combo)
        layout.addWidget(g_provider)

        # Autonomy
        g_auto = make_group("Autonomy")
        vl2 = QVBoxLayout(g_auto)
        self.chk_background_think = QCheckBox("Enable background thinking")
        self.chk_background_think.setChecked(True)
        self.chk_background_think.setStyleSheet("color: #ccc;")
        vl2.addWidget(self.chk_background_think)
        self.chk_autonomous = QCheckBox("Enable autonomous mode")
        self.chk_autonomous.setStyleSheet("color: #ccc;")
        vl2.addWidget(self.chk_autonomous)
        layout.addWidget(g_auto)

        # Voice
        g_voice = make_group("Voice")
        vl3 = QVBoxLayout(g_voice)
        self.chk_voice = QCheckBox("Enable voice input")
        self.chk_voice.setStyleSheet("color: #ccc;")
        vl3.addWidget(self.chk_voice)
        layout.addWidget(g_voice)

        # Theme
        g_theme = make_group("Theme")
        vl4 = QVBoxLayout(g_theme)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setCurrentText("Dark")
        self.theme_combo.setStyleSheet(
            "QComboBox { background: #2a2a2a; color: #ccc; border: 1px solid #3a3a3a; border-radius: 6px; padding: 6px; }"
        )
        vl4.addWidget(self.theme_combo)
        layout.addWidget(g_theme)

        # Max tokens
        g_gen = make_group("Generation")
        vl5 = QVBoxLayout(g_gen)
        h = QWidget()
        hl = QVBoxLayout(h)
        hl.setSpacing(8)
        lbl = QLabel("Max tokens per response:")
        lbl.setStyleSheet("color: #888;")
        hl.addWidget(lbl)
        self.spin_max_tokens = QSpinBox()
        self.spin_max_tokens.setRange(256, 8192)
        self.spin_max_tokens.setValue(2048)
        self.spin_max_tokens.setStyleSheet(
            "QSpinBox { background: #2a2a2a; color: #ccc; border: 1px solid #3a3a3a; border-radius: 6px; padding: 4px; }"
        )
        hl.addWidget(self.spin_max_tokens)
        vl5.addWidget(h)
        layout.addWidget(g_gen)

        layout.addStretch()


# ─────────────────────────────────────────────────────────────────
# STATUS BAR WIDGET
# ─────────────────────────────────────────────────────────────────
class StatusIndicator(QWidget):
    """Colored dot + text status bar widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(4)
        hl = QWidget()
        hlo = QVBoxLayout(hl)
        hlo.setSpacing(0)
        hlo.setContentsMargins(0, 0, 0, 0)
        self.dot = QLabel("●")
        self.dot.setStyleSheet("color: #555; font-size: 10px;")
        self.label = QLabel("Connecting…")
        self.label.setStyleSheet("color: #777; font-size: 12px;")
        hlo.addWidget(self.dot)
        hlo.addWidget(self.label)
        hl.setLayout(hlo)
        layout.addWidget(hl)

    def set_status(self, text: str, ok: bool = False, error: bool = False):
        self.label.setText(text)
        if error:
            self.dot.setStyleSheet("color: #e74c3c; font-size: 10px;")
        elif ok:
            self.dot.setStyleSheet("color: #2ecc71; font-size: 10px;")
        else:
            self.dot.setStyleSheet("color: #f39c12; font-size: 10px;")


# ─────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────
class EDIATHWindow(QMainWindow):
    """EDIATH Desktop AI Assistant — PyQt6 main window."""

    def __init__(self, backend_host: str = "127.0.0.1", backend_port: int = 8000):
        super().__init__()
        self.backend_host = backend_host
        self.backend_port = backend_port
        self._ws_thread: Optional[QThread] = None
        self._connected = False

        self._init_ui()
        self._start_ws_thread()
        self._connect_signals()

    def _init_ui(self):
        self.setWindowTitle("EDIATH AI Assistant")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)

        # Dark palette
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#141414"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#e0e0e0"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#1e1e1e"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1a1a1a"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#2c2c2c"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#e0e0e0"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#2c2c2c"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e0e0e0"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#3a3a3a"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        self.setPalette(palette)

        # Central widget: splitter sidebar + content
        central = QWidget()
        clayout = QVBoxLayout(central)
        clayout.setContentsMargins(0, 0, 0, 0)
        clayout.setSpacing(0)

        # Horizontal splitter
        hsplitter = QSplitter(Qt.Orientation.Horizontal)
        hsplitter.setSizes([180, 700])

        # ── SIDEBAR ──────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(180)
        sidebar.setStyleSheet("background: #1a1a1a; border-right: 1px solid #2a2a2a;")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 12, 0, 8)
        sl.setSpacing(4)

        brand = QLabel("EDIATH")
        brand.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand.setStyleSheet("color: #4a9eff; background: transparent; padding: 8px 0;")
        sl.addWidget(brand)

        self.nav_buttons = {}
        nav_items = [
            ("Dashboard", "dashboard", "Ctrl+D"),
            ("Chat", "chat", "Ctrl+C"),
            ("Memory", "memory", "Ctrl+M"),
            ("Tasks", "tasks", "Ctrl+T"),
            ("Settings", "settings", "Ctrl+,"),
        ]
        for label, key, shortcut in nav_items:
            btn = QPushButton(label)
            btn.setShortcut(shortcut)
            btn.setFixedHeight(38)
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            if key == "dashboard":
                btn.setChecked(True)
            btn.setStyleSheet(
                """
                QPushButton {
                    background: transparent;
                    color: #888;
                    border: none;
                    border-radius: 6px;
                    text-align: left;
                    padding-left: 16px;
                    font-size: 13px;
                }
                QPushButton:checked {
                    background: #2a2a2a;
                    color: #e0e0e0;
                    border-left: 3px solid #4a9eff;
                }
                QPushButton:hover:not(:checked) { background: #222; color: #aaa; }
                """
            )
            btn.clicked.connect(lambda _, k=key: self._show_screen(k))
            sl.addWidget(btn)
            self.nav_buttons[key] = btn

        sl.addStretch()

        # Connection status at bottom of sidebar
        self.status_indicator = StatusIndicator()
        self.status_indicator.label.setStyleSheet("color: #555; font-size: 11px;")
        sl.addWidget(self.status_indicator, alignment=Qt.AlignmentFlag.AlignCenter)

        hsplitter.addWidget(sidebar)

        # ── CONTENT STACK ────────────────────────────────────────────
        self.content_stack = QStackedWidget()
        self.dashboard_panel = DashboardPanel()
        self.chat_panel = ChatPanel()
        self.memory_panel = MemoryPanel()
        self.tasks_panel = TasksPanel()
        self.settings_panel = SettingsPanel()

        for panel in [
            self.dashboard_panel,
            self.chat_panel,
            self.memory_panel,
            self.tasks_panel,
            self.settings_panel,
        ]:
            self.content_stack.addWidget(panel)

        hsplitter.addWidget(self.content_stack)

        # Make sidebar not stretch
        hsplitter.setStretchFactor(0, 0)
        hsplitter.setStretchFactor(1, 1)

        clayout.addWidget(hsplitter)

        # ── STATUS BAR ───────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            "QStatusBar { background: #111; color: #777; border-top: 1px solid #222; }"
        )
        self.status_bar.showMessage("EDIATH AI Assistant ready")
        self.setStatusBar(self.status_bar)

        self.setCentralWidget(central)

        # Toolbar
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setFixedHeight(36)
        toolbar.setStyleSheet(
            "QToolBar { background: #1e1e1e; border: none; border-bottom: 1px solid #2a2a2a; spacing: 4px; }"
        )
        self.addToolBar(toolbar)

        self.action_new_chat = QAction("New Chat", self)
        self.action_new_chat.setShortcut("Ctrl+N")
        self.action_new_chat.triggered.connect(self._on_new_chat)
        toolbar.addAction(self.action_new_chat)

        self.action_clear = QAction("Clear Chat", self)
        self.action_clear.setShortcut("Ctrl+L")
        self.action_clear.triggered.connect(lambda: self.chat_panel._on_clear())
        toolbar.addAction(self.action_clear)

        toolbar.addSeparator()

        self.action_dashboard = QAction("Dashboard", self)
        self.action_dashboard.triggered.connect(lambda: self._show_screen("dashboard"))
        toolbar.addAction(self.action_dashboard)

        self.status_label = QLabel(f"Backend: {self.backend_host}:{self.backend_port}")
        self.status_label.setStyleSheet("color: #555; font-size: 11px; padding: 0 8px;")
        toolbar.addWidget(self.status_label)

    def _show_screen(self, key: str):
        screen_map = {
            "dashboard": 0,
            "chat": 1,
            "memory": 2,
            "tasks": 3,
            "settings": 4,
        }
        if key in screen_map:
            self.content_stack.setCurrentIndex(screen_map[key])
            for nav_key, btn in self.nav_buttons.items():
                btn.setChecked(nav_key == key)

    def _connect_signals(self):
        """Wire ChatPanel.sendRequested to HTTP POST to backend."""
        self.chat_panel.sendRequested.connect(self._on_send_message)

        # Quick actions from dashboard
        self.dashboard_panel.btn_new_chat.clicked.connect(lambda: self._show_screen("chat"))
        self.dashboard_panel.btn_explore.clicked.connect(self._on_explore)
        self.dashboard_panel.btn_schedule.clicked.connect(lambda: self._show_screen("tasks"))

    def _on_send_message(self, text: str):
        """Send message to backend via HTTP POST."""
        self.dashboard_panel.add_activity(f"You: {text[:60]}")
        self.status_bar.showMessage(f"Sending: {text[:40]}…")

        def post():
            try:
                import urllib.request
                import urllib.error

                url = f"http://{self.backend_host}:{self.backend_port}/chat"
                data = json.dumps({"text": text}).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())
                    if not result.get("ok"):
                        QTimer.singleShot(
                            0,
                            lambda: self.chat_panel.add_message(
                                "SYSTEM", result.get("error", "Unknown error"), time.time()
                            ),
                        )
            except Exception as exc:
                logger.error("Send message failed: %s", exc)
                QTimer.singleShot(
                    0,
                    lambda: self.chat_panel.add_message(
                        "SYSTEM", f"Request failed: {exc}", time.time()
                    ),
                )

        threading.Thread(target=post, daemon=True).start()

    def _on_explore(self):
        self._show_screen("chat")
        self._on_send_message(
            "Explore my recent memories and summarize what you found."
        )

    def _on_new_chat(self):
        self._show_screen("chat")
        self.chat_panel._on_clear()

    def _start_ws_thread(self):
        """Connect WebSocket in a background thread and push messages to the chat panel."""
        self._ws_thread = QThread()
        self._ws_thread.run = self._ws_loop
        self._ws_thread.start()

    def _ws_loop(self):
        """WebSocket receive loop — runs in QThread."""
        import websocket

        while not getattr(self, "_ws_running", True):
            time.sleep(0.1)

        ws_url = f"ws://{self.backend_host}:{self.backend_port}/ws"
        backoff = 2
        max_backoff = 30

        while getattr(self, "_ws_running", True):
            try:
                ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=self._ws_on_message,
                    on_error=self._ws_on_error,
                    on_close=self._ws_on_close,
                    on_open=self._ws_on_open,
                )
                self._ws = ws
                QTimer.singleShot(0, lambda: self.status_indicator.set_status("Connected", ok=True))
                self._connected = True
                backoff = 2
                ws.run_forever(ping_interval=10, ping_timeout=5)
            except Exception as exc:
                logger.warning("WebSocket error: %s", exc)
                QTimer.singleShot(0, lambda: self.status_indicator.set_status("Reconnecting…"))
                self._connected = False
            time.sleep(backoff)
            backoff = min(backoff * 1.5, max_backoff)

    def _ws_on_open(self, ws):
        pass

    def _ws_on_message(self, ws, data: str):
        try:
            msg = json.loads(data)
        except Exception:
            return

        def handle():
            t = msg.get("type", "")
            if t == "history":
                messages = msg.get("messages", [])
                self.chat_panel.set_messages(messages)
            elif t == "message":
                m = msg.get("message", {})
                role = m.get("role", "AI")
                content = m.get("content", "")
                ts = m.get("ts", time.time())
                self.chat_panel.add_message(role, content, ts)
                self.dashboard_panel.add_activity(f"AI: {content[:60]}")
                self.status_bar.showMessage("Response received")
        QTimer.singleShot(0, handle)

    def _ws_on_error(self, ws, error):
        logger.warning("WebSocket error: %s", error)

    def _ws_on_close(self, ws, code, msg):
        self._connected = False
        QTimer.singleShot(0, lambda: self.status_indicator.set_status("Disconnected", error=True))

    def closeEvent(self, event):
        self._ws_running = False
        if hasattr(self, "_ws"):
            try:
                self._ws.close()
            except Exception:
                pass
        if self._ws_thread:
            self._ws_thread.quit()
        event.accept()


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
def run_ui(
    backend_host: str = "127.0.0.1",
    backend_port: int = 8000,
    dark: bool = True,
):
    """Launch the EDIATH PyQt6 UI."""
    app = QApplication(sys.argv)
    app.setApplicationName("EDIATH AI Assistant")
    app.setFont(QFont("Segoe UI", 10))

    window = EDIATHWindow(backend_host=backend_host, backend_port=backend_port)
    window.show()

    return app.exec()