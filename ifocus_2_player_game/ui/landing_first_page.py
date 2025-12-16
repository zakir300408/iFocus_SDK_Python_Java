# iFocus Game device selection UI (PySide6)
# pip install pyside6
# Requires: assets/logo.png

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
from typing import List, Optional, Set

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QPainter, QColor, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QGridLayout,
)


# -----------------------------
# Stubs you will wire up later
# -----------------------------
def scan_devices_stub() -> List[dict]:
    demo = [
        {"name": "Player 1", "mac": "00:11:22:33:44:55", "rssi": random.randint(-90, -40)},
        {"name": "Player 2", "mac": "66:77:88:99:AA:BB", "rssi": random.randint(-90, -40)},
        {"name": "Player 3", "mac": "CC:DD:EE:FF:00:11", "rssi": random.randint(-90, -40)},
        {"name": "Player 4", "mac": "22:33:44:55:66:77", "rssi": random.randint(-90, -40)},
    ]
    random.shuffle(demo)
    return demo[: random.randint(1, 4)]


def connect_device_stub(mac: str) -> None:
    print(f"[CONNECT] {mac}")


def disconnect_device_stub(mac: str) -> None:
    print(f"[DISCONNECT] {mac}")


# -----------------------------
# Models
# -----------------------------
@dataclass
class DeviceInfo:
    name: str
    mac: str
    rssi: int


# -----------------------------
# Card widget
# -----------------------------
class AvatarWidget(QWidget):
    def __init__(self, ui_scale: float, parent: QWidget | None = None):
        super().__init__(parent)
        self.ui_scale = ui_scale
        self.accent: QColor = QColor("#60A5FA")
        self.setObjectName("Avatar")
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_accent(self, accent: QColor):
        self.accent = accent
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        if not p.isActive():
            return

        p.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        cx, cy = rect.center().x(), rect.center().y()
        r = min(rect.width(), rect.height()) // 2 - max(2, int(2 * self.ui_scale))

        p.setPen(Qt.NoPen)
        p.setBrush(self.accent)
        p.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)

        # sparkles
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(
            cx - r // 2,
            cy - r // 2,
            max(6, int(8 * self.ui_scale)),
            max(6, int(8 * self.ui_scale)),
        )
        p.drawEllipse(
            cx + r // 4,
            cy + r // 4,
            max(5, int(6 * self.ui_scale)),
            max(5, int(6 * self.ui_scale)),
        )


class PlayerCard(QFrame):
    toggled = Signal(object)  # emits self

    def __init__(self, ui_scale: float, parent: QWidget | None = None):
        super().__init__(parent)
        self.ui_scale = ui_scale
        self.device: Optional[DeviceInfo] = None
        self.selected: bool = False
        self.accent: QColor = QColor("#60A5FA")

        self.setObjectName("PlayerCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.PointingHandCursor)

        avatar_size = int(72 * ui_scale)

        self.avatar = AvatarWidget(ui_scale, self)
        self.avatar.setFixedSize(avatar_size, avatar_size)

        self.name_lbl = QLabel("", self)
        self.name_lbl.setObjectName("NameLabel")
        self.name_lbl.setAlignment(Qt.AlignHCenter)

        self.mac_lbl = QLabel("", self)
        self.mac_lbl.setObjectName("MetaLabel")
        self.mac_lbl.setAlignment(Qt.AlignHCenter)

        self.rssi_lbl = QLabel("", self)
        self.rssi_lbl.setObjectName("MetaLabel")
        self.rssi_lbl.setAlignment(Qt.AlignHCenter)

        self.pill = QLabel("Connected", self)
        self.pill.setObjectName("Pill")
        self.pill.setAlignment(Qt.AlignCenter)
        self.pill.setVisible(False)

        layout = QVBoxLayout(self)
        m = int(18 * ui_scale)
        layout.setContentsMargins(m, m, m, m)
        layout.setSpacing(int(10 * ui_scale))
        layout.addWidget(self.avatar, 0, Qt.AlignHCenter)
        layout.addWidget(self.name_lbl)
        layout.addWidget(self.mac_lbl)
        layout.addWidget(self.rssi_lbl)
        layout.addStretch(1)
        layout.addWidget(self.pill, 0, Qt.AlignHCenter)

        self._apply_fonts()
        self._refresh_style()

    def _apply_fonts(self):
        name_font = QFont()
        name_font.setPointSize(max(12, int(16 * self.ui_scale)))
        name_font.setWeight(QFont.Bold)
        self.name_lbl.setFont(name_font)

        meta_font = QFont()
        meta_font.setPointSize(max(9, int(11 * self.ui_scale)))
        self.mac_lbl.setFont(meta_font)
        self.rssi_lbl.setFont(meta_font)

        pill_font = QFont()
        pill_font.setPointSize(max(8, int(10 * self.ui_scale)))
        pill_font.setWeight(QFont.Bold)
        self.pill.setFont(pill_font)

    def set_device(self, device: DeviceInfo, accent_hex: str):
        self.device = device
        self.selected = False
        self.accent = QColor(accent_hex)
        self.avatar.set_accent(self.accent)
        self.name_lbl.setText(device.name)
        self.mac_lbl.setText(device.mac)
        self.rssi_lbl.setText(f"{device.rssi} dBm")
        self.pill.setVisible(False)
        self._refresh_style()

    def mousePressEvent(self, event):
        if not self.device:
            return
        if event.button() == Qt.LeftButton:
            self.selected = not self.selected
            self.pill.setVisible(self.selected)
            self._refresh_style()
            self.toggled.emit(self)

    def _refresh_style(self):
        self.setProperty("selected", "true" if self.selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

# -----------------------------
# Main window
# -----------------------------
class IFocusWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Compute a simple scale factor from screen size
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        w, h = geo.width(), geo.height()

        # 1440x900 is a decent “baseline laptop-ish” reference
        self.ui_scale = max(0.85, min(1.45, min(w / 1440.0, h / 900.0)))

        self.setWindowTitle("iFocus Game")
        self.selected_macs: Set[str] = set()
        self.cards: List[PlayerCard] = []

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(int(18 * self.ui_scale), int(18 * self.ui_scale),
                                 int(18 * self.ui_scale), int(18 * self.ui_scale))
        outer.setSpacing(int(12 * self.ui_scale))

        # Header: logo stays left, title/subtitle centered
        header = QHBoxLayout()
        header.setSpacing(int(12 * self.ui_scale))

        self.logo_lbl = QLabel()
        self.logo_lbl.setObjectName("Logo")
        self.logo_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        logo_path = (Path(__file__).resolve().parents[1] / "assets" / "logo.png")
        self._load_logo(str(logo_path))

        title_box = QVBoxLayout()
        title_box.setSpacing(int(4 * self.ui_scale))

        self.title = QLabel("iFocus Game")
        self.title.setObjectName("Title")
        self.title.setAlignment(Qt.AlignHCenter)
        title_font = QFont()
        title_font.setPointSize(max(18, int(28 * self.ui_scale)))
        title_font.setWeight(QFont.DemiBold)
        self.title.setFont(title_font)

        self.subtitle = QLabel("Search and tap players to connect (max 4)")
        self.subtitle.setObjectName("Subtitle")
        self.subtitle.setAlignment(Qt.AlignHCenter)
        subtitle_font = QFont()
        subtitle_font.setPointSize(max(10, int(12 * self.ui_scale)))
        self.subtitle.setFont(subtitle_font)

        title_box.addWidget(self.title)
        title_box.addWidget(self.subtitle)

        # Right spacer matches logo width so the title stays visually centered.
        right_spacer = QWidget()
        right_spacer.setFixedWidth(self.logo_lbl.sizeHint().width())

        header.addWidget(self.logo_lbl, 0, Qt.AlignVCenter)
        header.addStretch(1)
        header.addLayout(title_box)
        header.addStretch(1)
        header.addWidget(right_spacer)
        outer.addLayout(header)

        # Search row
        row = QHBoxLayout()
        row.addStretch(1)

        self.search_btn = QPushButton("Search")
        self.search_btn.setObjectName("SearchButton")
        self.search_btn.setCursor(Qt.PointingHandCursor)
        self.search_btn.clicked.connect(self.on_search)

        row.addWidget(self.search_btn)
        row.addStretch(1)
        outer.addLayout(row)

        self.status_lbl = QLabel("Not scanning")
        self.status_lbl.setObjectName("Status")
        self.status_lbl.setAlignment(Qt.AlignHCenter)
        status_font = QFont()
        status_font.setPointSize(max(9, int(11 * self.ui_scale)))
        self.status_lbl.setFont(status_font)
        outer.addWidget(self.status_lbl, 0, Qt.AlignHCenter)

        # Cards area
        self.cards_container = QWidget()
        self.cards_container.setObjectName("CardsContainer")
        self.grid = QGridLayout(self.cards_container)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(int(14 * self.ui_scale))
        self.grid.setVerticalSpacing(int(14 * self.ui_scale))
        self.cards_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(self.cards_container, 1)

        # Play button
        self.play_btn = QPushButton("Play")
        self.play_btn.setObjectName("PlayButton")
        self.play_btn.setCursor(Qt.PointingHandCursor)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.on_play)
        outer.addWidget(self.play_btn)

        self._apply_styles()
        self._clear_cards()

        # Use a normal maximized window so the OS title bar + close button remain visible.
        # Fullscreen hides window chrome on Windows.
        self.showMaximized()

    def _load_logo(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            # fallback: show company name if file missing
            self.logo_lbl.setText("Niantong")
            fallback_font = QFont()
            fallback_font.setPointSize(max(12, int(16 * self.ui_scale)))
            fallback_font.setWeight(QFont.Bold)
            self.logo_lbl.setFont(fallback_font)
            return

        # Scale logo to a sensible size relative to screen
        target_h = int(56 * self.ui_scale)
        scaled = pix.scaledToHeight(target_h, Qt.SmoothTransformation)
        self.logo_lbl.setPixmap(scaled)
        self.logo_lbl.setFixedSize(scaled.size())

    def _apply_styles(self):
        # Sizes derived from scale for responsiveness
        search_pad_v = int(10 * self.ui_scale)
        search_pad_h = int(22 * self.ui_scale)
        play_pad_v = int(16 * self.ui_scale)
        play_pad_h = int(18 * self.ui_scale)
        card_radius = int(22 * self.ui_scale)
        btn_radius = int(18 * self.ui_scale)
        search_radius = int(14 * self.ui_scale)

        self.setStyleSheet(
            f"""
            QWidget#Root {{
                background: #F5F5F7;
            }}

            QLabel#Title {{ color: #111827; }}
            QLabel#Subtitle {{ color: #6B7280; }}
            QLabel#Status {{ color: #6B7280; }}

            QPushButton#SearchButton {{
                background: #FFFFFF;
                color: #111827;
                border: 1px solid #E5E7EB;
                border-radius: {search_radius}px;
                padding: {search_pad_v}px {search_pad_h}px;
                font-size: {max(12, int(14 * self.ui_scale))}px;
                font-weight: 700;
                min-width: {int(160 * self.ui_scale)}px;
            }}
            QPushButton#SearchButton:hover {{
                background: #F2F2F7;
            }}
            QPushButton#SearchButton:pressed {{
                background: #EDEDF2;
            }}
            QPushButton#SearchButton:disabled {{
                color: #9CA3AF;
                background: #F9FAFB;
            }}

            QPushButton#PlayButton {{
                background: #007AFF;
                color: #FFFFFF;
                border: none;
                border-radius: {btn_radius}px;
                padding: {play_pad_v}px {play_pad_h}px;
                font-size: {max(14, int(18 * self.ui_scale))}px;
                font-weight: 800;
                min-height: {int(56 * self.ui_scale)}px;
            }}
            QPushButton#PlayButton:pressed {{
                background: #0063CC;
            }}
            QPushButton#PlayButton:disabled {{
                background: #93C5FD;
                color: rgba(255,255,255,0.9);
            }}

            QFrame#PlayerCard {{
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: {card_radius}px;
            }}
            QFrame#PlayerCard[selected="true"] {{
                background: #F2F7FF;
                border: 1px solid #007AFF;
            }}

            QLabel#NameLabel {{ color: #111827; }}
            QLabel#MetaLabel {{ color: #6B7280; }}

            QLabel#Pill {{
                background: #34C759;
                color: white;
                padding: {int(6 * self.ui_scale)}px {int(10 * self.ui_scale)}px;
                border-radius: {int(12 * self.ui_scale)}px;
            }}
            """
        )

    # -----------------------------
    # Card management + dynamic layout
    # -----------------------------
    def _clear_cards(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        self.cards = []
        self.selected_macs.clear()
        self._update_play_state()

    def _layout_cards(self, devices: List[DeviceInfo]):
        self._clear_cards()

        n = min(len(devices), 4)
        devices = devices[:n]

        palette = ["#60A5FA", "#34D399", "#FBBF24", "#F472B6"]

        # Stretch so cards expand with window
        for r in range(3):
            self.grid.setRowStretch(r, 1)
        for c in range(2):
            self.grid.setColumnStretch(c, 1)

        def add_card(i: int, dev: DeviceInfo) -> PlayerCard:
            card = PlayerCard(self.ui_scale)
            card.set_device(dev, palette[i % len(palette)])
            card.toggled.connect(self.on_card_toggled)
            self.cards.append(card)
            return card

        if n == 1:
            self.grid.addWidget(add_card(0, devices[0]), 0, 0, 2, 2)
        elif n == 2:
            self.grid.addWidget(add_card(0, devices[0]), 0, 0, 1, 1)
            self.grid.addWidget(add_card(1, devices[1]), 0, 1, 1, 1)
        elif n == 3:
            self.grid.addWidget(add_card(0, devices[0]), 0, 0, 1, 1)
            self.grid.addWidget(add_card(1, devices[1]), 0, 1, 1, 1)
            self.grid.addWidget(add_card(2, devices[2]), 1, 0, 1, 2)
        else:
            self.grid.addWidget(add_card(0, devices[0]), 0, 0, 1, 1)
            self.grid.addWidget(add_card(1, devices[1]), 0, 1, 1, 1)
            self.grid.addWidget(add_card(2, devices[2]), 1, 0, 1, 1)
            self.grid.addWidget(add_card(3, devices[3]), 1, 1, 1, 1)

        self._update_play_state()

    # -----------------------------
    # Events
    # -----------------------------
    def on_search(self):
        self._clear_cards()  # clear immediately per your requirement
        self.status_lbl.setText("Scanning...")
        self.search_btn.setEnabled(False)

        QTimer.singleShot(450, self._finish_scan)

    def _finish_scan(self):
        found = scan_devices_stub()
        devices = [DeviceInfo(d["name"], d["mac"], d["rssi"]) for d in found][:4]

        if not devices:
            self.status_lbl.setText("No devices found. Try Search again.")
        else:
            self.status_lbl.setText(f"Found {len(devices)} device(s). Tap to connect.")
            self._layout_cards(devices)

        self.search_btn.setEnabled(True)

    def on_card_toggled(self, card: PlayerCard):
        if not card.device:
            return

        mac = card.device.mac
        if card.selected:
            self.selected_macs.add(mac)
            connect_device_stub(mac)
        else:
            self.selected_macs.discard(mac)
            disconnect_device_stub(mac)

        self._update_play_state()

    def _update_play_state(self):
        self.play_btn.setEnabled(len(self.selected_macs) > 0)

    def on_play(self):
        macs = sorted(self.selected_macs)
        print("[PLAY] Connected MACs:", macs)
        self.status_lbl.setText(f"Ready: {len(macs)} player(s) connected.")


if __name__ == "__main__":
    app = QApplication([])
    win = IFocusWindow()
    app.exec()
