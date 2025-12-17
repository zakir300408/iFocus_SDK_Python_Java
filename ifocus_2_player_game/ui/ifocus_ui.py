from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPainter, QColor, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QGridLayout,
)


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

        self.setObjectName("PlayerCard")
        self.setCursor(Qt.PointingHandCursor)

        avatar_size = int(72 * ui_scale)

        self.avatar = AvatarWidget(ui_scale, self)
        self.avatar.setFixedSize(avatar_size, avatar_size)

        self.name_lbl = QLineEdit(self)
        self.name_lbl.setObjectName("NameEdit")
        self.name_lbl.setAlignment(Qt.AlignHCenter)
        self.name_lbl.setPlaceholderText("Enter name")

        # Wearing status indicator
        self.wearing_indicator = QLabel(self)
        self.wearing_indicator.setObjectName("WearingIndicator")
        indicator_size = int(12 * ui_scale)
        self.wearing_indicator.setFixedSize(indicator_size, indicator_size)

        self.wearing_text = QLabel("Wearing Status", self)
        self.wearing_text.setObjectName("WearingText")

        self.mac_lbl = QLabel(self)
        self.mac_lbl.setObjectName("MetaLabel")

        self.rssi_lbl = QLabel(self)
        self.rssi_lbl.setObjectName("MetaLabel")

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

        # Wearing status row
        wearing_row = QHBoxLayout()
        wearing_row.setSpacing(int(6 * ui_scale))
        wearing_row.addStretch(1)
        wearing_row.addWidget(self.wearing_indicator, 0, Qt.AlignVCenter)
        wearing_row.addWidget(self.wearing_text, 0, Qt.AlignVCenter)
        wearing_row.addStretch(1)
        layout.addLayout(wearing_row)

        # MAC address
        layout.addWidget(self.mac_lbl, 0, Qt.AlignHCenter)
        layout.addWidget(self.rssi_lbl, 0, Qt.AlignHCenter)
        layout.addStretch(1)
        layout.addWidget(self.pill, 0, Qt.AlignHCenter)

        self._apply_fonts()
        self._refresh_style()

    def _apply_fonts(self):
        def make_font(point: int, bold: bool = False) -> QFont:
            f = QFont()
            f.setPointSize(point)
            if bold:
                f.setWeight(QFont.Bold)
            return f

        self.name_lbl.setFont(make_font(max(12, int(16 * self.ui_scale)), bold=True))

        meta = make_font(max(9, int(11 * self.ui_scale)))
        self.mac_lbl.setFont(meta)
        self.rssi_lbl.setFont(meta)
        self.wearing_text.setFont(meta)

        self.pill.setFont(make_font(max(8, int(10 * self.ui_scale)), bold=True))

    def set_device(self, device: DeviceInfo, accent_hex: str):
        self.device = device
        self.selected = False
        accent = QColor(accent_hex)
        self.avatar.set_accent(accent)
        self.name_lbl.setText(device.name)
        self.mac_lbl.setText(device.mac)
        self.rssi_lbl.setText(f"{device.rssi} dBm")
        self.pill.setVisible(False)
        self.set_wearing_status(True)
        self._refresh_style()

    def set_connection_state(self, state: str):
        """
        state: 'disconnected', 'connecting', 'connected'
        """
        if state == "disconnected":
            self.selected = False
            self.pill.setVisible(False)
        elif state == "connecting":
            self.selected = True
            self.pill.setText("Connecting...")
            self.pill.setProperty("state", "connecting")
            self.pill.setVisible(True)
        elif state == "connected":
            self.selected = True
            self.pill.setText("Connected")
            self.pill.setProperty("state", "connected")
            self.pill.setVisible(True)
        
        self.pill.style().unpolish(self.pill)
        self.pill.style().polish(self.pill)
        self._refresh_style()

    def set_wearing_status(self, is_wearing: bool):
        self.wearing_indicator.setProperty("wearing", "good" if is_wearing else "bad")
        self.wearing_indicator.style().unpolish(self.wearing_indicator)
        self.wearing_indicator.style().polish(self.wearing_indicator)

    def set_selected(self, selected: bool):
        # Deprecated in favor of set_connection_state for logic, 
        # but kept for compatibility if needed, though we should avoid using it for logic now.
        self.selected = selected
        self.pill.setVisible(self.selected)
        self._refresh_style()

    def mousePressEvent(self, event):
        if not self.device:
            return
        if event.button() == Qt.LeftButton:
            # self.set_selected(not self.selected) # Removed immediate toggle
            self.toggled.emit(self)

    def _refresh_style(self):
        self.setProperty("selected", "true" if self.selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)


# -----------------------------
# Main window (pure UI)
# -----------------------------
class IFocusWindow(QMainWindow):
    # UI signals (hooks file listens to these)
    searchRequested = Signal()
    deviceToggled = Signal(str, bool)  # mac, selected
    playRequested = Signal(object)  # list[str]

    def __init__(self):
        super().__init__()

        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        w, h = geo.width(), geo.height()
        self.ui_scale = max(0.85, min(1.45, min(w / 1440.0, h / 900.0)))

        self.setWindowTitle("iFocus Game")
        self.selected_macs: Set[str] = set()
        self.cards: List[PlayerCard] = []

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(
            int(18 * self.ui_scale),
            int(18 * self.ui_scale),
            int(18 * self.ui_scale),
            int(18 * self.ui_scale),
        )
        outer.setSpacing(int(12 * self.ui_scale))

        # Header
        header = QHBoxLayout()
        header.setSpacing(int(12 * self.ui_scale))

        self.logo_lbl = QLabel()
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
        self.search_btn.clicked.connect(self._on_search_clicked)

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
        self.grid = QGridLayout(self.cards_container)
        self.grid.setContentsMargins(0, 0, 0, 0)
        spacing = int(14 * self.ui_scale)
        self.grid.setHorizontalSpacing(spacing)
        self.grid.setVerticalSpacing(spacing)
        outer.addWidget(self.cards_container, 1)

        # Play button
        self.play_btn = QPushButton("Play")
        self.play_btn.setObjectName("PlayButton")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._on_play_clicked)
        outer.addWidget(self.play_btn)

        self._apply_styles()
        self.clear_cards()

        self.showMaximized()

    # -----------------------------
    # Public UI methods (controller calls these)
    # -----------------------------
    def begin_scan_ui(self):
        self.clear_cards()
        self.set_status("Scanning...")
        self.search_btn.setEnabled(False)

    def end_scan_ui(self):
        self.search_btn.setEnabled(True)

    def set_status(self, text: str):
        self.status_lbl.setText(text)

    def set_devices(self, devices: List[DeviceInfo]):
        self._layout_cards(devices)
        
    def set_connection_state(self, mac: str, state: str):
        """
        Update the UI state for a specific device.
        state: 'disconnected', 'connecting', 'connected'
        """
        for c in self.cards:
            if c.device and c.device.mac == mac:
                c.set_connection_state(state)
                if state == "connected":
                    self.selected_macs.add(mac)
                else:
                    self.selected_macs.discard(mac)
                break
        self._update_play_state()

    def clear_cards(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        self.cards = []
        self.selected_macs.clear()
        self._update_play_state()

    def set_wearing_status(self, mac: str, is_wearing: bool):
        for c in self.cards:
            if c.device and c.device.mac == mac:
                c.set_wearing_status(is_wearing)
                break

    # -----------------------------
    # Internal UI slots
    # -----------------------------
    def _on_search_clicked(self):
        # Clear immediately, then hand control to hooks
        self.begin_scan_ui()
        self.searchRequested.emit()

    def _on_play_clicked(self):
        # Collect selected devices with their names
        players_info = []
        for mac in sorted(self.selected_macs):
            for c in self.cards:
                if c.device and c.device.mac == mac:
                    name = c.name_lbl.text().strip() or c.device.name
                    players_info.append({"mac": mac, "name": name})
                    break
        self.playRequested.emit(players_info)

    # -----------------------------
    # Card management + dynamic layout
    # -----------------------------
    def _layout_cards(self, devices: List[DeviceInfo]):
        self.clear_cards()

        n = min(len(devices), 4)
        devices = devices[:n]

        palette = ["#60A5FA", "#34D399", "#FBBF24", "#F472B6"]

        for r in range(2):
            self.grid.setRowStretch(r, 1)
        for c in range(2):
            self.grid.setColumnStretch(c, 1)

        def add_card(i: int, dev: DeviceInfo) -> PlayerCard:
            card = PlayerCard(self.ui_scale)
            card.set_device(dev, palette[i % len(palette)])
            card.toggled.connect(self._on_card_toggled)
            self.cards.append(card)
            return card

        if n == 1:
            self.grid.addWidget(add_card(0, devices[0]), 0, 0, 2, 2)
        elif n == 2:
            self.grid.addWidget(add_card(0, devices[0]), 0, 0)
            self.grid.addWidget(add_card(1, devices[1]), 0, 1)
        elif n == 3:
            self.grid.addWidget(add_card(0, devices[0]), 0, 0)
            self.grid.addWidget(add_card(1, devices[1]), 0, 1)
            self.grid.addWidget(add_card(2, devices[2]), 1, 0, 1, 2)
        else:
            self.grid.addWidget(add_card(0, devices[0]), 0, 0)
            self.grid.addWidget(add_card(1, devices[1]), 0, 1)
            self.grid.addWidget(add_card(2, devices[2]), 1, 0)
            self.grid.addWidget(add_card(3, devices[3]), 1, 1)

        self._update_play_state()

    def _on_card_toggled(self, card: PlayerCard):
        if not card.device:
            return
        mac = card.device.mac
        
        # Determine intent based on current selection state
        # If currently selected (connected/connecting), we want to disconnect (False)
        # If currently unselected, we want to connect (True)
        intent_selected = not card.selected

        # We do NOT update selected_macs here anymore, we wait for connection success
        self.deviceToggled.emit(mac, intent_selected)

    def _update_play_state(self):
        self.play_btn.setEnabled(len(self.selected_macs) > 0)

    # -----------------------------
    # Styling + logo
    # -----------------------------
    def _load_logo(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            self.logo_lbl.setText("Niantong")
            fallback_font = QFont()
            fallback_font.setPointSize(max(12, int(16 * self.ui_scale)))
            fallback_font.setWeight(QFont.Bold)
            self.logo_lbl.setFont(fallback_font)
            return

        target_h = int(56 * self.ui_scale)
        scaled = pix.scaledToHeight(target_h, Qt.SmoothTransformation)
        self.logo_lbl.setPixmap(scaled)
        self.logo_lbl.setFixedSize(scaled.size())

    def _apply_styles(self):
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

            QLineEdit#NameEdit {{
                background: transparent;
                color: #111827;
                border: none;
                border-bottom: 1px solid #E5E7EB;
                padding: {int(4 * self.ui_scale)}px {int(8 * self.ui_scale)}px;
                font-size: {max(12, int(16 * self.ui_scale))}px;
                font-weight: 700;
            }}
            QLineEdit#NameEdit:focus {{
                border-bottom: 2px solid #007AFF;
            }}

            QLabel#WearingIndicator {{
                border-radius: {int(6 * self.ui_scale)}px;
            }}
            QLabel#WearingIndicator[wearing="good"] {{
                background: #34C759;
            }}
            QLabel#WearingIndicator[wearing="bad"] {{
                background: #FF3B30;
            }}

            QLabel#WearingText {{ color: #6B7280; }}
            QLabel#MetaLabel {{ color: #6B7280; }}

            QLabel#Pill {{
                background: #34C759;
                color: white;
                padding: {int(6 * self.ui_scale)}px {int(10 * self.ui_scale)}px;
                border-radius: {int(12 * self.ui_scale)}px;
            }}
            """
        )
