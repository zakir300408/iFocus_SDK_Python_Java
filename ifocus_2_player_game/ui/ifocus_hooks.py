from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtWidgets import QApplication

from ifocus_ui import IFocusWindow, DeviceInfo


ScanFn = Callable[[], List[DeviceInfo]]
ConnectFn = Callable[[str], None]
DisconnectFn = Callable[[str], None]


class IFocusController:
    def __init__(self, window: IFocusWindow):
        self.window = window

        self.scan_devices_fn: Optional[ScanFn] = None
        self.connect_device_fn: Optional[ConnectFn] = None
        self.disconnect_device_fn: Optional[DisconnectFn] = None

        self.window.searchRequested.connect(self.on_search_requested)
        self.window.deviceToggled.connect(self.on_device_toggled)
        self.window.playRequested.connect(self.on_play_requested)

    def set_scan_devices_callback(self, fn: ScanFn) -> None:
        self.scan_devices_fn = fn

    def set_connection_callbacks(self, connect_fn: ConnectFn, disconnect_fn: DisconnectFn) -> None:
        self.connect_device_fn = connect_fn
        self.disconnect_device_fn = disconnect_fn

    def on_search_requested(self):
        if self.scan_devices_fn is None:
            self.window.set_status("Device scanning is not configured.")
            self.window.end_scan_ui()
            return

        try:
            devices = self.scan_devices_fn()[:4]
        except Exception as exc:
            self.window.set_status(f"Scan failed: {exc}")
            self.window.end_scan_ui()
            return

        if not devices:
            self.window.set_status("No devices found.")
        else:
            self.window.set_status(f"Found {len(devices)} device(s). Tap to connect.")
            self.window.set_devices(devices)

        self.window.end_scan_ui()

    def on_device_toggled(self, mac: str, selected: bool):
        try:
            if selected:
                if self.connect_device_fn is not None:
                    self.connect_device_fn(mac)
            else:
                if self.disconnect_device_fn is not None:
                    self.disconnect_device_fn(mac)
        except Exception as exc:
            self.window.set_status(f"Connection error for {mac}: {exc}")

    def on_play_requested(self, macs: List[str]):
        print("[PLAY] Connected MACs:", macs)
        self.window.set_status(f"Ready: {len(macs)} player(s) connected.")


def main():
    app = QApplication([])
    win = IFocusWindow()
    controller = IFocusController(win)

    # Wire real SDK hooks here.
    # controller.set_scan_devices_callback(your_scan_fn)
    # controller.set_connection_callbacks(your_connect_fn, your_disconnect_fn)

    app.exec()


if __name__ == "__main__":
    main()
