"""Quick iFocus connect smoke test.

Scans briefly, connects to the strongest iFocus device, prints what it connected to,
then disconnects.

Run:
  python ifocus_sdk/quick_connect_demo.py

Notes:
  - Uses the SDK single-device APIs (same pattern as ifocus_sdk/demo.py).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Optional


# Ensure workspace root is importable so `ifocus_sdk` resolves when running directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ifocus_sdk.APIs.FocusComs import startDeviceScan, stopDeviceScan, getCurrentRssi  # noqa: E402
from ifocus_sdk.APIs.calibrationControl import calibrationControl, setWearingCheckEnabled  # noqa: E402


async def quick_connect(scan_seconds: float = 2.5, retries: int = 2) -> tuple[str, str] | None:
    """Scan briefly and connect to the strongest iFocus device.

    Returns (device_id, name) on success, else None.
    """
    found_devices: dict[str, tuple[str, int]] = {}

    def on_device(name: str, device_id: str, rssi: Optional[int]) -> None:
        found_devices[device_id] = (name, int(rssi) if rssi is not None else -999)

    for attempt in range(1, retries + 1):
        if attempt > 1:
            print(f"Retrying scan ({attempt}/{retries})...")

        await startDeviceScan(on_device)
        await asyncio.sleep(scan_seconds)
        await stopDeviceScan()

        if found_devices:
            break

    if not found_devices:
        print("No iFocus devices found.")
        return None

    device_id, (name, rssi) = max(found_devices.items(), key=lambda kv: kv[1][1])
    current_rssi = getCurrentRssi(device_id)

    print(f"Selected: {name}  id={device_id}  rssi={current_rssi if current_rssi is not None else rssi} dBm")
    print(f"Connecting to: {name} ({device_id})...")

    connected = await calibrationControl("connect", device_id)
    if not connected:
        print("Connection failed.")
        return None

    print(f"Connected to: {name} ({device_id})")
    return device_id, name


async def main() -> int:
    parser = argparse.ArgumentParser(description="Quick connect to an iFocus device and print the selected device.")
    parser.add_argument("--scan-seconds", type=float, default=2.5, help="How long to scan per attempt.")
    parser.add_argument("--retries", type=int, default=2, help="How many scan attempts to perform.")
    args = parser.parse_args()

    # Keep this consistent with the game/dev setup: do not block on wearing checks.
    setWearingCheckEnabled(False)

    result = await quick_connect(scan_seconds=args.scan_seconds, retries=args.retries)
    if result is None:
        return 1

    print("Disconnecting...")
    await calibrationControl("disconnect")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
