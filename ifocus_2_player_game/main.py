from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ifocus_sdk.APIs.calibrationControl import calibrationControl, setWearingCheckEnabled
from ifocus_sdk.APIs.FocusInference import startFocusInference, stopFocusInference
from ifocus_sdk.APIs.trainFocusModel import trainFocusModel


async def main() -> None:
    try:
        # Connect two devices selected by scan (strongest RSSI).
        if not await calibrationControl('connect', nDevices=2, subjectId=["device1", "device2"]):
            print("Connect failed")
            return

        # Disable wearing status check.
        # (Per request: proceed even if electrodes have no contact.)
        setWearingCheckEnabled(False)

        d1, d2 = "device1", "device2"

        # Start 5s FOCUS for both devices.
        await asyncio.gather(
            calibrationControl('start', subjectId=d1, stateLabel='FOCUS'),
            calibrationControl('start', subjectId=d2, stateLabel='FOCUS'),
        )
        await asyncio.sleep(5)
        await asyncio.gather(
            calibrationControl('stop', subjectId=d1),
            calibrationControl('stop', subjectId=d2),
        )

        # Start 5s RELAX for both devices.
        await asyncio.gather(
            calibrationControl('start', subjectId=d1, stateLabel='RELAX'),
            calibrationControl('start', subjectId=d2, stateLabel='RELAX'),
        )
        await asyncio.sleep(5)
        await asyncio.gather(
            calibrationControl('stop', subjectId=d1),
            calibrationControl('stop', subjectId=d2),
        )

        # Train and report accuracy.
        results: dict[str, str] = {}

        def cb(subject: str):
            def _inner(success: bool, message: str, n_samples: int) -> None:
                results[subject] = message
                print(f"[{subject}] {message}")

            return _inner

        trainFocusModel(d1, cb(d1))
        trainFocusModel(d2, cb(d2))

        # Inference (both subjects, both models)
        update_hz = 5.0
        infer_seconds = 10.0

        c1 = await calibrationControl('client', subjectId=d1)
        c2 = await calibrationControl('client', subjectId=d2)
        if c1 is None or c2 is None:
            print("Missing BLE clients for inference")
            return

        def icb(subject: str):
            def _inner(label: str, strength: int, timestamp: str) -> None:
                print(f"[{subject}] {timestamp} label={label} strength={strength}")

            return _inner

        await asyncio.gather(
            startFocusInference(d1, c1, update_hz, icb(d1)),
            startFocusInference(d2, c2, update_hz, icb(d2)),
        )

        await asyncio.sleep(infer_seconds)
        await stopFocusInference()
    finally:
        await calibrationControl('disconnect')
        print("Disconnected")


if __name__ == "__main__":
    asyncio.run(main())

