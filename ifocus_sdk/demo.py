"""
iFocus Calibration SDK - Example Usage

This demo shows all available operations:
- BLE device scanning and connection
- Starting/stopping calibration sessions
- Automatic data saving to BDF files
- Retrieving collected data
- Training a subject specific focus model
- Running live focus inference using the trained model
- Stopping and resuming inference
- Clearing all data for a subject

==============================================================================
PUBLIC APIs
==============================================================================

1. BLE Device Management (from APIs.ifocus_ble):
   ---------------------------------------------
   - startDeviceScan(callback) -> bool
       Scan for iFocus BLE devices. Callback receives (name, device_id, rssi).
       Returns True if scan started.
   
   - stopDeviceScan() -> bool
       Stop BLE scanning. Returns True if stopped.
   
   - getCurrentRssi(device_id=None) -> int | None
       Get last known RSSI for a device.

2. Calibration Control (from APIs.ifocus_callibrate):
   --------------------------------------------------
   - calibrationControl(action, deviceId, subjectId, stateLabel, outputDir)
       Unified API for all calibration operations.
       
       Actions:
         'connect'    - Connect to device. Requires deviceId. Returns bool.
         'disconnect' - Disconnect from device. Returns bool.
         'start'      - Start calibration. Requires subjectId, stateLabel. Returns bool.
         'stop'       - Stop calibration, save BDF. Returns Path or None.
         'get'        - Get stored segments. Requires subjectId, stateLabel. Returns List.
         'reset'      - Clear all data + models for subject. Requires subjectId. Returns bool.
   
   - getWearingStatus() -> bool
       Get headset wearing status. Returns True if electrodes have contact.
   
   - setWearingCheckEnabled(enabled: bool) -> None
       Enable/disable wearing check. When disabled, data collection and inference
       proceed even without electrode contact. Developer override for testing.
   
   - isWearingCheckEnabled() -> bool
       Check if wearing verification is enabled (default: True).

3. Model Training (from APIs.trainFocusModel):
   -------------------------------------------
   - trainFocusModel(subjectId, callback)
       Train SVM model from calibration BDF files.
       Callback receives (success: bool, message: str, n_samples: int).

4. Live Inference (from APIs.focus_inference):
   -------------------------------------------
   - startFocusInference(subjectId, client, updateHz, callback) -> bool
       Start live focus inference. Requires trained model.
       Callback receives (label: str, focus_strength: int, timestamp: str).
   
   - stopFocusInference()
       Stop live inference.

==============================================================================
"""

import asyncio
import logging
from typing import Optional
import os
import sys

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ifocus_sdk.APIs.FocusComs import startDeviceScan, stopDeviceScan, getCurrentRssi, _get_client
from ifocus_sdk.APIs.calibrationControl import calibrationControl, getWearingStatus, setWearingCheckEnabled, isWearingCheckEnabled
from APIs.trainFocusModel import trainFocusModel, _get_latest_model_path
from ifocus_sdk.APIs.FocusInference import startFocusInference, stopFocusInference


def _training_callback(success: bool, message: str, n_samples: int) -> None:
    status = "SUCCESS" if success else "FAIL"
    print("\n=== Training result ===")
    print(f"Status        : {status}")
    print(f"Message       : {message}")
    print(f"Samples used  : {n_samples}")
    print("======================\n")


def _focus_callback(label: str, focus_strength: int, timestamp: str) -> None:
    wearing = getWearingStatus()
    wearing_indicator = "✓" if wearing else "⚠"
    print(f"[FOCUS] {timestamp}  label={label:>6}  strength={focus_strength:3d}  wearing={wearing_indicator}")


async def scan_and_connect(found_devices: dict) -> tuple[str, str] | None:
    """Scan for iFocus devices and connect to the strongest one.
    
    Returns (device_id, name) or None if failed.
    """
    print("Scanning for iFocus devices...")
    
    def on_device(name: str, device_id: str, rssi: Optional[int]):
        found_devices[device_id] = (name, rssi or -999)
    
    max_retries = 3
    scan_duration = 5.0
    
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"Retrying scan (attempt {attempt + 1}/{max_retries})...")
        
        await startDeviceScan(on_device)
        await asyncio.sleep(scan_duration)
        await stopDeviceScan()
        
        if found_devices:
            break
    
    if not found_devices:
        print("No devices found after multiple scans.")
        return None
    
    # Pick device with strongest RSSI
    device_id, (name, rssi) = max(found_devices.items(), key=lambda kv: kv[1][1])
    current_rssi = getCurrentRssi(device_id)
    print(f"Selected {name} (RSSI: {current_rssi} dBm)")
    
    # Connect
    print(f"Connecting to {name}...")
    if not await calibrationControl('connect', device_id):
        print("Connection failed.")
        return None
    
    print(f"Connected to {name}\n")
    return device_id, name


async def check_wearing_status():
    """Check and display wearing status for a few seconds.
    
    Returns True if headset is being worn, False otherwise.
    """
    print("\n--- Checking wearing status (3 seconds) ---")
    print("Please ensure the headset is properly worn...")
    print(f"Wearing check enabled: {isWearingCheckEnabled()}")
    
    wearing_count = 0
    total_checks = 6
    
    for i in range(total_checks):
        wearing = getWearingStatus()
        if wearing:
            wearing_count += 1
        status = "✓ WEARING" if wearing else "✗ NOT WEARING"
        print(f"  [{i*0.5:.1f}s] {status}")
        await asyncio.sleep(0.5)
    
    final_status = getWearingStatus()
    if not final_status:
        print("\n⚠ Warning: Headset may not be worn properly. Please check electrode contact.")
        print("  Data collection will be paused until electrodes have contact.")
        print("  Use setWearingCheckEnabled(False) to bypass this check (dev only).")
    else:
        print("\n✓ Headset is properly worn. Data collection is active.")
    return final_status


async def demo_calibration():
    """
    PART 1: Calibration + Training
    
    1. Scan and connect to iFocus headset
    2. Collect FOCUS / RELAX data in alternating blocks
    3. Save data automatically to BDF files
    4. Train a subject specific SVM focus model
    5. Disconnect (keeping data and model intact)
    """
    subject_id = "subject1"
    found_devices: dict[str, tuple[str, int]] = {}
    
    # 1. Scan and connect
    result = await scan_and_connect(found_devices)
    if not result:
        return
    device_id, name = result
    
    # 1.5. Check wearing status before calibration
    await check_wearing_status()
    
    # 2. Run calibration in alternating blocks: FOCUS -> RELAX -> FOCUS -> RELAX
    state_sequence = [
        ("FOCUS", 10),
        ("RELAX", 10),
        ("FOCUS", 10),
        ("RELAX", 10),
    ]
    
    for state_label, duration in state_sequence:
        print(f"\n=== {state_label} calibration ({duration}s) ===")
        
        started = await calibrationControl('start', None, subject_id, state_label)
        if not started:
            print("✗ Start failed")
            continue
        
        print(f"Recording {state_label}... please maintain state.")
        
        # Monitor wearing status during recording
        for elapsed in range(duration):
            wearing = getWearingStatus()
            if not wearing and isWearingCheckEnabled():
                print(f"  [{elapsed}s] ⚠ Not wearing - data paused")
            else:
                print(f"  [{elapsed}s] ✓ Recording...")
            await asyncio.sleep(1)
        
        # Stop calibration and save data
        bdf_path = await calibrationControl('stop')
        if bdf_path:
            print(f"✓ Saved: {bdf_path.name}")
        else:
            print("✗ Save failed")
    
    # 3. Show calibration summary
    print("\n--- Calibration summary ---")
    ordered_labels = list(dict.fromkeys(label for label, _ in state_sequence))
    for state_label in ordered_labels:
        segments = await calibrationControl('get', None, subject_id, state_label)
        if segments:
            total_samples = sum(len(frame["eeg"]) for seg in segments for frame in seg)
            print(f"  {state_label}: {total_samples} samples across {len(segments)} segment(s)")
        else:
            print(f"  {state_label}: 0 samples")
    
    # 4. Train focus model
    print("\n--- Training subject specific focus model ---")
    trainFocusModel(subject_id, _training_callback)
    
    # 5. Disconnect (keep data and model)
    print("\n--- Disconnecting (data and model preserved) ---")
    await calibrationControl('disconnect')
    print("Disconnected. Calibration data and trained model are saved.\n")
    
    return subject_id, found_devices


async def demo_inference(subject_id: str, found_devices: dict):
    """
    PART 2: Load Model + Live Inference
    
    1. Check if model exists for subject
    2. Reconnect to iFocus headset
    3. Run live focus inference
    4. Pause inference (user input)
    5. Resume inference
    6. Stop and cleanup
    """
    
    # 1. Check if model exists
    print("=" * 60)
    print("PART 2: Loading existing model and running inference")
    print("=" * 60)
    
    model_path = _get_latest_model_path(subject_id)
    if model_path is None:
        print(f"No trained model found for '{subject_id}'. Run calibration first.")
        return
    
    print(f"Found trained model: {model_path.name}")
    
    # 2. Reconnect to device
    result = await scan_and_connect(found_devices)
    if not result:
        return
    device_id, name = result
    
    # 3. Run live inference (first session)
    print("\n--- Live focus inference (5 seconds) ---")
    update_hz = 2.0
    ble_client = _get_client()
    
    try:
        started = await startFocusInference(subject_id, ble_client, update_hz, _focus_callback)
        if not started:
            print("Failed to start focus inference.")
            return
        
        await asyncio.sleep(5.0)
        
        # 4. Pause inference
        await stopFocusInference()
        print("\n--- Inference paused ---")
        
        # Wait for user input
        print("\nPress Enter to resume inference...")
        await asyncio.get_event_loop().run_in_executor(None, input)
        
        # 5. Resume inference (second session)
        print("\n--- Resuming inference (5 seconds) ---")
        started = await startFocusInference(subject_id, ble_client, update_hz, _focus_callback)
        if not started:
            print("Failed to resume focus inference.")
        else:
            await asyncio.sleep(5.0)
        
    finally:
        await stopFocusInference()
        print("\nFocus inference stopped.")
    
    # 6. Cleanup - delete everything
    print("\n--- Cleanup: Deleting all data and models ---")
    if await calibrationControl('reset', None, subject_id):
        print(f"✓ Cleared all calibration data and models for {subject_id}")
    else:
        print(f"✗ Failed to clear data for {subject_id}")
    
    # Disconnect
    print("\nDisconnecting from device...")
    await calibrationControl('disconnect')
    print("Done.")


async def main():
    """
    Complete demo workflow:
    
    PART 1: Calibration + Training
      - Connect, collect data, train model, disconnect
    
    PART 2: Load Model + Inference  
      - Reconnect, load existing model, run inference with pause/resume, cleanup
    """
    print("=" * 60)
    print("iFocus SDK Demo - Calibration and Inference Workflow")
    print("=" * 60)
    
    # Part 1: Calibration and training
    print("\nPART 1: Calibration and Model Training")
    print("-" * 40)
    
    result = await demo_calibration()
    if result is None:
        print("Calibration failed. Exiting.")
        return
    
    subject_id, found_devices = result
    
    # Simulate "restarting" - wait for user
    print("\n" + "=" * 60)
    print("Simulating application restart...")
    print("Press Enter to continue to Part 2 (inference with saved model)...")
    await asyncio.get_event_loop().run_in_executor(None, input)
    
    # Part 2: Load model and run inference
    await demo_inference(subject_id, found_devices)
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
