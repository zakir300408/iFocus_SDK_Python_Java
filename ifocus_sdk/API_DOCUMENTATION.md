# iFocus SDK API Documentation

This document provides comprehensive documentation for the iFocus SDK Python API, which enables BLE communication with iFocus EEG headsets, calibration data collection, model training, and real-time focus inference.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Module Overview](#module-overview)
4. [API Reference](#api-reference)
   - [FocusComs - BLE Device Management](#focuscoms---ble-device-management)
   - [calibrationControl - Calibration & Data Collection](#calibrationcontrol---calibration--data-collection)
   - [trainFocusModel - Model Training](#trainfocusmodel---model-training)
   - [FocusInference - Live Inference](#focusinference---live-inference)
5. [Data Flow](#data-flow)
6. [Error Handling](#error-handling)
7. [Examples](#examples)

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `bleak` - BLE communication
- `numpy` - Numerical computing
- `scipy` - Signal processing
- `pyedflib` - EDF/BDF file handling
- `scikit-learn` - Machine learning
- `joblib` - Model persistence

---

## Quick Start

```python
import asyncio
from ifocus_sdk.APIs.FocusComs import startDeviceScan, stopDeviceScan, connect, disconnect
from ifocus_sdk.APIs.calibrationControl import calibrationControl
from ifocus_sdk.APIs.trainFocusModel import trainFocusModel
from ifocus_sdk.APIs.FocusInference import startFocusInference, stopFocusInference

async def main():
    # 1. Scan for devices
    devices = []
    def on_device(name, device_id, rssi):
        print(f"Found: {name} ({device_id}) RSSI: {rssi}")
        devices.append(device_id)
    
    await startDeviceScan(on_device)
    await asyncio.sleep(5)  # Scan for 5 seconds
    await stopDeviceScan()
    
    # 2. Connect to first device
    if devices:
        await calibrationControl('connect', deviceId=devices[0])
        
        # 3. Collect calibration data for BOTH states (minimum 2 required)
        # Record FOCUS state
        await calibrationControl('start', subjectId='user1', stateLabel='FOCUS')
        await asyncio.sleep(60)  # Record for 60 seconds
        await calibrationControl('stop')
        
        # Record RELAX state (REQUIRED - model needs at least 2 different states)
        await calibrationControl('start', subjectId='user1', stateLabel='RELAX')
        await asyncio.sleep(60)  # Record for 60 seconds
        await calibrationControl('stop')
        
        # 4. Train model (requires calibration data from 2+ states)
        def on_train_complete(success, message, n_samples):
            print(f"Training: {message}")
        trainFocusModel('user1', on_train_complete)
        
        # 5. Run inference
        def on_prediction(label, strength, timestamp):
            print(f"{timestamp}: {label} (strength: {strength})")
        
        client = _get_client()  # Get BleakClient
        await startFocusInference('user1', client, updateHz=2.0, callback=on_prediction)
        
        # Run for some time...
        await asyncio.sleep(60)
        
        # 6. Stop inference when done
        await stopFocusInference()
        
        # 7. Disconnect
        await calibrationControl('disconnect')

asyncio.run(main())
```

---

## Module Overview

| Module | Purpose |
|--------|---------|
| `FocusComs` | BLE device discovery, connection management |
| `calibrationControl` | EEG data collection for calibration |
| `trainFocusModel` | Train subject-specific focus classification models |
| `FocusInference` | Real-time focus state prediction |
| `iFocusParser` | Low-level BLE packet parsing |
| `_preprocess_extract_features` | EEG signal processing and feature extraction |

---

## API Reference

### FocusComs - BLE Device Management

Module for discovering and connecting to iFocus BLE devices.

#### `startDeviceScan(callback)`

Start scanning for iFocus BLE devices.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `callback` | `Callable[[str, str, Optional[int]], None]` | Function called for each discovered device |

**Callback Arguments:**
- `device_name` (str): Name of the device (e.g., "iFocus_XXXX")
- `device_id` (str): BLE address/identifier
- `rssi` (Optional[int]): Signal strength in dBm

**Returns:** `bool` - `True` if scan started, `False` if already scanning

**Example:**
```python
async def scan_for_devices():
    found_devices = []
    
    def on_device_found(name, device_id, rssi):
        print(f"Discovered: {name} at {device_id} (RSSI: {rssi} dBm)")
        found_devices.append({'name': name, 'id': device_id, 'rssi': rssi})
    
    await startDeviceScan(on_device_found)
    await asyncio.sleep(10)  # Scan for 10 seconds
    await stopDeviceScan()
    
    return found_devices
```

---

#### `stopDeviceScan()`

Stop the BLE device scan.

**Returns:** `bool` - `True` if stopped, `False` if not scanning

**Example:**
```python
await stopDeviceScan()
```

---

#### `connect(device_id, auto_reconnect, callback)`

Connect to an iFocus BLE device.

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `device_id` | `str` | *required* | BLE address of the device |
| `auto_reconnect` | `bool` | `False` | Automatically reconnect on disconnection |
| `callback` | `Optional[Callable[[bool], None]]` | `None` | Called with connection status changes |

**Returns:** `bool` - `True` if connected successfully

**Example:**
```python
def on_connection_change(connected):
    if connected:
        print("Connected to device!")
    else:
        print("Disconnected from device")

success = await connect(
    device_id="AA:BB:CC:DD:EE:FF",
    auto_reconnect=True,
    callback=on_connection_change
)
```

---

#### `disconnect(device_id)`

Disconnect from a BLE device.

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `device_id` | `Optional[str]` | `None` | Device to disconnect. If `None`, disconnects current device |

**Returns:** `bool` - `True` if disconnected successfully

**Example:**
```python
await disconnect("AA:BB:CC:DD:EE:FF")
# or disconnect from current device
await disconnect()
```

---

#### `getCurrentRssi(device_id)`

Get the last known RSSI (signal strength) for a device.

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `device_id` | `Optional[str]` | `None` | Device address. If `None` and only one device cached, returns that |

**Returns:** `Optional[int]` - RSSI in dBm, or `None` if not available

**Example:**
```python
rssi = getCurrentRssi("AA:BB:CC:DD:EE:FF")
if rssi:
    print(f"Signal strength: {rssi} dBm")
```

---

#### `_get_client()`

Get the current BleakClient instance (internal use, but exposed for advanced usage).

**Returns:** `Optional[BleakClient]` - The connected client or `None`

---

### calibrationControl - Calibration & Data Collection

Module for collecting EEG calibration data and managing calibration sessions.

#### `calibrationControl(action, deviceId, subjectId, stateLabel, outputDir)`

Unified API for all calibration operations.

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `action` | `str` | *required* | Operation to perform (see below) |
| `deviceId` | `Optional[str]` | `None` | BLE device address (for 'connect') |
| `subjectId` | `Optional[str]` | `None` | Subject identifier |
| `stateLabel` | `Optional[str]` | `None` | Mental state label ('FOCUS', 'RELAX', etc.) |
| `outputDir` | `Optional[str\|Path]` | `None` | Custom output directory for BDF files |

**Actions:**

| Action | Description | Required Params | Returns |
|--------|-------------|-----------------|---------|
| `'connect'` | Connect to BLE device | `deviceId` | `bool` |
| `'disconnect'` | Disconnect from device | - | `bool` |
| `'start'` | Start recording calibration data | `subjectId`, `stateLabel` | `bool` |
| `'stop'` | Stop recording and save BDF file | `outputDir` (optional) | `Optional[Path]` |
| `'get'` | Get stored calibration segments | `subjectId`, `stateLabel` | `List[Segment]` |
| `'reset'` | Clear all data for a subject | `subjectId` | `bool` |

**Action Details:**

**`'connect'`** - Establishes BLE connection to the iFocus headset.

**`'disconnect'`** - Closes BLE connection and stops any active streaming.

**`'start'`** - Begins recording EEG data for calibration:
- Starts BLE notifications to receive EEG data from headset
- Associates incoming data with the specified `subjectId` and `stateLabel`
- Data is buffered in memory until `'stop'` is called
- If headset is not worn (wearing check enabled), data collection pauses automatically

**`'stop'`** - **MUST be called after each recording session.** This action:
1. Stops the current recording session
2. Saves all recorded EEG data to a BDF file at `data/calibration/cal_{subjectId}_{stateLabel}_{timestamp}_{duration}s.bdf`
3. Stops BLE data streaming from the headset
4. Returns the `Path` to the saved file (or `None` if no data was recorded)

> ⚠️ **Important:** If you call `'start'` again without calling `'stop'`, the previous session data will be lost! The BDF files are required for model training.

**`'get'`** - Retrieves calibration data segments from memory (useful for inspection before saving).

**`'reset'`** - Clears all calibration data and model files for a subject (both in memory and on disk).

**Example - Complete Calibration Session:**
```python
async def run_calibration(device_id: str, subject_id: str):
    # Connect to device
    connected = await calibrationControl('connect', deviceId=device_id)
    if not connected:
        print("Failed to connect")
        return
    
    # Record FOCUS state (60 seconds)
    print("Recording FOCUS state...")
    await calibrationControl('start', subjectId=subject_id, stateLabel='FOCUS')
    await asyncio.sleep(60)
    focus_file = await calibrationControl('stop')
    print(f"Saved: {focus_file}")
    
    # Record RELAX state (60 seconds)
    print("Recording RELAX state...")
    await calibrationControl('start', subjectId=subject_id, stateLabel='RELAX')
    await asyncio.sleep(60)
    relax_file = await calibrationControl('stop')
    print(f"Saved: {relax_file}")
    
    # Disconnect
    await calibrationControl('disconnect')
    
    return [focus_file, relax_file]
```

---

#### `iFocusRealTimeReader(client, chunk_sec)`

Async generator for streaming real-time EEG data.

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `client` | `BleakClient` | *required* | Connected BleakClient instance |
| `chunk_sec` | `float` | `1.0` | Approximate chunk duration in seconds |

**Yields:** `np.ndarray` - 1D array of EEG samples

**Example:**
```python
async def stream_eeg(client):
    async for chunk in iFocusRealTimeReader(client, chunk_sec=0.5):
        print(f"Received {len(chunk)} samples")
        # Process chunk...
```

---

#### `getWearingStatus()`

Check if the headset is being worn (electrodes have contact).

**Returns:** `bool` - `True` if electrodes have contact, `False` otherwise

**Example:**
```python
if getWearingStatus():
    print("Headset is worn properly")
else:
    print("Please put on the headset")
```

---

#### `setWearingCheckEnabled(enabled)`

Enable or disable wearing status verification.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `enabled` | `bool` | `True` to require wearing (default), `False` to bypass |

**Note:** Disabling is intended for developer testing only. Data quality may be affected.

**Example:**
```python
# For testing without wearing the headset
setWearingCheckEnabled(False)

# Re-enable for production
setWearingCheckEnabled(True)
```

---

#### `isWearingCheckEnabled()`

Check if wearing verification is enabled.

**Returns:** `bool` - `True` if enabled, `False` if bypassed

---

### trainFocusModel - Model Training

Module for training subject-specific focus classification models.

#### `trainFocusModel(subjectId, callback)`

Train an SVM model from calibration BDF files.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `subjectId` | `str` | Subject identifier (must match calibration data) |
| `callback` | `Callable[[bool, str, int], None]` | Called when training completes |

**Callback Arguments:**
- `success` (bool): `True` if training succeeded
- `message` (str): Status message or error description
- `n_samples` (int): Number of samples used for training

**Model Output:**
- Saved to `data/calibration/model_{subjectId}_{timestamp}.joblib`
- Contains trained SVM pipeline, feature names, and metadata

**Example:**
```python
def on_training_complete(success, message, n_samples):
    if success:
        print(f"✓ Model trained with {n_samples} samples")
        print(f"  {message}")
    else:
        print(f"✗ Training failed: {message}")

# Train after collecting calibration data
trainFocusModel('user1', on_training_complete)
```

**Requirements:**
- **At least 2 different mental states** (e.g., FOCUS and RELAX) - The SVM classifier needs multiple classes to train
- Calibration BDF files must exist in `data/calibration/` directory (created by `calibrationControl('stop')`)
- Minimum recommended: 60 seconds of recording per state

> ⚠️ **Common Error:** If you only record one state (e.g., only FOCUS), training will fail with: `"Need at least 2 states, found: ['FOCUS']"`

---

#### `_get_latest_model_path(subject_id)`

Get the path to the most recent model file for a subject.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `subject_id` | `str` | Subject identifier |

**Returns:** `Optional[Path]` - Path to model file, or `None` if not found

---

### FocusInference - Live Inference

Module for real-time focus state prediction using trained models.

#### `startFocusInference(subjectId, client, updateHz, callback)`

Start live focus inference.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `subjectId` | `str` | Subject with a trained model |
| `client` | `BleakClient` | Connected BleakClient instance |
| `updateHz` | `float` | Prediction updates per second (e.g., 2.0 = every 500ms) |
| `callback` | `Callable[[str, int, str], None]` | Called with each prediction |

**Callback Arguments:**
- `label` (str): Predicted state ('FOCUS', 'RELAX', etc.)
- `strength` (int): Focus strength score (1-100)
- `timestamp` (str): ISO format UTC timestamp

**Returns:** `bool` - `True` if started successfully

**Raises:**
- `RuntimeError`: If inference already running or no model found
- `ValueError`: If `updateHz <= 0` or `client` is `None`

**Example:**
```python
async def run_inference(subject_id: str, client):
    predictions = []
    
    def on_prediction(label, strength, timestamp):
        predictions.append({
            'label': label,
            'strength': strength,
            'timestamp': timestamp
        })
        
        # Visual indicator
        bar = '█' * (strength // 5)
        print(f"[{timestamp}] {label}: {strength:3d} |{bar}")
    
    # Start inference at 2 Hz
    await startFocusInference(
        subjectId=subject_id,
        client=client,
        updateHz=2.0,
        callback=on_prediction
    )
    
    # Run for 60 seconds
    await asyncio.sleep(60)
    
    # Stop inference
    await stopFocusInference()
    
    return predictions
```

---

#### `stopFocusInference()`

Stop live focus inference.

**Returns:** `None`

**Example:**
```python
await stopFocusInference()
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           iFocus SDK Data Flow                          │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐      BLE       ┌──────────────┐
  │ iFocus       │ ─────────────► │ FocusComs    │
  │ Headset      │                │ (connect)    │
  └──────────────┘                └──────┬───────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
          ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
          │ calibration     │  │ iFocusParser    │  │ iFocusReal      │
          │ Control         │  │ (parse BLE      │  │ TimeReader      │
          │ (collect data)  │  │  packets)       │  │ (stream EEG)    │
          └────────┬────────┘  └─────────────────┘  └────────┬────────┘
                   │                                         │
                   ▼                                         │
          ┌─────────────────┐                                │
          │ Save BDF Files  │                                │
          │ data/calibration│                                │
          └────────┬────────┘                                │
                   │                                         │
                   ▼                                         │
          ┌─────────────────┐                                │
          │ trainFocusModel │                                │
          │ (train SVM)     │                                │
          └────────┬────────┘                                │
                   │                                         │
                   ▼                                         ▼
          ┌─────────────────┐                       ┌─────────────────┐
          │ Save Model      │ ────────────────────► │ FocusInference  │
          │ .joblib file    │                       │ (predictions)   │
          └─────────────────┘                       └────────┬────────┘
                                                             │
                                                             ▼
                                                    ┌─────────────────┐
                                                    │ Callback with   │
                                                    │ label, strength │
                                                    │ timestamp       │
                                                    └─────────────────┘
```

---

## Error Handling

### Common Exceptions

| Exception | Cause | Solution |
|-----------|-------|----------|
| `ValueError("'connect' requires deviceId")` | Missing device ID | Provide `deviceId` parameter |
| `ValueError("'start' requires subjectId and stateLabel")` | Missing calibration params | Provide both parameters |
| `RuntimeError("Not connected. Use 'connect' first.")` | Starting calibration without connection | Call `connect` first |
| `RuntimeError("No model found for 'X'")` | No trained model exists | Run `trainFocusModel` first |
| `RuntimeError("Inference already running.")` | Starting inference twice | Call `stopFocusInference` first |
| `RuntimeError("Headset is not being worn")` | Wearing check failed | Wear headset or disable check |

### Best Practices

```python
import asyncio
import logging

# Enable logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def safe_calibration(device_id, subject_id, state_label):
    try:
        # Connect with timeout
        connected = await asyncio.wait_for(
            calibrationControl('connect', deviceId=device_id),
            timeout=30.0
        )
        
        if not connected:
            raise RuntimeError("Connection failed")
        
        # Check wearing status before starting
        if not getWearingStatus():
            logger.warning("Headset not worn - waiting...")
            await asyncio.sleep(5)
            
            if not getWearingStatus():
                raise RuntimeError("Please wear the headset")
        
        # Start calibration
        await calibrationControl('start', 
                                 subjectId=subject_id, 
                                 stateLabel=state_label)
        
        # Record for 60 seconds
        await asyncio.sleep(60)
        
        # Stop and save
        bdf_path = await calibrationControl('stop')
        logger.info(f"Saved calibration data to {bdf_path}")
        
        return bdf_path
        
    except asyncio.TimeoutError:
        logger.error("Connection timeout")
        return None
        
    except Exception as e:
        logger.exception(f"Calibration failed: {e}")
        return None
        
    finally:
        # Always cleanup
        await calibrationControl('disconnect')
```

---

## Examples

### Example 1: Full Workflow Script

```python
"""Complete iFocus SDK workflow example."""
import asyncio
from ifocus_sdk.APIs.FocusComs import (
    startDeviceScan, stopDeviceScan, connect, disconnect, _get_client
)
from ifocus_sdk.APIs.calibrationControl import (
    calibrationControl, getWearingStatus, setWearingCheckEnabled
)
from ifocus_sdk.APIs.trainFocusModel import trainFocusModel
from ifocus_sdk.APIs.FocusInference import startFocusInference, stopFocusInference


async def main():
    SUBJECT_ID = "demo_user"
    CALIBRATION_DURATION = 60  # seconds per state
    
    # ===== Step 1: Discover Devices =====
    print("Scanning for iFocus devices...")
    devices = []
    
    def on_device(name, device_id, rssi):
        print(f"  Found: {name} ({device_id})")
        devices.append(device_id)
    
    await startDeviceScan(on_device)
    await asyncio.sleep(5)
    await stopDeviceScan()
    
    if not devices:
        print("No devices found!")
        return
    
    device_id = devices[0]
    print(f"\nUsing device: {device_id}")
    
    # ===== Step 2: Connect =====
    print("\nConnecting...")
    await calibrationControl('connect', deviceId=device_id)
    print("Connected!")
    
    # ===== Step 3: Calibration =====
    states = ['FOCUS', 'RELAX']
    
    for state in states:
        print(f"\n{'='*50}")
        print(f"Recording {state} state for {CALIBRATION_DURATION} seconds")
        print(f"{'='*50}")
        
        # Wait for headset to be worn
        while not getWearingStatus():
            print("Please wear the headset...")
            await asyncio.sleep(2)
        
        await calibrationControl('start', 
                                 subjectId=SUBJECT_ID, 
                                 stateLabel=state)
        
        # Show countdown
        for remaining in range(CALIBRATION_DURATION, 0, -10):
            print(f"  {remaining} seconds remaining...")
            await asyncio.sleep(10)
        
        bdf_path = await calibrationControl('stop')
        print(f"  Saved: {bdf_path}")
    
    # ===== Step 4: Train Model =====
    print("\n" + "="*50)
    print("Training model...")
    print("="*50)
    
    training_done = asyncio.Event()
    training_result = {}
    
    def on_training(success, message, n_samples):
        training_result['success'] = success
        training_result['message'] = message
        training_result['n_samples'] = n_samples
        training_done.set()
    
    trainFocusModel(SUBJECT_ID, on_training)
    await training_done.wait()
    
    if not training_result['success']:
        print(f"Training failed: {training_result['message']}")
        await calibrationControl('disconnect')
        return
    
    print(f"Model trained with {training_result['n_samples']} samples")
    print(training_result['message'])
    
    # ===== Step 5: Live Inference =====
    print("\n" + "="*50)
    print("Starting live inference (30 seconds)...")
    print("="*50)
    
    client = _get_client()
    
    def on_prediction(label, strength, timestamp):
        bar = '█' * (strength // 5) + '░' * (20 - strength // 5)
        print(f"  {label:8s} [{bar}] {strength:3d}%")
    
    await startFocusInference(
        subjectId=SUBJECT_ID,
        client=client,
        updateHz=2.0,
        callback=on_prediction
    )
    
    await asyncio.sleep(30)
    await stopFocusInference()
    
    # ===== Cleanup =====
    print("\nDisconnecting...")
    await calibrationControl('disconnect')
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2: Inference-Only Mode

```python
"""Run inference with a pre-trained model."""
import asyncio
from ifocus_sdk.APIs.FocusComs import startDeviceScan, stopDeviceScan, connect, _get_client
from ifocus_sdk.APIs.calibrationControl import calibrationControl
from ifocus_sdk.APIs.FocusInference import startFocusInference, stopFocusInference


async def inference_session(subject_id: str, duration_seconds: int = 300):
    """Run inference for a specified duration."""
    
    # Find and connect to device
    device_id = None
    
    def on_device(name, did, rssi):
        nonlocal device_id
        if device_id is None:
            device_id = did
    
    await startDeviceScan(on_device)
    await asyncio.sleep(3)
    await stopDeviceScan()
    
    if not device_id:
        raise RuntimeError("No iFocus device found")
    
    await calibrationControl('connect', deviceId=device_id)
    client = _get_client()
    
    # Collect predictions
    predictions = []
    
    def on_prediction(label, strength, timestamp):
        predictions.append({
            'label': label,
            'strength': strength,
            'timestamp': timestamp
        })
    
    try:
        await startFocusInference(
            subjectId=subject_id,
            client=client,
            updateHz=1.0,
            callback=on_prediction
        )
        
        await asyncio.sleep(duration_seconds)
        
    finally:
        await stopFocusInference()
        await calibrationControl('disconnect')
    
    # Compute statistics
    focus_preds = [p for p in predictions if p['label'] == 'FOCUS']
    avg_strength = sum(p['strength'] for p in predictions) / len(predictions)
    focus_ratio = len(focus_preds) / len(predictions) * 100
    
    print(f"\nSession Summary:")
    print(f"  Duration: {duration_seconds} seconds")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Focus time: {focus_ratio:.1f}%")
    print(f"  Average strength: {avg_strength:.1f}")
    
    return predictions


if __name__ == "__main__":
    asyncio.run(inference_session("user1", duration_seconds=60))
```

---

## File Structure

```
ifocus_sdk/
├── APIs/
│   ├── __init__.py
│   ├── FocusComs.py           # BLE device management
│   ├── calibrationControl.py   # Calibration & data collection
│   ├── trainFocusModel.py      # Model training
│   ├── FocusInference.py       # Live inference
│   ├── iFocusParser.py         # BLE packet parsing
│   ├── _preprocess_extract_features.py  # Signal processing
│   └── __epoch_maker.py        # Sliding window epochs
├── demo.py                     # Example usage
└── data/
    └── calibration/            # BDF files and models
        ├── cal_user1_FOCUS_*.bdf
        ├── cal_user1_RELAX_*.bdf
        └── model_user1_*.joblib
```

---

## Constants & Configuration

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_SAMPLE_RATE` | 500 Hz | EEG sampling frequency - **DO NOT CHANGE** (hardware-defined by iFocus headset) |
| `TRAIN_WIN_SEC` | 3.0 s | Training epoch window size - each sample uses 3 seconds of EEG data |
| `TRAIN_UPDATE_HZ` | 2.0 Hz | Training epoch extraction rate - extracts 2 overlapping epochs per second from calibration data |
| `CALIBRATION_DATA_DIR` | `data/calibration/` | Default data storage path |

### Understanding Update Hz (Prediction Rate)

The `updateHz` parameter in `startFocusInference()` and `TRAIN_UPDATE_HZ` control how many predictions you get per second:

| updateHz Value | Meaning | Use Case |
|----------------|---------|----------|
| `1.0` | 1 prediction per second | Low CPU usage, smooth UI updates |
| `2.0` | 2 predictions per second (every 500ms) | **Recommended** - good balance |
| `4.0` | 4 predictions per second (every 250ms) | More responsive, higher CPU usage |

**How it works:**
- The EEG data is processed in sliding windows of `TRAIN_WIN_SEC` (3 seconds)
- At `updateHz=2.0`, windows overlap by 50% - a new prediction is made every 0.5 seconds
- Higher `updateHz` = more overlap = more predictions = faster response but similar accuracy

```
Example with updateHz=2.0 (2 predictions/second):

Time:     0s      1s      2s      3s      4s      5s
          |-------|-------|-------|-------|-------|
Window 1: [=======3 sec=======]
                    ↓ Prediction 1
Window 2:     [=======3 sec=======]
                        ↓ Prediction 2
Window 3:         [=======3 sec=======]
                            ↓ Prediction 3
...and so on
```

> ⚠️ **Important:** Do not modify `DEFAULT_SAMPLE_RATE` (500 Hz). This is the hardware sampling rate of the iFocus headset and cannot be changed.

---

## Changelog

- **v1.0.0** - Initial release
  - BLE device discovery and connection
  - EEG calibration data collection
  - SVM model training
  - Real-time focus inference
  - Wearing status detection

---

## Support

For issues and feature requests, please open an issue on the GitHub repository.
