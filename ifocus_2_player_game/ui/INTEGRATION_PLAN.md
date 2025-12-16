# iFocus UI Integration Plan

## Overview
This document outlines the plan to integrate the iFocus SDK with the landing page UI to make it fully functional.

---

## Available SDK APIs

### 1. **Device Scanning** (`FocusComs.py`)
```python
from ifocus_sdk.APIs.FocusComs import startDeviceScan, stopDeviceScan, getCurrentRssi

# Start scanning
async def on_device_found(name: str, device_id: str, rssi: int):
    print(f"Found: {name} ({device_id}) RSSI: {rssi}")

await startDeviceScan(on_device_found)
await asyncio.sleep(5)  # Scan for 5 seconds
await stopDeviceScan()

# Get RSSI for a device
rssi = getCurrentRssi(device_id)
```

### 2. **Device Connection** (`calibrationControl.py`)
```python
from ifocus_sdk.APIs.calibrationControl import calibrationControl

# Connect to specific device
success = await calibrationControl('connect', deviceId="device_mac_address")

# Connect to multiple devices (auto-scan and select strongest RSSI)
success = await calibrationControl('connect', nDevices=2, subjectId=["player1", "player2"])

# Disconnect
await calibrationControl('disconnect')

# Get BleakClient for a device
client = await calibrationControl('client', subjectId="player1")
```

### 3. **Wearing Status** (`calibrationControl.py`)
```python
from ifocus_sdk.APIs.calibrationControl import getWearingStatus, setWearingCheckEnabled

# Get current wearing status
is_wearing = getWearingStatus()  # True = good contact, False = poor/no contact

# Disable wearing check (for testing)
setWearingCheckEnabled(False)
```

### 4. **RSSI Monitoring**
```python
from ifocus_sdk.APIs.FocusComs import getCurrentRssi

# Get last known RSSI for connected device
rssi = getCurrentRssi(device_id)
```

---

## Integration Steps

### Step 1: Replace Stub Functions
**Current stubs in `landing_first_page.py`:**
- `scan_devices_stub()` → Replace with real SDK scanning
- `connect_device_stub()` → Replace with `calibrationControl('connect')`
- `disconnect_device_stub()` → Replace with `calibrationControl('disconnect')`

### Step 2: Add Async Support to UI
**Challenge:** PySide6 UI is synchronous, SDK is async

**Solutions:**
1. **Option A: Use `asyncio` with QEventLoop integration**
   - Create async event loop that cooperates with Qt
   - Use `qasync` library for Qt + asyncio integration
   
2. **Option B: Run SDK in separate thread**
   - Use `QThread` to run async operations
   - Communicate back to UI via signals

**Recommended:** Option A with `qasync` (cleaner integration)

### Step 3: Implement Real Scanning
**Changes to `on_search()`:**
```python
async def on_search(self):
    self.status_lbl.setText("Scanning...")
    self.search_btn.setEnabled(False)
    
    found_devices = []
    
    def on_device(name: str, device_id: str, rssi: int):
        # Add to list if not already present
        if not any(d['mac'] == device_id for d in found_devices):
            found_devices.append({
                'name': name,
                'mac': device_id,
                'rssi': rssi
            })
            # Update UI in real-time
            self._update_device_list(found_devices)
    
    await startDeviceScan(on_device)
    await asyncio.sleep(5)  # Scan for 5 seconds
    await stopDeviceScan()
    
    if not found_devices:
        self.status_lbl.setText("No devices found. Try Search again.")
    else:
        self.status_lbl.setText(f"Found {len(found_devices)} device(s). Tap to connect.")
    
    self.search_btn.setEnabled(True)
```

### Step 4: Implement Real Connection/Disconnection
**Changes to `on_card_toggled()`:**
```python
async def on_card_toggled(self, card: PlayerCard):
    if not card.device:
        return
    
    mac = card.device.mac
    
    if card.selected:
        # Connect to device
        self.status_lbl.setText(f"Connecting to {card.device.name}...")
        success = await calibrationControl('connect', deviceId=mac)
        
        if success:
            self.selected_macs.add(mac)
            self.status_lbl.setText(f"Connected to {card.device.name}")
            # Start wearing status monitoring
            self._start_wearing_monitor(card, mac)
        else:
            card.selected = False
            card.pill.setVisible(False)
            card._refresh_style()
            self.status_lbl.setText(f"Failed to connect to {card.device.name}")
    else:
        # Disconnect
        self.selected_macs.discard(mac)
        self._stop_wearing_monitor(mac)
        await calibrationControl('disconnect', deviceId=mac)
        self.status_lbl.setText(f"Disconnected from {card.device.name}")
    
    self._update_play_state()
```

### Step 5: Implement Wearing Status Monitoring
**Add periodic monitoring:**
```python
class IFocusWindow(QMainWindow):
    def __init__(self):
        # ... existing code ...
        self.wearing_timers: Dict[str, QTimer] = {}
    
    def _start_wearing_monitor(self, card: PlayerCard, device_id: str):
        """Start periodic wearing status check for a device."""
        timer = QTimer(self)
        
        async def check_wearing():
            # Get wearing status from SDK
            is_wearing = getWearingStatus()
            
            # Update card UI
            card.set_wearing_status(is_wearing)
            
            # Update status label if not wearing
            if not is_wearing:
                self.status_lbl.setText(f"⚠️ {card.device.name}: Poor electrode contact")
        
        # Check every 500ms
        timer.timeout.connect(lambda: asyncio.create_task(check_wearing()))
        timer.start(500)
        self.wearing_timers[device_id] = timer
    
    def _stop_wearing_monitor(self, device_id: str):
        """Stop wearing status monitoring."""
        timer = self.wearing_timers.pop(device_id, None)
        if timer:
            timer.stop()
```

### Step 6: Implement Real RSSI Updates
**Add periodic RSSI updates during scanning:**
```python
def _start_rssi_monitor(self, card: PlayerCard, device_id: str):
    """Update RSSI values for connected devices."""
    timer = QTimer(self)
    
    def update_rssi():
        rssi = getCurrentRssi(device_id)
        if rssi is not None:
            card.rssi_lbl.setText(f"{rssi} dBm")
    
    timer.timeout.connect(update_rssi)
    timer.start(2000)  # Update every 2 seconds
    self.rssi_timers[device_id] = timer
```

### Step 7: Handle Player Name Editing
**Store custom names:**
```python
class IFocusWindow(QMainWindow):
    def __init__(self):
        # ... existing code ...
        self.device_names: Dict[str, str] = {}  # mac -> custom name
    
    def _on_name_edited(self, card: PlayerCard):
        """Called when user edits player name."""
        if card.device:
            custom_name = card.name_lbl.text().strip()
            if custom_name:
                self.device_names[card.device.mac] = custom_name
```

---

## Required Dependencies

### Install `qasync` for Qt + asyncio integration:
```bash
pip install qasync
```

### Update main application entry point:
```python
import sys
import asyncio
from qasync import QEventLoop, asyncSlot
from PySide6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set up asyncio event loop for Qt
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    win = IFocusWindow()
    win.show()
    
    with loop:
        loop.run_forever()
```

---

## Data Flow

### Scanning Flow:
```
User clicks "Search" 
  → startDeviceScan() 
  → on_device() callback for each device found
  → Update UI cards in real-time
  → stopDeviceScan() after timeout
  → Display results
```

### Connection Flow:
```
User taps card (select)
  → calibrationControl('connect', deviceId=mac)
  → Update card state (Connected pill visible)
  → Start wearing status timer
  → Start RSSI update timer
  → Enable Play button if devices connected
```

### Wearing Status Flow:
```
Timer fires every 500ms
  → getWearingStatus() for device
  → Update card.set_wearing_status(is_wearing)
  → Green circle = good contact
  → Red circle = poor/no contact
```

### RSSI Update Flow:
```
Timer fires every 2s during scan
  → getCurrentRssi(device_id)
  → Update card RSSI label
```

---

## Error Handling

### 1. **Connection Failures**
- Show error message in status label
- Unselect card if connection fails
- Allow retry

### 2. **Disconnection**
- Detect unexpected disconnections
- Update UI state
- Show notification

### 3. **No Devices Found**
- Show helpful message
- Suggest checking device power/range

### 4. **Wearing Status Issues**
- Visual indicator (red circle)
- Optional: Disable "Play" if not worn
- Show helpful message

---

## UI State Management

### States to track:
1. **Scanning** - Search button disabled, status shows "Scanning..."
2. **Devices Found** - Cards displayed, selectable
3. **Connecting** - Status shows "Connecting to..."
4. **Connected** - Green pill shown, wearing monitor active
5. **Ready to Play** - At least one device connected, Play enabled

---

## Testing Strategy

### 1. **Mock Mode** (for UI testing without hardware)
- Keep stub functions as fallback
- Add environment variable to toggle real/mock mode
- Useful for layout/design testing

### 2. **Single Device Testing**
- Test with one device first
- Verify all features work

### 3. **Multi-Device Testing**
- Test with 2-4 devices
- Verify concurrent connections
- Test wearing status for multiple devices

---

## Next Steps

1. ✅ **Plan Complete** (this document)
2. ⬜ Install `qasync` dependency
3. ⬜ Add async support to UI window
4. ⬜ Replace scan stub with real SDK
5. ⬜ Replace connect/disconnect stubs
6. ⬜ Implement wearing status monitoring
7. ⬜ Implement RSSI updates
8. ⬜ Test with real devices
9. ⬜ Add error handling
10. ⬜ Polish UX (loading states, animations)

---

## Notes

- **Multi-device support**: SDK supports connecting to multiple devices simultaneously
- **Wearing status**: Available via `getWearingStatus()` which reads from parser
- **RSSI**: Cached during scan, use `getCurrentRssi(device_id)`
- **Subject IDs**: Can bind devices to custom subject IDs for game logic
- **Async requirement**: All SDK calls are async, need proper event loop integration
