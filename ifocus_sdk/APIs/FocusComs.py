"""iFocus BLE Device Discovery and Connection.

Public API:
    - startDeviceScan(callback): Scan for iFocus devices
    - stopDeviceScan(): Stop scanning
    - connect(device_id, auto_reconnect, callback): Connect to device
    - disconnect(device_id): Disconnect from device
    - getCurrentRssi(device_id): Get last known RSSI
"""

import asyncio
import logging
from typing import Callable, Optional

from bleak import BleakScanner, BleakClient

logger = logging.getLogger(__name__)

# Module state
_scanner: Optional[BleakScanner] = None
_client: Optional[BleakClient] = None
_rssi_cache: dict[str, int] = {}
_connection_callback: Optional[Callable[[bool], None]] = None
_auto_reconnect: bool = False
_target_device_id: Optional[str] = None


async def startDeviceScan(callback: Callable[[str, str, Optional[int]], None]) -> bool:
    """Start BLE scanning for iFocus devices.
    
    Args:
        callback: Called with (device_name, device_id, rssi) for each iFocus device.
    
    Returns:
        True if scan started, False if already scanning.
    """
    global _scanner
    if _scanner:
        return False

    def on_device(device, adv_data):
        name = device.name or getattr(adv_data, "local_name", None) or ""
        if not name.startswith("iFocus"):
            return
        
        rssi = getattr(adv_data, "rssi", None) or getattr(device, "rssi", None)
        if rssi is not None:
            _rssi_cache[device.address] = rssi
        
        try:
            callback(name, device.address, rssi)
        except Exception:
            pass  # Protect scanner from callback errors

    _scanner = BleakScanner(detection_callback=on_device)
    await _scanner.start()
    return True


async def stopDeviceScan() -> bool:
    """Stop BLE scanning.
    
    Returns:
        True if stopped, False if not scanning.
    """
    global _scanner
    if not _scanner:
        return False
    try:
        await _scanner.stop()
    finally:
        _scanner = None
    return True


async def connect(
    device_id: str,
    auto_reconnect: bool = False,
    callback: Optional[Callable[[bool], None]] = None
) -> bool:
    """Connect to a BLE device.
    
    Args:
        device_id: BLE address of the device.
        auto_reconnect: If True, auto-reconnect on unexpected disconnection.
        callback: Called with connection status changes.
    
    Returns:
        True if connected successfully.
    """
    global _client, _connection_callback, _auto_reconnect, _target_device_id

    # Already connected to this device
    if _client and _client.is_connected and _client.address == device_id:
        if callback:
            callback(True)
        return True

    # Disconnect from different device first
    if _client and _client.is_connected:
        await disconnect()

    _target_device_id = device_id
    _auto_reconnect = auto_reconnect
    _connection_callback = callback

    def on_disconnect(_):
        global _client
        if _auto_reconnect and _client is not None:
            asyncio.create_task(_reconnect_loop())
        else:
            if _connection_callback:
                try:
                    _connection_callback(False)
                except Exception:
                    pass
            _client = None

    _client = BleakClient(device_id, disconnected_callback=on_disconnect)

    try:
        await _client.connect()
        connected = _client.is_connected
        if callback:
            callback(connected)
        return connected
    except Exception as e:
        logger.debug("Connection failed: %s", e)
        if callback:
            callback(False)
        _client = None
        return False


async def _reconnect_loop() -> None:
    """Auto-reconnect loop."""
    global _client
    while _auto_reconnect and _target_device_id:
        try:
            _client = BleakClient(_target_device_id, disconnected_callback=lambda _: None)
            await _client.connect()
            if _client.is_connected:
                if _connection_callback:
                    _connection_callback(True)
                return
        except Exception:
            await asyncio.sleep(2.0)


async def disconnect(device_id: Optional[str] = None) -> bool:
    """Disconnect from a device.
    
    Args:
        device_id: If provided, only disconnect if connected to this device.
    
    Returns:
        True if disconnected successfully.
    """
    global _client, _auto_reconnect, _target_device_id

    if device_id is not None and (_client is None or _client.address != device_id):
        return False

    _auto_reconnect = False
    _target_device_id = None

    if _client:
        try:
            await _client.disconnect()
        except Exception:
            pass
        finally:
            _client = None

    if _connection_callback:
        try:
            _connection_callback(False)
        except Exception:
            pass

    return True


def getCurrentRssi(device_id: Optional[str] = None) -> Optional[int]:
    """Get last known RSSI for a device.
    
    Args:
        device_id: Device address. If None and only one device cached, returns that.
    
    Returns:
        RSSI in dBm or None.
    """
    if device_id:
        return _rssi_cache.get(device_id)
    if len(_rssi_cache) == 1:
        return next(iter(_rssi_cache.values()))
    return None


def _get_client() -> Optional[BleakClient]:
    """Get the current BleakClient instance."""
    return _client
