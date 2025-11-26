"""iFocus EEG Calibration API.

Provides a unified API for EEG calibration data collection and management.

Public API:
    - calibrationControl(action, ...): Unified control function
    - iFocusRealTimeReader(client, chunk_sec): Async generator for live EEG
    - getWearingStatus(): Get headset wearing/electrode contact status
    - setWearingCheckEnabled(enabled): Enable/disable wearing check (developer override)
    - isWearingCheckEnabled(): Check if wearing check is enabled
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .iFocusParser import Parser
from .FocusComs import connect, disconnect, _get_client

logger = logging.getLogger(__name__)

# BLE UUIDs for iFocus headset
DATA_CHAR_UUID = "0000ffe2-3c17-d293-8e48-14fe2e4da212"
COMMAND_CHAR_UUID = "0000ffe3-3c17-d293-8e48-14fe2e4da212"

# EEG Configuration
DEFAULT_SAMPLE_RATE = 500

# Data storage path
CALIBRATION_DATA_DIR = Path("data") / "calibration"

# Type aliases
Frame = Dict[str, List[float]]
Segment = List[Frame]


def _sanitize_name(name: str) -> str:
    """Sanitize a string for use in filenames."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "unknown"


# ----------------------------------------------------------------------
# Real-time EEG Reader
# ----------------------------------------------------------------------

async def iFocusRealTimeReader(client, chunk_sec: float = 1.0):
    """Async generator yielding real-time EEG chunks.
    
    Args:
        client: Connected BleakClient.
        chunk_sec: Approximate chunk duration in seconds.
    
    Yields:
        np.ndarray: 1D array of EEG samples.
    
    Raises:
        ValueError: If client is None.
        RuntimeError: If client is not connected.
    """
    if client is None:
        raise ValueError("client is required")
    if not client.is_connected:
        raise RuntimeError("Client is not connected")

    chunk_samples = max(1, int(DEFAULT_SAMPLE_RATE * chunk_sec))
    # Use a local parser but update the module-level parser's wearing status
    local_parser = Parser()
    queue: asyncio.Queue[List[float]] = asyncio.Queue()
    buffer: List[float] = []

    def on_notify(_, data: bytes):
        for frame in local_parser.parse_data(data):
            # Sync wearing status to module-level parser so getWearingStatus() works
            _parser._wearing_status = local_parser.wearing_status
            
            eeg_samples = [float(ch[0]) if isinstance(ch, list) else float(ch) 
                          for ch in frame[:-1]]
            if eeg_samples:
                try:
                    queue.put_nowait(eeg_samples)
                except asyncio.QueueFull:
                    pass

    try:
        await client.start_notify(DATA_CHAR_UUID, on_notify)
        await client.write_gatt_char(COMMAND_CHAR_UUID, b"\x01", response=True)
        
        while True:
            samples = await queue.get()
            buffer.extend(samples)
            
            while len(buffer) >= chunk_samples:
                yield np.array(buffer[:chunk_samples], dtype=float)
                del buffer[:chunk_samples]
                
    finally:
        try:
            await client.write_gatt_char(COMMAND_CHAR_UUID, b"\x02", response=True)
            await client.stop_notify(DATA_CHAR_UUID)
        except Exception:
            pass


# ----------------------------------------------------------------------
# Calibration Manager
# ----------------------------------------------------------------------

class _CalibrationManager:
    """Manages calibration data collection."""
    
    def __init__(self):
        self._session: Optional[Tuple[str, str, List[Frame]]] = None
        self._data: Dict[Tuple[str, str], List[Segment]] = defaultdict(list)
        self._was_wearing = True  # Track wearing state transitions
        self._last_warning_time = 0
        self._warning_interval = 2.0  # Warn every 2 seconds when not wearing
    
    def on_frames(self, frames: List[Frame], wearing: bool) -> None:
        """Process incoming frames during active session.
        
        Args:
            frames: List of data frames to process.
            wearing: Current wearing status. Frames are skipped if not wearing
                     and wearing check is enabled.
        """
        if self._session:
            import time
            
            # Check wearing status and log warning if not worn
            if _wearing_check_enabled and not wearing:
                current_time = time.time()
                if current_time - self._last_warning_time >= self._warning_interval:
                    logger.warning(
                        "HEADSET NOT WORN - Electrodes have no contact. "
                        "Data collection paused until headset is worn properly."
                    )
                    self._last_warning_time = current_time
                self._was_wearing = False
                return
            
            # Log when wearing resumes
            if not self._was_wearing and wearing:
                logger.info("Headset worn again - resuming data collection.")
            self._was_wearing = True
            
            subj, label, segment = self._session
            for f in frames:
                segment.append({"eeg": f["eeg"][:]})
    
    def start(self, subject_id: str, state_label: str) -> None:
        """Start a new calibration session."""
        if self._session:
            self.stop()  # Finalize previous without saving
        self._session = (subject_id, state_label, [])
        self._was_wearing = True  # Reset wearing state on new session
        self._last_warning_time = 0
    
    def stop(self, output_dir: Optional[Path] = None) -> Optional[Path]:
        """Stop session, store data, and optionally save to BDF."""
        if not self._session or not self._session[2]:
            self._session = None
            return None
        
        subject_id, state_label, segment = self._session
        self._data[(subject_id, state_label)].append([{"eeg": f["eeg"][:]} for f in segment])
        self._session = None
        
        if output_dir is not None or output_dir == "":
            return self._save_bdf(subject_id, state_label, output_dir or CALIBRATION_DATA_DIR)
        return self._save_bdf(subject_id, state_label, CALIBRATION_DATA_DIR)
    
    def get_data(self, subject_id: str, state_label: str) -> List[Segment]:
        """Get stored segments for subject/state."""
        return list(self._data.get((subject_id, state_label), []))
    
    def clear(self, subject_id: str) -> None:
        """Clear all data and files for a subject."""
        # Remove from memory
        keys_to_remove = [k for k in self._data if k[0] == subject_id]
        for key in keys_to_remove:
            del self._data[key]
        
        if self._session and self._session[0] == subject_id:
            self._session = None
        
        # Remove files
        safe_subj = _sanitize_name(subject_id)
        if CALIBRATION_DATA_DIR.exists():
            for pattern in [f"cal_{safe_subj}_*.bdf", f"model_{safe_subj}_*.joblib"]:
                for f in CALIBRATION_DATA_DIR.glob(pattern):
                    try:
                        f.unlink()
                    except Exception:
                        pass
    
    def _save_bdf(self, subject_id: str, state_label: str, output_dir: Path) -> Optional[Path]:
        """Save the latest segment to BDF format."""
        try:
            segments = self._data.get((subject_id, state_label), [])
            if not segments:
                return None
            
            samples = []
            for frame in segments[-1]:
                samples.extend(frame["eeg"])
            
            if not samples:
                return None
            
            import pyedflib
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            safe_subj = _sanitize_name(subject_id)
            safe_state = _sanitize_name(state_label)
            duration = len(samples) / DEFAULT_SAMPLE_RATE
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            bdf_path = output_dir / f"cal_{safe_subj}_{safe_state}_{timestamp}_{int(duration)}s.bdf"
            
            pmin, pmax = min(samples), max(samples)
            if pmax == pmin:
                pmax = pmin + 1.0
            
            with pyedflib.EdfWriter(str(bdf_path), n_channels=1, file_type=pyedflib.FILETYPE_BDFPLUS) as writer:
                writer.setSignalHeader(0, {
                    "label": f"{safe_subj}-{safe_state}",
                    "dimension": "uV",
                    "sample_frequency": float(DEFAULT_SAMPLE_RATE),
                    "physical_min": float(str(pmin)[:8]),
                    "physical_max": float(str(pmax)[:8]),
                    "digital_min": -32768,
                    "digital_max": 32767,
                })
                writer.writeSamples([np.asarray(samples, dtype=np.float64)])
            
            return bdf_path
        except Exception as e:
            logger.error("Failed to save BDF: %s", e)
            return None


# ----------------------------------------------------------------------
# Module State
# ----------------------------------------------------------------------

_parser = Parser()
_manager: Optional[_CalibrationManager] = None
_streaming = False
_connected = False
_device_id: Optional[str] = None
_wearing_check_enabled = True  # Developer override flag


def _get_manager() -> _CalibrationManager:
    global _manager
    if _manager is None:
        _manager = _CalibrationManager()
    return _manager


def _transform_frames(raw_frames: List[list]) -> List[Frame]:
    """Convert parser output to Frame format."""
    result = []
    for frame in raw_frames:
        if not frame:
            continue
        eeg = [float(ch[0]) if isinstance(ch, list) else float(ch) for ch in frame[:-1]]
        imu = frame[-1] if isinstance(frame[-1], list) else [float(frame[-1])]
        result.append({"eeg": eeg, "imu": [float(v) for v in imu]})
    return result


def _notification_handler(_, data: bytes) -> None:
    """BLE notification handler."""
    frames = _parser.parse_data(data)
    if frames:
        _get_manager().on_frames(_transform_frames(frames), _parser.wearing_status)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

async def calibrationControl(
    action: str,
    deviceId: Optional[str] = None,
    subjectId: Optional[str] = None,
    stateLabel: Optional[str] = None,
    outputDir: Optional[str | Path] = None
) -> bool | List[Segment] | Optional[Path]:
    """Unified calibration control API.
    
    Args:
        action: 'connect', 'disconnect', 'start', 'stop', 'get', or 'reset'
        deviceId: BLE device address (for 'connect')
        subjectId: Subject identifier
        stateLabel: Mental state label ('FOCUS', 'RELAX', etc.)
        outputDir: Custom output directory for BDF files
    
    Returns:
        - 'connect': bool (success)
        - 'disconnect': bool (success)
        - 'start': bool (success)
        - 'stop': Path or None (saved file path)
        - 'get': List[Segment]
        - 'reset': bool (success)
    """
    global _streaming, _device_id, _connected
    mgr = _get_manager()
    _client = _get_client()
    
    # Validate parameters
    _validate_action_params(action, deviceId, subjectId, stateLabel)
    
    if action == 'connect':
        if _connected:
            return True
        
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        def on_connect(connected: bool):
            if not future.done():
                loop.call_soon_threadsafe(future.set_result, connected)
        
        try:
            await connect(deviceId, auto_reconnect=False, callback=on_connect)
            if await future:
                _device_id = deviceId
                _connected = True
                return True
        except Exception as e:
            logger.error("Connection failed: %s", e)
        return False
    
    elif action == 'disconnect':
        if _device_id:
            try:
                await disconnect(_device_id)
            except Exception:
                pass
            _device_id = None
            _connected = False
            _streaming = False
        return True
    
    elif action == 'start':
        if not _connected or not _client:
            raise RuntimeError("Not connected. Use 'connect' first.")
        
        if not _streaming:
            try:
                await _client.start_notify(DATA_CHAR_UUID, _notification_handler)
                await _client.write_gatt_char(COMMAND_CHAR_UUID, b"\x01", response=True)
                _streaming = True
            except Exception as e:
                raise RuntimeError(f"Failed to start streaming: {e}") from e
        
        mgr.start(subjectId, stateLabel)
        return True
    
    elif action == 'stop':
        result = mgr.stop(Path(outputDir) if outputDir else None)
        
        if _streaming and _client and _client.is_connected:
            try:
                await _client.write_gatt_char(COMMAND_CHAR_UUID, b"\x02", response=True)
                await _client.stop_notify(DATA_CHAR_UUID)
            except Exception:
                pass
            _streaming = False
        
        return result
    
    elif action == 'get':
        return mgr.get_data(subjectId, stateLabel)
    
    elif action == 'reset':
        mgr.clear(subjectId)
        return True
    
    return False


def _validate_action_params(action: str, deviceId, subjectId, stateLabel) -> None:
    """Validate parameters for calibrationControl actions."""
    if action == 'connect' and not deviceId:
        raise ValueError("'connect' requires deviceId")
    elif action == 'start' and (not subjectId or not stateLabel):
        raise ValueError("'start' requires subjectId and stateLabel")
    elif action == 'get' and (not subjectId or not stateLabel):
        raise ValueError("'get' requires subjectId and stateLabel")
    elif action == 'reset' and not subjectId:
        raise ValueError("'reset' requires subjectId")
    elif action not in ('connect', 'disconnect', 'start', 'stop', 'get', 'reset'):
        raise ValueError(f"Unknown action: {action}")


def getWearingStatus() -> bool:
    """Get the current headset wearing status.
    
    Returns:
        True if the headset electrodes have contact (being worn),
        False otherwise. Also returns False if no data has been received yet.
    """
    return _parser.wearing_status


def setWearingCheckEnabled(enabled: bool) -> None:
    """Enable or disable wearing status check.
    
    When disabled, calibration and inference will proceed even if the headset
    is not being worn. This is intended for developer testing only.
    
    Args:
        enabled: True to require wearing (default), False to bypass check.
    """
    global _wearing_check_enabled
    _wearing_check_enabled = enabled
    if not enabled:
        logger.warning("Wearing check DISABLED - data quality may be affected")


def isWearingCheckEnabled() -> bool:
    """Check if wearing status verification is enabled.
    
    Returns:
        True if wearing check is enabled (default), False if bypassed.
    """
    return _wearing_check_enabled


def _check_wearing_required() -> None:
    """Raise RuntimeError if wearing check is enabled and headset is not worn."""
    if _wearing_check_enabled and not _parser.wearing_status:
        raise RuntimeError(
            "Headset is not being worn (electrodes have no contact). "
            "Please wear the headset properly, or use setWearingCheckEnabled(False) to bypass."
        )



