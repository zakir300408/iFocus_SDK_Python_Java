"""iFocus BLE Packet Parser.

Parses raw BLE data from the iFocus headset into EEG and IMU values.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Protocol constants
_EEG_BYTES = 3              # Bytes per EEG sample
_IMU_BYTES = 2              # Bytes per IMU axis
_EEG_CHANNELS = 5           # Number of EEG channels per frame
_IMU_AXES = 3               # Number of IMU axes (X, Y, Z)
_HEADER_SIZE = 2            # Frame header size

_EEG_SCALE = 0.02235174     # EEG raw to µV conversion
_IMU_SCALE = 0.01           # IMU raw to g conversion


class Parser:
    """Stateful parser for iFocus BLE packets.
    
    Maintains an internal buffer and extracts EEG/IMU frames with
    checksum validation and packet loss detection.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        
        # Precompute byte offsets
        eeg_block = _EEG_CHANNELS * _EEG_BYTES
        imu_block = _IMU_AXES * _IMU_BYTES
        
        self._eeg_indices = [_HEADER_SIZE + i * _EEG_BYTES for i in range(_EEG_CHANNELS)]
        self._wearing_status_idx = _HEADER_SIZE + eeg_block  # Byte 17: electrode contact detection
        self._eeg_checksum_idx = _HEADER_SIZE + eeg_block + 1
        self._eeg_seq_idx = self._eeg_checksum_idx + 1
        
        self._imu_start = self._eeg_seq_idx + _HEADER_SIZE + 1
        self._imu_indices = [self._imu_start + i * _IMU_BYTES for i in range(_IMU_AXES)]
        self._imu_checksum_idx = self._imu_start + imu_block
        self._imu_seq_idx = self._imu_checksum_idx + 1
        
        self._min_frame_size = self._imu_seq_idx + 1
        self._pattern = re.compile(
            b"\xbb\xaa.{%d}\xdd\xcc.{8}" % (eeg_block + 3),
            flags=re.DOTALL
        )
        
        self._reset_sequence_tracking()

    def _reset_sequence_tracking(self) -> None:
        """Reset sequence counters for packet loss detection."""
        self._eeg_seq_last = 255
        self._imu_seq_last = 255
        self._eeg_drops = 0
        self._imu_drops = 0
        self._wearing_status = False  # Track latest wearing status

    def clear_buffer(self) -> None:
        """Clear internal buffer and reset sequence tracking."""
        self._buffer.clear()
        self._reset_sequence_tracking()

    def parse_data(self, data: bytes) -> list[list]:
        """Parse incoming BLE data and extract complete frames.
        
        Args:
            data: Raw bytes from BLE notification.
            
        Returns:
            List of frames. Each frame is [eeg_ch0, eeg_ch1, ..., eeg_ch4, imu_xyz]
            where eeg values are [float] and imu_xyz is [x, y, z].
            Returns empty list if no complete frames available.
        """
        self._buffer.extend(data)
        
        if len(self._buffer) < self._min_frame_size:
            return []
        
        frames = []
        last_end = 0
        
        for match in self._pattern.finditer(self._buffer):
            frame_bytes = memoryview(match.group())
            
            # Validate EEG checksum
            eeg_sum = sum(frame_bytes[_HEADER_SIZE:self._eeg_checksum_idx]) & 0xFF
            if frame_bytes[self._eeg_checksum_idx] != eeg_sum:
                logger.debug("EEG checksum mismatch, dropping packet")
                continue
            
            # Validate IMU checksum
            imu_sum = sum(frame_bytes[self._imu_start:self._imu_checksum_idx]) & 0xFF
            if frame_bytes[self._imu_checksum_idx] != imu_sum:
                logger.debug("IMU checksum mismatch, dropping packet")
                continue
            
            # Check for packet loss (sequence gaps)
            eeg_seq = frame_bytes[self._eeg_seq_idx]
            expected_eeg = (self._eeg_seq_last + 1) % 256
            if eeg_seq != expected_eeg:
                self._eeg_drops += 1
                logger.debug("EEG packet loss: expected %d, got %d (total drops: %d)",
                           expected_eeg, eeg_seq, self._eeg_drops)
            self._eeg_seq_last = eeg_seq
            
            imu_seq = frame_bytes[self._imu_seq_idx]
            expected_imu = (self._imu_seq_last + 1) % 256
            if imu_seq != expected_imu:
                self._imu_drops += 1
                logger.debug("IMU packet loss: expected %d, got %d (total drops: %d)",
                           expected_imu, imu_seq, self._imu_drops)
            self._imu_seq_last = imu_seq
            
            # Extract wearing status
            # "脱落检测" (detachment detection): 0 = electrodes have contact (wearing), non-zero = detached (not wearing)
            self._wearing_status = frame_bytes[self._wearing_status_idx] == 0
            
            # Extract EEG values
            eeg = [
                [int.from_bytes(frame_bytes[i:i + _EEG_BYTES], signed=True, byteorder="little") * _EEG_SCALE]
                for i in self._eeg_indices
            ]
            
            # Extract IMU values
            imu = [
                int.from_bytes(frame_bytes[i:i + _IMU_BYTES], signed=True, byteorder="little") * _IMU_SCALE
                for i in self._imu_indices
            ]
            
            eeg.append(imu)
            frames.append(eeg)
            last_end = match.end()
        
        # Trim processed data from buffer
        if last_end > 0:
            del self._buffer[:last_end]
        
        return frames

    @property
    def drop_stats(self) -> dict:
        """Return packet loss statistics."""
        return {"eeg_drops": self._eeg_drops, "imu_drops": self._imu_drops}

    @property
    def wearing_status(self) -> bool:
        """Return True if headset is being worn (electrodes have contact)."""
        return self._wearing_status
