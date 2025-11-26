"""Sliding window epoch generator for EEG data.

Public API:
    - EpochMaker: Class for building fixed-size epochs from streaming EEG
        - offline(eeg, win_sec, update_frequency): Create epochs from full array
        - push(x): Add samples and return complete epochs
        - reset(): Clear internal buffer
"""

from typing import List
import numpy as np

# Default sampling rate (can be overridden by importing modules)
DEFAULT_FS = 500


class EpochMaker:
    """Sliding window epoch generator for single-channel EEG.
    
    Args:
        win_sec: Window length in seconds.
        update_frequency: Epoch update rate in Hz (epochs per second).
        fs: Sampling frequency in Hz (default 500).
    
    Example (offline):
        epochs = EpochMaker.offline(eeg_array, win_sec=3.0, update_frequency=2.0)
    
    Example (streaming):
        maker = EpochMaker(win_sec=3.0, update_frequency=2.0)
        for chunk in data_stream:
            for epoch in maker.push(chunk):
                process(epoch)
    """

    def __init__(self, win_sec: float = 3.0, update_frequency: float = 2.0, fs: float = DEFAULT_FS):
        self.fs = float(fs)
        self.win_samples = int(round(win_sec * self.fs))
        
        if self.win_samples <= 0:
            raise ValueError("win_sec too small")
        
        if update_frequency <= 0:
            raise ValueError("update_frequency must be > 0")
        
        hop = self.fs / update_frequency
        if hop < 1:
            raise ValueError(f"update_frequency={update_frequency} too high for fs={fs}")
        
        self.hop_samples = int(round(hop))
        if self.hop_samples > self.win_samples:
            raise ValueError("update_frequency too low (hop > window)")
        
        self._buffer = np.empty(0, dtype=float)

    def reset(self) -> None:
        """Clear internal buffer."""
        self._buffer = np.empty(0, dtype=float)

    def push(self, x) -> List[np.ndarray]:
        """Add samples and return any complete epochs.
        
        Args:
            x: 1D array-like of new samples.
        
        Returns:
            List of epoch arrays, each of length win_samples.
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.size == 0:
            return []
        
        self._buffer = np.concatenate([self._buffer, x])
        
        epochs = []
        while len(self._buffer) >= self.win_samples:
            epochs.append(self._buffer[:self.win_samples].copy())
            self._buffer = self._buffer[self.hop_samples:]
        
        return epochs

    @classmethod
    def offline(
        cls,
        eeg: np.ndarray,
        win_sec: float,
        update_frequency: float,
        fs: float = DEFAULT_FS
    ) -> List[np.ndarray]:
        """Create epochs from a complete EEG array.
        
        Args:
            eeg: 1D array of EEG samples.
            win_sec: Window length in seconds.
            update_frequency: Epoch update rate in Hz.
            fs: Sampling frequency in Hz.
        
        Returns:
            List of epoch arrays.
        """
        maker = cls(win_sec=win_sec, update_frequency=update_frequency, fs=fs)
        eeg = np.asarray(eeg, dtype=float).ravel()
        return maker.push(eeg)
