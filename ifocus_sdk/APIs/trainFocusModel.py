"""Subject-specific focus model training.

Public API:
    - trainFocusModel(subjectId, callback): Train and save model
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pyedflib
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .calibrationControl import CALIBRATION_DATA_DIR, DEFAULT_SAMPLE_RATE, _sanitize_name
from ._preprocess_extract_features import extract_features
from .__epoch_maker import EpochMaker

logger = logging.getLogger(__name__)

# Training configuration
TRAIN_WIN_SEC = 3.0
TRAIN_UPDATE_HZ = 2.0


def _get_latest_model_path(subject_id: str) -> Optional[Path]:
    """Get path to most recent model file for subject, or None if not found."""
    safe_id = _sanitize_name(subject_id)
    if not CALIBRATION_DATA_DIR.exists():
        return None
    
    models = sorted(CALIBRATION_DATA_DIR.glob(f"model_{safe_id}_*.joblib"))
    return models[-1] if models else None


def _get_calibration_files(subject_id: str) -> List[Path]:
    """Get sorted list of calibration BDF files for subject."""
    safe_id = _sanitize_name(subject_id)
    if not CALIBRATION_DATA_DIR.exists():
        return []
    return sorted(CALIBRATION_DATA_DIR.glob(f"cal_{safe_id}_*.bdf"))


def _parse_state_from_filename(path: Path, subject_id: str) -> str:
    """Extract state label from calibration filename.
    
    Expected: cal_{subject}_{state}_{date}_{time}_{duration}s.bdf
    """
    prefix = f"cal_{_sanitize_name(subject_id)}_"
    stem = path.stem
    
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected filename format: {path.name}")
    
    parts = stem[len(prefix):].split("_")
    if len(parts) < 4:
        raise ValueError(f"Cannot parse state from: {path.name}")
    
    # State is everything before the last 3 parts (date, time, duration)
    return "_".join(parts[:-3])


def _load_bdf(path: Path) -> np.ndarray:
    """Load first channel from BDF file as 1D float array."""
    with pyedflib.EdfReader(str(path)) as reader:
        if reader.signals_in_file < 1:
            raise ValueError(f"No signals in {path}")
        return reader.readSignal(0).astype(float)


def trainFocusModel(subjectId: str, callback: Callable[[bool, str, int], None]) -> None:
    """Train a focus classification model for a subject.
    
    Args:
        subjectId: Subject identifier used during calibration.
        callback: Called with (success, message, n_samples) when complete.
    """
    n_samples = 0
    
    try:
        files = _get_calibration_files(subjectId)
        if not files:
            callback(False, f"No calibration files for '{subjectId}'", 0)
            return
        
        CALIBRATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        X_list, y_list = [], []
        feature_names = None
        
        for path in files:
            try:
                state = _parse_state_from_filename(path, subjectId)
                signal = _load_bdf(path)
                epochs = EpochMaker.offline(signal, TRAIN_WIN_SEC, TRAIN_UPDATE_HZ)
            except Exception as e:
                logger.warning("Skipping %s: %s", path.name, e)
                continue
            
            if not epochs:
                continue
            
            for epoch in epochs:
                feats = extract_features(epoch, fs=DEFAULT_SAMPLE_RATE)
                
                if feature_names is None:
                    feature_names = sorted(feats.keys())
                elif set(feats.keys()) != set(feature_names):
                    raise RuntimeError("Inconsistent features between epochs")
                
                X_list.append([feats[k] for k in feature_names])
                y_list.append(state)
        
        if not X_list:
            callback(False, f"No usable samples for '{subjectId}'", 0)
            return
        
        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=object)
        n_samples = len(X)
        
        classes = np.unique(y)
        if len(classes) < 2:
            callback(False, f"Need at least 2 states, found: {classes.tolist()}", n_samples)
            return
        
        # Create model
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True))
        ])
        
        # Train/test split if enough data
        test_metrics = None
        class_counts = {c: int((y == c).sum()) for c in classes}
        
        if n_samples >= 10 and min(class_counts.values()) >= 2:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                model.fit(X_train, y_train)
                acc = accuracy_score(y_test, model.predict(X_test))
                test_metrics = {"accuracy": acc, "n_train": len(X_train), "n_test": len(X_test)}
            except Exception as e:
                logger.warning("Train/test split failed: %s", e)
        
        # Final fit on all data
        model.fit(X, y)
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_id = _sanitize_name(subjectId)
        model_path = CALIBRATION_DATA_DIR / f"model_{safe_id}_{timestamp}.joblib"
        
        joblib.dump({
            "subject_id": subjectId,
            "feature_names": feature_names,
            "classes": classes.tolist(),
            "fs": DEFAULT_SAMPLE_RATE,
            "win_sec": TRAIN_WIN_SEC,
            "n_samples": n_samples,
            "test_metrics": test_metrics,
            "model": model,
            "trained_at": timestamp,
        }, model_path)
        
        # Build result message
        if test_metrics:
            msg = (f"Model saved to {model_path}. "
                   f"Test accuracy: {test_metrics['accuracy']:.3f} "
                   f"(train={test_metrics['n_train']}, test={test_metrics['n_test']})")
        else:
            msg = f"Model saved to {model_path}. Not enough data for test split."
        
        callback(True, msg, n_samples)
        
    except Exception as e:
        logger.exception("Training failed")
        callback(False, f"Training failed: {e}", n_samples)
