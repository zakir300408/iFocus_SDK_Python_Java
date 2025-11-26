"""Live focus inference on iFocus EEG data.

Public API:
    - startFocusInference(subjectId, client, updateHz, callback): Start inference
    - stopFocusInference(): Stop inference
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, List, Optional

import numpy as np
import joblib

from .calibrationControl import DEFAULT_SAMPLE_RATE, iFocusRealTimeReader, getWearingStatus, isWearingCheckEnabled
from .__epoch_maker import EpochMaker
from .trainFocusModel import _get_latest_model_path
from ._preprocess_extract_features import extract_features

logger = logging.getLogger(__name__)

_inference_task: Optional[asyncio.Task] = None
_stop_event: Optional[asyncio.Event] = None


def _load_model(subject_id: str) -> dict:
    """Load trained model for subject."""
    path = _get_latest_model_path(subject_id)
    if path is None:
        raise RuntimeError(f"No model found for '{subject_id}'. Train first with trainFocusModel().")
    
    payload = joblib.load(path)
    if "model" not in payload or "feature_names" not in payload:
        raise RuntimeError(f"Invalid model file: {path}")
    return payload


def _compute_focus_strength(features: dict, proba: np.ndarray, classes: List) -> int:
    """Compute focus strength score (1-100) from EEG features and model output."""
    eps = 1e-12
    beta = features.get("bp_rel_beta", 0.0)
    alpha = features.get("bp_rel_alpha", 0.0)
    theta = features.get("bp_rel_theta", 0.0)
    
    if beta > 0.01:
        engagement = beta / (alpha + theta + eps)
        engagement_norm = 1.0 / (1.0 + np.exp(-2.0 * (engagement - 1.0)))
        
        tbr = min(theta / (beta + eps), 10.0)
        tbr_focus = np.clip(1.0 - (tbr - 0.5) / 3.5, 0.0, 1.0)
        
        weights = (0.4, 0.3, 0.3)
    else:
        engagement_norm = tbr_focus = 0.5
        weights = (0.0, 0.0, 1.0)
    
    focus_idx = next((i for i, c in enumerate(classes) if str(c).lower().startswith("focus")), None)
    p_focus = float(proba[focus_idx]) if focus_idx is not None else float(np.max(proba))
    
    score = weights[0] * engagement_norm + weights[1] * tbr_focus + weights[2] * p_focus
    return max(1, min(100, int(round(1 + 99 * score))))


async def _inference_loop(
    subject_id: str,
    client,
    update_hz: float,
    callback: Callable[[str, int, str], None],
    stop_event: asyncio.Event
) -> None:
    """Main inference loop."""
    try:
        payload = _load_model(subject_id)
        model = payload["model"]
        feature_names = payload["feature_names"]
        classes = payload.get("classes") or list(getattr(model, "classes_", []))
        fs = float(payload.get("fs", DEFAULT_SAMPLE_RATE))
        win_sec = float(payload.get("win_sec", 3.0))
        
        epoch_maker = EpochMaker(win_sec=win_sec, update_frequency=update_hz)
        chunk_sec = min(0.5, win_sec / 4.0)
        
        import time
        last_wearing_warning_time = 0
        warning_interval = 2.0
        was_wearing = True
        
        async for chunk in iFocusRealTimeReader(client, chunk_sec=chunk_sec):
            if stop_event.is_set():
                break
            
            if chunk.size == 0:
                continue
            
            is_wearing = getWearingStatus()
            
            if isWearingCheckEnabled() and not is_wearing:
                current_time = time.time()
                if current_time - last_wearing_warning_time >= warning_interval:
                    logger.warning(
                        "HEADSET NOT WORN - Electrodes have no contact. "
                        "Predictions paused until headset is worn properly."
                    )
                    last_wearing_warning_time = current_time
                was_wearing = False
                continue
            
            if not was_wearing and is_wearing:
                epoch_maker.reset()
                logger.info("Headset worn again - cleared buffer, resuming predictions.")
            was_wearing = True
            
            for epoch in epoch_maker.push(chunk):
                feats = extract_features(epoch, fs=fs)
                x = np.array([[feats[k] for k in feature_names]])
                
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(x)[0]
                else:
                    df = model.decision_function(x)
                    df = np.vstack([-df, df]).T if df.ndim == 1 else df
                    ex = np.exp(df - np.max(df))
                    proba = (ex / np.sum(ex, axis=1, keepdims=True))[0]
                
                try:
                    label = str(model.predict(x)[0])
                except Exception:
                    label = str(classes[np.argmax(proba)]) if classes else "UNKNOWN"
                
                strength = _compute_focus_strength(feats, proba, classes)
                timestamp = datetime.now(timezone.utc).isoformat()
                
                try:
                    callback(label, strength, timestamp)
                except Exception as e:
                    logger.warning("Callback error: %s", e)
    
    except Exception as e:
        logger.exception("Inference loop failed: %s", e)


async def startFocusInference(
    subjectId: str,
    client,
    updateHz: float,
    callback: Callable[[str, int, str], None]
) -> bool:
    """Start live focus inference.
    
    Args:
        subjectId: Subject with trained model.
        client: Connected BleakClient.
        updateHz: Focus score updates per second.
        callback: Called with (label, focus_strength, timestamp).
    
    Returns:
        True if started successfully.
    """
    global _inference_task, _stop_event
    
    if _inference_task and not _inference_task.done():
        raise RuntimeError("Inference already running.")
    if updateHz <= 0:
        raise ValueError("updateHz must be > 0")
    if client is None:
        raise ValueError("client is required")
    
    _load_model(subjectId)
    
    _stop_event = asyncio.Event()
    _inference_task = asyncio.create_task(
        _inference_loop(subjectId, client, updateHz, callback, _stop_event)
    )
    return True


async def stopFocusInference() -> None:
    """Stop live focus inference."""
    global _inference_task, _stop_event
    
    if _inference_task is None:
        return
    
    if _stop_event:
        _stop_event.set()
    
    try:
        await _inference_task
    finally:
        _inference_task = None
        _stop_event = None
