#preprocess_extract_features.py
"""EEG preprocessing and feature extraction.

Public API:
    - extract_features(epoch, fs): Main feature extraction pipeline
    - preprocess_eeg(x, fs): Bandpass filter EEG signal
    - compute_bandpowers(x, fs): Compute spectral band powers

Constants:
    - BANDS: Canonical EEG frequency bands
    - DEFAULT_SAMPLE_RATE: 500 Hz
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

__all__ = ['extract_features', 'preprocess_eeg', 'compute_bandpowers', 'BANDS']

# =========================
# Canonical EEG bands
# =========================
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
TOTAL_BAND = (0.5, 45.0)


# =========================
# Filtering
# =========================
def filter_data(x, fs, hp, lp, order=4, axis=None):
    """Zero phase band pass (hp..lp)."""
    x = np.asarray(x, dtype=float)
    if axis is None:
        axis = -1 if (x.ndim == 1 or x.shape[-1] >= x.shape[0]) else 0
    
    nyq = 0.5 * fs
    if not (0 < hp < lp < nyq):
        raise ValueError(f"Cutoffs must satisfy 0 < hp < lp < {nyq} Hz.")
        
    sos_bp = butter(order, [hp, lp], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos_bp, x, axis=axis)


def preprocess_eeg(x, fs, hp=TOTAL_BAND[0], lp=TOTAL_BAND[1], order=4):
    """Preprocess: float cast, squeeze, broad bandpass."""
    x = np.asarray(x, dtype=float).squeeze()
    if x.ndim != 1:
        raise ValueError("Input epoch must be a 1D signal.")
    return filter_data(x, fs, hp, lp, order=order, axis=-1)


# =========================
# PSD / Bandpowers
# =========================
def compute_psd_welch(x, fs, nperseg=None, noverlap=None):
    x = np.asarray(x, dtype=float).squeeze()
    if nperseg is None:
        nperseg = min(int(2 * fs), len(x))
        nperseg = max(min(256, len(x)), nperseg)
        nperseg = min(nperseg, len(x))
    if noverlap is None:
        noverlap = nperseg // 2

    return welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
                 detrend="constant", scaling="density", average="mean")


def bandpower_from_psd(f, Pxx, band):
    lo, hi = band
    idx = (f >= lo) & (f < hi)
    if not np.any(idx): return 0.0
    return float(np.trapz(Pxx[idx], f[idx]))


def bandmax_from_psd(f, Pxx, band):
    lo, hi = band
    idx = (f >= lo) & (f < hi)
    if not np.any(idx): return 0.0
    return float(np.max(Pxx[idx]))


def compute_bandpowers(x, fs, bands=BANDS, total_band=TOTAL_BAND):
    f, Pxx = compute_psd_welch(x, fs)
    bp_abs = {name: bandpower_from_psd(f, Pxx, rng) for name, rng in bands.items()}
    total = bandpower_from_psd(f, Pxx, total_band)
    bp_rel = {name: (p / (total + 1e-12)) for name, p in bp_abs.items()}
    return bp_abs, bp_rel, f, Pxx # Returning f, Pxx to reuse for max power


# =========================
# Entropy & Ratios
# =========================
def _prob_normalize(vals, eps=1e-12):
    v = np.asarray(vals, dtype=float)
    v = np.clip(v, eps, np.inf)
    return v / (v.sum() + eps)

def shannon_entropy(p):
    p = _prob_normalize(p)
    return float(-np.sum(p * np.log2(p)))

def renyi_entropy(p, alpha=3.0):
    p = _prob_normalize(p)
    if alpha == 1.0: return shannon_entropy(p)
    return float(1.0 / (1.0 - alpha) * np.log2(np.sum(p ** alpha)))

def tsallis_entropy(p, alpha=3.0):
    p = _prob_normalize(p)
    if alpha == 1.0: return shannon_entropy(p)
    return float((1.0 / (alpha - 1.0)) * (1.0 - np.sum(p ** alpha)))

def kl_divergence(p, q, eps=1e-12):
    p = _prob_normalize(p, eps)
    q = _prob_normalize(q, eps)
    return float(np.sum(p * (np.log2(p) - np.log2(q))))

def mental_state_entropy_features(bp_rel, relax_bands=("alpha", "theta"), focus_bands=("beta", "gamma")):
    feats = {}
    def vec(names): return np.array([bp_rel.get(b, 0.0) for b in names], dtype=float)
    
    r, f = vec(relax_bands), vec(focus_bands)
    
    if r.size >= 2:
        p_r = _prob_normalize(r)
        feats["relax_shannon"] = shannon_entropy(p_r)
        feats["relax_renyi"] = renyi_entropy(p_r)
        feats["relax_tsallis"] = tsallis_entropy(p_r)

    if f.size >= 2:
        p_f = _prob_normalize(f)
        feats["focus_shannon"] = shannon_entropy(p_f)
        feats["focus_renyi"] = renyi_entropy(p_f)
        feats["focus_tsallis"] = tsallis_entropy(p_f)

    if r.size >= 2 and f.size == r.size:
        p_r, p_f = _prob_normalize(r), _prob_normalize(f)
        feats["kl_relax_to_focus"] = kl_divergence(p_r, p_f)
        feats["kl_focus_to_relax"] = kl_divergence(p_f, p_r)
    return feats


# =========================
# Time Domain
# =========================
def signal_shannon_entropy(x, bins=64, eps=1e-12):
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0: return 0.0
    hist, _ = np.histogram(x, bins=bins, density=False)
    p = hist.astype(float)
    p /= (p.sum() + eps)
    p = np.clip(p, eps, np.inf)
    return float(-np.sum(p * np.log2(p)))

def compute_band_time_features(x, fs, bands=BANDS, entropy_bins=64):
    feats = {}
    for name, (lo, hi) in bands.items():
        xb = filter_data(x, fs, lo, hi)
        # Note: Mean is removed because bandpass signals are zero-mean
        feats[f"{name}_std"] = float(np.std(xb))
        feats[f"{name}_ptp"] = float(np.ptp(xb))
        feats[f"{name}_entropy_amp"] = signal_shannon_entropy(xb, bins=entropy_bins)
    return feats


# =========================
# MAIN EXTRACTOR
# =========================
def extract_features(
    epoch, 
    fs=500, 
    bands=BANDS, 
    total_band=TOTAL_BAND, 
    pre_hp=TOTAL_BAND[0], 
    pre_lp=TOTAL_BAND[1]
):
    """
    Main pipeline.
    Usage: features = extract_features(my_numpy_array)
    """
    # 1) Preprocess
    x = preprocess_eeg(epoch, fs, hp=pre_hp, lp=pre_lp)

    feats = {}

    # 2) Spectral Features
    bp_abs, bp_rel, f, Pxx = compute_bandpowers(x, fs, bands=bands, total_band=total_band)
    
    for k, v in bp_abs.items(): feats[f"bp_abs_{k}"] = v
    for k, v in bp_rel.items(): feats[f"bp_rel_{k}"] = v

    # Max power per band (re-using computed Pxx)
    for name, rng in bands.items():
        feats[f"bp_max_{name}"] = bandmax_from_psd(f, Pxx, rng)

    # 3) Time Domain Features
    feats.update(compute_band_time_features(x, fs, bands=bands))

    # 4) Ratios
    beta = bp_rel.get("beta", 0.0)
    alpha = bp_rel.get("alpha", 0.0)
    theta = bp_rel.get("theta", 0.0)
    eps = 1e-12
    feats["ratio_engagement"] = float(beta / (alpha + theta + eps))
    feats["ratio_beta_alpha"] = float(beta / (alpha + eps))
    feats["ratio_theta_beta"] = float(theta / (beta + eps))

    # 5) Entropy Features
    feats.update(mental_state_entropy_features(bp_rel))

    return feats
