import json
import os
import numpy as np
import tempfile
from datetime import datetime

# Drift history stored as a simple JSON file alongside the DB
DRIFT_FILE = os.path.join(tempfile.gettempdir(), "embedding_drift.json")

# EMA alpha — how much weight the current scan gets vs the stored embedding.
# 0.1 = slow adaptation (10% new, 90% old). Good for gradual face changes.
EMA_ALPHA = 0.10

# Drift warning thresholds (cosine distance from original embedding)
DRIFT_WARNING_THRESHOLD = 0.12   # Amber — face changing noticeably
DRIFT_ALERT_THRESHOLD   = 0.20   # Red   — suggest re-enrollment


def _load_drift_data():
    """Load drift history from JSON file. Returns empty dict if not found."""
    if os.path.exists(DRIFT_FILE):
        try:
            with open(DRIFT_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_drift_data(data):
    """Persist drift history to JSON file."""
    with open(DRIFT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def compute_drift(old_emb: np.ndarray, new_emb: np.ndarray) -> float:
    """
    Compute cosine distance between two embeddings.
    Returns a value between 0.0 (identical) and 2.0 (opposite).
    Typical values:
        < 0.05  → negligible drift (same conditions)
        0.05–0.12 → minor drift (different lighting/angle)
        0.12–0.20 → noticeable drift (facial changes)
        > 0.20  → significant drift (re-enrollment recommended)
    """
    old_norm = old_emb / (np.linalg.norm(old_emb) + 1e-8)
    new_norm = new_emb / (np.linalg.norm(new_emb) + 1e-8)
    cosine_similarity = float(np.dot(old_norm, new_norm))
    # Clamp to [-1, 1] to handle floating point edge cases
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
    return round(1.0 - cosine_similarity, 4)


def apply_ema_update(stored_emb: np.ndarray, current_emb: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    """
    Apply Exponential Moving Average update to the stored embedding.

    Formula: new = (1 - alpha) * stored + alpha * current

    The result is re-normalized to unit length so it stays on the
    hypersphere that cosine similarity operates on.

    Args:
        stored_emb:  The currently stored face embedding (numpy array)
        current_emb: The embedding from this recognition event
        alpha:       EMA weight for the new observation (default 0.10)

    Returns:
        Updated, unit-normalized embedding (numpy array, float32)
    """
    updated = (1.0 - alpha) * stored_emb + alpha * current_emb
    norm = np.linalg.norm(updated)
    if norm > 1e-8:
        updated = updated / norm
    return updated.astype(np.float32)


def record_drift(user_id: int, drift_score: float):
    """
    Append a drift observation to the user's history log.
    Keeps the last 30 observations per user.
    """
    data = _load_drift_data()
    key = str(user_id)

    if key not in data:
        data[key] = {"history": [], "first_seen": datetime.now().isoformat()}

    data[key]["history"].append({
        "drift": drift_score,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    # Rolling window: keep last 30 observations
    data[key]["history"] = data[key]["history"][-30:]
    data[key]["last_drift"] = drift_score
    data[key]["avg_drift"] = round(
        float(np.mean([h["drift"] for h in data[key]["history"]])), 4
    )

    _save_drift_data(data)


def get_drift_status(user_id: int) -> dict:
    """
    Return the drift status for a user.

    Returns a dict with:
        status:     "stable" | "warning" | "alert"
        avg_drift:  Average cosine distance across recent scans
        last_drift: Most recent cosine distance
        scan_count: Number of recognition events recorded
        label:      Human-readable status label
    """
    data = _load_drift_data()
    key = str(user_id)

    if key not in data or not data[key].get("history"):
        return {
            "status": "stable",
            "avg_drift": 0.0,
            "last_drift": 0.0,
            "scan_count": 0,
            "label": "No data yet",
        }

    entry = data[key]
    avg   = entry.get("avg_drift", 0.0)
    last  = entry.get("last_drift", 0.0)
    count = len(entry.get("history", []))

    if avg >= DRIFT_ALERT_THRESHOLD:
        status = "alert"
        label  = "Re-enrollment recommended"
    elif avg >= DRIFT_WARNING_THRESHOLD:
        status = "warning"
        label  = "Face changing — monitor"
    else:
        status = "stable"
        label  = "Embedding stable"

    return {
        "status":     status,
        "avg_drift":  avg,
        "last_drift": last,
        "scan_count": count,
        "label":      label,
    }


def get_all_drift_statuses() -> dict:
    """Return drift status for every user in the drift file."""
    data = _load_drift_data()
    return {int(uid): get_drift_status(int(uid)) for uid in data.keys()}


def clear_drift_for_user(user_id: int):
    """Remove drift history for a user (call on deletion or re-enrollment)."""
    data = _load_drift_data()
    data.pop(str(user_id), None)
    _save_drift_data(data)
