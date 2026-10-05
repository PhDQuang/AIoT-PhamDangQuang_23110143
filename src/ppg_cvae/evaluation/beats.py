"""Complete foot-to-foot beats using the same detector for all methods."""
import numpy as np
from scipy.signal import find_peaks
from .heart_rate import estimate_hr


def extract_beats(signal, fs=64, min_distance=20, prominence=None):
    x = np.asarray(signal, float).squeeze()
    if not estimate_hr(x, fs, min_distance, prominence).measurable:
        return [], 0
    peaks, _ = find_peaks(x, distance=min_distance, prominence=prominence)
    feet = [a+int(np.argmin(x[a:b+1])) for a, b in zip(peaks[:-1], peaks[1:])]
    beats = []
    for a, b in zip(feet[:-1], feet[1:]):
        if b-a > 2:
            beat = x[a:b+1]
            peak = int(np.argmax(beat))
            if 0 < peak < len(beat)-1 and np.ptp(beat) > 1e-8:
                beats.append(beat)
    return beats, max(0, len(feet)-1)
