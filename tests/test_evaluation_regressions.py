import numpy as np
from ppg_cvae.evaluation.heart_rate import generation_metrics
from ppg_cvae.evaluation.morphology import pulse_features
from ppg_cvae.evaluation.diversity import aligned_beat_diversity


def test_a3_counts_failures_in_denominator():
    sine = np.sin(2*np.pi*np.arange(512)/64)
    _, metrics = generation_metrics([sine, np.zeros(512)], 60)
    assert metrics['V'] == metrics['A3'] == .5
    assert metrics['failed_windows'] == 1


def test_complete_beats_have_positive_rise_times():
    sine = np.sin(2*np.pi*np.arange(512)/64)
    features = pulse_features(sine)
    assert features['rise_time_s'] == .5
    assert features['rise_cycle_ratio'] == .5
    assert features['beat_extraction_success'] == 1


def test_phase_shift_not_reported_as_shape_diversity():
    t = np.arange(512)/64
    signals = np.array([np.sin(2*np.pi*t+phase) for phase in (0, np.pi/2, np.pi)])
    assert aligned_beat_diversity(signals)['mean_temporal_std'] < 1e-6
