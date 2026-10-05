import base64
import io

import numpy as np
import pytest
import torch

from ppg_cvae.models.cvae import Decoder
from ppg_cvae.web import Demo, read_signal
from ppg_cvae.baselines.gaussian import GaussianParams


def test_upload_selects_ppg_without_time_column():
    content = 'time,ppg\n' + '\n'.join(f'{i/64},{np.sin(i/8)}' for i in range(512))
    signal = read_signal('sensor.csv', content.encode())
    assert signal.shape == (512,)
    assert signal[100] == pytest.approx(np.sin(100/8))


def test_upload_rejects_unrelated_windows_and_invalid_values():
    buffer = io.BytesIO()
    np.savez(buffer, ppg=np.ones((2, 1, 512)))
    with pytest.raises(ValueError, match='liên tục'):
        read_signal('windows.npz', buffer.getvalue())
    content = '\n'.join(['nan'] + [str(np.sin(i)) for i in range(511)])
    with pytest.raises(ValueError):
        read_signal('bad.txt', content.encode())


def test_generation_is_reproducible_and_preserves_input(tmp_path):
    checkpoint = tmp_path / 'decoder.pt'
    torch.save(Decoder().state_dict(), checkpoint)
    demo = Demo(checkpoint, None)
    signal = np.sin(np.arange(512) / 8).astype(np.float32)
    buffer = io.BytesIO()
    np.save(buffer, signal)
    body = dict(name='ppg.npy', data=base64.b64encode(buffer.getvalue()).decode(),
                hr=80, segments=2, seed=42, normalized=True)
    first = demo.generate(body)
    second = demo.generate(body)
    assert first['input'] == signal.tolist()
    assert len(first['generated']) == 1024
    assert first['generated'] == second['generated']
    body['seed'] = 43
    assert demo.generate(body)['generated'] != first['generated']
    body['hr'] = 90
    with pytest.raises(ValueError):
        demo.generate(body)


def test_continuation_holds_out_future_without_leakage(tmp_path):
    checkpoint = tmp_path / 'decoder.pt'
    torch.save(Decoder().state_dict(), checkpoint)
    demo = Demo(checkpoint, None)
    signal = np.sin(2*np.pi*(80/60)*np.arange(1536)/64).astype(np.float32)

    def run(values, normalized):
        buffer = io.BytesIO()
        np.save(buffer, values)
        return demo.generate(dict(name='24s.npy', data=base64.b64encode(buffer.getvalue()).decode(),
                                  hr=80, segments=2, seed=42, normalized=normalized, mode='continuation'))

    first = run(signal, True)
    assert len(first['input']) == 512
    assert first['reference'] == signal[512:].tolist()
    assert len(first['generated']) == 1024
    assert first['comparison_start_seconds'] == 8
    assert first['rmse'] == pytest.approx(np.sqrt(np.mean((np.array(first['reference'])-first['generated'])**2)))
    changed = signal.copy()
    changed[512:] = changed[512:]*100+1000
    for normalized in (True, False):
        a, b = run(signal, normalized), run(changed, normalized)
        assert a['input'] == b['input']
        assert a['generated'] == b['generated']
        assert a['target_hr'] == b['target_hr']
        assert a['reference'] != b['reference']
    with pytest.raises(ValueError):
        run(signal[:512], True)


def test_hr_demo_samples_prior_and_reports_independent_hr(tmp_path):
    checkpoint = tmp_path / 'decoder.pt'
    torch.save(Decoder().state_dict(), checkpoint)
    demo = Demo(checkpoint, None)
    demo.bank = [GaussianParams()]
    body = {'hr': 80, 'seed': 42, 'count': 5}
    first, second = demo.synthesize(body), demo.synthesize(body)
    assert np.asarray(first['cvae']).shape == (5, 512)
    assert np.asarray(first['gaussian']).shape == (5, 512)
    assert first['cvae'] == second['cvae']
    assert first['gaussian'] == second['gaussian']
    assert first['cvae_metrics']['failures'] == sum(not m['measurable'] for m in first['cvae_metrics']['measurements'])
    errors = [abs(m['hr']-80) for m in first['gaussian_metrics']['measurements'] if m['measurable']]
    assert first['gaussian_metrics']['mae'] == pytest.approx(np.mean(errors))
    for bad in ({'hr': 90}, {'count': 0}, {'seed': -1}):
        with pytest.raises(ValueError):
            demo.synthesize(body | bad)
