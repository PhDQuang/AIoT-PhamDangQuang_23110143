"""Local web demo backed by a trained PPG decoder (no extra dependencies)."""
from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch

from .models.cvae import Decoder
from .evaluation.heart_rate import estimate_hr
from .baselines.gaussian import GaussianParams, synthesize_two_gaussian

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / '.krun/jobs/20261004-151954-483f31/output/krun_outputs/artifacts'


def read_signal(name: str, data: bytes, column: str = '') -> np.ndarray:
    """Read a single continuous vector; never flatten unrelated NPZ windows."""
    suffix = Path(name).suffix.lower()
    if suffix in {'.npz', '.npy'}:
        loaded = np.load(io.BytesIO(data), allow_pickle=False)
        if suffix == '.npz':
            with loaded as archive:
                key = column or next((k for k in ('ppg', 'PPG', 'signal') if k in archive), '')
                if key not in archive:
                    raise ValueError('NPZ cần khóa ppg, PPG hoặc signal; có thể nhập tên khóa bên dưới.')
                values = archive[key]
        else:
            values = loaded
        values = np.asarray(values).squeeze()
        if values.ndim != 1:
            raise ValueError('Cần một chuỗi PPG liên tục, không dùng ma trận nhiều cửa sổ.')
    elif suffix in {'.csv', '.txt'}:
        import pandas as pd
        content = data.decode('utf-8-sig')
        separator = ',' if ',' in content.splitlines()[0] else ( ';' if ';' in content.splitlines()[0] else r'\s+')
        first = content.splitlines()[0].replace(';', ' ').replace(',', ' ').split()
        try:
            [float(v) for v in first]
            header = None
        except ValueError:
            header = 0
        frame = pd.read_csv(io.StringIO(content), sep=separator, header=header)
        if column:
            key = column if header == 0 else int(column)
            if key not in frame:
                raise ValueError('Không tìm thấy cột PPG đã chọn.')
        else:
            key = next((k for k in frame.columns if str(k).lower() in {'ppg', 'signal', 'bvp'}), None)
            if key is None:
                if len(frame.columns) != 1:
                    raise ValueError('File nhiều cột: hãy nhập tên cột PPG (hoặc số cột từ 0 nếu không có tiêu đề).')
                key = frame.columns[0]
        values = frame[key].to_numpy(dtype=float)
    else:
        raise ValueError('Hỗ trợ CSV, TXT, NPY và NPZ chứa tín hiệu PPG, không hỗ trợ ảnh hoặc video.')
    values = np.asarray(values, dtype=np.float32)
    if not 512 <= values.size <= 64000:
        raise ValueError('Tín hiệu cần từ 512 đến 64.000 mẫu (8–1.000 giây ở 64 Hz).')
    if not np.isfinite(values).all() or np.ptp(values) < 1e-8:
        raise ValueError('Tín hiệu phải hữu hạn và có biến thiên, không chứa NaN/Inf.')
    return values


class Demo:
    def __init__(self, checkpoint: Path, detector: Path | None):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.latent_dim = state['fc.0.weight'].shape[1] - 1
        self.decoder = Decoder(self.latent_dim)
        self.decoder.load_state_dict(state)
        self.decoder.eval()
        self.checkpoint = str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else checkpoint.name
        self.detector = json.loads(detector.read_text(encoding='utf-8')) if detector and detector.exists() else {}
        bank_path = checkpoint.parents[3] / 'gaussian/fold_01/seed_42/gaussian_fit.json'
        self.bank = [GaussianParams(**p) for p in json.loads(bank_path.read_text(encoding='utf-8'))['parameters']] if bank_path.exists() else None

    def synthesize(self, body):
        hr, seed = float(body['hr']), int(body.get('seed', 42))
        count = int(body.get('count', 1))
        if hr not in (60, 80, 120) or not 0 <= seed <= 2147483647 or not 1 <= count <= 20:
            raise ValueError('HR: 60/80/120 bpm; seed: 0–2147483647; số mẫu: 1–20.')
        rng = torch.Generator(device='cpu').manual_seed(seed)
        with torch.inference_mode():
            generated = self.decoder(torch.randn(count, self.latent_dim, generator=rng), torch.full((count,), hr/60)).numpy()[:, 0]
        gaussian = synthesize_two_gaussian(hr, count, params=self.bank, seed=seed)[:, 0] if self.bank else None
        def summarize(signals):
            rows = [estimate_hr(s, **self.detector).__dict__ for s in signals]
            errors = [abs(m['hr']-hr) for m in rows if m['measurable']]
            return dict(measurements=rows, mae=float(np.mean(errors)) if errors else None,
                        V=len(errors)/count, A3=sum(e <= 3 for e in errors)/count,
                        failures=count-len(errors))
        return dict(fs=64, duration_seconds=8, target_hr=hr, seed=seed, count=count,
                    cvae=generated.tolist(), gaussian=gaussian.tolist() if gaussian is not None else None,
                    cvae_metrics=summarize(generated), gaussian_metrics=summarize(gaussian) if gaussian is not None else None,
                    checkpoint=self.checkpoint, gaussian_source='Ngân hàng tham số khớp trên train, cùng fold' if self.bank else 'Không tìm thấy ngân hàng Gaussian',
                    detector=self.detector, normalization='Biên độ chuẩn hóa của mô hình; không phải điện áp cảm biến')

    def generate(self, body):
        data = base64.b64decode(body['data'], validate=True)
        signal = read_signal(body['name'], data, body.get('column', '').strip())
        mode = body.get('mode', 'generate')
        if mode not in {'generate', 'continuation'}:
            raise ValueError('Chế độ không hợp lệ.')
        hr, segments, seed = float(body['hr']), int(body['segments']), int(body['seed'])
        if hr not in (60, 80, 120) or not 1 <= segments <= 8 or not 0 <= seed <= 2147483647:
            raise ValueError('HR phải là 60/80/120; số đoạn 1–8; seed 0–2147483647.')
        # Display normalization is explicit and does not pretend to be training normalization.
        normalized = bool(body.get('normalized', True))
        reference = None
        if mode == 'continuation':
            if not 1024 <= signal.size <= 4608 or signal.size % 512:
                raise ValueError('So sánh tiếp diễn cần file 16–72 giây, gồm các đoạn đủ 8 giây; file 24 giây cần 1.536 mẫu ở 64 Hz.')
            context = signal[:512]
            if np.ptp(context) < 1e-8:
                raise ValueError('8 giây đầu không có biến thiên để lấy HR hoặc chuẩn hóa.')
            # Fit display scaling on context only: future data must not affect inference.
            shown = signal if normalized else (signal - context.mean()) / context.std()
            reference = shown[512:].copy()
            shown = shown[:512]
            segments = reference.size // 512
            initial_hr = estimate_hr(shown, **self.detector)
            if not initial_hr.measurable:
                raise ValueError('Không đo được HR trong 8 giây đầu; cần tín hiệu có ít nhất 3 đỉnh hợp lệ.')
            hr = initial_hr.hr
        else:
            shown = signal if normalized else (signal - signal.mean()) / signal.std()
        if not np.isfinite(shown).all() or (reference is not None and not np.isfinite(reference).all()):
            raise ValueError('8 giây đầu không đủ biến thiên để chuẩn hóa.')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        with torch.inference_mode():
            z = torch.randn(segments, self.latent_dim, generator=generator)
            generated = self.decoder(z, torch.full((segments,), hr / 60)).numpy()[:, 0]
        measurements = [estimate_hr(s, **self.detector).__dict__ for s in generated]
        original_hr = estimate_hr(shown[-512:], **self.detector).__dict__
        return dict(input=shown.tolist(), generated=generated.reshape(-1).tolist(), fs=64,
                    target_hr=hr, measurements=measurements, input_hr=original_hr,
                    mode=mode, reference=reference.tolist() if reference is not None else None,
                    comparison_start_seconds=8 if reference is not None else 0,
                    rmse=float(np.sqrt(np.mean((reference-generated.reshape(-1))**2))) if reference is not None else None,
                    checkpoint=self.checkpoint, seed=seed,
                    normalization='Đầu vào đã chuẩn hóa' if normalized else ('Chuẩn hóa theo 8 giây đầu, chỉ để hiển thị' if mode == 'continuation' else 'Đầu vào z-score theo file, chỉ để so sánh hình ảnh'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--detector-config', type=Path)
    args = parser.parse_args(argv)
    roots = [ROOT / 'results', FINAL, ROOT / 'artifacts']
    checkpoint = args.checkpoint or next((r / 'cvae/fold_01/seed_42/decoder_state_dict.pt' for r in roots
                                          if (r / 'cvae/fold_01/seed_42/decoder_state_dict.pt').exists()), None)
    if checkpoint is None or not checkpoint.is_file():
        parser.error('Không tìm thấy checkpoint. Truyền --checkpoint đường/dẫn/decoder_state_dict.pt')
    detector = args.detector_config or checkpoint.parents[3] / 'proxy/fold_01/detector_config.json'
    torch.set_num_threads(1)
    demo = Demo(checkpoint, detector)

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, data, content_type):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/':
                self.send(200, (ROOT / 'web/index.html').read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/api/status':
                self.send(200, json.dumps({'checkpoint': demo.checkpoint, 'gaussian_available': demo.bank is not None}).encode(), 'application/json')
            elif self.path == '/app.js':
                self.send(200, (ROOT / 'web/app.js').read_bytes(), 'application/javascript; charset=utf-8')
            else:
                self.send(404, b'Not found', 'text/plain')

        def do_POST(self):
            if self.path not in {'/api/generate', '/api/synthesize'}:
                self.send(404, b'Not found', 'text/plain')
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 5_000_000:
                    raise ValueError('File quá lớn (giới hạn request 5 MB).')
                body = json.loads(self.rfile.read(length))
                result = demo.synthesize(body) if self.path == '/api/synthesize' else demo.generate(body)
                self.send(200, json.dumps(result, allow_nan=False).encode(), 'application/json')
            except (ValueError, KeyError, TypeError, IndexError, UnicodeError, OSError) as error:
                self.send(400, json.dumps({'error': str(error)}, ensure_ascii=False).encode(), 'application/json')

    server = HTTPServer((args.host, args.port), Handler)
    print(f'PPG Studio: http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
