"""Package source, actual weights and readable results without raw data or secrets."""
from pathlib import Path
import csv
import hashlib
import io
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'.krun/jobs/20261004-151954-483f31/output/krun_outputs/artifacts'


def main():
    target = ROOT/'delivery/PPG_CVAE_G2_Ban_giao.zip'
    target.parent.mkdir(exist_ok=True)
    files = {}
    for directory in ['src', 'scripts', 'tests', 'notebooks', 'configs', 'docs']:
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.pkl'}:
                files[p.relative_to(ROOT).as_posix()] = p
    for name in ['README.md', 'pyproject.toml', 'requirements.txt', 'requirements-kaggle.txt',
                 'requirements-motion.txt', 'krun.yaml', '.krunignore', '.gitignore']:
        p = ROOT/name
        if p.exists(): files[name] = p
    for p in ART.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.pt', '.csv', '.json', '.yaml', '.png', '.txt'}:
            files['results/'+p.relative_to(ART).as_posix()] = p
    for p in (ROOT/'artifacts/completion').rglob('*'):
        if p.is_file(): files['completion/'+p.relative_to(ROOT/'artifacts/completion').as_posix()] = p
    for p in (ROOT/'reports/work/figures').glob('*.png'):
        files['reports/work/figures/'+p.name] = p
    for name in ['report_content.json', 'local_machine.json', 'metrics.json']:
        p = ROOT/'reports/work'/name
        if p.exists(): files['reports/work/'+name] = p
    files['reports/Bao_cao_PPG_CVAE_AI_cho_IoT.docx'] = ROOT/'reports/Bao_cao_PPG_CVAE_AI_cho_IoT.docx'
    instructions = '''# Hướng dẫn bàn giao G2

Báo cáo: reports/Bao_cao_PPG_CVAE_AI_cho_IoT.docx. Năm học 2026–2027.
Đây là kết quả thực nghiệm có mục tiêu HR chưa đạt; không phải mô hình y khoa.

## Chạy CPU sau khi giải nén

Mở PowerShell trong thư mục gói, chạy:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD/src"
.venv/Scripts/python.exe -m ppg_cvae.cli.generate --checkpoint results/cvae/fold_01/seed_42/decoder_state_dict.pt --hr 80 --seed 42 --num-samples 10 --detector-config results/proxy/fold_01/detector_config.json --output demo/generated_ppg.npz
.venv/Scripts/python.exe -m pytest -q
```

CLI in HR mục tiêu, HR đo và measurable; NPZ lưu waveform. HR đo không thay bằng target.
completion/demo là đầu ra CPU đã chạy. completion/local_cpu_* là benchmark laptop,
results/summary/resource_metrics.csv là benchmark Kaggle; không trộn hai môi trường.
Các environment.json ghi phiên bản môi trường thực nghiệm; requirements là file cài đặt,
không phải bảo đảm bitwise giữa hệ điều hành/phần cứng khác nhau.

## Nội dung kết quả

results có checkpoint cVAE/ablation/proxy, bank Gaussian, cấu hình, manifest,
normalization, detector, history, bảng từng mẫu và bảng/hình tổng hợp.
completion có thống kê bổ sung, độ phủ, kiểm tra ACC, KL snapshot và benchmark CPU.
MANIFEST_SHA256.csv lưu SHA256/byte của mọi file bàn giao, trừ chính manifest.
Gói không chứa API token, .venv, credentials hoặc dữ liệu cảm biến thô.

## Dataset và tái tạo thực nghiệm

Nguồn thô: https://www.kaggle.com/datasets/ameersifat53/ppg-dalia-dataset
Prepared fold 1: https://www.kaggle.com/datasets/hcsinhgoethe/ppg-cvae-prepared-fold01
Prepared fold 2–5: https://www.kaggle.com/datasets/hcsinhgoethe/ppg-cvae-prepared-folds02-05
Job cuối: 20261004-151954-483f31. Xem README và docs/OVERNIGHT_RUN.md.
Không đóng lại toàn bộ train/val/test NPZ hoặc 120 NPZ sinh lớn; có manifest,
tham số/seed và mã tái tạo. Giữ output gốc ngoài gói nếu cần truy nguyên từng waveform.
scripts/complete_delivery.py dùng vị trí output gốc trong project; gói chạy demo bằng results/.

## Giới hạn

MAE cVAE tại 60/80/120 khoảng 17,88/8,47/34,09 bpm, chưa đạt ≤3.
V cao không nghĩa đúng HR. Chưa triển khai MCU, INT8 hoặc chứng minh tăng cường dữ liệu.
Phân tầng motion là audit mô tả sau thực nghiệm, không đổi tập hợp lệ.
KL từng chiều bổ sung là snapshot checkpoint cuối; history cũ không được dựng lại.
Theo phạm vi đề cương, CPU máy thử được phép; web và slide không bắt buộc.
'''
    entries = []
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, p in sorted(files.items()):
            archive.write(p, name)
            entries.append([name, p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()])
        extra = instructions.encode('utf-8')
        archive.writestr('HUONG_DAN_BAN_GIAO.md', extra)
        entries.append(['HUONG_DAN_BAN_GIAO.md', len(extra), hashlib.sha256(extra).hexdigest()])
        buf = io.StringIO(); writer = csv.writer(buf)
        writer.writerow(['path', 'bytes', 'sha256']); writer.writerows(entries)
        archive.writestr('MANIFEST_SHA256.csv', buf.getvalue().encode('utf-8-sig'))
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad: raise ValueError(f'ZIP CRC failed: {bad}')
        manifest = csv.DictReader(io.StringIO(archive.read('MANIFEST_SHA256.csv').decode('utf-8-sig')))
        for row in manifest:
            data = archive.read(row['path'])
            if len(data) != int(row['bytes']) or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError(f'Hash mismatch {row["path"]}')
    result = {'zip': str(target.relative_to(ROOT)), 'files': len(entries)+1,
              'bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'crc_and_all_file_hashes': 'passed'}
    (target.parent/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
