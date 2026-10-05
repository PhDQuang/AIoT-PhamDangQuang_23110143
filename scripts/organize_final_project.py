"""Collect successful notebook evidence and portable experiment artifacts."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT/'.krun/jobs/20261004-151954-483f31/output/krun_outputs/artifacts'
out = ROOT/'notebooks/executed'
out.mkdir(exist_ok=True)
selected = {}
for file in sorted((ROOT/'.krun/jobs').glob('*/output/krun_outputs/notebooks/*.executed.ipynb')):
    doc = json.loads(file.read_text(encoding='utf-8'))
    cells = [c for c in doc['cells'] if c['cell_type'] == 'code']
    errors = [o for c in cells for o in c.get('outputs', []) if o.get('output_type') == 'error']
    if cells and not errors and all(c.get('execution_count') is not None for c in cells):
        selected[file.name] = file
rows = []
for name, file in selected.items():
    shutil.copy2(file, out/name)
    rows.append({'notebook': name, 'job': file.parts[-5], 'status': 'executed_successfully'})
(out/'execution_index.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
lines = ['# Notebook và kết quả thực thi', '', 'Notebook nguồn ở thư mục này phục vụ chạy lại quy trình trên Kaggle.',
         'Thư mục `executed/` lưu bản thực thi thành công mới nhất tìm thấy trong output gốc, giữ nguyên output và execution count.', '',
         '| Notebook đã chạy | Job nguồn |', '|---|---|']
lines += [f"| [{r['notebook']}](executed/{r['notebook']}) | {r['job']} |" for r in rows]
lines += ['', 'Các notebook không có bản thực thi trong `executed/` được giữ dưới dạng mã nguồn. Kết quả tổng hợp cuối được lưu trong `results/summary/` cùng các cấu hình và bảng theo fold.',
          'Không tạo output hoặc execution count giả cho notebook nguồn.']
(ROOT/'notebooks/README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
count = 0
for file in FINAL.rglob('*'):
    if file.is_file() and file.suffix.lower() in {'.pt', '.csv', '.json', '.yaml', '.png', '.txt'}:
        target = ROOT/'results'/file.relative_to(FINAL)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
        count += 1
for file in (ROOT/'artifacts/completion').rglob('*'):
    if file.is_file() and file.suffix.lower() in {'.csv', '.json', '.png', '.md', '.txt'}:
        target = ROOT/'completion'/file.relative_to(ROOT/'artifacts/completion')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
print(f'Collected {len(rows)} executed notebooks and {count} portable artifact files.')
