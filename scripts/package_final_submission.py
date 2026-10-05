"""Package the named LMS files after the repository has been pushed."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import subprocess
import shutil
import hashlib
import csv
import io
import json

ROOT=Path(__file__).resolve().parents[1]
git=['git','-c',f'safe.directory={ROOT.as_posix()}']
commit=subprocess.check_output(git+['rev-parse','HEAD'],cwd=ROOT,text=True).strip()
remote=subprocess.check_output(git+['ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0]
assert commit == remote, 'Package only the commit verified on GitHub.'
target=ROOT/'delivery/PhamDangQuang_final'
target.mkdir(parents=True,exist_ok=True)
names=subprocess.check_output(git+['ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
demo_files={
 'results/cvae/fold_01/seed_42/decoder_state_dict.pt',
 'results/proxy/fold_01/detector_config.json',
 'results/gaussian/fold_01/seed_42/gaussian_fit.json',
 'results/data/fold_01/normalization.json',
}
files=[]
for name in names:
    if not name: continue
    if name.startswith('results/') and not (name in demo_files or name.startswith('results/summary/')): continue
    files.append(name)
archive_path=target/'PhamDangQuang_final.zip'
rows=[]
with ZipFile(archive_path,'w',ZIP_DEFLATED,compresslevel=6) as archive:
    for name in files:
        file=ROOT/name
        data=file.read_bytes()
        archive.writestr('PhamDangQuang_final/'+name,data)
        rows.append([name,len(data),hashlib.sha256(data).hexdigest()])
    buf=io.StringIO(); writer=csv.writer(buf)
    writer.writerow(['path','bytes','sha256']); writer.writerows(rows)
    archive.writestr('PhamDangQuang_final/MANIFEST_SHA256.csv',buf.getvalue().encode('utf-8-sig'))
    archive.writestr('PhamDangQuang_final/GITHUB_COMMIT.txt',(commit+'\n').encode())
    archive.writestr('PhamDangQuang_final/NOI_DUNG_GOI.md',(
      '# Gói mã nguồn nộp LMS\n\n'
      'Gói gồm mã nguồn, notebook nguồn và đã thực thi, cấu hình, tài liệu, kết quả tổng hợp và trọng số cần chạy web demo fold 1 seed 42.\n\n'
      'Các checkpoint của toàn bộ fold/seed và bảng theo từng run được lưu trên GitHub: https://github.com/PhDQuang/AIoT-PhamDangQuang_23110143 .\n\n'
      'Chạy demo theo README.md. Không cần dữ liệu thô cho suy luận. Khi chạy lại huấn luyện, tải dataset theo hướng dẫn notebook.\n').encode('utf-8'))
with ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    for name,size,sha in rows:
        assert hashlib.sha256(archive.read('PhamDangQuang_final/'+name)).hexdigest() == sha
for suffix in ('.docx','.pptx'):
    shutil.copy2(ROOT/('reports/PhamDangQuang_final'+suffix),target/('PhamDangQuang_final'+suffix))
shutil.copy2(ROOT/'HUONG_DAN_NOP_BAI.md',target/'HUONG_DAN_NOP_BAI.md')
record={'commit':commit,'repository':'https://github.com/PhDQuang/AIoT-PhamDangQuang_23110143',
        'zip_files':len(rows),'zip_bytes':archive_path.stat().st_size,'crc':'passed','sha256':'passed'}
(ROOT/'delivery/final_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
