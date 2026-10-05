"""Create the named final report with a clickable source repository link."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import re
import html

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'reports/Bao_cao_PPG_CVAE_AI_cho_IoT_Day_Du.docx'
target=ROOT/'reports/PhamDangQuang_final.docx'
url='https://github.com/PhDQuang/AIoT-PhamDangQuang_23110143'
with ZipFile(source) as z:
    files={n:z.read(n) for n in z.namelist()}
xml=files['word/document.xml'].decode('utf-8')
rels=files['word/_rels/document.xml.rels'].decode('utf-8')
rid='rIdFinalGithub'
rels=rels.replace('</Relationships>',f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="{url}" TargetMode="External"/></Relationships>')
paragraph=f'''<w:p><w:pPr><w:spacing w:after="120"/><w:jc w:val="left"/></w:pPr><w:r><w:t>Mã nguồn và notebook của đề tài được công bố tại GitHub:</w:t></w:r></w:p>
<w:p><w:pPr><w:spacing w:after="120"/><w:jc w:val="left"/></w:pPr><w:hyperlink r:id="{rid}"><w:r><w:rPr><w:color w:val="0563C1"/><w:u w:val="single"/></w:rPr><w:t>{html.escape(url)}</w:t></w:r></w:hyperlink></w:p>
<w:p><w:r><w:t>Repository gồm mã nguồn, notebook nguồn và bản thực thi thành công, cấu hình, checkpoint, kết quả thực nghiệm và hướng dẫn chạy web. Gói PhamDangQuang_final.zip đính kèm trên LMS lưu bản mã nguồn cùng các tệp cần thiết để chạy demo.</w:t></w:r></w:p>'''
inserted=False
def insert(match):
    global inserted
    p=match.group()
    if not inserted and 'Heading2' in p and re.search(r'<w:t[^>]*>7\.2\.',p):
        inserted=True
        return paragraph+p
    return p
xml=re.sub(r'<w:p(?:\s[^>]*)?>.*?</w:p>',insert,xml,flags=re.S)
assert inserted
xml=xml.replace('reports/Bao_cao_PPG_CVAE_AI_cho_IoT_Day_Du.docx','reports/PhamDangQuang_final.docx')
files['word/document.xml']=xml.encode('utf-8')
files['word/_rels/document.xml.rels']=rels.encode('utf-8')
with ZipFile(target,'w',ZIP_DEFLATED) as z:
    for n,b in files.items(): z.writestr(n,b)
print(target)
