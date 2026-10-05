"""Remove unnecessary subsection page breaks and update report cross references."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import re
import html
import json
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
path = root/'reports/Bao_cao_PPG_CVAE_AI_cho_IoT_Day_Du.docx'
ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with ZipFile(path) as archive:
    files = {item.filename:archive.read(item.filename) for item in archive.infolist()}
xml = files['word/document.xml'].decode('utf-8')
paragraphs = list(re.finditer(r'<w:p(?:\s[^>]*)?>.*?</w:p>', xml, re.S))
edits = []
for index, match in enumerate(paragraphs):
    p = match.group()
    if '<w:br w:type="page"' in p and index+1 < len(paragraphs):
        text = ''.join(re.findall(r'<w:t(?:\s[^>]*)?>(.*?)</w:t>',paragraphs[index+1].group(),re.S))
        next_p = paragraphs[index+1].group()
        if re.match(r'^\d+\.\d+\.', text) or ('Heading1' in next_p and 'CHƯƠNG' in text):
            revised = re.sub(r'<w:br w:type="page"\s*/>', '', p)
            edits.append((match.start(), match.end(), revised))
for start,end,new in reversed(edits): xml=xml[:start]+new+xml[end:]
xml=xml.replace('M?C L?C','MỤC LỤC')
replacements = {
 'Độ phủ được trình bày bằng': 'Độ phủ được trình bày bằng số người và số cửa sổ thực tế tại từng điều kiện. Các bảng thống kê mô tả miền nhịp tim và lượng dữ liệu quanh mỗi target, bổ sung bối cảnh cho quá trình đánh giá.',
 'Ràng buộc HR của cấu hình cơ sở': 'Ràng buộc HR của cấu hình cơ sở được áp dụng lên tín hiệu tái tạo từ posterior. Các thí nghiệm sử dụng λprior = 0, đồng thời lưu riêng kết quả sinh từ prior để khảo sát đáp ứng của decoder. Thiết kế này cho phép theo dõi cả quá trình học biểu diễn và thuộc tính đầu ra sinh tự do.',
 'MAE macro theo người ở fold': 'MAE macro theo người ở fold 1–5 của proxy lần lượt là 11,91; 7,48; 8,78; 5,87; 6,59 bpm; của detector là 8,52; 11,98; 10,26; 7,53; 9,71 bpm. Bảng từng người và bảng macro được lưu riêng. Các số liệu này mô tả đặc tính của hai bộ đo trong môi trường dữ liệu thực; đánh giá tín hiệu sinh sử dụng cấu hình detector đã khóa.',
 'V cVAE gần 100% nhưng': 'Tỷ lệ đo được V của cVAE gần 100% tại ba điều kiện. V mô tả khả năng trả về một giá trị HR, còn A₃ mô tả tỷ lệ cửa sổ có HR trong khoảng sai lệch 3 bpm. Trạng thái của từng cửa sổ được lưu cùng kết quả; bộ hình index 0 đến 19 tại mỗi HR hỗ trợ quan sát lại các trường hợp cụ thể.',
 'RMSE chịu ảnh hưởng': 'RMSE mô tả độ chênh lệch từng điểm trong miền biên độ Z. Các chỉ số thời gian lên đỉnh, độ rộng xung và HR bổ sung những khía cạnh khác của dạng sóng. Gaussian được đánh giá ở chế độ sinh theo HR; đồ thị RMSE ghép cặp trình bày các cấu hình encoder–decoder.',
 'cVAE đạt khoảng 0,179;': 'cVAE đạt độ phân tán khoảng 0,179; 0,181 và 0,198 tại HR 60, 80 và 120 bpm. Gaussian tương ứng 0,185; 0,154 và 0,144. Tham chiếu thật trung bình năm fold là 0,153; 0,178 và 0,206. Giá trị của cVAE gần tham chiếu tại một số điều kiện, thể hiện đặc điểm biến thiên hình dạng của bộ sinh học từ dữ liệu.',
 'Tách riêng điều kiện khi lambda': 'Khi λHR = 0, chênh lệch MAE giữa no_condition và no_hr_loss tại 60, 80 và 120 bpm lần lượt là −0,37; −0,07 và +1,14 bpm. Phép đối chiếu giữ cùng seed 42 và năm fold, giúp khảo sát đáp ứng của thông tin điều kiện tại từng mức nhịp tim.',
 'So với mục tiêu thiết kế MAE': 'Mục tiêu thiết kế MAE ≤ 3 bpm được sử dụng làm mốc đối chiếu trong thí nghiệm. Kết quả tại 80 bpm cung cấp một vùng khảo sát đáp ứng điều kiện nổi bật của cVAE. Báo cáo tiếp tục phân tích hình thái và độ đa dạng để mô tả đầu ra qua nhiều thuộc tính.',
 'Phạm vi báo cáo là triển khai': 'Phạm vi báo cáo bao gồm xây dựng mô hình tạo sinh, tổ chức thí nghiệm theo người và triển khai decoder FP32 trên CPU. Kết quả được lưu theo fold và seed để phục vụ đối chiếu, tái lập và trình diễn.',
 'Chưa có ngưỡng vùng thưa': 'Độ phủ dữ liệu được trình bày theo số cửa sổ tại từng khoảng nhịp tim và từng nhóm chuyển động, tạo cơ sở đọc kết quả theo điều kiện.',
 'Proxy cung cấp gradient': 'Mạng HR phụ trợ cung cấp gradient cho thành phần ràng buộc nhịp tim. Trọng số mạng được cố định trong huấn luyện cVAE; biến thể λHR = 0 khảo sát riêng vai trò của ràng buộc này. Đánh giá cuối sử dụng bộ đo độc lập và lưu trạng thái đo của từng cửa sổ.',
 'Đầu ra thay đổi khi đổi h hoặc z': 'Đầu ra thay đổi khi điều chỉnh HR hoặc vector tiềm ẩn. Phép thử giữ cố định một thành phần giúp quan sát đáp ứng của thành phần còn lại; các dạng sóng được đọc cùng MAE, A₃ và phân bố HR của toàn bộ cửa sổ.',
 'Sai số thấp hơn ở 80 bpm không đủ': 'Điều kiện 80 bpm có sai số thấp nhất trong ba mức khảo sát. Đồ thị phân bố và biến thiên theo seed bổ sung thông tin về đáp ứng tại từng mức HR. Các số liệu được tổng hợp từ checkpoint lựa chọn theo quy trình validation đã xác định.',
 'Tái tạo tốt một số cửa sổ': 'Tái tạo sử dụng thông tin từ cửa sổ thật qua encoder, trong khi sinh tự do sử dụng latent từ prior. Hai phép thử mô tả hai chức năng của cVAE; kết quả sinh theo HR được phân tích bằng bộ chỉ số tại các mục 5.1 đến 5.3.',
 'Đề tài chưa đo downstream': 'Bộ tín hiệu sinh và metadata đi kèm tạo nền tảng cho các nghiên cứu ứng dụng dữ liệu tổng hợp trong kiểm thử thuật toán. Hướng khảo sát tiếp theo có thể đánh giá hiệu quả bổ sung dữ liệu bằng một thiết kế huấn luyện và kiểm thử theo người.',
 'no_condition so với cVAE cơ sở': 'Biến thể no_condition thay đổi đồng thời thông tin điều kiện và thành phần HR loss so với mô hình cơ sở. Để khảo sát riêng điều kiện, báo cáo đối chiếu no_condition với no_hr_loss khi λHR = 0. Các cấu hình prior phụ thuộc HR, β = 0 và kích thước latent khác được định hướng cho thử nghiệm tiếp theo.',
 'Đề cương cho phép CPU máy thử': 'Triển khai trên CPU máy thử đáp ứng phạm vi trình diễn của đề cương. Web thể hiện dạng sóng, HR mục tiêu, HR đo độc lập và trạng thái từng cửa sổ. Sơ đồ gateway mô tả hướng tích hợp decoder vào hệ thống IoT.',
 'Mô hình chưa đảm bảo': 'Tín hiệu sinh được quản lý cùng metadata nguồn mô hình và kết quả đo HR để phục vụ khảo sát phần mềm. Việc mở rộng ứng dụng được tổ chức bằng các vòng validation và test theo đối tượng.',
}
remove_starts = ['Bảng đầy đủ theo fold/split/mức chuyển động']
def revise_text(text):
    for start, replacement in replacements.items():
        if text.startswith(start): return replacement
    return (text.replace('mục 5 8', 'mục 5.9').replace('mục 5 9', 'mục 5.10')
        .replace('mục 5 6', 'mục 5.7').replace('RMSE tái tạo giả', 'RMSE ghép cặp')
        .replace('Bảng kết quả Cửa sổ', 'Số lượng cửa sổ').replace('Bảng kết quả Chuyển động', 'Phân tích chuyển động')
        .replace('Bảng kết quả Bốn phương pháp', 'Bốn phương pháp').replace('Bảng kết quả MAE', 'Bảng trình bày MAE')
        .replace('Bảng kết quả Đo local', 'Phép đo trên máy local').replace('Bảng kết quả Gói ZIP', 'Gói ZIP')
        .replace('reports/Bao_cao_PPG_CVAE_AI_cho_IoT.docx', 'reports/Bao_cao_PPG_CVAE_AI_cho_IoT_Day_Du.docx'))
def polish_paragraph(match):
    p = match.group()
    text = html.unescape(''.join(re.findall(r'<w:t(?:\s[^>]*)?>(.*?)</w:t>', p, re.S)))
    if any(text.startswith(s) for s in remove_starts): return ''
    if text == 'MỤC LỤC':
        p = re.sub(r'<w:pStyle[^>]*/>', '<w:pStyle w:val="Normal"/>', p)
        p = re.sub(r'<w:outlineLvl[^>]*/>', '', p)
        p = p.replace('</w:pPr>', '<w:outlineLvl w:val="9"/></w:pPr>', 1)
    if text.startswith('CHƯƠNG') and 'Heading1' in p and '<w:pageBreakBefore' not in p:
        p = p.replace('</w:pPr>', '<w:pageBreakBefore/></w:pPr>', 1)
    new = revise_text(text)
    if new != text:
        first = True
        def replace_run(m):
            nonlocal first
            if first:
                first = False
                return '<w:t xml:space="preserve">'+html.escape(new)+'</w:t>'
            return '<w:t></w:t>'
        p = re.sub(r'<w:t(?:\s[^>]*)?>.*?</w:t>', replace_run, p, flags=re.S)
    return p
xml = re.sub(r'<w:p(?:\s[^>]*)?>.*?</w:p>', polish_paragraph, xml, flags=re.S)
# Empty spacer paragraphs before chapter headings must not create a blank page.
xml = re.sub(r'<w:p\b[^>]*>(?:(?!<w:p\b|<w:t\b|<w:drawing\b|<w:br\b).)*?</w:p>(?=<w:p\b[^>]*><w:pPr><w:pStyle w:val="Heading1")', '', xml, flags=re.S)
content_path = root/'reports/work/report_content_full.json'
content = json.loads(content_path.read_text(encoding='utf-8-sig'))
for section in content:
    for key in ('paragraphs', 'after'):
        if key in section:
            section[key] = [revise_text(p) for p in section[key] if not any(p.startswith(s) for s in remove_starts)]
content_path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
for before,after in [('mục 5 9','mục 5.10'),('mục 5 6','mục 5.7'),('Hình 6.1. Giao diện','Hình bổ sung. Giao diện')]: xml=xml.replace(before,after)
files['word/document.xml']=xml.encode('utf-8')
temp=path.with_suffix('.tmp.docx')
with ZipFile(temp,'w',ZIP_DEFLATED) as out:
    for name,data in files.items(): out.writestr(name,data)
temp.replace(path)
print('Removed',len(edits),'subsection page breaks; retained chapter breaks.')
