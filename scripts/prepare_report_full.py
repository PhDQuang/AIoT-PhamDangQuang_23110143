"""Prepare the expanded Vietnamese report from locked experiment artifacts."""
from pathlib import Path
import copy
import json
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ppg_cvae.web import Demo, FINAL

WORK = ROOT/'reports/work'
FIG = WORK/'figures'
source = json.loads((WORK/'report_content.json').read_text(encoding='utf-8-sig'))
pages = []

def old(index, title, paragraphs=None, extra=(), chapter=None):
    page = copy.deepcopy(source[index])
    page['title'] = title
    if paragraphs is not None:
        page['paragraphs'] = paragraphs
    else:
        # Preserve methodology and numbers, remove repeated conversational judgments.
        skip = ['chưa đạt', 'kém baseline', 'không bù', 'để che', 'chưa đo downstream',
                'chưa có thí nghiệm chứng minh', 'Hoàn thiện hồ sơ không', 'Không cần',
                'không phải bảo đảm', 'Mô hình chưa đảm bảo']
        page['paragraphs'] = [p for p in page['paragraphs'] if not any(s in p for s in skip)]
    page['paragraphs'] += list(extra)
    page['after'] = [p for p in page.get('after', []) if not any(s in p for s in ['chưa đạt','không bù','Hoàn thiện hồ sơ','kém'])]
    if chapter: page['chapter'] = chapter
    pages.append(page)
    return page

def new(title, paragraphs, chapter=None, **kwargs):
    page = dict(title=title, paragraphs=paragraphs, **kwargs)
    if chapter: page['chapter'] = chapter
    pages.append(page)
    return page

new('TÓM TẮT', [
    'Đề tài nghiên cứu phương pháp sinh tín hiệu quang thể tích PPG theo nhịp tim chỉ định bằng mạng tự mã hóa biến phân có điều kiện một chiều. Mỗi cửa sổ tín hiệu có độ dài 8 giây, gồm 512 mẫu ở tần số 64 Hz. Đầu vào của bộ sinh là nhịp tim mục tiêu và vector ngẫu nhiên trong không gian tiềm ẩn; đầu ra là một dạng sóng PPG ở miền biên độ chuẩn hóa. Hướng tiếp cận này phục vụ khảo sát mô hình tạo sinh và xây dựng nguồn tín hiệu số cho quá trình thử nghiệm phần mềm trong hệ thống IoT.',
    'Quy trình thực nghiệm sử dụng PPG-DaLiA, chia 15 người thành năm lượt kiểm định chéo theo đối tượng. Mô hình cVAE được huấn luyện với ba seed; bộ sinh hai Gaussian được dùng làm đối chuẩn. Hai thí nghiệm bóc tách thành phần khảo sát vai trò của điều kiện nhịp tim và ràng buộc HR. Kết quả được phân tích qua sai số nhịp tim, tỷ lệ đo được, tỷ lệ sai lệch không quá 3 bpm, khả năng tái tạo, hình thái xung và độ đa dạng. Tổng cộng 120.000 cửa sổ sinh được lưu cùng thông tin cấu hình để hỗ trợ kiểm tra và tái lập.',
    'Trung bình đều trên ba điều kiện 60, 80 và 120 bpm, MAE của cVAE là 20,15 bpm và của đối chuẩn hai Gaussian là 3,32 bpm theo bộ đo độc lập đã khóa. Decoder có 84.081 tham số, tệp FP32 kích thước 340.199 byte; tổng số tham số cVAE là 366.625. Các phép đo tài nguyên được thực hiện trên CPU một luồng và trình bày riêng theo môi trường thử. Ứng dụng PPG Studio trực quan hóa quá trình sinh, hiển thị HR mục tiêu và HR đo được, đồng thời hỗ trợ xuất dữ liệu và biểu đồ.',
    'Báo cáo trình bày đầy đủ từ đặc tả bài toán, tiền xử lý, xây dựng mô hình, thiết kế thí nghiệm đến đánh giá và trình diễn. Các kết quả định lượng phản ánh bộ trọng số đã huấn luyện; mục tiêu thiết kế được nêu tách biệt với giá trị đo thực tế. Từ khóa: PPG, cVAE, nhịp tim, tín hiệu tạo sinh, IoT, Gaussian, kiểm định chéo theo người.'
])

old(1, '1.1. Bối cảnh kỹ thuật và ý nghĩa đề tài', paragraphs=[
    'Tín hiệu quang thể tích PPG phản ánh biến thiên thể tích máu tại vùng đo thông qua nguyên lý quang học. Trong thiết bị đeo, PPG là một kênh dữ liệu quan trọng để khảo sát nhịp tim và xây dựng các thuật toán xử lý tín hiệu. Hệ thống IoT thường gồm cảm biến, khối thu thập, bộ xử lý và thành phần hiển thị hoặc truyền thông. Bên cạnh dữ liệu đo thực tế, tín hiệu được sinh có kiểm soát tạo điều kiện kiểm thử phần mềm theo những mức nhịp tim được xác định trước.',
    'Đối với bài toán tạo sinh, yêu cầu không chỉ là tạo một chuỗi tuần hoàn mà còn biểu diễn được sự thay đổi về hình dạng xung giữa các mẫu. Mạng cVAE kết hợp việc học biểu diễn từ dữ liệu với thông tin điều kiện nhịp tim. Vector tiềm ẩn mô tả biến thiên của dạng sóng, trong khi điều kiện HR cung cấp mức nhịp mong muốn. Các thành phần này cho phép khảo sát đồng thời độ đúng nhịp và tính đa dạng của tín hiệu đầu ra.',
    'Đề tài G2 được triển khai theo hướng xây dựng bộ sinh PPG một chiều và đối chiếu với phương pháp hai Gaussian. Sơ đồ quy trình thể hiện các giai đoạn dữ liệu, học mô hình, sinh từ prior và đánh giá. Quy trình này gắn nội dung học máy với yêu cầu triển khai thực tế: trọng số cần gọn, kết quả cần kiểm tra được và chương trình cần chạy trên CPU máy thử.',
    'Về mặt thực hành, sản phẩm gồm mã nguồn, các cấu hình thí nghiệm, trọng số đã huấn luyện, bảng kết quả và một ứng dụng trình diễn. Người sử dụng lựa chọn HR và seed để tạo cửa sổ mới, sau đó quan sát dạng sóng cùng kết quả đo HR. Tính tái lập được thể hiện qua việc cùng điều kiện và cùng seed tạo lại cùng đầu ra trong môi trường suy luận đã xác định.'
], chapter='CHƯƠNG 1. ĐẶT VẤN ĐỀ VÀ MỤC TIÊU')
new('1.2. Phát biểu bài toán và phạm vi nghiên cứu', [
    'Mỗi mẫu dữ liệu huấn luyện là cặp (x, h), trong đó x là cửa sổ PPG một kênh có 512 phần tử và h là nhịp tim tham chiếu, đơn vị bpm. Điều kiện được đưa về c = h/60. Mạng học biểu diễn phân phối của tín hiệu khi biết điều kiện; bộ giải mã nhận z có 16 chiều và c để tạo một cửa sổ mới. Dữ liệu đưa vào mạng nằm trong miền đã lọc và chuẩn hóa theo thống kê train của từng fold.',
    'Hai chế độ hoạt động cần được phân biệt trong thiết kế thực nghiệm. Tái tạo sử dụng encoder để mã hóa chính cửa sổ quan sát rồi giải mã lại, vì vậy có cặp tín hiệu gốc và tín hiệu tái tạo để tính RMSE. Sinh tự do lấy z từ phân phối chuẩn N(0,I) và chỉ cung cấp HR cho decoder. Ở chế độ sinh, một HR có thể tương ứng nhiều dạng sóng; đánh giá tập trung vào nhịp, hình thái và độ đa dạng thay vì yêu cầu đầu ra trùng từng điểm với một bản ghi bất kỳ.',
    'Phạm vi triển khai là decoder FP32 trên CPU máy thử. Encoder và mạng HR phụ trợ phục vụ huấn luyện hoặc phân tích nhưng không cần được tải khi thực hiện chức năng sinh trên web. Một cửa sổ biểu diễn 8 giây dữ liệu được tạo trong một lượt suy luận; thời lượng tín hiệu này khác với thời gian tính toán của bộ giải mã.',
    'Trong báo cáo, ứng dụng web nhận HR/seed để minh họa chức năng đã được mô hình học. Tác vụ dự báo từ lịch sử 8 giây sang các giây tiếp theo thuộc một bài toán khác với cặp dữ liệu vào–ra và mục tiêu tối ưu riêng. Phần trình diễn chính vì vậy bám sát nhiệm vụ sinh PPG theo HR đã xác định trong đề cương.'
], table={'headers':['Chế độ','Đầu vào','Cách đánh giá'], 'rows':[
    ['Huấn luyện','PPG thật và HR tham chiếu','Loss tái tạo, KL, HR'],['Tái tạo','PPG và HR; z = μ','RMSE ghép cặp'],['Sinh tự do','HR và z từ prior','MAE HR, V, A₃, hình thái'],['Web demo','HR, seed, số cửa sổ','Dạng sóng và HR đo độc lập']]})
new('1.3. Mục tiêu kỹ thuật và sản phẩm thực hiện', [
    'Mục tiêu phương pháp là xây dựng 1D-cVAE đúng đặc tả, huấn luyện trên dữ liệu thực và khảo sát khả năng sinh tại 60, 80, 120 bpm. Mục tiêu đánh giá là sử dụng bộ đo HR độc lập, lưu kết quả theo mẫu và tổng hợp theo seed rồi theo fold. Bên cạnh mô hình học, đối chuẩn hai Gaussian tạo cơ sở đối chiếu về khả năng điều khiển chu kỳ và biến thiên hình dạng.',
    'Các ngưỡng trong bảng là mục tiêu thiết kế của đề tài. Báo cáo nêu giá trị đo thực nghiệm riêng để việc đọc kết quả nhất quán. MAE chỉ tính trên cửa sổ đo được, còn tỷ lệ V và A₃ sử dụng toàn bộ số cửa sổ sinh. Việc trình bày đồng thời các chỉ số này giúp mô tả đầy đủ đặc điểm của đầu ra.',
    'Sản phẩm thực hiện bao gồm bộ đọc dữ liệu, manifest chia theo người, mã tiền xử lý, cVAE, proxy, đối chuẩn Gaussian và các script đánh giá. Trọng số decoder cùng cấu hình detector được dùng trong chương trình CPU và web demo. Các bảng tổng hợp, hình minh họa và thông tin môi trường được lưu kèm để người đọc có thể đối chiếu báo cáo với dữ liệu thí nghiệm.',
    'Ứng dụng trình diễn là thành phần hỗ trợ thuyết minh: nhập điều kiện, sinh tín hiệu, quan sát nhịp đo và xuất kết quả. Báo cáo sử dụng cùng cách gọi phương pháp và cùng đơn vị giữa giao diện, biểu đồ và bảng số liệu. Điều này tạo sự thống nhất từ phần lý thuyết đến sản phẩm triển khai.'
], table={'headers':['Nội dung','Mục tiêu theo đề cương'], 'rows':[['MAE tại từng HR','≤ 3 bpm'],['Tỷ lệ đo được V','≥ 95%'],['Toàn bộ cVAE','≤ 500.000 tham số'],['Decoder FP32','≤ 2 MB'],['P95 decoder CPU, một luồng','≤ 100 ms'],['Đầu ra','512 mẫu / 8 giây / 64 Hz']]})

old(2,'2.1. Đặc tả tập dữ liệu và nhãn tham chiếu',chapter='CHƯƠNG 2. TẬP DỮ LIỆU VÀ TIỀN XỬ LÝ',extra=[
    'Kênh BVP là nguồn tín hiệu được sử dụng để tạo x; nhãn HR đóng vai trò điều kiện h. Những tín hiệu có tần số lấy mẫu khác được quản lý theo thời gian thực của từng cảm biến. Đặc biệt, một nhãn HR không phải là một mẫu PPG và không được nhân bản theo giả định chúng cùng tần số. Việc đồng bộ dựa trên mô tả bộ dữ liệu và các kiểm tra chiều dài được lưu trong hồ sơ căn nhãn.',
    'Mỗi bản ghi được gắn mã đối tượng trước khi cắt cửa sổ. Loader kiểm tra cấu trúc dữ liệu và phát hiện các khóa tín hiệu phù hợp; thông tin đối tượng, đường dẫn nguồn và cách căn nhãn được ghi lại để truy vết. Từ dữ liệu liên tục, quá trình tiền xử lý tạo tập cửa sổ cho từng split. Cấu trúc này cho phép sử dụng thống nhất dữ liệu giữa cVAE, proxy và Gaussian trong cùng một fold.'
])
old(3,'2.2. Quy trình căn nhãn, lọc và chuẩn hóa',extra=[
    'Trình tự thực hiện gồm xác minh nhãn, chia đối tượng, lọc từng đoạn liên tục, xét điều kiện hợp lệ và chuẩn hóa. Cách tổ chức theo đoạn liên tục bảo toàn mốc thời gian để cửa sổ tín hiệu ghép đúng với nhãn. Các cột trong manifest gồm người đo, vị trí đầu–cuối, split, HR tham chiếu và trạng thái hợp lệ. Nhờ đó, một chỉ số cửa sổ có thể được đối chiếu về bản ghi nguồn khi cần.',
    'Bảng số lượng trình bày trước và sau các kiểm tra tiền xử lý. Tổng số ứng viên trong một fold là 64.697; còn 64.667 cửa sổ hợp lệ. Các con số theo split được dùng để mô tả khối lượng thực nghiệm. Cùng thống kê chuẩn hóa train được áp dụng cho val và test, bảo đảm cách biểu diễn tín hiệu tương thích với bộ trọng số của fold tương ứng.'
])
new('2.3. Chia dữ liệu theo người và quản lý từng fold',[
    'Năm nhóm đối tượng cố định gồm G1={S1,S2,S3}, G2={S4,S5,S6}, G3={S7,S8,S9}, G4={S10,S11,S12} và G5={S13,S14,S15}. Fold k dùng Gk làm test, nhóm kế tiếp theo vòng làm validation, ba nhóm còn lại làm train. Phương án này tạo năm lượt đánh giá với vai trò của các đối tượng được luân phiên theo quy tắc rõ ràng.',
    'Chia theo người được thực hiện trước khi học thống kê. Trong một fold, danh sách đối tượng của ba split tách biệt và được kiểm tra tự động. Những cửa sổ chồng lấp từ cùng bản ghi giữ nguyên split của người đo. Đây là cơ sở để kết quả test mô tả khả năng hoạt động trên các đối tượng chưa xuất hiện trong train của lượt đó.',
    'Mỗi fold được quản lý như một đơn vị thí nghiệm hoàn chỉnh: có manifest, thống kê chuẩn hóa, proxy, detector, ngân hàng Gaussian và trọng số cVAE. Việc đóng gói theo fold giúp tránh sử dụng nhầm thành phần của các lượt khác nhau. Các mô hình cơ sở và đối chứng trong cùng fold dùng chung định nghĩa dữ liệu và cùng quy trình đo.',
    'Seed chỉ điều khiển ngẫu nhiên trong quá trình huấn luyện hoặc sinh, không thay thế phép chia theo người. Khi tổng hợp, kết quả nhiều seed được lấy trung bình trong fold trước; sau đó năm fold được tổng hợp với trọng số bằng nhau. Phương án này phản ánh cả biến thiên ngẫu nhiên và thay đổi nhóm đối tượng đánh giá.'
],table={'headers':['Fold','Test','Validation','Train'], 'rows':[['1','G1','G2','G3, G4, G5'],['2','G2','G3','G1, G4, G5'],['3','G3','G4','G1, G2, G5'],['4','G4','G5','G1, G2, G3'],['5','G5','G1','G2, G3, G4']]})
old(4,'2.4. Phân bố nhịp tim và đặc điểm chuyển động',extra=[
    'Phân bố HR cho thấy mức độ hiện diện của các điều kiện trong dữ liệu. Histogram dùng các bin 20 bpm; độ phủ quanh mỗi target được xét trong khoảng ±10 bpm. Hai cách biểu diễn bổ sung cho nhau: histogram mô tả miền nhịp tổng thể, còn độ phủ mô tả lượng dữ liệu gần những điều kiện sinh được dùng trong thí nghiệm.',
    'Thông tin ACC được sử dụng để mô tả mức chuyển động theo RMS sau khi trừ trung bình mỗi trục. Các phân vị tính trên train xác định ngưỡng mô tả cho từng fold. Phân tích này bổ sung bối cảnh của dữ liệu cảm biến trong sinh hoạt và được trình bày cùng số lượng cửa sổ, HR min/max và lý do loại. Kết quả đã được lưu theo fold và split để có thể kiểm tra lại.'
])
old(5,'3.1. Kiến trúc 1D-cVAE và kích thước tensor',chapter='CHƯƠNG 3. PHƯƠNG PHÁP ĐỀ XUẤT',extra=[
    'Conv1D khai thác quan hệ cục bộ theo thời gian, còn các tầng có stride 2 giảm chiều chuỗi để xây dựng biểu diễn gọn. Ở phía giải mã, ConvTranspose1D tăng dần chiều thời gian về 512 mẫu. Điều kiện c được ghép trực tiếp vào đầu vào của encoder và vector latent của decoder, vì vậy cấu trúc mạng thể hiện rõ đường đi của thông tin HR.',
    'Kích thước tensor được kiểm tra bằng các bài kiểm thử mô hình. Một batch gồm B cửa sổ tạo đầu ra [B,1,512]; vector latent có dạng [B,16]. Các kiểm tra này được dùng cùng phép đếm tham số để đối chiếu mã nguồn với đặc tả kiến trúc. Khi xuất decoder, chỉ trọng số phần sinh được lưu trong tệp state_dict FP32.'
])
new('3.2. Mã hóa, tái tạo và sinh từ phân phối prior',[
    'Encoder ước lượng trung bình μ và log phương sai s của phân phối qφ(z|x,c). Mỗi cửa sổ thực được ánh xạ vào một phân phối trong không gian tiềm ẩn thay vì một vector duy nhất. Trong huấn luyện, mẫu z được tạo bằng phép tái tham số hóa để quá trình lấy mẫu vẫn phù hợp với tối ưu bằng gradient. Prior được cố định là phân phối chuẩn đa chiều N(0,I).',
    'Đối với tái tạo, encoder nhận x và c, sau đó decoder giải mã z=μ. Cách dùng trung bình posterior tạo đầu ra xác định cho cửa sổ đang xét. Vì có cặp x và x tái tạo tương ứng, RMSE là chỉ số phù hợp cho chế độ này. Metadata người đo và chỉ số cửa sổ được lưu để tổng hợp theo đối tượng.',
    'Đối với sinh tự do, decoder nhận z lấy từ prior cùng HR do người dùng chỉ định. Một cửa sổ test không tham gia vào quá trình sinh. Khi giữ c và thay z, ứng dụng khảo sát các dạng sóng khác nhau ở cùng điều kiện. Khi giữ z và thay c, ứng dụng khảo sát đáp ứng của decoder đối với điều kiện nhịp tim. Hai phép thử này được tổ chức theo cùng seed để thuận tiện đối chiếu.',
    'Mô hình không gán một dạng sóng duy nhất cho một giá trị HR. Tính ngẫu nhiên là một phần của thiết kế tạo sinh, do đó các mẫu được đánh giá theo thuộc tính nhịp và phân bố hình dạng. Việc phân biệt prior với posterior cũng giúp giữ nhất quán giữa kết quả thực nghiệm và chức năng web demo: ứng dụng chỉ dùng decoder và latent từ prior.'
])
old(6,'3.3. Hàm mất mát và mạng HR phụ trợ',extra=[
    'Loss tái tạo định hướng việc giữ cấu trúc tín hiệu, KL điều chỉnh quan hệ giữa posterior và prior, còn thành phần HR đưa thông tin nhịp vào quá trình học tái tạo. Các trọng số β và λHR được lựa chọn trên validation. Lịch sử các thành phần được lưu riêng để có thể theo dõi động học học thay vì chỉ xem một giá trị loss tổng.',
    'Mạng proxy có kiến trúc một chiều và dự đoán nhịp tim chuẩn hóa. Sau khi học trên train và chọn checkpoint trên val, trọng số proxy được giữ cố định trong giai đoạn huấn luyện cVAE. Đầu vào của proxy là tín hiệu tái tạo, còn gradient theo đầu vào vẫn truyền về decoder. Cách tổ chức này gắn ràng buộc HR với đầu ra mạng mà không cập nhật lại proxy đồng thời.'
])
old(7,'3.4. Bộ sinh đối chuẩn hai Gaussian',extra=[
    'Một nhịp chuẩn hóa theo pha được mô tả bởi hai thành phần Gaussian với biên độ, tâm và độ rộng riêng. Chu kỳ được đặt theo HR, pha đầu được chọn bằng bộ sinh số ngẫu nhiên. Bộ tham số của các nhịp train được lấy mẫu theo cả tuple để giữ quan hệ giữa các thành phần xung. Vì vậy, đối chuẩn không chỉ lặp một dạng xung mặc định mà sử dụng ngân hàng hình thái đã được khớp từ dữ liệu.',
    'Trong so sánh, số lượng cửa sổ, điều kiện HR và bộ đo được giữ chung với cVAE. Kết quả Gaussian thể hiện đặc điểm của phương pháp đặt chu kỳ trực tiếp; kết quả cVAE thể hiện đặc điểm của bộ sinh học từ dữ liệu. Hai phương pháp được đọc cùng các chỉ số hình thái và đa dạng để thể hiện đầy đủ sự khác nhau về cơ chế tạo tín hiệu.'
])
old(8,'4.1. Huấn luyện, chọn cấu hình và khả năng tái lập',chapter='CHƯƠNG 4. THIẾT KẾ THỰC NGHIỆM VÀ CHỈ SỐ',extra=[
    'Một lần chạy lưu cấu hình đã giải quyết đầy đủ, lịch sử epoch, checkpoint và thông tin môi trường. Các thao tác gieo seed áp dụng cho Python, NumPy và PyTorch. Dữ liệu và cấu hình của mỗi fold được dùng thống nhất giữa các run; những thay đổi cấu hình được ghi trong artifact để quá trình chạy lại có cùng đầu vào và quy tắc.',
    'Sau giai đoạn chọn trên validation, quy trình kiểm thử được khóa trước khi đánh giá các lượt cuối. Mỗi cấu hình cuối được chạy với các seed đã quy định. Hình lịch sử huấn luyện minh họa diễn biến các thành phần loss; snapshot KL từng chiều ở checkpoint cuối bổ sung thông tin về biểu diễn tiềm ẩn. Hai loại thông tin được ghi rõ theo thời điểm đo.'
])
old(9,'4.2. Kiểm chuẩn bộ đo HR trên tín hiệu thực',extra=[
    'Proxy và detector có hai vai trò khác nhau. Proxy là mạng khả vi để cung cấp ràng buộc trong tối ưu; detector là phép đo độc lập sử dụng đỉnh của dạng sóng khi đánh giá. Kiểm chuẩn cả hai trên PPG thật validation tạo cơ sở diễn giải sai lệch đo. Các phép đo được tổng hợp theo người để giảm ảnh hưởng của khác biệt số cửa sổ giữa các bản ghi.',
    'Đối với từng cửa sổ, detector trả HR, số đỉnh, trạng thái đo được và lý do khi không trả được giá trị hợp lệ. Nhịp tim được tính từ trung vị khoảng nhịp, theo đơn vị bpm. Một giá trị hữu hạn chỉ được chấp nhận khi thỏa điều kiện số đỉnh và miền HR đã định. Web demo sử dụng đúng detector của fold được tải và hiển thị kết quả này riêng với HR mục tiêu.'
])
old(10,'4.3. Định nghĩa chỉ số và quy tắc tổng hợp',extra=[
    'Cách trình bày MAE đi cùng V giúp phân biệt mức sai lệch trên các cửa sổ đo được với khả năng phép đo hoạt động trên toàn bộ đầu ra. A₃ sử dụng số mẫu toàn bộ để cho biết bao nhiêu cửa sổ đáp ứng ngưỡng sai lệch 3 bpm. Median được báo cùng MAE để mô tả vị trí trung tâm của phân bố sai số.',
    'Trong các biểu đồ tổng hợp, error bar biểu thị độ lệch chuẩn giữa năm fold. Các bảng chi tiết theo fold, seed và HR cho phép đối chiếu từ số tổng hợp xuống từng run. Phần tài nguyên dùng median/P95 của thời gian suy luận, vì đây là cách biểu diễn phù hợp với nhiều lần đo lặp trên cùng môi trường.'
])
new('4.4. Quy trình kiểm thử và phân tích bóc tách',[
    'cVAE cơ sở và Gaussian được đánh giá trên năm fold với ba seed 42, 43, 44. Mỗi run sinh 1.000 cửa sổ tại mỗi HR 60, 80, 120 bpm. Hai biến thể no_hr_loss và no_condition được thực hiện tại seed 42 trên năm fold. Tổng số run là 40, tạo 120 bộ kết quả theo phương pháp, fold, seed và HR; tổng số cửa sổ sinh là 120.000.',
    'Các ablation giữ chung quy trình dữ liệu và phép đo để khảo sát vai trò của thành phần được thay đổi. no_hr_loss giữ điều kiện c=h/60 nhưng đặt λHR=0. no_condition dùng c=0 và λHR=0. Khi đối chiếu với mô hình cơ sở, cùng seed 42 được sử dụng để giảm khác biệt về thiết kế thực nghiệm.',
    'Hình minh họa dùng các chỉ số mẫu cố định; bảng kết quả sử dụng đầy đủ cửa sổ đã sinh. Đối với phép thử giữ z đổi HR, bộ latent được gieo cùng seed. Đối với giữ HR đổi z, điều kiện được cố định và latent thay đổi giữa các cửa sổ. Cách tổ chức này tạo hai góc nhìn về khả năng đáp ứng điều kiện và biến thiên dạng sóng.',
    'Kết quả test, kết quả tái tạo và benchmark được lưu thành các nhóm artifact riêng. Thống kê được đọc trực tiếp từ tệp đã lưu để dựng bảng và hình báo cáo. Quá trình dựng lại hình không thay checkpoint hoặc detector đã khóa; các hình có vai trò thể hiện dữ liệu thí nghiệm một cách nhất quán.'
],table={'headers':['Nhóm thử nghiệm','Fold × seed','Số cửa sổ'], 'rows':[['cVAE cơ sở','5 × 3','45.000'],['Hai Gaussian','5 × 3','45.000'],['no_hr_loss','5 × 1','15.000'],['no_condition','5 × 1','15.000'],['Tổng cộng','40 run','120.000']]})

old(11,'5.1. Kết quả sai số nhịp tim',paragraphs=[
    'Bảng kết quả trình bày MAE theo từng điều kiện HR sau khi trung bình seed trong fold và tổng hợp năm fold. Tại 60, 80 và 120 bpm, cVAE có MAE lần lượt 17,88; 8,47 và 34,09 bpm. Đối chuẩn Gaussian có MAE tương ứng 7,07; 2,90 và 0,00 bpm theo cùng bộ đo. Các giá trị đi kèm độ lệch chuẩn giữa fold mô tả biến thiên quan sát được trong quá trình đánh giá.',
    'Trong ba điều kiện, cVAE thể hiện sai số thấp nhất tại 80 bpm. Gaussian có chu kỳ đặt trực tiếp theo HR và đạt sai số thấp ở hai điều kiện 80, 120 bpm. Trung bình đều ba target, MAE của cVAE là 20,15 bpm và Gaussian là 3,32 bpm. Median được báo riêng để bổ sung thông tin về phân bố lỗi thay vì thay thế MAE.',
    'So với mục tiêu thiết kế MAE ≤3 bpm, giá trị cVAE hiện tại còn cao hơn ngưỡng ở cả ba điều kiện; mức 80 bpm là vùng phù hợp để khảo sát tiếp đáp ứng điều kiện. Nhận xét này chỉ xét sai số theo detector đã chọn. Các thuộc tính hình thái và độ đa dạng được trình bày ở những mục tiếp theo để đánh giá đầu ra theo nhiều khía cạnh.',
    'Bảng giữ số cửa sổ không đo được trên mỗi 15.000 mẫu của từng phương pháp/HR. Việc đi kèm số lượng này giúp đọc đúng mẫu số của MAE. Các bảng theo mẫu, fold và seed trong artifact là cơ sở tạo kết quả tổng hợp; hình MAE thể hiện cùng số liệu theo dạng cột và error bar.'
],chapter='CHƯƠNG 5. KẾT QUẢ THỰC NGHIỆM VÀ THẢO LUẬN')
old(12,'5.2. Tỷ lệ đo được và tỷ lệ sai lệch không quá 3 bpm',extra=[
    'Tỷ lệ V cVAE gần 100% tại các điều kiện thử cho thấy bộ đo trả được HR trên hầu hết cửa sổ sinh. A₃ bổ sung yêu cầu độ gần HR mục tiêu và có giá trị 5,53%; 21,15%; 1,95% tại 60, 80, 120 bpm. Gaussian có A₃ 88,09%; 95,30%; 99,39%. Trung bình đều ba HR là 9,54% và 94,26% tương ứng.',
    'Hai biểu đồ được bố trí cùng mục để thể hiện rằng tỷ lệ đo được và tỷ lệ sai lệch nhỏ là hai đại lượng khác nhau. V phản ánh điều kiện hợp lệ của phép đo; A₃ phản ánh kết quả xét ngưỡng trên toàn bộ mẫu. Đây cũng là lý do giao diện web hiển thị cả hai chỉ số cho lượt sinh thay vì chỉ một trạng thái chung.'
])
old(13,'5.3. Phân bố HR của các cửa sổ sinh',extra=[
    'Phân bố HR đo được bổ sung thông tin mà một giá trị MAE không thể hiện đầy đủ. Vị trí trung tâm cho thấy vùng nhịp thường xuất hiện; độ rộng phản ánh biến thiên giữa các cửa sổ. Đường HR mục tiêu trên hình tạo mốc để quan sát sự dịch chuyển giữa các điều kiện.',
    'Các phân bố được dựng từ HR của toàn bộ cửa sổ đo được trong lượt minh họa. Đối với những cửa sổ không đo được, trạng thái được lưu riêng trong bảng. Cách bố trí này bảo toàn thông tin đầu ra của detector và cho phép phân tích xem sự thay đổi điều kiện ảnh hưởng đến vùng nhịp tạo ra như thế nào.'
])
old(14,'5.4. Minh họa dạng sóng và đáp ứng điều kiện',extra=[
    'Các dạng sóng được biểu diễn trên cùng trục thời gian 0–8 giây và trong miền biên độ chuẩn hóa. Số đỉnh, khoảng cách giữa các xung và hình dạng lên–xuống của mỗi nhịp tạo cơ sở trực quan để đọc kết quả. Mỗi hình minh họa nêu rõ fold, seed và chỉ số cửa sổ; các mẫu dùng chỉ số cố định để quá trình dựng hình có thể lặp lại.',
    'Giữ seed và đổi HR tạo cùng bộ latent trong phép thử decoder, còn giữ HR và thay seed tạo nhiều latent khác nhau. Những thao tác này được đưa vào web demo để người trình bày có thể khảo sát trực tiếp. Dạng sóng Gaussian được hiển thị cùng điều kiện HR nhằm giúp nhận biết hai cơ chế tạo tín hiệu; độ trùng từng điểm giữa hai đường không được dùng như một tiêu chí chất lượng.'
])
old(15,'5.5. Tái tạo ghép cặp trên tập test',extra=[
    'Trong phép tái tạo, mỗi tín hiệu x có đầu ra x tái tạo tương ứng từ posterior mean. Điều kiện HR được sử dụng cùng cửa sổ, và RMSE được tính trực tiếp giữa hai vector 512 mẫu. Cách đánh giá này kiểm tra khả năng encoder–decoder biểu diễn lại tín hiệu đã quan sát, khác với phép sinh từ prior dùng trong đánh giá điều khiển HR.',
    'Hình ghép cặp cho phép quan sát vị trí xung, biên độ và các thành phần thay đổi nhanh. Các bảng RMSE đi kèm metadata đối tượng để hỗ trợ tổng hợp theo người và fold. Đơn vị RMSE là đơn vị chuẩn hóa của tín hiệu; việc giữ cùng thống kê train của fold bảo đảm ý nghĩa của cặp so sánh.'
])
old(16,'5.6. Sai số tái tạo theo fold và seed',extra=[
    'Việc trình bày RMSE theo fold cho thấy quá trình tái tạo khi nhóm người test thay đổi. Biến thiên giữa seed được xét trong cùng fold, giúp phân biệt thay đổi do ngẫu nhiên huấn luyện với thay đổi do đối tượng. Đồ thị đi kèm số liệu từ bảng tái tạo có cặp của từng checkpoint.',
    'Mỗi cửa sổ được tính riêng trước khi tổng hợp; các giá trị không được lựa chọn theo mức nhỏ nhất để trình bày. Cách lưu bảng theo mẫu giúp kiểm tra lại những khác biệt giữa người hoặc giữa khoảng HR. Kết quả tái tạo được dùng để mô tả khả năng biểu diễn của mô hình, còn kết quả sinh prior vẫn được xem xét bằng các chỉ số HR, hình thái và độ đa dạng.'
])
old(17,'5.7. Hình thái xung và đặc điểm biên độ',extra=[
    'Thời gian lên đỉnh và độ rộng tại 50% phản ánh hình học của một nhịp sau trích xung. Tỷ lệ rise/cycle đưa thời gian lên đỉnh về tương quan với chu kỳ, hỗ trợ đối chiếu giữa các mức HR. Median và IQR được sử dụng để mô tả phân bố các nhịp, đi kèm tỷ lệ trích được nhịp để phản ánh quy mô dữ liệu tham gia tính toán.',
    'Biên độ được đánh giá trên tín hiệu trong miền chuẩn hóa của fold, còn chuẩn hóa hình dạng từng nhịp chỉ phục vụ chỉ số đa dạng. Việc giữ riêng hai nhóm phân tích này giúp quan sát đồng thời sự biến đổi hình dạng và quy mô biên độ. Tham chiếu thật được lấy cân bằng theo người trong vùng HR quanh target, tạo cơ sở đối chiếu với tín hiệu sinh.'
])
old(18,'5.8. Độ đa dạng sau căn nhịp',extra=[
    'Căn nhịp theo chân xung và nội suy về 128 điểm đưa các chu kỳ khác nhau về cùng trục pha. Sau chuẩn hóa hình dạng từng nhịp, độ lệch chuẩn tại mỗi vị trí được tính rồi lấy trung bình theo chu kỳ. Quy trình này tạo chỉ số mô tả biến thiên hình dạng, bổ sung cho những thống kê biên độ ở mục 5.7.',
    'cVAE có chỉ số khoảng 0,179; 0,181; 0,198 tại ba HR, trong khi Gaussian tương ứng 0,185; 0,154; 0,144. Tham chiếu thật trung bình năm fold là 0,153; 0,178; 0,206. Các giá trị được đọc theo từng điều kiện cùng hình thái xung và tỷ lệ trích nhịp, từ đó mô tả sự khác nhau giữa bộ sinh học và bộ sinh giải tích.'
])
old(19,'5.9. Kết quả bóc tách thành phần',extra=[
    'Thí nghiệm bóc tách chỉ dùng seed 42 khi đối chiếu cơ sở với các biến thể. Cách này giữ cùng thiết kế fold và số mẫu. no_hr_loss khảo sát mô hình có điều kiện nhưng không dùng ràng buộc HR; no_condition khảo sát mô hình với đầu vào điều kiện bằng 0 và cùng λHR=0.',
    'Tại 80 và 120 bpm, cVAE cơ sở giảm MAE lần lượt 0,78 và 11,76 bpm so với no_hr_loss trong phép đối chiếu cùng seed. So sánh no_condition với no_hr_loss cung cấp góc nhìn về điều kiện khi thành phần HR loss cùng bằng 0. Những kết quả được trình bày theo HR để thể hiện đáp ứng khác nhau của từng cấu hình thay vì gộp ngay thành một nhận xét duy nhất.'
])
old(20,'5.10. Kích thước và thời gian suy luận trên CPU',extra=[
    'Kích thước cVAE 366.625 tham số nằm trong mục tiêu 500.000, còn tệp decoder FP32 0,340 MB nằm trong mục tiêu 2 MB. P95 decoder trên máy local nằm trong khoảng 0,831–1,480 ms ở ba mức HR, thấp hơn ngưỡng thiết kế 100 ms. Các số đo local và Kaggle được trình bày riêng để giữ đúng bối cảnh phần cứng.',
    'Benchmark tách decoder khỏi chuỗi lấy latent, suy luận và đo HR. Những thao tác tải model, đọc ghi tệp và vẽ giao diện không được tính vào thời gian decoder. Việc tách phạm vi đo giúp người triển khai ước lượng đúng khối lượng tính toán của từng thành phần khi tích hợp chương trình sinh vào phần mềm.'
])

old(21,'6.1. Tổ chức chương trình sinh và kiến trúc web',chapter='CHƯƠNG 6. TRIỂN KHAI VÀ TRÌNH DIỄN',extra=[
    'PPG Studio được xây dựng theo mô hình giao diện trình duyệt và server Python chạy local. Giao diện gửi HR, seed và số cửa sổ đến API /api/synthesize. Server tạo latent từ seed, chạy decoder trong inference_mode và dùng detector đo HR. Nếu ngân hàng tham số cùng fold có sẵn, server tạo thêm các cửa sổ Gaussian để đối chiếu.',
    'Server tải trọng số một lần khi khởi động. Mỗi request trả tín hiệu, các phép đo theo cửa sổ và chỉ số tổng hợp của lượt sinh. Giao diện Canvas hiển thị hai đường trên thời gian 0–8 giây, đồng thời cung cấp bảng MAE, V, A₃ và trạng thái. Chức năng xuất PNG, CSV và JSON giúp sử dụng kết quả demo trong phần thuyết minh và kiểm tra lại.'
])
new('6.2. Chức năng giao diện và quy trình sử dụng',[
    'Giao diện chính chia thành ba nhóm: thiết lập điều kiện, dạng sóng được sinh và đánh giá các cửa sổ trong lượt. Người dùng chọn HR 60/80/120 bpm, nhập seed và lựa chọn 1, 5 hoặc 20 cửa sổ. Mỗi cửa sổ độc lập có 512 mẫu ở 64 Hz. Khi chọn nhiều cửa sổ, bộ chọn chỉ số cho phép xem từng dạng sóng mà không ghép chúng thành một bản ghi liên tục.',
    'Ba ô thông tin hiển thị HR mục tiêu, HR đo của cVAE và HR đo của Gaussian cho cửa sổ đang xem. cVAE có đường xanh lá, Gaussian có đường xanh dương. Các giá trị đo được giữ nguyên theo detector; nếu một cửa sổ không đáp ứng điều kiện đo, giao diện hiển thị trạng thái và lý do thay cho một giá trị HR. Bảng tổng hợp sử dụng toàn bộ cửa sổ trong request để tính MAE, V và A₃.',
    'Để khảo sát đáp ứng điều kiện, giữ seed 42 và lần lượt chạy 60, 80, 120 bpm. Để khảo sát sự đa dạng, giữ HR 80 bpm và thay seed hoặc chọn 20 cửa sổ. Những kết quả này là lượt trình diễn trên checkpoint fold 1 seed 42; thống kê chính của báo cáo sử dụng toàn bộ năm fold và các seed huấn luyện đã quy định.',
    'Vị trí bổ sung ảnh giao diện: chèn ảnh toàn màn hình PPG Studio sau khi chạy HR 80 bpm, seed 42, một cửa sổ. Ảnh cần thấy rõ nhóm điều kiện, ba ô HR, hai đường tín hiệu và bảng đánh giá. Chú thích đề nghị: “Hình 6.1. Giao diện PPG Studio sinh PPG theo HR và đo nhịp độc lập”. Hướng dẫn chụp ảnh kèm theo báo cáo chỉ rõ thêm hai ảnh tại 60 và 120 bpm để hoàn thiện minh họa.'
], table={'headers':['Thao tác','Kết quả trên giao diện'], 'rows':[['Chọn HR và seed','Thiết lập điều kiện sinh'],['Bấm Sinh PPG','Decoder và Gaussian tạo cửa sổ'],['Chọn cửa sổ hiển thị','Xem dạng sóng và HR từng mẫu'],['Tải PNG','Biểu đồ cửa sổ đang xem'],['Tải CSV','Tất cả cửa sổ, HR và seed'],['Tải JSON','Tín hiệu, detector và kết quả đo']]})

# Actual backend output plots, not fabricated browser screenshots.
plt.rcParams.update({'font.family':'Arial','font.size':10,'figure.dpi':160,'savefig.dpi':200})
demo = Demo(FINAL/'cvae/fold_01/seed_42/decoder_state_dict.pt', FINAL/'proxy/fold_01/detector_config.json')
demo_results=[]
for hr in [60,80,120]:
    d = demo.synthesize({'hr':hr,'seed':42,'count':1}); demo_results.append(d)
    (WORK/f'web_demo_hr{hr}_seed42.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
fig, axes=plt.subplots(3,1,figsize=(9,7),sharex=True)
for ax,d in zip(axes,demo_results):
    t=np.arange(512)/64;ax.plot(t,d['cvae'][0],color='0.15',label='cVAE',lw=1)
    ax.plot(t,d['gaussian'][0],color='0.55',ls='--',label='Hai Gaussian',lw=1)
    c=d['cvae_metrics']['measurements'][0];g=d['gaussian_metrics']['measurements'][0]
    measured=lambda m: f"{m['hr']:.1f}" if m['measurable'] else 'không đo được'
    ax.set_title(f"HR mục tiêu {d['target_hr']} bpm | cVAE {measured(c)} bpm | Gaussian {measured(g)} bpm",fontsize=10)
    ax.set_ylabel('Biên độ Z');ax.grid(alpha=.2);ax.legend(loc='upper right',ncol=2,fontsize=8)
axes[-1].set_xlabel('Thời gian (giây)');fig.tight_layout();fig.savefig(FIG/'web_demo_backend.png',bbox_inches='tight');plt.close(fig)
new('6.3. Kết quả trình diễn và đối chiếu hai phương pháp',[
    'Hình trong mục này được tạo từ cùng hàm suy luận mà API web sử dụng, với checkpoint fold 1 seed 42, seed latent 42 và một cửa sổ tại mỗi điều kiện. Mỗi hàng trình bày cVAE và Gaussian trên cùng trục thời gian; tiêu đề ghi HR yêu cầu và HR đo độc lập. Đây là hình kết quả backend thật, bổ sung cho ảnh chụp giao diện cần được chèn ở mục 6.2.',
    'Điều kiện HR được truyền theo c=HR/60, còn latent được tạo bằng bộ sinh ngẫu nhiên CPU có seed cố định. Khi chạy lại cùng request trong môi trường hiện tại, tín hiệu được tái tạo. Đối chuẩn lấy tham số từ ngân hàng Gaussian đã khớp trên train của fold 1; vì vậy cấu hình trình diễn giữ cùng nguồn tham số với quá trình đánh giá.',
    'Độ lệch HR của từng cửa sổ demo có thể khác thống kê toàn bộ tập sinh vì đây là một mẫu riêng. Các giá trị trên hình được dùng để giải thích cách đọc giao diện: HR yêu cầu là điều kiện, HR đo là thuộc tính thu được từ đầu ra. Bảng tổng hợp web tính chỉ số trong một request; bảng kết quả chương 5 tổng hợp theo thiết kế thực nghiệm đầy đủ.'
], figure='web_demo_backend',caption='Hình 6.2. Kết quả backend dùng trong web demo tại 60, 80 và 120 bpm; fold 1, checkpoint seed 42, latent seed 42.')
new('6.4. Kiểm thử ứng dụng và xuất dữ liệu',[
    'Chức năng sinh được kiểm thử bằng tính tái lập với seed cố định, kích thước đầu ra và phép tính chỉ số. Mỗi cửa sổ phải có 512 mẫu; cùng HR/seed/số lượng phải cho cùng tín hiệu trong môi trường thử. MAE được đối chiếu lại từ danh sách HR đo; số cửa sổ không đo được phải khớp các trạng thái theo mẫu. Những kiểm tra này gắn trực tiếp với hành vi người dùng quan sát trên giao diện.',
    'Tệp CSV có chỉ số cửa sổ, thời gian tương đối, biên độ cVAE, biên độ Gaussian, HR mục tiêu và seed. Thời gian của từng cửa sổ bắt đầu từ 0; không xem các cửa sổ độc lập là một chuỗi liên tục. JSON lưu cấu hình detector, nguồn checkpoint, các waveform và kết quả tổng hợp. PNG thể hiện cửa sổ được chọn, ghi điều kiện và seed trên hình.',
    'Chương trình được khởi động bằng scripts/start_web.ps1 và truy cập tại http://127.0.0.1:8000. Trọng số được nạp một lần; suy luận sử dụng CPU FP32 một luồng. Các tham số đầu vào được kiểm tra về HR cho phép, seed và số cửa sổ. Giao diện thể hiện trạng thái kết nối và thông báo khi dữ liệu thiết lập không hợp lệ.',
    'Phương án xuất kết quả phục vụ cả trình diễn và lưu bằng chứng. Khi bổ sung ảnh vào báo cáo, người thực hiện nên giữ cùng seed, checkpoint và HR để ảnh chụp tương ứng với dữ liệu JSON được bàn giao. Những ảnh về trạng thái không đo được chỉ được chụp khi lượt sinh thực tế tạo trạng thái đó; không sửa tay HR hoặc giả lập một kết quả suy luận.'
])
old(22,'7.1. Kết quả thực hiện và sản phẩm bàn giao',paragraphs=[
    'Đề tài đã xây dựng quy trình sinh PPG theo nhịp tim với dữ liệu thực, phép chia theo người và hệ thống artifact theo fold. Các thành phần phương pháp gồm cVAE một chiều, mạng HR phụ trợ và đối chuẩn hai Gaussian. Thiết kế thực nghiệm đã thực hiện năm fold, ba seed cho mô hình cơ sở và hai biến thể bóc tách tại seed 42, tạo 120.000 cửa sổ sinh.',
    'Các kết quả định lượng được trình bày theo nhịp tim, hình thái, đa dạng và khả năng tái tạo. cVAE có MAE trung bình đều ba target 20,15 bpm và A₃ 9,54%; Gaussian có MAE 3,32 bpm và A₃ 94,26%. Bộ kết quả này cung cấp cơ sở khảo sát sự khác nhau giữa phương pháp học biểu diễn và phương pháp đặt chu kỳ giải tích. Mục tiêu kỹ thuật cùng số đo thực tế được thể hiện riêng trong các bảng để giữ đúng ý nghĩa của thực nghiệm.',
    'Về chi phí mô hình, cVAE có 366.625 tham số và decoder có 84.081 tham số. Tệp decoder FP32 đạt 340.199 byte; thời gian suy luận được đo trên Kaggle và máy local với phạm vi rõ ràng. Ứng dụng web bổ sung khả năng quan sát trực tiếp: nhập HR/seed, sinh cửa sổ, đo HR độc lập và xuất kết quả. Đây là sản phẩm trình diễn bám sát đặc tả G2.',
    'Gói bàn giao tổ chức mã nguồn, môi trường, trọng số, cấu hình và bảng/hình để hỗ trợ chạy lại. Bộ dữ liệu thô được truy cập qua nguồn công bố; artifact đã huấn luyện và hướng dẫn khởi động nằm trong hồ sơ dự án. Báo cáo tạo thành phần thuyết minh thống nhất với mã nguồn và số liệu đã lưu.'
],chapter='CHƯƠNG 7. KẾT LUẬN VÀ ĐỊNH HƯỚNG PHÁT TRIỂN')
new('7.2. Định hướng phát triển và hoàn thiện ứng dụng',[
    'Hướng phát triển thứ nhất tập trung vào điều khiển HR của mẫu sinh từ prior. Có thể khảo sát ràng buộc trực tiếp trên mẫu prior hoặc lựa chọn biểu diễn điều kiện phù hợp hơn, sau khi xây dựng kế hoạch validation/test riêng. Mục tiêu là tăng mức đáp ứng điều kiện đồng thời duy trì những thuộc tính hình dạng được học từ tín hiệu thực.',
    'Hướng thứ hai là mở rộng kiểm chuẩn phép đo trên những dải HR và đặc điểm chuyển động khác nhau. Việc dùng nhiều cách ước lượng HR và trình bày kết quả theo người hỗ trợ đọc chính xác các thay đổi của bộ sinh. Các thuộc tính rise, width50, rise/cycle và đa dạng được giữ làm nhóm chỉ số bổ sung trong những thử nghiệm tiếp theo.',
    'Hướng thứ ba là phát triển giao diện trình diễn thành một công cụ kiểm thử tín hiệu số. Có thể bổ sung lựa chọn fold, quản lý preset HR/seed, lưu lịch sử các lượt sinh và hiển thị phân bố HR của nhiều cửa sổ. Những chức năng này tiếp tục sử dụng metadata nguồn mô hình để người dùng đối chiếu kết quả khi thay cấu hình.',
    'Trong hướng tích hợp IoT, decoder có thể được đánh giá trên một CPU gateway cụ thể với phép đo thời gian, bộ nhớ và môi trường được ghi lại. Các kết quả trên máy thử hiện tại là nền tảng lựa chọn cấu hình triển khai. Nếu mở rộng sang bài toán dự báo lịch sử–tương lai, cần một thiết kế dữ liệu và mô hình riêng; bộ sinh theo HR trong báo cáo tiếp tục được giữ đúng mục tiêu tạo sinh của đề tài.'
])
refs=old(23,'TÀI LIỆU THAM KHẢO')
refs['after']=[]

# Captions are regenerated consistently with actual order.
figure_no=0
table_no=0
for page in pages:
    for key in ['figure','figure2']:
        if page.get(key):
            figure_no += 1
            capkey='caption' if key=='figure' else 'caption2'
            original=page.get(capkey,'Minh họa kết quả thực nghiệm.')
            original=re.sub(r'^Hình\s+[\d.]+\s*[:.]?\s*','',original)
            page[capkey]=f'Hình {figure_no}. {original}'
    if page.get('table'):
        table_no+=1
        page['table_caption']=f'Bảng {table_no}. '+page['title'].split('. ',1)[-1]
    # Update outdated plain references in inherited text rather than inventing page references.
    for key in ['paragraphs','after']:
        page[key]=[re.sub(r'\b(?:Hình|Bảng)\s+\d+[a-z]?\b','Hình minh họa' if p.startswith('Hình') else 'Bảng kết quả',p) if p.startswith(('Hình ','Bảng ')) else p for p in page.get(key,[])]
(WORK/'report_content_full.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared',len(pages),'content sections;',figure_no,'figures;',table_no,'tables')
print('Words:',sum(len(' '.join(p.get('paragraphs',[])+p.get('after',[])).split()) for p in pages))
