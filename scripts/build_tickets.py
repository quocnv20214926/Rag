"""Generate the evaluation set data/eval/tickets.jsonl (150 labelled Vietnamese tickets).

Ticket types and the expected action:
  normal        (80)  answer    - enough evidence, answer with citations
  missing_info  (25)  clarify   - required information is missing, ask the customer
  no_evidence   (25)  escalate  - the knowledge base has nothing on it, do not conclude, escalate
  conflict      (20)  escalate  - active documents contradict each other, escalate

Each ticket has gold_sections (doc_id#section_id holding the evidence), key_facts (facts the answer must
contain, matched after number normalisation), reference (expected answer/behaviour) and missing_slots.

    python scripts/build_tickets.py
"""

from __future__ import annotations

import json
import random
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B25, DT, HT, GH = "CS-BAOHANH-2025", "CS-DOITRA", "CS-HOANTIEN", "CS-GIAOHANG"
TG, TTn, TV, CP, TC, BM, KM = "CS-TRAGOP", "CS-THANHTOAN", "CS-THANHVIEN", "CS-CAREPLUS", "CS-THUCU", "CS-BAOMAT", "KM-THANG10-2025"
HANG, FAQ = "TT-BAOHANH-HANG", "FAQ-CHUNG"


def n(tid, text, gold, facts, ref):
    return dict(id=tid, type="normal", expected_decision="answer", ticket=text, gold_sections=gold, key_facts=facts,
                reference=ref, missing_slots=[])


NORMAL = [
    n("N01", "Chào shop, tôi mua tủ lạnh Kiwa được 3 tháng thì bị hỏng. Nếu gửi bảo hành thì mất bao lâu mới xong?",
      [f"{B25}#thoi-gian-xu-ly", f"{B25}#tai-nha"], ["15 ngày làm việc"],
      "Thời gian xử lý bảo hành tối đa 15 ngày làm việc; tủ lạnh được bảo hành tại nhà, kỹ thuật viên đến trong 48 giờ (nội thành HN, HCM, ĐN) hoặc 72 giờ (tỉnh khác)."),
    n("N02", "Laptop Lumio Go 14 của mình hết bảo hành rồi, mang ra cửa hàng kiểm tra lỗi có mất phí không?",
      [f"{B25}#sua-chua-co-phi"], ["100.000"],
      "Phí kiểm tra 100.000 đồng/sản phẩm, được miễn nếu khách đồng ý sửa tại Lam Sơn (không áp dụng mức 150.000 đồng của chính sách 2023 đã hết hiệu lực)."),
    n("N03", "Điện thoại Sora Nova 8 tôi gửi bảo hành ở Lam Sơn đã 20 ngày làm việc mà chưa xong, tôi có được mượn máy dùng tạm không?",
      [f"{B25}#thoi-gian-xu-ly"], ["mượn máy"],
      "Quá 15 ngày làm việc chưa xong, khách mua điện thoại được mượn máy dùng tạm miễn phí; quá 30 ngày làm việc được đổi sản phẩm mới cùng model hoặc tương đương."),
    n("N04", "Tôi làm mất hóa đơn mua tivi rồi, giờ tivi bị lỗi thì có bảo hành được không?",
      [f"{B25}#dieu-kien", f"{FAQ}#quen-hoa-don"], ["số điện thoại"],
      "Vẫn bảo hành được nếu tra cứu được lịch sử mua hàng bằng số điện thoại đã dùng khi mua hoặc mã đơn hàng."),
    n("N05", "Máy giặt Vinto nhà tôi bị ngập nước do lụt, giờ không chạy. Có được bảo hành miễn phí không?",
      [f"{B25}#tu-choi"], ["có phí"],
      "Không được bảo hành miễn phí vì sản phẩm bị vào nước; khách có thể chọn sửa chữa có phí, được báo giá trước."),
    n("N06", "Pin điện thoại Sora được bảo hành bao lâu vậy?", [f"{B25}#thoi-han", f"{HANG}#sora"], ["6 tháng"],
      "Pin điện thoại được bảo hành 6 tháng kể từ ngày mua."),
    n("N07", "Tivi Sora 65 inch nhà mình ở Cầu Giấy, Hà Nội bị lỗi. Kỹ thuật viên có đến tận nhà không, bao lâu thì đến?",
      [f"{B25}#tai-nha"], ["48 giờ"], "Tivi từ 43 inch được bảo hành tại nhà; nội thành Hà Nội kỹ thuật viên đến trong vòng 48 giờ, miễn phí đi lại."),
    n("N08", "Tôi ở Nghệ An, điều hòa Kiwa đang bảo hành bị hỏng thì bao lâu có người đến sửa?", [f"{B25}#tai-nha"], ["72 giờ"],
      "Điều hòa được bảo hành tại nhà; với tỉnh ngoài HN, HCM, ĐN kỹ thuật viên đến trong vòng 72 giờ."),
    n("N09", "Làm sao để tôi tra cứu thời hạn bảo hành của máy tính bảng đã mua?", [f"{B25}#tra-cuu"], ["tra cứu bảo hành"],
      "Vào mục Tra cứu bảo hành trên website, nhập số điện thoại mua hàng, mã đơn, IMEI hoặc serial."),
    n("N10", "Mua điện thoại được 20 ngày thì bị rè loa do lỗi máy, có được đổi máy mới không?", [f"{DT}#loi-nsx"], ["30 ngày"],
      "Được đổi mới cùng model miễn phí vì lỗi nhà sản xuất trong 30 ngày đầu."),
    n("N11", "Tủ lạnh mua được 20 ngày thì không lạnh, kỹ thuật viên xác nhận lỗi máy. Có được đổi tủ mới không hay phải bảo hành?",
      [f"{DT}#loi-nsx"], ["15 ngày"], "Tủ lạnh chỉ được đổi mới do lỗi trong 15 ngày đầu; đã 20 ngày nên xử lý theo chính sách bảo hành."),
    n("N12", "Mình mua nồi cơm điện Haru được 3 ngày, chưa dùng, còn nguyên hộp tem, giờ không thích nữa muốn trả lại được không, mất phí gì không?",
      [f"{DT}#doi-y"], ["7 ngày", "10%"], "Được trả trong 7 ngày khi còn nguyên hộp tem, phí đổi trả đổi ý 10% giá trị hóa đơn."),
    n("N13", "Điều hòa nhà tôi lắp xong hôm qua nhưng giờ muốn đổi sang loại công suất lớn hơn vì không còn hợp, được không?",
      [f"{DT}#doi-y"], ["lắp đặt"], "Không áp dụng đổi ý vì hàng điện lạnh đã được lắp đặt; nếu có lỗi kỹ thuật thì xử lý theo bảo hành/đổi mới."),
    n("N14", "Tôi mua tai nghe Sora Buds 3, đã bóc niêm phong dùng thử 1 ngày, thấy không hợp tai nên muốn trả lại.",
      [f"{DT}#khong-ap-dung"], ["không áp dụng"], "Tai nghe nhét tai đã bóc niêm phong không áp dụng đổi trả đổi ý vì lý do vệ sinh; vẫn được bảo hành nếu lỗi."),
    n("N15", "Máy giặt bị lỗi trong thời gian đổi mới nhưng tôi lỡ vứt hộp rồi thì có đổi được không?", [f"{DT}#khau-tru"], ["5%"],
      "Vẫn đổi được nhưng bị khấu trừ 5% giá trị sản phẩm do thiếu hộp."),
    n("N16", "Đặt hàng online giờ muốn trả hàng vì đổi ý thì làm thế nào, có tốn phí vận chuyển không?", [f"{DT}#online"], ["50.000"],
      "Tạo yêu cầu ở mục Đơn hàng của tôi, nhân viên lấy hàng trong 3 ngày làm việc; đổi ý chịu phí vận chuyển 50.000 đồng ngoài phí 10%."),
    n("N17", "Tôi trả máy lọc nước, lúc mua thanh toán bằng ví điện tử thì bao lâu nhận lại tiền?", [f"{HT}#thoi-gian"], ["1 đến 3 ngày"],
      "Hoàn về ví điện tử trong 1 đến 3 ngày làm việc kể từ khi yêu cầu hoàn tiền được duyệt."),
    n("N18", "Mình trả lại laptop mua trả góp qua công ty tài chính vì lỗi nhà sản xuất, phí tất toán khoản vay ai chịu?",
      [f"{HT}#tra-gop"], ["Lam Sơn chịu"], "Trả hàng do lỗi nhà sản xuất thì phí phạt tất toán do Lam Sơn chịu."),
    n("N19", "Nếu tôi trả lại sản phẩm thì điểm thành viên đã tích từ đơn đó có bị mất không?", [f"{HT}#diem-qua"], ["thu hồi"],
      "Điểm tích từ đơn bị trả sẽ bị thu hồi; điểm đã dùng để thanh toán được hoàn lại."),
    n("N20", "Ở quận 3 TP.HCM, mình đặt máy sấy tóc lúc 5 giờ chiều thì có giao trong 2 tiếng được không, phí bao nhiêu?",
      [f"{GH}#thoi-gian"], ["30.000"], "Được, giao nhanh 2 giờ áp dụng nội thành HCM cho đơn dưới 10 kg đặt trước 19:00, phí 30.000 đồng."),
    n("N21", "Giao hàng ra Cần Thơ thì mất mấy ngày?", [f"{GH}#thoi-gian"], ["2 đến 5 ngày"], "Các tỉnh ngoài HN, HCM, ĐN giao trong 2 đến 5 ngày làm việc."),
    n("N22", "Shipper giao 2 lần nhà mình đều không có ai, lần thứ 3 có mất phí không?", [f"{GH}#giao-lai"], ["50.000"],
      "Hai lần giao lại đầu miễn phí, từ lần thứ 3 phí 50.000 đồng/lần."),
    n("N23", "Nhận tivi về mở ra thấy viền bị trầy, mình đã ký biên bản nhận hàng rồi thì có đổi được không?", [f"{GH}#kiem-tra"], ["ký"],
      "Sau khi đã ký biên bản giao nhận, khiếu nại về ngoại quan không được giải quyết."),
    n("N24", "Mua tivi Sora 55 inch muốn treo tường thì tốn thêm bao nhiêu tiền?", [f"{GH}#lap-tivi"], ["450.000"],
      "Công lắp treo tường miễn phí, giá treo cho tivi từ 55 inch tính 450.000 đồng (khuyến mãi tháng 10 chỉ tặng giá treo cho tivi từ 65 inch)."),
    n("N25", "Lắp điều hòa nhà tôi cần khoảng 5 mét ống đồng thì phải trả thêm bao nhiêu?", [f"{GH}#lap-dieu-hoa"], ["360.000"],
      "Gói lắp miễn phí 3 mét đầu; 2 mét vượt tính 180.000 đồng/mét, tổng 360.000 đồng."),
    n("N26", "Lắp máy rửa bát âm tủ có mất phí không?", [f"{GH}#lap-khac"], ["300.000"], "Máy rửa bát âm tủ tính phí lắp 300.000 đồng."),
    n("N27", "Mình muốn mua trả góp qua thẻ tín dụng thì có những kỳ hạn nào, có mất phí gì không?", [f"{TG}#the-tin-dung"], ["12 tháng", "2,5%"],
      "Trả góp 0% kỳ hạn 3, 6, 9, 12 tháng; phí chuyển đổi 0% đến 2,5% tùy ngân hàng."),
    n("N28", "Tôi năm nay 62 tuổi, có mua trả góp qua công ty tài chính được không?", [f"{TG}#cong-ty-tai-chinh"], ["60 tuổi"],
      "Không, trả góp qua công ty tài chính áp dụng cho khách từ 20 đến 60 tuổi; có thể cân nhắc trả góp qua thẻ tín dụng."),
    n("N29", "Trả góp qua công ty tài chính khoản vay 25 triệu cần giấy tờ gì?", [f"{TG}#cong-ty-tai-chinh"], ["căn cước", "giấy phép lái xe"],
      "Căn cước công dân gắn chip, và vì khoản vay trên 20 triệu cần thêm giấy phép lái xe hoặc sao kê lương 3 tháng."),
    n("N30", "Tôi bị trễ một kỳ trả góp, có thể đóng tiền ở cửa hàng Lam Sơn được không?", [f"{TG}#thanh-toan-tre-han"], ["công ty tài chính"],
      "Không, khách thanh toán trực tiếp cho ngân hàng hoặc công ty tài chính; phí trễ hạn do đơn vị cho vay quy định."),
    n("N31", "Đơn hàng 60 triệu có thanh toán khi nhận hàng được không?", [f"{TTn}#phuong-thuc"], ["30%"],
      "COD tối đa 50 triệu; đơn trên mức này phải thanh toán trước tối thiểu 30%."),
    n("N32", "Hôm qua tôi mua laptop, giờ muốn xuất hóa đơn công ty được không?", [f"{TTn}#hoa-don-cong-ty"], ["trong ngày mua"],
      "Không, thông tin xuất hóa đơn công ty phải cung cấp chậm nhất trong ngày mua; sau ngày mua không xuất lại từ cá nhân sang công ty."),
    n("N33", "Tài khoản ngân hàng của tôi bị trừ tiền 2 lần cho cùng một đơn hàng, xử lý thế nào?", [f"{TTn}#loi-thanh-toan"], ["2 ngày làm việc"],
      "Gửi ảnh chụp giao dịch cho CSKH; Lam Sơn đối soát trong 2 ngày làm việc và hoàn khoản trừ trùng."),
    n("N34", "Chi tiêu bao nhiêu thì được lên hạng Vàng?", [f"{TV}#hang"], ["20.000.000"], "Chi tiêu từ 20 triệu đến dưới 50 triệu đồng trong 12 tháng."),
    n("N35", "Tôi đang là hạng Vàng, mua đơn 10 triệu thì được bao nhiêu điểm?", [f"{TV}#tich-diem"], ["150 điểm"],
      "Hạng Vàng nhận 1,5 điểm mỗi 100.000 đồng: 10 triệu được 150 điểm (tương đương 150.000 đồng)."),
    n("N36", "Điểm thành viên tích trong năm 2024 thì khi nào hết hạn?", [f"{TV}#su-dung-diem"], ["31/12/2025"], "Hết hạn ngày 31/12/2025."),
    n("N37", "Hạng Kim cương có ưu đãi gì về đổi trả không?", [f"{TV}#uu-dai"], ["15 ngày"],
      "Hạng Kim cương được đổi trả đổi ý trong 15 ngày và miễn phí đổi trả 10%."),
    n("N38", "Gói Care+ 24 giá bao nhiêu nếu mua cho tủ lạnh 20 triệu?", [f"{CP}#cac-goi"], ["1.800.000"], "Care+ 24 bằng 9% giá sản phẩm: 1.800.000 đồng."),
    n("N39", "Dùng quyền lợi Care+ Rơi vỡ để thay màn hình thì tôi phải trả bao nhiêu?", [f"{CP}#dong-chi-tra"], ["20%", "300.000"],
      "Đồng chi trả 20% chi phí sửa chữa, tối thiểu 300.000 đồng."),
    n("N40", "Tôi mua gói Care+ được 10 ngày, chưa dùng, giờ muốn hủy có được hoàn tiền không?", [f"{CP}#huy-goi"], ["100%"],
      "Được hoàn 100% vì hủy trong 15 ngày và chưa dùng quyền lợi."),
    n("N41", "Điện thoại cũ của tôi bị vỡ màn hình thì có mang đi thu cũ đổi mới được không?", [f"{TC}#san-pham"], ["màn hình"],
      "Không, sản phẩm thu cũ phải không bể vỡ màn hình."),
    n("N42", "Đổi laptop cũ hãng khác sang Lumio Air 13 thì được trợ giá thêm bao nhiêu?", [f"{TC}#tro-gia"], ["500.000"],
      "Đổi sang sản phẩm khác hãng được trợ giá 500.000 đồng ngoài giá thu."),
    n("N43", "Có người tự xưng nhân viên Lam Sơn gọi xin mã OTP để xác nhận đơn hàng của tôi, có đúng không?", [f"{BM}#xac-minh"], ["OTP"],
      "Không đúng, nhân viên Lam Sơn tuyệt đối không yêu cầu mã OTP, mật khẩu, số thẻ hoặc CVV."),
    n("N44", "Tôi muốn xóa toàn bộ dữ liệu cá nhân của mình khỏi hệ thống Lam Sơn thì mất bao lâu?", [f"{BM}#quyen"], ["30 ngày"],
      "Yêu cầu xóa dữ liệu được xử lý trong 30 ngày, trừ dữ liệu hóa đơn phải lưu theo luật."),
    n("N45", "Tivi Sora 75 inch mua trong tháng 10 này được giảm bao nhiêu?", [f"{KM}#tivi-sora"], ["10%", "2.000.000"],
      "Giảm 10%, tối đa 2.000.000 đồng, và được tặng giá treo tường vì từ 65 inch."),
    n("N46", "Mua tủ lạnh Kiwa 16 triệu trong tháng này có quà gì không?", [f"{KM}#tu-lanh-kiwa"], ["Polar Mini"],
      "Tặng máy lọc không khí Polar Mini trị giá 1.290.000 đồng."),
    n("N47", "Điện thoại mới nhận 3 ngày, màn hình có 2 điểm chết ở góc dưới, có được đổi mới không?", ["QT-DOA#man-hinh"], ["3 điểm chết"],
      "Chưa đủ tiêu chí DOA: cần từ 3 điểm chết trở lên hoặc 1 điểm chết ở vùng trung tâm."),
    n("N48", "Laptop mới nhận hôm qua không lên nguồn, nếu cửa hàng hết hàng cùng model thì tôi có những lựa chọn nào?", ["QT-DOA#xu-ly"], ["5 ngày", "100%"],
      "Chờ hàng tối đa 5 ngày, đổi model khác bù trừ chênh lệch, hoặc hoàn tiền 100%."),
    n("N49", "Đơn hàng đang giao rồi, tôi muốn đổi địa chỉ nhận có được không?", ["HD-DONHANG#thay-doi"], ["thời gian nhận"],
      "Khi đơn đang giao chỉ được đổi thời gian nhận trong ngày, không đổi địa chỉ."),
    n("N50", "Đặt online chọn nhận tại cửa hàng thì cửa hàng giữ hàng cho mình mấy ngày?", ["HD-DONHANG#nhan-tai-cua-hang"], ["3 ngày"],
      "Giữ hàng 3 ngày kể từ khi nhận tin nhắn hàng đã sẵn sàng."),
    n("N51", "Công ty tôi muốn mua 15 máy tính xách tay cho nhân viên, có được chiết khấu không?", ["CS-DOANHNGHIEP#dieu-kien", "CS-DOANHNGHIEP#chiet-khau"], ["3%"],
      "Đủ điều kiện khách hàng doanh nghiệp (từ 10 sản phẩm); chiết khấu 3% đến 8% tùy giá trị đơn."),
    n("N52", "Mình đặt trước điện thoại rồi, giờ muốn hủy trước ngày mở bán thì có lấy lại tiền cọc không?", ["CS-DATTRUOC#huy"], ["100%"],
      "Hủy trước ngày mở bán chính thức được hoàn 100% tiền cọc."),
    n("N53", "Cửa hàng Lam Sơn mở cửa đến mấy giờ?", [f"{FAQ}#gio-mo-cua"], ["22:00"], "Mở cửa từ 8:00 đến 22:00 tất cả các ngày."),
    n("N54", "Máy giặt Vinto nhà mình báo lỗi E21 là bị gì vậy?", ["HD-LOI-MAYGIAT#e21"], ["thoát nước"],
      "E21 là lỗi thoát nước: vệ sinh lọc bơm xả, kiểm tra ống xả không gập hoặc đặt quá cao 1 mét."),
    n("N55", "Máy giặt mới lắp tuần trước, khi vắt rung lắc rất mạnh và kêu to", ["HD-LOI-MAYGIAT#rung-lac"], ["bu lông"],
      "Kiểm tra đã tháo 4 bu lông vận chuyển và cân bằng chân máy."),
    n("N56", "Điều hòa Kiwa nhà tôi hiện lỗi E6 là sao?", ["HD-LOI-DIEUHOA#e6"], ["giao tiếp"],
      "E6 là lỗi giao tiếp giữa dàn nóng và dàn lạnh; cần đặt lịch kỹ thuật viên."),
    n("N57", "Điều hòa lắp được 2 tháng thì bị chảy nước ở dàn lạnh, sửa có mất phí không?", ["HD-LOI-DIEUHOA#chay-nuoc"], ["6 tháng"],
      "Trong 6 tháng đầu sau khi Lam Sơn lắp đặt, lỗi chảy nước do lắp đặt được xử lý miễn phí."),
    n("N58", "Tivi Sora của tôi đèn đỏ cứ nháy 6 lần rồi lặp lại", ["HD-LOI-TIVI#den-do-nhap-nhay"], ["đèn nền"],
      "Nháy 6 lần là lỗi đèn nền màn hình, cần đặt lịch bảo hành tại nhà."),
    n("N59", "tu lanh kiwa nha minh hien chu F2 la loi gi", ["HD-LOI-TULANH#f1-f2"], ["ngăn mát"],
      "F2 là lỗi cảm biến nhiệt ngăn mát; rút điện 10 phút, nếu còn lỗi thì đặt lịch bảo hành tại nhà."),
    n("N60", "Laptop Lumio của em không lên màn hình, đèn báo nháy cam 3 lần", ["HD-LOI-DIENTHOAI-LAPTOP#laptop-khong-len"], ["bo mạch"],
      "Nháy cam 3 lần là lỗi bo mạch, cần mang đến trung tâm bảo hành."),
    n("N61", "Robot hút bụi Zenko mua được 8 tháng giờ pin chai chỉ chạy 20 phút, pin còn bảo hành không?",
      [f"{HANG}#nexa-polar-zenko", f"{B25}#thoi-han"], ["6 tháng"], "Pin robot hút bụi chỉ bảo hành 6 tháng nên đã hết bảo hành pin sau 8 tháng."),
    n("N62", "Lõi lọc của máy lọc nước Kiwa có được bảo hành không?", [f"{HANG}#kiwa"], ["vật tư tiêu hao"],
      "Không, lõi lọc là vật tư tiêu hao không được bảo hành."),
    n("N63", "Máy giặt Vinto cửa trước thì động cơ được bảo hành mấy năm?", [f"{HANG}#vinto"], ["10 năm"],
      "Động cơ truyền động trực tiếp được bảo hành 10 năm."),
    n("N64", "Tủ lạnh Kiwa thì máy nén bảo hành bao lâu?", [f"{HANG}#kiwa"], ["10 năm"], "Máy nén tủ lạnh Kiwa bảo hành 10 năm."),
    n("N65", "Tôi có thể gửi máy bảo hành ở cửa hàng Lam Sơn gần nhà thay vì mang ra trung tâm của hãng không?", [f"{HANG}#noi-gui"], ["cửa hàng"],
      "Được, khách gửi tại bất kỳ cửa hàng Lam Sơn nào, Lam Sơn chuyển đến hãng và báo tiến độ."),
    n("N66", "Mâm từ bếp từ Haru được bảo hành bao lâu?", [f"{HANG}#haru"], ["36 tháng"], "Mâm từ bếp từ Haru được bảo hành 36 tháng."),
]


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def vnd(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def product_tickets() -> list[dict]:
    catalog = json.loads((ROOT / "data" / "catalog.json").read_text(encoding="utf-8"))
    pool = [p for p in catalog if not p.get("conflict")]
    rng = random.Random(7)
    picks = rng.sample(pool, 14)
    price_t = ["Giá {name} bao nhiêu vậy shop?", "{name} hiện đang bán giá bao nhiêu ạ?", "cho minh hoi gia {plain}"]
    warr_t = ["{name} được bảo hành bao lâu?", "Mua {name} thì bảo hành mấy tháng vậy?", "{plain} bao hanh bao lau"]
    out = []
    for i, p in enumerate(picks):
        name = p["name"]
        gold = [f"{p['doc_id']}#{p['section']}"]
        if i % 2 == 0:
            text = rng.choice(price_t).format(name=name, plain=strip_accents(name).lower())
            facts, ref = [vnd(p["price"])], f"Giá niêm yết {vnd(p['price'])} đồng (mã {p['sku']})."
        else:
            text = rng.choice(warr_t).format(name=name, plain=strip_accents(name).lower())
            facts = [f"{p['warranty']} tháng"]
            ref = f"Bảo hành chính hãng {p['warranty']} tháng" + (f", {p['extra_warranty']}" if p.get("extra_warranty") else "") + "."
        out.append(n(f"N{67 + i:02d}", text, gold, facts, ref))
    return out


def m(tid, text, slots):
    return dict(id=tid, type="missing_info", expected_decision="clarify", ticket=text, gold_sections=[], key_facts=[],
                reference="Hỏi lại khách: " + ", ".join(slots) + ". Không kết luận khi thiếu thông tin.", missing_slots=slots)


MISSING = [
    m("M01", "Máy của tôi bị lỗi, có được đổi máy mới không?", ["sản phẩm", "ngày mua", "tình trạng lỗi"]),
    m("M02", "Đơn hàng của tôi đến đâu rồi?", ["mã đơn hàng hoặc số điện thoại"]),
    m("M03", "Bảo hành được bao lâu vậy shop?", ["sản phẩm"]),
    m("M04", "Tôi muốn mua trả góp.", ["sản phẩm hoặc giá trị đơn", "hình thức trả góp"]),
    m("M05", "Hủy đơn giúp tôi với.", ["mã đơn hàng"]),
    m("M06", "Tôi muốn trả lại hàng đã mua.", ["sản phẩm", "ngày mua", "lý do trả"]),
    m("M07", "Giá bao nhiêu vậy shop?", ["sản phẩm"]),
    m("M08", "Giao hàng mất bao lâu vậy?", ["khu vực giao hàng"]),
    m("M09", "Bao giờ tôi nhận được tiền hoàn?", ["mã đơn hàng", "phương thức thanh toán"]),
    m("M10", "Phí lắp đặt bao nhiêu?", ["sản phẩm"]),
    m("M11", "Máy không lên nguồn, giờ phải làm sao?", ["sản phẩm"]),
    m("M12", "Tôi có được trợ giá thu cũ không?", ["sản phẩm cũ", "sản phẩm mới"]),
    m("M13", "Cái máy mua tháng trước hỏng rồi.", ["sản phẩm", "tình trạng lỗi"]),
    m("M14", "Tôi muốn đổi sang model khác.", ["sản phẩm hiện tại", "ngày mua", "lý do đổi"]),
    m("M15", "Sản phẩm này có được bảo hành tại nhà không?", ["sản phẩm"]),
    m("M16", "Kiểm tra giúp tôi máy còn bảo hành không.", ["số điện thoại, mã đơn hoặc IMEI/serial"]),
    m("M17", "Mình muốn xuất hóa đơn VAT.", ["mã đơn hàng", "thông tin công ty"]),
    m("M18", "may giat nha minh bao loi, sua sao a", ["mã lỗi"]),
    m("M19", "Đổi máy có mất phí không?", ["sản phẩm", "lý do đổi (lỗi hay đổi ý)", "ngày mua"]),
    m("M20", "Tôi được giảm bao nhiêu tiền?", ["sản phẩm", "chương trình hoặc hạng thành viên"]),
    m("M21", "Shipper hẹn mà không giao, tôi phải làm gì?", ["mã đơn hàng"]),
    m("M22", "Tôi muốn đổi địa chỉ nhận hàng.", ["mã đơn hàng"]),
    m("M23", "Bảo hành bao giờ xong vậy?", ["số phiếu bảo hành hoặc số điện thoại"]),
    m("M24", "Có được trả góp 0% không?", ["sản phẩm hoặc giá trị đơn"]),
    m("M25", "Tôi muốn mua gói bảo hành mở rộng.", ["sản phẩm", "ngày mua"]),
]


def e(tid, text):
    return dict(id=tid, type="no_evidence", expected_decision="escalate", ticket=text, gold_sections=[], key_facts=[],
                reference="Kho tri thức không có căn cứ: từ chối kết luận, không bịa thông tin, chuyển nhân viên.", missing_slots=[])


NO_EVIDENCE = [
    e("E01", "Lam Sơn có bán xe máy điện không?"),
    e("E02", "Cửa hàng có tuyển nhân viên bán hàng part-time không?"),
    e("E03", "Tôi muốn mua máy ảnh Canon EOS R50, bên mình có hàng không?"),
    e("E04", "Lam Sơn có nhận thay màn hình điện thoại mua ở chỗ khác không, giá bao nhiêu?"),
    e("E05", "Tháng 11 có chương trình Black Friday không, giảm bao nhiêu?"),
    e("E06", "Tôi muốn thuê 5 cái tivi trong 3 ngày để tổ chức sự kiện được không?"),
    e("E07", "Mua hàng có giao sang Lào được không?"),
    e("E08", "Tôi muốn đặt cọc giữ giá tủ lạnh trong 2 tháng tới được không?"),
    e("E09", "Cửa hàng Lam Sơn có chỗ đỗ ô tô không?"),
    e("E10", "Bên mình có bán điều hòa Daikin không?"),
    e("E11", "Lam Sơn có bảo hành pin xe đạp điện không?"),
    e("E12", "Tôi muốn làm đại lý phân phối cho Lam Sơn ở Thái Bình thì liên hệ ai?"),
    e("E13", "Bên mình có dịch vụ vệ sinh máy giặt tại nhà không, giá bao nhiêu?"),
    e("E14", "Giá vàng hôm nay bao nhiêu?"),
    e("E15", "Lam Sơn có ứng dụng trên điện thoại không, tải ở đâu?"),
    e("E16", "Tôi muốn mua máy chạy bộ cho gia đình."),
    e("E17", "Lam Sơn có nhận thanh toán bằng Bitcoin không?"),
    e("E18", "Tivi Sora có xem được Netflix không?"),
    e("E19", "Khi nào Sora Prime 13 ra mắt vậy?"),
    e("E20", "Tôi bị mất điện thoại vừa mua, Lam Sơn có định vị giúp được không?"),
    e("E21", "Cửa hàng Lam Sơn ở Hà Giang địa chỉ ở đâu?"),
    e("E22", "Mua hàng có được tặng kèm bảo hiểm cháy nổ không?"),
    e("E23", "Lam Sơn có lắp đặt pin năng lượng mặt trời cho nhà dân không?"),
    e("E24", "Laptop Lumio Air 13 có chạy được AutoCAD 2025 mượt không?"),
    e("E25", "Thanh toán bằng thẻ Visa phát hành ở nước ngoài thì phí chuyển đổi ngoại tệ là bao nhiêu?"),
]


def k(tid, text, gold, ref):
    return dict(id=tid, type="conflict", expected_decision="escalate", ticket=text, gold_sections=gold, key_facts=[],
                reference=ref, missing_slots=[])


C1, C2, C3 = [f"{KM}#doi-tra", f"{DT}#doi-y"], ["SP-LAPTOP#lm-a14", f"{HANG}#lumio"], [f"{GH}#phi-giao", f"{FAQ}#giao-hang"]
C4, C5, C6 = [f"{CP}#thoi-han-dang-ky", f"{FAQ}#care-plus"], ["SP-DIENTHOAI#sr-dt01", f"{TG}#dieu-kien-chung"], [f"{HT}#thoi-gian", "QT-KHIEUNAI#hoan-tien"]
C7, C8 = ["SP-DIEUHOA#kw-dh03", f"{HANG}#kiwa"], [f"{TC}#dieu-kien", f"{KM}#thu-cu-sora"]
CONFLICT = [
    k("K01", "Tôi mua tivi Sora 55 inch ngày 05/10 trong đợt Mừng Thu Sale, giờ được 10 ngày, không còn nhu cầu nên muốn trả lại có được không?", C1,
      "Thể lệ khuyến mãi cho đổi ý 15 ngày, chính sách đổi trả chung cho 7 ngày: mâu thuẫn, chuyển nhân viên xác nhận."),
    k("K02", "Mình mua tủ lạnh Kiwa đợt khuyến mãi tháng 10, chưa lắp, đã 12 ngày, giờ đổi ý muốn trả được không?", C1,
      "Mâu thuẫn 15 ngày (khuyến mãi) và 7 ngày (đổi trả chung): chuyển nhân viên."),
    k("K03", "Hàng mua trong chương trình khuyến mãi tháng 10 thì được đổi ý trong mấy ngày?", C1, "Mâu thuẫn 15 ngày và 7 ngày: chuyển nhân viên."),
    k("K04", "Laptop Lumio Air 14 bảo hành mấy tháng?", C2, "Trang sản phẩm ghi 24 tháng, thông tin hãng ghi dòng Air 12 tháng: chuyển nhân viên."),
    k("K05", "Lumio Air 14 của tôi mua được 15 tháng thì hỏng bàn phím, còn bảo hành không?", C2, "Mâu thuẫn 24 và 12 tháng: chuyển nhân viên."),
    k("K06", "Bảo hành của máy Lumio Air 14 là 12 hay 24 tháng vậy?", C2, "Mâu thuẫn: chuyển nhân viên."),
    k("K07", "Nhà mình cách cửa hàng 15 km, đơn nồi cơm điện 790 nghìn có được miễn phí giao không?", C3,
      "Chính sách giao hàng: miễn phí 20 km cho đơn từ 500.000; FAQ: miễn phí 10 km mọi đơn: mâu thuẫn."),
    k("K08", "Đơn hàng 300 nghìn ở gần cửa hàng có được giao miễn phí không?", C3, "Chính sách tính 20.000 đồng, FAQ ghi miễn phí mọi đơn trong 10 km: mâu thuẫn."),
    k("K09", "Lam Sơn miễn phí giao hàng trong bán kính bao nhiêu km?", C3, "20 km (chính sách) và 10 km (FAQ): mâu thuẫn."),
    k("K10", "Tôi mua điện thoại được 20 ngày, giờ mua gói Care+ còn kịp không?", C4, "Chính sách Care+ 30 ngày, FAQ 7 ngày: mâu thuẫn."),
    k("K11", "Gói Care+ phải mua trong bao lâu kể từ ngày mua máy?", C4, "30 ngày và 7 ngày: mâu thuẫn."),
    k("K12", "Mình muốn mua điện thoại Sora Lite A3 theo hình thức trả góp 0% được không?", C5,
      "Giá 2.690.000 đồng; trang sản phẩm ghi trả góp từ 2 triệu nhưng chính sách yêu cầu đơn từ 3 triệu: mâu thuẫn."),
    k("K13", "Đơn hàng Sora Lite A3 2,69 triệu có trả góp được không? Mình thấy trang sản phẩm ghi trả góp từ 2 triệu.", C5, "Mâu thuẫn mức tối thiểu 2 và 3 triệu."),
    k("K14", "Tôi trả hàng, lúc mua thanh toán bằng thẻ tín dụng thì bao lâu nhận lại tiền?", C6,
      "Chính sách hoàn tiền: 7 đến 15 ngày làm việc; quy trình khiếu nại: 5 ngày làm việc: mâu thuẫn."),
    k("K15", "Khiếu nại của tôi đã được duyệt hoàn tiền về thẻ tín dụng 6 ngày làm việc rồi mà chưa thấy tiền, như vậy có đúng hạn không?", C6,
      "Quy trình khiếu nại cam kết 5 ngày, chính sách hoàn tiền cho 7 đến 15 ngày: mâu thuẫn."),
    k("K16", "Điều hòa Kiwa Inverter 1.5 HP máy nén được bảo hành mấy năm?", C7, "Trang sản phẩm ghi 10 năm, thông tin hãng ghi 5 năm: mâu thuẫn."),
    k("K17", "Máy nén điều hòa Kiwa Inverter 1.5 HP nhà tôi hỏng sau 6 năm sử dụng, còn bảo hành không?", C7, "10 năm và 5 năm: mâu thuẫn."),
    k("K18", "Thu cũ điện thoại lên đời Sora Nova 8 trong tháng 10 có được dùng thêm mã giảm 5% không?", C8,
      "Thu cũ không áp dụng đồng thời khuyến mãi khác, nhưng thể lệ tháng 10 cho phép cộng mã 5%: mâu thuẫn."),
    k("K19", "Đổi máy cũ lấy Sora Prime 12 trong tháng 10 thì được trợ giá và giảm giá cộng dồn không?", C8, "Mâu thuẫn về cộng dồn ưu đãi."),
    k("K20", "Chương trình thu cũ đổi mới có áp dụng cùng khuyến mãi Mừng Thu Sale không?", C8, "Mâu thuẫn."),
]


def main():
    tickets = NORMAL + product_tickets() + MISSING + NO_EVIDENCE + CONFLICT
    ids = [t["id"] for t in tickets]
    assert len(ids) == len(set(ids)) == 150, (len(ids), len(set(ids)))
    # every gold section must exist in the knowledge base
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from rag_cskh.ingest import load_chunks
    sections = {c.section_key for c in load_chunks(ROOT / "data" / "kb")}
    missing = [(t["id"], g) for t in tickets for g in t["gold_sections"] if g not in sections]
    assert not missing, f"unknown gold sections: {missing}"
    out = ROOT / "data" / "eval" / "tickets.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for t in tickets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(tickets), "tickets", dict(Counter(t["type"] for t in tickets)))


if __name__ == "__main__":
    main()
