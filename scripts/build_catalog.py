"""Generate the 25 product catalog documents (data/kb/SP-*.md) and data/catalog.json for the fictional retailer.

All brands, models, prices and specs are synthetic. Generation is deterministic (fixed seed).
Warranty periods match TT-BAOHANH-HANG and CS-BAOHANH-2025 except for 3 planted conflicts
(marked `conflict` in catalog.json) that test the "conflicting information -> escalate" branch.

    python scripts/build_catalog.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "kb"
rng = random.Random(20251009)


def vnd(x: int) -> str:
    return f"{x:,}".replace(",", ".") + " đồng"


def round_price(x: float) -> int:
    return int(round(x / 10_000) * 10_000 - 10_000)


# --------------------------------------------------------------------------- #
# Categories: (code, category name, default brand, warranty months, product generator)
# --------------------------------------------------------------------------- #
def tivi():
    out = []
    lines = [("Sora Vivid", "LED 4K", 0.9), ("Sora QLED", "QLED 4K", 1.25), ("Sora OLED", "OLED 4K", 1.9)]
    sizes = [32, 43, 50, 55, 65, 75]
    for series, panel, k in lines:
        for s in (sizes[:4] if series == "Sora Vivid" else sizes[2:]):
            if series == "Sora OLED" and s == 50:
                continue
            code = f"{series.split()[1][0]}{s}"
            price = round_price((3_500_000 + s ** 2.05 * 1_900) * k)
            out.append(dict(
                name=f"Tivi {series} {s} inch {code}", sku=f"SR-TV{code}", price=price,
                specs=[f"Kích thước màn hình: {s} inch", f"Công nghệ màn hình: {panel}",
                       f"Tần số quét: {120 if k > 1 else 60} Hz", "Hệ điều hành: Sora OS 6",
                       f"Công suất loa: {20 if s < 55 else 40} W", f"Cổng HDMI: {3 if s < 55 else 4} cổng (1 cổng eARC)"],
                warranty=24, extra_warranty="tấm nền OLED bảo hành 36 tháng" if "OLED" in series else None,
                notes=[f"Kích thước có chân đế khoảng {round(s * 2.2)} x {round(s * 1.45)} cm.",
                       "Hỗ trợ điều khiển giọng nói tiếng Việt và chiếu màn hình từ điện thoại."]))
    return out


def dien_thoai():
    out = []
    models = [("Sora Lite A3", 2_690_000, "4 GB", "64 GB", 5000, 18), ("Sora Lite A5", 3_490_000, "6 GB", "128 GB", 5000, 25),
              ("Sora Nova 7", 5_990_000, "8 GB", "128 GB", 5000, 33), ("Sora Nova 7 Plus", 7_490_000, "8 GB", "256 GB", 5200, 45),
              ("Sora Nova 8", 8_990_000, "12 GB", "256 GB", 5000, 45), ("Sora Prime 12", 15_990_000, "12 GB", "256 GB", 4800, 65),
              ("Sora Prime 12 Pro", 21_990_000, "16 GB", "512 GB", 5000, 80), ("Sora Prime Fold", 36_990_000, "16 GB", "512 GB", 4400, 45),
              ("Sora Prime Flip", 22_990_000, "12 GB", "256 GB", 3900, 33), ("Sora Active X", 9_490_000, "8 GB", "256 GB", 6000, 33),
              ("Sora Lite A3s", 2_990_000, "4 GB", "128 GB", 5000, 18), ("Sora Nova 8 Pro", 11_990_000, "12 GB", "512 GB", 5100, 65)]
    for i, (name, price, ram, rom, bat, chg) in enumerate(models):
        sku = f"SR-DT{i + 1:02d}"
        notes = [f"Pin {bat} mAh, sạc nhanh {chg} W.", "Bảo hành pin 6 tháng theo chính sách chung."]
        if name == "Sora Lite A3":
            notes.append("Sản phẩm hỗ trợ trả góp 0% cho đơn hàng từ 2.000.000 đồng.")      # conflict C5 (policy minimum is 3M)
        out.append(dict(name=f"Điện thoại {name}", sku=sku, price=price,
                        specs=[f"RAM {ram}, bộ nhớ trong {rom}", f"Màn hình {rng.choice(['6,5', '6,6', '6,7'])} inch",
                               f"Camera chính {rng.choice([50, 64, 108, 200])} MP", "Hỗ trợ 2 sim, 5G" if price > 5_000_000 else "Hỗ trợ 2 sim, 4G"],
                        warranty=12, notes=notes, conflict="C5-tra-gop" if name == "Sora Lite A3" else None))
    return out


def may_tinh_bang():
    out = []
    for i, (name, price, size) in enumerate([("Sora Tab S6 Lite", 6_490_000, 10.4), ("Sora Tab S7", 9_990_000, 11),
                                             ("Sora Tab S7 FE", 8_490_000, 12.4), ("Sora Tab S8", 14_990_000, 11),
                                             ("Sora Tab S8 Plus", 19_990_000, 12.4), ("Sora Tab Kids", 3_990_000, 8.7),
                                             ("Sora Tab A9", 4_990_000, 11), ("Sora Tab S8 Ultra", 26_990_000, 14.6)]):
        out.append(dict(name=f"Máy tính bảng {name}", sku=f"SR-TAB{i + 1:02d}", price=price,
                        specs=[f"Màn hình {size} inch", f"RAM {rng.choice([4, 6, 8, 12])} GB",
                               f"Pin {rng.choice([7040, 8000, 10090, 11200])} mAh", "Hỗ trợ bút cảm ứng" if price > 8_000_000 else "Không hỗ trợ bút"],
                        warranty=12, notes=["Bảo hành pin 6 tháng."]))
    return out


def laptop():
    out = []
    rows = [("Lumio Go 14", 9_990_000, "Core i3", 8, 256, "Go"), ("Lumio Go 15", 11_490_000, "Core i5", 8, 512, "Go"),
            ("Lumio Air 13", 16_990_000, "Core Ultra 5", 16, 512, "Air"), ("Lumio Air 14", 19_490_000, "Core Ultra 5", 16, 512, "Air"),
            ("Lumio Air 15", 21_990_000, "Core Ultra 7", 16, 1024, "Air"), ("Lumio Pro 14", 29_990_000, "Core Ultra 7", 32, 1024, "Pro"),
            ("Lumio Pro 16", 37_990_000, "Core Ultra 9", 32, 1024, "Pro"), ("Lumio Pro 16 Creator", 49_990_000, "Core Ultra 9", 64, 2048, "Pro"),
            ("Lumio Go 14 Student", 8_490_000, "Core i3", 8, 256, "Go"), ("Lumio Air 13 OLED", 23_490_000, "Core Ultra 7", 16, 1024, "Air"),
            ("Lumio Pro 14 Business", 32_990_000, "Core Ultra 7", 32, 1024, "Pro"), ("Lumio Go 15 Plus", 13_490_000, "Core i5", 16, 512, "Go")]
    for name, price, cpu, ram, ssd, line in rows:
        warranty = 24 if line == "Pro" else 12
        conflict = None
        if name == "Lumio Air 14":
            warranty, conflict = 24, "C2-lumio-air"                               # conflict C2 (brand page: Air 12 months)
        out.append(dict(name=f"Máy tính xách tay {name}", sku="LM-" + name.split()[1][0] + name.split()[2] + ("".join(w[0] for w in name.split()[3:])),
                        price=price, specs=[f"CPU {cpu}", f"RAM {ram} GB", f"SSD {ssd} GB", f"Trọng lượng {rng.choice([1.2, 1.3, 1.4, 1.6, 1.8, 2.1])} kg",
                                            "Màn hình OLED" if "OLED" in name or "Creator" in name else "Màn hình IPS"],
                        warranty=warranty, conflict=conflict,
                        notes=["Cài sẵn Windows 11 Home bản quyền.", "Tặng balo và chuột không dây khi mua trong tháng 10/2025." if line == "Air" else "Hỗ trợ nâng cấp RAM và SSD."]))
    return out


def man_hinh():
    out = []
    for i, (size, res, hz, price) in enumerate([(22, "Full HD", 75, 2_290_000), (24, "Full HD", 100, 2_790_000), (24, "Full HD", 165, 3_690_000),
                                                (27, "2K", 75, 4_990_000), (27, "2K", 165, 6_490_000), (27, "4K", 60, 7_990_000),
                                                (32, "4K", 144, 12_990_000), (34, "UltraWide 2K", 100, 10_490_000),
                                                (27, "4K", 144, 11_490_000), (24, "2K", 75, 3_990_000)]):
        out.append(dict(name=f"Màn hình Lumio View {size} inch {res} {hz}Hz", sku=f"LM-MH{size}{i}", price=price,
                        specs=[f"Kích thước {size} inch", f"Độ phân giải {res}", f"Tần số quét {hz} Hz", f"Tấm nền {rng.choice(['IPS', 'VA', 'IPS'])}"],
                        warranty=36, notes=["Đổi mới nếu có từ 1 điểm chết trong 30 ngày đầu theo cam kết của hãng Lumio."]))
    return out


def tai_nghe():
    out = []
    rows = [("Sora Buds Lite", 690_000, "nhét tai", "Sora"), ("Sora Buds 3", 1_490_000, "nhét tai", "Sora"), ("Sora Buds 3 Pro", 2_990_000, "nhét tai", "Sora"),
            ("Sora Wave Over-ear", 3_490_000, "chụp tai", "Sora"), ("Nexa Pods", 890_000, "nhét tai", "Nexa"), ("Nexa Pods ANC", 1_790_000, "nhét tai", "Nexa"),
            ("Nexa Studio H1", 2_490_000, "chụp tai", "Nexa"), ("Nexa Studio H2", 4_290_000, "chụp tai", "Nexa"), ("Nexa Sport Neck", 590_000, "đeo cổ", "Nexa"),
            ("Nexa Kids", 490_000, "chụp tai", "Nexa"), ("Sora Buds Open", 1_990_000, "mở", "Sora"), ("Nexa Game G5", 1_290_000, "chụp tai", "Nexa")]
    for i, (name, price, kind, brand) in enumerate(rows):
        notes = ["Tai nghe nhét tai đã bóc niêm phong không áp dụng đổi trả đổi ý."] if kind == "nhét tai" else ["Kèm hộp đựng và cáp sạc USB-C."]
        out.append(dict(name=f"Tai nghe {name}", sku=f"{brand[:2].upper()}-TN{i + 1:02d}", price=price,
                        specs=[f"Kiểu {kind}", f"Thời lượng pin {rng.choice([6, 8, 24, 30, 40])} giờ", "Chống ồn chủ động" if "ANC" in name or "Pro" in name or "H2" in name else "Không chống ồn chủ động",
                               f"Bluetooth {rng.choice(['5.3', '5.4'])}"],
                        warranty=12, brand=brand, notes=notes))
    return out


def loa():
    out = []
    for i, (name, price, w) in enumerate([("Nexa Go Mini", 590_000, 5), ("Nexa Go 2", 1_190_000, 10), ("Nexa Boom 3", 2_490_000, 30),
                                          ("Nexa Party 100", 5_990_000, 100), ("Nexa Soundbar S2", 3_990_000, 120), ("Nexa Soundbar S5", 8_990_000, 400),
                                          ("Nexa Home Smart", 1_990_000, 20), ("Nexa Karaoke K1", 4_490_000, 80), ("Nexa Boom 3 Pro", 3_490_000, 40),
                                          ("Nexa Soundbar S7 Atmos", 14_990_000, 600)]):
        out.append(dict(name=f"Loa {name}", sku=f"NX-L{i + 1:02d}", price=price,
                        specs=[f"Công suất {w} W", f"Chống nước {rng.choice(['IPX4', 'IPX5', 'IP67', 'không'])}", "Kết nối Bluetooth, AUX" + (", HDMI eARC" if "Soundbar" in name else "")],
                        warranty=12, notes=["Kèm 2 micro không dây." if "Karaoke" in name else "Hỗ trợ ghép đôi 2 loa cùng dòng."]))
    return out


def dong_ho():
    return [dict(name=f"Đồng hồ thông minh {n}", sku=f"SR-W{i + 1:02d}", price=p,
                 specs=[f"Màn hình {rng.choice(['1,4', '1,5', '1,9'])} inch AMOLED", f"Pin {rng.choice([2, 5, 7, 14])} ngày", "Đo nhịp tim, SpO2", "Chống nước 5 ATM"],
                 warranty=12, notes=["Dây đeo là phụ kiện, bảo hành 6 tháng."])
            for i, (n, p) in enumerate([("Sora Watch 6", 5_490_000), ("Sora Watch 6 Classic", 7_990_000), ("Sora Watch Fit 3", 1_990_000),
                                        ("Sora Watch Kids", 1_490_000), ("Sora Watch Ultra", 13_990_000), ("Sora Band 9", 890_000),
                                        ("Sora Watch 6 LTE", 6_990_000), ("Sora Watch Active", 3_490_000)])]


def tu_lanh():
    out = []
    rows = [(150, "ngăn đá trên", 5_490_000), (180, "ngăn đá trên", 6_490_000), (230, "ngăn đá trên", 7_990_000), (260, "ngăn đá dưới", 10_490_000),
            (320, "ngăn đá dưới", 12_990_000), (380, "ngăn đá dưới", 15_490_000), (450, "side by side", 18_990_000), (520, "side by side", 22_990_000),
            (560, "multi door", 26_990_000), (640, "multi door", 34_990_000), (90, "mini", 3_290_000), (400, "ngăn đá dưới", 16_990_000)]
    for i, (lit, kind, price) in enumerate(rows):
        out.append(dict(name=f"Tủ lạnh Kiwa Inverter {lit} lít {kind}", sku=f"KW-TL{lit}{i}", price=price,
                        specs=[f"Dung tích {lit} lít", f"Kiểu {kind}", "Công nghệ không đóng tuyết" if lit >= 180 else "Làm lạnh trực tiếp",
                               f"Điện năng tiêu thụ khoảng {round(0.6 + lit / 400, 1)} kWh/ngày"],
                        warranty=24, extra_warranty="máy nén bảo hành 10 năm",
                        notes=["Được tặng máy lọc không khí Polar Mini khi mua trong chương trình Mừng Thu Sale." if price >= 15_000_000 else "Lắp đặt miễn phí."]))
    return out


def dieu_hoa():
    out = []
    rows = [("1 HP", 9_000, 6_990_000, False), ("1 HP", 9_000, 7_990_000, True), ("1.5 HP", 12_000, 9_490_000, True), ("1.5 HP", 12_000, 8_490_000, False),
            ("2 HP", 18_000, 13_990_000, True), ("2.5 HP", 24_000, 18_990_000, True), ("1 HP 2 chiều", 9_000, 9_990_000, True),
            ("1.5 HP 2 chiều", 12_000, 11_990_000, True), ("2 HP 2 chiều", 18_000, 16_490_000, True), ("1 HP Wi-Fi", 9_000, 8_990_000, True),
            ("1.5 HP Wi-Fi", 12_000, 10_990_000, True), ("Tủ đứng 4 HP", 36_000, 32_990_000, True)]
    for i, (cap, btu, price, inv) in enumerate(rows):
        name = f"Điều hòa Kiwa {'Inverter ' if inv else ''}{cap}"
        extra, conflict = "máy nén bảo hành 5 năm", None
        if name == "Điều hòa Kiwa Inverter 1.5 HP":
            extra, conflict = "máy nén bảo hành 10 năm", "C7-may-nen"               # conflict C7 (brand page: 5 years)
        out.append(dict(name=name, sku=f"KW-DH{i + 1:02d}", price=price,
                        specs=[f"Công suất làm lạnh {btu:,} BTU".replace(",", "."), f"Phù hợp phòng {btu // 600}-{btu // 450} m²",
                               "Inverter tiết kiệm điện" if inv else "Không Inverter", "Gas R32"],
                        warranty=24, extra_warranty=extra, conflict=conflict,
                        notes=["Miễn phí công lắp đặt, gồm 3 mét ống đồng."]))
    return out


def may_loc_nuoc():
    return [dict(name=f"Máy lọc nước Kiwa {n}", sku=f"KW-LN{i + 1:02d}", price=p,
                 specs=[f"{c} lõi lọc", f"Công suất lọc {rng.choice([10, 15, 20])} lít/giờ", "Có chế độ nóng lạnh" if "Nóng lạnh" in n else "Không có chế độ nóng lạnh"],
                 warranty=12, notes=["Lõi lọc là vật tư tiêu hao, không được bảo hành; nên thay lõi số 1 sau 6 tháng."])
            for i, (n, c, p) in enumerate([("RO 9 lõi", 9, 5_990_000), ("RO 10 lõi", 10, 6_990_000), ("RO Nóng lạnh 10 lõi", 10, 9_490_000),
                                          ("Hydrogen 11 lõi", 11, 10_990_000), ("Âm tủ bếp 8 lõi", 8, 7_490_000), ("RO 7 lõi Mini", 7, 4_490_000),
                                          ("Nóng lạnh Slim", 9, 8_490_000), ("Kiềm 12 lõi", 12, 12_990_000)])]


def may_giat():
    out = []
    rows = [(8, "cửa trên", 5_490_000), (9, "cửa trên", 6_290_000), (10, "cửa trên", 7_490_000), (8, "cửa trước", 7_990_000), (9, "cửa trước", 8_990_000),
            (10, "cửa trước", 10_490_000), (12, "cửa trước", 13_990_000), (9, "giặt sấy", 15_990_000), (11, "giặt sấy", 19_990_000),
            (14, "cửa trên", 9_990_000), (8.5, "cửa trước", 8_490_000), (13, "giặt sấy", 23_990_000)]
    for i, (kg, kind, price) in enumerate(rows):
        dd = kind != "cửa trên"
        out.append(dict(name=f"Máy giặt Vinto {kg} kg {kind}", sku=f"VT-MG{i + 1:02d}", price=price,
                        specs=[f"Khối lượng giặt {kg} kg", f"Kiểu {kind}", "Động cơ truyền động trực tiếp" if dd else "Động cơ dây curoa",
                               f"Tốc độ vắt {rng.choice([700, 1200, 1400])} vòng/phút"],
                        warranty=24, extra_warranty="động cơ truyền động trực tiếp bảo hành 10 năm" if dd else None,
                        notes=["Lắp đặt miễn phí; khi lắp lần đầu cần tháo 4 bu lông vận chuyển."]))
    return out


def may_say():
    return [dict(name=f"Máy sấy quần áo Vinto {n}", sku=f"VT-MS{i + 1:02d}", price=p,
                 specs=[f"Khối lượng sấy {kg} kg", f"Công nghệ {t}"], warranty=24, notes=["Nên vệ sinh lưới lọc xơ vải sau mỗi lần sấy."])
            for i, (n, kg, t, p) in enumerate([("7 kg thông hơi", 7, "sấy thông hơi", 5_990_000), ("8 kg ngưng tụ", 8, "sấy ngưng tụ", 9_490_000),
                                              ("9 kg bơm nhiệt", 9, "bơm nhiệt", 15_990_000), ("10 kg bơm nhiệt", 10, "bơm nhiệt", 18_490_000),
                                              ("8 kg bơm nhiệt", 8, "bơm nhiệt", 13_990_000), ("7.5 kg ngưng tụ", 7.5, "sấy ngưng tụ", 8_290_000)])]


def may_rua_bat():
    return [dict(name=f"Máy rửa bát Vinto {n}", sku=f"VT-RB{i + 1:02d}", price=p,
                 specs=[f"{b} bộ chén đĩa", f"Kiểu {k}", f"{rng.choice([6, 7, 8])} chương trình rửa"], warranty=24,
                 notes=["Lắp máy âm tủ tính phí 300.000 đồng." if "âm tủ" in k else "Lắp đặt miễn phí."])
            for i, (n, b, k, p) in enumerate([("Mini 6 bộ", 6, "để bàn", 6_990_000), ("12 bộ độc lập", 12, "độc lập", 11_990_000),
                                              ("14 bộ độc lập", 14, "độc lập", 14_490_000), ("13 bộ âm tủ", 13, "âm tủ toàn phần", 16_990_000),
                                              ("15 bộ âm tủ sấy khí nóng", 15, "âm tủ toàn phần", 21_990_000), ("10 bộ bán âm", 10, "bán âm tủ", 12_490_000)])]


def haru(kind, code, rows, specs_fn, note):
    return [dict(name=f"{kind} Haru {n}", sku=f"HR-{code}{i + 1:02d}", price=p,
                 specs=specs_fn(n), warranty=12, notes=[note]) for i, (n, p) in enumerate(rows)]


def noi_com():
    return haru("Nồi cơm điện", "NC", [("1.0 lít cơ", 590_000), ("1.8 lít cơ", 790_000), ("1.8 lít điện tử", 1_490_000), ("1.5 lít cao tần", 2_990_000),
                                 ("1.8 lít cao tần", 3_990_000), ("1.0 lít mini", 690_000), ("1.8 lít áp suất", 4_990_000), ("2.2 lít điện tử", 1_890_000)],
                lambda n: [f"Dung tích {n.split()[0]} lít", f"Công nghệ nấu {' '.join(n.split()[2:])}", f"Lòng nồi phủ chống dính {rng.choice(['2', '5', '7'])} lớp"],
                "Lòng nồi bị trầy xước do dùng muỗng kim loại không được bảo hành.")


def lo_vi_song():
    return haru("Lò vi sóng", "LV", [("20 lít cơ", 1_490_000), ("20 lít điện tử", 1_890_000), ("23 lít có nướng", 2_490_000), ("25 lít có nướng", 2_990_000),
                               ("28 lít đối lưu", 4_490_000), ("32 lít đối lưu", 5_990_000), ("20 lít mini", 1_290_000), ("25 lít inverter", 3_790_000)],
                lambda n: [f"Dung tích {n.split()[0]} lít", f"Chức năng {' '.join(n.split()[2:])}", f"Công suất vi sóng {rng.choice([700, 800, 900, 1000])} W"],
                "Không dùng hộp kim loại trong chế độ vi sóng.")


def bep_tu():
    return haru("Bếp từ", "BT", [("đơn 2000W", 990_000), ("đôi 4000W", 6_990_000), ("đôi inverter", 8_990_000), ("đôi từ hồng ngoại", 7_490_000),
                           ("ba vùng nấu", 12_990_000), ("đơn cảm ứng", 1_290_000), ("đôi nhập khẩu", 15_990_000), ("đôi lắp âm", 9_490_000)],
                lambda n: [f"Kiểu bếp {n}", f"Mặt kính {rng.choice(['ceramic', 'Schott Ceran'])}", "Khóa trẻ em, hẹn giờ"],
                "Mâm từ bảo hành 36 tháng theo chính sách hãng Haru.")


def may_pha_cafe():
    return haru("Máy pha cà phê", "CF", [("Espresso Mini", 2_490_000), ("Espresso Pro 20 bar", 5_990_000), ("tự động Barista", 14_990_000),
                                   ("viên nén", 2_990_000), ("pha phin điện", 790_000), ("Espresso có máy xay", 9_490_000)],
                lambda n: [f"Kiểu máy {n}", f"Bình chứa nước {rng.choice([1.0, 1.2, 1.5, 1.8])} lít"],
                "Cần vệ sinh cặn canxi mỗi 3 tháng; hỏng do cặn canxi không được bảo hành.")


def quat():
    return haru("Quạt", "QT", [("đứng 5 cánh", 690_000), ("đứng DC inverter", 1_990_000), ("treo tường", 590_000), ("tháp", 1_490_000),
                         ("điều hòa hơi nước", 2_990_000), ("bàn mini", 390_000), ("trần 5 cánh có đèn", 3_490_000), ("sạc tích điện", 1_190_000)],
                lambda n: [f"Kiểu quạt {n}", f"Công suất {rng.choice([35, 45, 55, 120])} W", "Có điều khiển từ xa" if rng.random() > 0.4 else "Không có điều khiển"],
                "Quạt sạc tích điện: pin bảo hành 6 tháng.")


def may_loc_kk():
    return [dict(name=f"Máy lọc không khí Polar {n}", sku=f"PL-KK{i + 1:02d}", price=p,
                 specs=[f"Phòng đến {m} m²", "Màng lọc HEPA H13", "Cảm biến bụi PM2.5"], warranty=12,
                 notes=["Màng lọc HEPA là vật tư tiêu hao, không được bảo hành; thay sau 12 tháng."])
            for i, (n, m, p) in enumerate([("Mini", 18, 1_290_000), ("Air 30", 30, 2_990_000), ("Air 45", 45, 4_490_000), ("Air 60 Wi-Fi", 60, 6_990_000),
                                          ("Air 80 Pro", 80, 9_990_000), ("Humid 40 tạo ẩm", 40, 5_490_000), ("Car mini", 5, 890_000), ("Air 100 Max", 100, 13_990_000)])]


def robot():
    return [dict(name=f"Robot hút bụi Zenko {n}", sku=f"ZK-RB{i + 1:02d}", price=p,
                 specs=[f"Lực hút {s} Pa", "Lau nhà" if "Mop" in n or "Pro" in n else "Không lau nhà", "Bản đồ LiDAR" if p > 5_000_000 else "Điều hướng con quay"],
                 warranty=12, notes=["Pin bảo hành 6 tháng; chổi và giẻ lau là vật tư tiêu hao."])
            for i, (n, s, p) in enumerate([("S1", 2500, 2_990_000), ("S2 Mop", 4000, 4_990_000), ("X5 Pro", 6000, 9_990_000), ("X7 Pro tự giặt giẻ", 8000, 15_990_000),
                                          ("S2 Lite", 3000, 3_690_000), ("X5 Mop", 5000, 7_490_000), ("X9 Ultra", 12000, 24_990_000), ("Mini M1", 2000, 2_290_000)])]


def camera():
    return [dict(name=f"Camera an ninh Zenko {n}", sku=f"ZK-CM{i + 1:02d}", price=p,
                 specs=[f"Độ phân giải {r}", "Ngoài trời chống nước IP66" if "ngoài trời" in n else "Trong nhà, xoay 360 độ", "Hỗ trợ thẻ nhớ đến 256 GB"],
                 warranty=24, notes=["Gói lưu trữ đám mây tính phí riêng, không bao gồm trong giá máy."])
            for i, (n, r, p) in enumerate([("C2 trong nhà", "2 MP", 590_000), ("C3 trong nhà", "3 MP", 790_000), ("C5 trong nhà 2K", "5 MP", 1_190_000),
                                          ("O3 ngoài trời", "3 MP", 1_290_000), ("O5 ngoài trời 2K", "5 MP", 1_690_000), ("O8 ngoài trời 4K", "8 MP", 2_490_000),
                                          ("D1 chuông cửa", "2 MP", 1_490_000), ("P4 pin năng lượng mặt trời ngoài trời", "4 MP", 2_990_000)])]


def binh_nong_lanh():
    return [dict(name=f"Bình nóng lạnh Ferro {n}", sku=f"FR-BN{i + 1:02d}", price=p, brand="Ferro",
                 specs=[f"Dung tích {l} lít", f"Kiểu {k}", "Chống giật ELCB"], warranty=24, extra_warranty="ruột bình bảo hành 7 năm",
                 notes=["Phí lắp đặt 250.000 đồng, chưa gồm vật tư."])
            for i, (n, l, k, p) in enumerate([("15 lít vuông", 15, "gián tiếp", 2_290_000), ("20 lít vuông", 20, "gián tiếp", 2_590_000),
                                              ("30 lít ngang", 30, "gián tiếp", 3_190_000), ("Trực tiếp 4500W", 0, "trực tiếp", 2_990_000),
                                              ("Trực tiếp có bơm", 0, "trực tiếp có bơm trợ lực", 3_790_000), ("20 lít Wi-Fi", 20, "gián tiếp", 3_490_000)])]


def may_in():
    return [dict(name=f"Máy in Printo {n}", sku=f"PT-MI{i + 1:02d}", price=p, brand="Printo",
                 specs=[f"Loại {t}", f"Tốc độ {s} trang/phút", "Kết nối Wi-Fi" if "Wi-Fi" in n else "Kết nối USB"], warranty=12,
                 notes=["Hộp mực đi kèm in được khoảng 700 trang; mực là vật tư tiêu hao."])
            for i, (n, t, s, p) in enumerate([("L110 laser trắng đen", "laser trắng đen", 20, 2_490_000), ("L210 Wi-Fi", "laser trắng đen", 30, 3_990_000),
                                              ("M300 đa năng Wi-Fi", "laser đa năng", 30, 6_490_000), ("C200 phun màu", "phun màu", 10, 3_290_000),
                                              ("C400 phun màu Wi-Fi", "phun màu", 15, 4_990_000), ("P50 in nhiệt", "in nhiệt", 40, 1_990_000)])]


def phu_kien():
    rows = [("Sạc nhanh Sora 25W", 390_000), ("Sạc nhanh Sora 45W", 690_000), ("Cáp USB-C Sora 1m", 190_000), ("Cáp USB-C Sora 2m", 290_000),
            ("Sạc dự phòng Nexa 10.000 mAh", 490_000), ("Sạc dự phòng Nexa 20.000 mAh", 790_000), ("Ốp lưng Sora Nova 8", 290_000),
            ("Chuột không dây Lumio M1", 290_000), ("Bàn phím không dây Lumio K1", 590_000), ("Bút cảm ứng Sora Pen", 1_490_000),
            ("Giá treo tivi Sora 32-55 inch", 250_000), ("Giá treo tivi Sora 55-85 inch", 450_000)]
    return [dict(name=f"Phụ kiện {n}", sku=f"PK-{i + 1:02d}", price=p, specs=["Hàng chính hãng"], warranty=6,
                 notes=["Phụ kiện bảo hành 6 tháng, 1 đổi 1 trong thời hạn bảo hành nếu lỗi nhà sản xuất."]) for i, (n, p) in enumerate(rows)]


CATEGORIES = [
    ("SP-TIVI", "Tivi Sora", "Sora", tivi), ("SP-DIENTHOAI", "Điện thoại Sora", "Sora", dien_thoai),
    ("SP-MAYTINHBANG", "Máy tính bảng Sora Tab", "Sora", may_tinh_bang), ("SP-LAPTOP", "Máy tính xách tay Lumio", "Lumio", laptop),
    ("SP-MANHINH", "Màn hình máy tính Lumio", "Lumio", man_hinh), ("SP-TAINGHE", "Tai nghe Sora và Nexa", "Sora/Nexa", tai_nghe),
    ("SP-LOA", "Loa và soundbar Nexa", "Nexa", loa), ("SP-DONGHO", "Đồng hồ thông minh Sora", "Sora", dong_ho),
    ("SP-TULANH", "Tủ lạnh Kiwa", "Kiwa", tu_lanh), ("SP-DIEUHOA", "Điều hòa Kiwa", "Kiwa", dieu_hoa),
    ("SP-MAYLOCNUOC", "Máy lọc nước Kiwa", "Kiwa", may_loc_nuoc), ("SP-MAYGIAT", "Máy giặt Vinto", "Vinto", may_giat),
    ("SP-MAYSAY", "Máy sấy quần áo Vinto", "Vinto", may_say), ("SP-MAYRUABAT", "Máy rửa bát Vinto", "Vinto", may_rua_bat),
    ("SP-NOICOM", "Nồi cơm điện Haru", "Haru", noi_com), ("SP-LOVISONG", "Lò vi sóng Haru", "Haru", lo_vi_song),
    ("SP-BEPTU", "Bếp từ Haru", "Haru", bep_tu), ("SP-MAYPHACAFE", "Máy pha cà phê Haru", "Haru", may_pha_cafe),
    ("SP-QUAT", "Quạt điện Haru", "Haru", quat), ("SP-MAYLOCKK", "Máy lọc không khí Polar", "Polar", may_loc_kk),
    ("SP-ROBOT", "Robot hút bụi Zenko", "Zenko", robot), ("SP-CAMERA", "Camera an ninh Zenko", "Zenko", camera),
    ("SP-BINHNONGLANH", "Bình nóng lạnh Ferro", "Ferro", binh_nong_lanh), ("SP-MAYIN", "Máy in Printo", "Printo", may_in),
    ("SP-PHUKIEN", "Phụ kiện", "nhiều hãng", phu_kien),
]



# --------------------------------------------------------------------------- #
# Extra description text per product group (like a real product page)
# --------------------------------------------------------------------------- #
PROFILE = {
    "SP-TIVI": dict(fit=["phòng khách gia đình", "phòng ngủ", "xem thể thao và chơi game console"],
                    box=["tivi", "chân đế", "điều khiển giọng nói", "2 pin AAA", "sách hướng dẫn"],
                    tips=["Khoảng cách xem lý tưởng bằng khoảng 1,5 lần đường chéo màn hình.", "Nên cập nhật phần mềm Sora OS khi có thông báo để sửa lỗi và bổ sung tính năng.",
                          "Không lau màn hình bằng cồn hoặc chất tẩy rửa; dùng khăn mềm khô.", "Bật chế độ Game để giảm độ trễ khi chơi trò chơi."]),
    "SP-DIENTHOAI": dict(fit=["học sinh, sinh viên", "người dùng cơ bản", "người chụp ảnh nhiều", "người chơi game"],
                         box=["điện thoại", "cáp USB-C", "que lấy sim", "sách hướng dẫn"],
                         tips=["Hộp không kèm củ sạc; nên dùng sạc Sora chính hãng để đạt tốc độ sạc nhanh.", "Sao lưu dữ liệu định kỳ lên tài khoản đám mây.",
                               "Không sạc qua đêm ở nơi nóng hoặc trên chăn gối.", "Khi mang đi bảo hành, khách nên tắt khóa tài khoản và sao lưu dữ liệu."]),
    "SP-MAYTINHBANG": dict(fit=["học trực tuyến", "giải trí", "ghi chú và vẽ"], box=["máy tính bảng", "cáp USB-C", "củ sạc 15 W", "que lấy sim"],
                           tips=["Bút cảm ứng bán riêng với các model không kèm bút.", "Bật chế độ trẻ em để giới hạn thời gian sử dụng."]),
    "SP-LAPTOP": dict(fit=["sinh viên", "nhân viên văn phòng", "lập trình viên", "thiết kế đồ họa"], box=["máy tính", "bộ sạc USB-C", "sách hướng dẫn"],
                      tips=["Lần đầu sử dụng nên cập nhật Windows và trình điều khiển từ Lumio Center.", "Không đặt máy trên chăn, đệm khi chạy tác vụ nặng.",
                            "Kích hoạt bản quyền Windows bằng tài khoản Microsoft để giữ bản quyền khi cài lại."]),
    "SP-MANHINH": dict(fit=["văn phòng", "chơi game", "đồ họa"], box=["màn hình", "chân đế", "cáp HDMI", "dây nguồn"],
                       tips=["Dùng cáp DisplayPort để đạt tần số quét tối đa.", "Hở sáng nhẹ ở viền là đặc tính của tấm nền."]),
    "SP-TAINGHE": dict(fit=["nghe nhạc", "gọi điện, họp trực tuyến", "tập thể thao"], box=["tai nghe", "hộp sạc hoặc túi đựng", "cáp sạc USB-C", "bộ nút tai các cỡ"],
                       tips=["Cập nhật firmware qua ứng dụng Sora Wear hoặc Nexa Sound.", "Vệ sinh nút tai định kỳ bằng khăn khô."]),
    "SP-LOA": dict(fit=["tiệc ngoài trời", "phòng khách", "karaoke gia đình"], box=["loa", "cáp sạc hoặc dây nguồn", "sách hướng dẫn"],
                   tips=["Không để loa chống nước ngâm trong nước quá thời gian công bố.", "Soundbar nên kết nối cổng HDMI eARC của tivi."]),
    "SP-DONGHO": dict(fit=["theo dõi sức khỏe", "luyện tập thể thao"], box=["đồng hồ", "dây đeo", "đế sạc"],
                      tips=["Chỉ số sức khỏe mang tính tham khảo, không thay thế thiết bị y tế.", "Không dùng đồng hồ khi tắm nước nóng hoặc xông hơi."]),
    "SP-TULANH": dict(fit=["gia đình 1-2 người", "gia đình 3-4 người", "gia đình đông người"], box=["tủ lạnh", "khay đá", "sách hướng dẫn"],
                      tips=["Sau khi vận chuyển, để tủ đứng yên 2 giờ trước khi cắm điện.", "Không để thực phẩm còn nóng vào tủ.",
                            "Đặt tủ cách tường tối thiểu 10 cm để tản nhiệt."]),
    "SP-DIEUHOA": dict(fit=["phòng ngủ", "phòng khách", "văn phòng nhỏ"], box=["dàn lạnh", "dàn nóng", "điều khiển", "giá treo dàn lạnh"],
                       tips=["Vệ sinh lưới lọc 2 tuần một lần.", "Cài đặt 26-27 độ C để tiết kiệm điện.", "Bảo dưỡng tổng thể 6 tháng một lần."]),
    "SP-MAYLOCNUOC": dict(fit=["gia đình dùng nước máy", "gia đình dùng nước giếng khoan đã qua xử lý sơ bộ"], box=["máy lọc", "bộ lõi lọc lắp sẵn", "vòi", "phụ kiện lắp đặt"],
                          tips=["Thay lõi số 1, 2, 3 sau 6 tháng; lõi RO sau 24 tháng.", "Xả bỏ 2 lít nước đầu tiên sau khi thay lõi."]),
    "SP-MAYGIAT": dict(fit=["gia đình 2-3 người", "gia đình 4-5 người", "gia đình đông người"], box=["máy giặt", "ống cấp nước", "ống xả", "nắp che bu lông"],
                       tips=["Tháo 4 bu lông vận chuyển trước khi sử dụng.", "Dùng nước giặt cho máy cửa trước để tránh tạo bọt.", "Vệ sinh lồng giặt mỗi tháng một lần."]),
    "SP-MAYSAY": dict(fit=["căn hộ chung cư", "mùa nồm ẩm"], box=["máy sấy", "ống thoát khí hoặc ống xả nước", "sách hướng dẫn"],
                      tips=["Vệ sinh lưới lọc xơ vải sau mỗi lần sấy.", "Không sấy đồ len, lụa ở nhiệt độ cao."]),
    "SP-MAYRUABAT": dict(fit=["gia đình 2-4 người", "gia đình đông người"], box=["máy rửa bát", "giỏ đựng", "ống cấp và xả", "muối làm mềm nước"],
                         tips=["Dùng muối làm mềm nước để tránh cặn canxi.", "Không rửa đồ gỗ, đồ gang trong máy."]),
    "SP-NOICOM": dict(fit=["gia đình nhỏ", "gia đình 4-6 người"], box=["nồi", "lòng nồi", "muỗng nhựa", "cốc đong"],
                      tips=["Dùng muỗng nhựa hoặc gỗ để bảo vệ lớp chống dính.", "Lau khô đáy lòng nồi trước khi nấu."]),
    "SP-LOVISONG": dict(fit=["hâm nóng nhanh", "nướng và rã đông"], box=["lò", "đĩa xoay thủy tinh", "vỉ nướng (dòng có nướng)"],
                        tips=["Không bật lò khi trong lò trống.", "Không dùng hộp kim loại ở chế độ vi sóng."]),
    "SP-BEPTU": dict(fit=["căn hộ", "nhà phố"], box=["bếp", "dây nguồn", "sách hướng dẫn"],
                     tips=["Chỉ dùng nồi đáy nhiễm từ.", "Lau mặt kính sau khi bếp nguội."]),
    "SP-MAYPHACAFE": dict(fit=["gia đình yêu cà phê", "văn phòng nhỏ"], box=["máy", "tay cầm pha", "muỗng nén"],
                          tips=["Vệ sinh cặn canxi mỗi 3 tháng.", "Dùng cà phê xay mịn cho espresso."]),
    "SP-QUAT": dict(fit=["phòng ngủ", "phòng khách"], box=["quạt", "điều khiển (nếu có)", "sách hướng dẫn"],
                    tips=["Vệ sinh cánh quạt định kỳ.", "Không để quạt hơi nước chạy khi bình chứa cạn."]),
    "SP-MAYLOCKK": dict(fit=["phòng ngủ", "nhà có trẻ nhỏ", "người bị dị ứng bụi"], box=["máy lọc", "màng lọc lắp sẵn", "sách hướng dẫn"],
                        tips=["Thay màng lọc HEPA sau 12 tháng hoặc khi đèn báo.", "Đặt máy cách tường ít nhất 20 cm."]),
    "SP-ROBOT": dict(fit=["căn hộ", "nhà có thú cưng"], box=["robot", "đế sạc", "chổi cạnh", "giẻ lau (dòng có lau)"],
                     tips=["Dọn dây điện và đồ vật nhỏ trên sàn trước khi chạy.", "Vệ sinh hộp bụi sau mỗi 2-3 lần chạy."]),
    "SP-CAMERA": dict(fit=["giám sát nhà ở", "cửa hàng nhỏ"], box=["camera", "bộ nguồn", "vít và tắc kê"],
                      tips=["Đổi mật khẩu mặc định ngay khi cài đặt.", "Dùng thẻ nhớ chuẩn Class 10 trở lên."]),
    "SP-BINHNONGLANH": dict(fit=["phòng tắm gia đình"], box=["bình", "van an toàn", "giá treo"],
                            tips=["Lắp aptomat chống giật riêng cho bình.", "Xả cặn bình 6 tháng một lần."]),
    "SP-MAYIN": dict(fit=["văn phòng nhỏ", "học tập tại nhà"], box=["máy in", "hộp mực khởi động", "dây nguồn", "cáp USB"],
                     tips=["Dùng mực chính hãng để không mất bảo hành.", "In thử trang kiểm tra khi máy lâu không dùng."]),
    "SP-PHUKIEN": dict(fit=["thay thế phụ kiện chính hãng"], box=["sản phẩm", "sách hướng dẫn"],
                       tips=["Kiểm tra tương thích model trước khi mua.", "Giữ hóa đơn để bảo hành 1 đổi 1."]),
}


def tier(price: int, doc_id: str, prices: list[int]) -> str:
    rank = sorted(prices).index(price) / max(1, len(prices) - 1)
    return "phổ thông" if rank < 0.34 else ("tầm trung" if rank < 0.67 else "cao cấp")


def enrich(p: dict, doc_id: str, prices: list[int]) -> list[str]:
    prof = PROFILE[doc_id]
    r = random.Random(p["sku"])
    fit = r.choice(prof["fit"])
    seg = tier(p["price"], doc_id, prices)
    paras = [f"Giới thiệu: {p['name']} thuộc phân khúc {seg} trong danh mục, phù hợp với {fit}. "
             f"Sản phẩm được phân phối chính hãng tại Lam Sơn, xuất hóa đơn điện tử và kích hoạt bảo hành theo số điện thoại mua hàng.",
             "Thông số chính:\n" + "\n".join(f"- {s}." for s in p["specs"]),
             "Trong hộp gồm: " + ", ".join(prof["box"]) + ".",
             "Lưu ý sử dụng:\n" + "\n".join(f"- {s}" for s in r.sample(prof["tips"], k=min(2, len(prof["tips"]))))]
    return paras


def slug(sku: str) -> str:
    return sku.lower().replace(".", "-")


def render(doc_id: str, title: str, brand: str, products: list[dict]) -> str:
    lo, hi = min(p["price"] for p in products), max(p["price"] for p in products)
    out = [f"---\ndoc_id: {doc_id}\ntitle: Catalog {title}\ncategory: san-pham\neffective_date: 2025-10-01\nstatus: active\n---\n",
           f"# Catalog {title}\n",
           "## Tổng quan danh mục {#tong-quan}\n",
           f"Danh mục gồm {len(products)} sản phẩm của hãng {brand}, giá niêm yết từ {vnd(lo)} đến {vnd(hi)}. "
           "Giá niêm yết đã gồm thuế giá trị gia tăng, chưa gồm ưu đãi khuyến mãi. Các model đang kinh doanh: "
           + "; ".join(f"{p['name']} ({p['sku']})" for p in products) + ".\n"]
    for p in products:
        w = f"Bảo hành chính hãng {p['warranty']} tháng"
        if p.get("extra_warranty"):
            w += f", {p['extra_warranty']}"
        prices = [q["price"] for q in products]
        extra = enrich(p, doc_id, prices)
        out += [f"## {p['name']} {{#{slug(p['sku'])}}}\n",
                f"Mã sản phẩm {p['sku']}. Giá niêm yết {vnd(p['price'])}. {w}.\n",
                extra[0] + "\n", extra[1] + "\n",
                extra[2] + "\n", extra[3] + "\n",
                "Ghi chú bán hàng: " + " ".join(p["notes"]) + "\n"]
    return "\n".join(out)


def main():
    catalog = []
    for doc_id, title, brand, fn in CATEGORIES:
        products = fn()
        for p in products:
            p.setdefault("brand", brand)
            p.update(doc_id=doc_id, section=slug(p["sku"]), category=title)
        (KB / f"{doc_id}.md").write_text(render(doc_id, title, brand, products), encoding="utf-8")
        catalog += products
    skus = [p["sku"] for p in catalog]
    assert len(skus) == len(set(skus)), "duplicate SKU"
    (ROOT / "data" / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(CATEGORIES)} catalog documents, {len(catalog)} products")


if __name__ == "__main__":
    main()
