from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
import os, re

# ===== GOOGLE GEMINI CONFIG =====
# LƯU Ý: VÌ BẠN ĐANG CHẠY CỤC BỘ, HÃY ĐẢM BẢO BIẾN MÔI TRƯỜNG GEMINI_API_KEY ĐÃ ĐƯỢC THIẾT LẬP
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyD-_LadDuBLs9eSmY2PGsuEVTrUS-SeTyQ")
genai.configure(api_key=GEMINI_API_KEY)

# ===== BẮT ĐẦU SỬA ĐỔI (YÊU CẦU ĐỊNH DẠNG MỚI) =====
# SỬA LỖI: Định nghĩa system_instruction ở đây
system_instruction = "Bạn là trợ lý y tế thân thiện và chuyên nghiệp. Hãy trả lời một cách tự nhiên và chi tiết. Hãy phân tích sâu về tình trạng bệnh nhân. Khi trả lời các mục chính, hãy dùng định dạng số (ví dụ: 1., 2.). Khi liệt kê các ý nhỏ bên trong, hãy dùng gạch đầu dòng (-). Tuyệt đối KHÔNG sử dụng các ký hiệu Markdown như ** (in đậm) hoặc * (dấu hoa thị)."
# ===== KẾT THÚC SỬA ĐỔI =====

# SỬA LỖI: Truyền system_instruction khi khởi tạo model
gemini_model = genai.GenerativeModel(
    "models/gemini-2.5-flash",
    system_instruction=system_instruction
) 

def call_gemini(prompt: str):
    """
    Gọi API Gemini và tự động dọn dẹp Markdown.
    system_instruction đã được nạp vào model lúc khởi tạo.
    """
    try:
        response = gemini_model.generate_content(
            contents=prompt
        )
        
        # ===== BẮT ĐẦU CHỖ SỬA MARKDOWN =====
        text = response.text
        # Xóa **bold** (thay thế **nội dung** bằng nội dung)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) 
        # Xóa *italic* (thay thế *nội dung* bằng nội dung)
        text = re.sub(r'\*(.*?)\*', r'\1', text)    
        
        return text.strip() # Trả về text đã dọn dẹp
        # ===== KẾT THÚC CHỖ SỬA MARKDOWN =====
        
    except Exception as e:
        return f"❌ Gemini API error: {type(e).__name__}: {str(e)}"


# ===== FLASK APP =====
app = Flask(__name__)

# ===== LOAD DATA =====
# Giả định file 6.csv tồn tại và chứa dữ liệu phù hợp
try:
    data = pd.read_csv("6.csv")
    data["SoLanNhapVienGoc"] = data["Số lần nhập viện"]
    data["Bệnh"] = data["Bệnh"].astype(str)
    data["tai_nhap_vien"] = ((data["Số lần nhập viện"] > 3) | (data["Nghiêm trọng"] == 1)).astype(int) 

    X = data[["Bệnh", "Bệnh nền", "Số lần nhập viện", "Tuổi"]].copy()
    y = data["tai_nhap_vien"]

    encoder_benh = LabelEncoder()
    X["Bệnh"] = encoder_benh.fit_transform(X["Bệnh"])
    encoder_benhnen = LabelEncoder()
    X["Bệnh nền"] = encoder_benhnen.fit_transform(X["Bệnh nền"])

    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X, y)

    importances = model_rf.feature_importances_
    factors = X.columns
    importance_ratio = importances / importances.sum()
except Exception as e:
    # Nếu không tìm thấy file 6.csv hoặc lỗi dữ liệu, ứng dụng sẽ không chạy
    print(f"Lỗi khi tải dữ liệu hoặc huấn luyện mô hình: {e}")
    # Tiếp tục chạy Flask nhưng các hàm dự đoán sẽ gặp lỗi nếu data không được load
    pass 


# ===== SANITIZE =====
def sanitize(obj):
    """Chuyển numpy.int64, numpy.float64 -> int/float để JSON serialize"""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    else:
        return obj


# ===== HÀM TÍNH XÁC SUẤT VÀ PHÂN TÍCH YẾU TỐ =====
def tinh_xac_suat(proba_model, tuoi, so_lan, nghiem_trong, benh_nen):
    """
    Sửa đổi (Logic thực tế): Hàm này giờ đóng vai trò "điều chỉnh"
    xác suất gốc từ mô hình.
    
    - Cập nhật: Thêm logic kiểm tra 'benh_nen'
    """
    
    # Bắt đầu với xác suất gốc từ mô hình
    proba = proba_model

    # === CÁC YẾU TỐ AN TOÀN (GIẢM RỦI RO) ===
    if nghiem_trong == 0:
        proba *= 0.7 
    if so_lan == 1:
        proba *= 0.8
    if tuoi < 40:
        proba *= 0.9

    # === CÁC YẾU TỐ RỦI RO (TĂNG RỦI RO) ===
    if nghiem_trong == 1:
        proba = proba + (1.0 - proba) * 0.33

    if tuoi >= 65:
        proba = proba + (1.0 - proba) * 0.20
    elif tuoi >= 50:
        proba = proba + (1.0 - proba) * 0.10

    if so_lan >= 3:
        proba = proba + (1.0 - proba) * 0.20

    # === CẬP NHẬT: LOGIC BỆNH NỀN (Nặng/Nhẹ) ===
    benh_nen_lower = str(benh_nen).lower()
    if "không" not in benh_nen_lower:
        # Nếu CÓ bệnh nền
        if "nặng" in benh_nen_lower or "tiểu đường" in benh_nen_lower or "cao huyết áp" in benh_nen_lower or "suy thận" in benh_nen_lower or "ung thư" in benh_nen_lower:
            # Bệnh nền nặng/nguy cơ cao, tăng 15%
            proba = proba + (1.0 - proba) * 0.15
        elif "nhẹ" not in benh_nen_lower:
            # Bệnh nền không rõ, tăng 5%
            proba = proba + (1.0 - proba) * 0.05
        # Nếu "nhẹ" thì không tăng
        
    # Đảm bảo xác suất luôn nằm trong khoảng hợp lý
    return min(max(proba, 0.01), 0.98)

def tinh_phan_tram(nghiem_trong, benh_nen, so_lan, tuoi):
    """
    Sửa đổi (Tông hợp lý): Giảm bớt các hệ số nhân
    và thêm logic phân biệt bệnh nền nặng/nhẹ.
    """
    global factors, importance_ratio 
    
    adjusted_importances = dict(zip(factors, importance_ratio))

    # === Yếu tố Bệnh (liên kết với Nghiêm trọng) ===
    if nghiem_trong == 1:
        adjusted_importances["Bệnh"] *= 1.5
    else:
        adjusted_importances["Bệnh"] *= 0.8

    # === CẬP NHẬT: LOGIC BỆNH NỀN NẶNG/NHẸ ===
    benh_nen_lower = str(benh_nen).lower()
    if "không" in benh_nen_lower:
        adjusted_importances["Bệnh nền"] *= 0.5  # Ít quan trọng
    elif "nặng" in benh_nen_lower or "tiểu đường" in benh_nen_lower or "cao huyết áp" in benh_nen_lower or "suy thận" in benh_nen_lower or "ung thư" in benh_nen_lower:
        adjusted_importances["Bệnh nền"] *= 1.5  # Quan trọng cao
    elif "nhẹ" in benh_nen_lower:
        adjusted_importances["Bệnh nền"] *= 1.0  # Quan trọng trung bình
    else:
        # Có bệnh nền nhưng không rõ mức độ (ví dụ: "hen suyễn")
        adjusted_importances["Bệnh nền"] *= 1.2  # Mặc định

    # === Yếu tố Số lần nhập viện ===
    scale_so_lan = 1.0 + max(0, min(so_lan - 1, 5)) * 0.15
    adjusted_importances["Số lần nhập viện"] *= scale_so_lan

    # === Yếu tố Tuổi ===
    distance_from_safe_age = abs(tuoi - 35)
    scale_tuoi = 1.0 + (distance_from_safe_age / 10.0) * 0.05
    adjusted_importances["Tuổi"] *= scale_tuoi

    # === Chuẩn hóa lại 100% ===
    total_importance = sum(adjusted_importances.values())
    
    if total_importance > 0:
        phan_tram_list = [(v / total_importance) * 100 for v in adjusted_importances.values()]
    else:
        phan_tram_list = [100.0 / len(factors)] * len(factors)
        
    return phan_tram_list


# ===== DỰ ĐOÁN BỆNH NHÂN CŨ =====
def du_doan_benh_nhan_cu(patient_id):
    row = data[data["ID"] == patient_id]
    if row.empty:
        return None, f"Không tìm thấy bệnh nhân ID={patient_id}"
    x_input = X.loc[row.index]
    proba_model = model_rf.predict_proba(x_input)[:, 1][0]

    ten = str(row["Tên bệnh nhân"].values[0])
    tuoi = int(row["Tuổi"].values[0])
    benh = str(row["Bệnh"].values[0])
    benh_nen = str(row["Bệnh nền"].values[0])
    so_lan = int(row["SoLanNhapVienGoc"].values[0])
    nghiem_trong = int(row["Nghiêm trọng"].values[0])

    # SỬA LỖI: Cập nhật lời gọi hàm
    proba_total = tinh_xac_suat(proba_model, tuoi, so_lan, nghiem_trong, benh_nen)
    phan_tram_list = tinh_phan_tram(nghiem_trong, benh_nen, so_lan, tuoi)

    info = {
        "ID": int(patient_id),
        "Tên": ten,
        "Tuổi": tuoi,
        "Bệnh": benh,
        "Bệnh nền": benh_nen,
        "Số lần nhập viện": so_lan,
        "Nghiêm trọng": nghiem_trong,
        "Nguy cơ": float(round(proba_total * 100, 2)),
        "factors": {str(k): float(round(v, 2)) for k, v in zip(factors, phan_tram_list)},
        "summary": "⚠ Có nguy cơ tái nhập viện." if proba_total >= 0.5 else "Nguy cơ thấp."
    }
    return info, None


# ===== DỰ ĐOÁN BỆNH NHÂN MỚI =====
def du_doan_benh_nhan_moi(ten, tuoi, benh, benh_nen, so_lan, nghiem_trong):
    global data
    so_lan_dk = 1 if so_lan > 3 else 0
    new_id = int(data["ID"].max()) + 1 if "ID" in data.columns and not data["ID"].empty else 1

    new_row = {
        "ID": new_id,
        "Tên bệnh nhân": ten,
        "Tuổi": int(tuoi),
        "Bệnh": benh,
        "Bệnh nền": benh_nen,
        "Số lần nhập viện": int(so_lan_dk),
        "SoLanNhapVienGoc": int(so_lan),
        "Nghiêm trọng": int(nghiem_trong)
    }
    # Thêm hàng mới vào DataFrame (không lưu ra file để tránh ghi đè)
    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True) 

    # Xử lý Label Encoding cho bệnh nhân mới
    encoded_benh = 0
    if benh in encoder_benh.classes_:
        encoded_benh = encoder_benh.transform([benh])[0]
    # else: bệnh mới, dùng label 0

    encoded_benh_nen = 0
    if benh_nen in encoder_benhnen.classes_:
        encoded_benh_nen = encoder_benhnen.transform([benh_nen])[0]
    # else: bệnh nền mới, dùng label 0


    x_input = pd.DataFrame([{
        "Bệnh": encoded_benh,
        "Bệnh nền": encoded_benh_nen,
        "Số lần nhập viện": so_lan_dk,
        "Tuổi": int(tuoi)
    }])

    proba_model = model_rf.predict_proba(x_input)[:, 1][0]
    
    # ===== BẮT ĐẦU ĐOẠN SỬA LỖI =====

    # Chuyển đổi kiểu dữ liệu sang SỐ một lần để đảm bảo
    i_tuoi = int(tuoi)
    i_so_lan = int(so_lan)
    i_nghiem_trong = int(nghiem_trong)

    # SỬA LỖI: Gọi hàm tính xác suất với 5 tham số
    proba_total = tinh_xac_suat(proba_model, i_tuoi, i_so_lan, i_nghiem_trong, benh_nen)
    
    # SỬA LỖI: Gọi hàm tính phần trăm với đúng 4 tham số mới
    # (nghiem_trong, benh_nen, so_lan, tuoi)
    phan_tram_list = tinh_phan_tram(i_nghiem_trong, benh_nen, i_so_lan, i_tuoi)
    
    # ===== KẾT THÚC ĐOẠN SỬA LỖI =====

    return {
        "ID": int(new_id),
        "Tên": str(ten),
        "Tuổi": i_tuoi,                # Trả về số đã chuyển đổi
        "Bệnh": str(benh),
        "Bệnh nền": str(benh_nen),
        "Số lần nhập viện": i_so_lan, # Trả về số đã chuyển đổi
        "Nghiêm trọng": i_nghiem_trong, # Trả về số đã chuyển đổi
        "Nguy cơ": float(round(proba_total * 100, 2)),
        "factors": {str(k): float(round(v, 2)) for k, v in zip(factors, phan_tram_list)},
        "summary": "⚠ Có nguy cơ tái nhập viện." if proba_total >= 0.5 else "Nguy cơ thấp."
    }


# ===== GỢI Ý CHĂM SÓC TỰ NHIÊN (ĐÃ CHUYỂN RA KHỎI ROUTE) =====
def tao_goi_y_tu_nhien(info):
    """Sinh gợi ý chăm sóc tự nhiên dựa trên mức nguy cơ."""
    nguy_co = info["Nguy cơ"]
    ten = info["Tên"]
    tuoi = info["Tuổi"]
    benh = info["Bệnh"]
    
    # SỬA LỖI GÕ PHÍM: (BBệnh -> Bệnh)
    benh_nen = info["Bệnh nền"]

    if nguy_co >= 75:
        muc = "rất cao"
        goi_y = (
            f"Bệnh nhân {ten}, {tuoi} tuổi, hiện đang có nguy cơ tái nhập viện rất cao "
            f"do mắc {benh.lower()} và có bệnh nền {benh_nen.lower()}. "
            "Tình trạng sức khỏe cần được theo dõi sát sao hàng ngày, nên có người thân hỗ trợ dùng thuốc đúng liều. "
            "Bệnh nhân nên tái khám mỗi 2–4 tuần, kiểm tra các chỉ số quan trọng như huyết áp, đường huyết, nhịp tim. "
            "Nếu thấy dấu hiệu bất thường (sốt cao, mệt mỏi, khó thở), hãy liên hệ ngay với bác sĩ điều trị hoặc bệnh viện gần nhất. "
            "Gia đình nên đồng hành và giữ liên lạc thường xuyên với cơ sở y tế để đảm bảo an toàn cho bệnh nhân."
        )
    elif nguy_co >= 50:
        muc = "trung bình"
        goi_y = (
            f"Bệnh nhân {ten}, {tuoi} tuổi, có nguy cơ tái nhập viện trung bình. "
            f"Bệnh {benh.lower()} hiện có thể kiểm soát nếu tuân thủ điều trị và duy trì sinh hoạt lành mạnh. "
            "Khuyến khích bệnh nhân theo dõi sức khỏe hằng ngày, giữ chế độ ăn phù hợp, hạn chế căng thẳng và ngủ đủ giấc. "
            "Nên tái khám định kỳ 1–2 tháng/lần để bác sĩ theo dõi diễn tiến bệnh. "
            f"Nếu có bệnh nền {benh_nen.lower()}, cần theo dõi kỹ vì có thể làm tăng nguy cơ biến chứng."
        )
    else:
        muc = "thấp"
        goi_y = (
            f"Bệnh nhân {ten}, {tuoi} tuổi, có nguy cơ tái nhập viện thấp. "
            f"Sức khỏe hiện ổn định, bệnh {benh.lower()} đang được kiểm soát tốt. "
            "Tuy nhiên, vẫn nên tái khám định kỳ 3–6 tháng/lần để phòng ngừa tái phát. "
            "Duy trì thói quen sinh hoạt điều độ, tập thể dục nhẹ nhàng, và tuân thủ hướng dẫn điều trị của bác sĩ. "
            "Gia đình có thể yên tâm nhưng vẫn nên khuyến khích bệnh nhân theo dõi sức khỏe thường xuyên."
        )

    return muc, goi_y


# ===== PROMPT FORMATTER =====
def format_patient_info(info):
    """
    Định dạng thông tin bệnh nhân cho Gemini API.
    SỬA LỖI: Chuyển đổi 1/0 thành Có/Không để Gemini hiểu đúng.
    """
    
    # Chuyển đổi 1/0 thành "Có" / "Không"
    nghiem_trong_text = "Có" if info['Nghiêm trọng'] == 1 else "Không"
    
    return f"""
    Bệnh nhân: {info['Tên']}
    Tuổi: {info['Tuổi']}
    Bệnh: {info['Bệnh']}
    Bệnh nền: {info['Bệnh nền']}
    Số lần nhập viện: {info['Số lần nhập viện']}
    Tình trạng nghiêm trọng: {nghiem_trong_text}
    Nguy cơ tái nhập viện: {info['Nguy cơ']}%
    {info['summary']}
    """


# ===== ROUTES =====
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").strip().lower()

    # ================== 1. PHÁT HIỆN THÔNG TIN BỆNH NHÂN MỚI ==================
    if any(kw in user_msg for kw in ["bệnh nhân", "tuổi", "bệnh", "nhập viện", "nghiêm trọng", "bệnh nền"]):
        try:
            # --- Trích xuất tên ---
            ten_match = re.search(r"(?:bệnh nhân|tên)\s+([a-zA-ZÀ-ỹ\s]+?)(?:,|\d|bị|tuổi|$)", user_msg)
            ten = ten_match.group(1).strip().title() if ten_match else "Bệnh nhân mới"

            # --- Tuổi ---
            tuoi_match = re.search(r"(\d{1,2})\s*(?:tuổi|age)?", user_msg)
            tuoi = int(tuoi_match.group(1)) if tuoi_match else 40

            # --- Bệnh ---
            benh_match = re.search(
                r"(?:bị bệnh|mắc bệnh|mắc phải|bị mắc|bị|mắc)\s+([a-zA-ZÀ-ỹ\s]+?)(?:,|\.|và|có|về|bệnh nền|nhập viện|tình trạng|nghiêm trọng|mắc bệnh|bị bệnh|mắc phải|bị mắc|$)", 
                user_msg
            )
            benh = benh_match.group(1).strip() if benh_match else "Không rõ"

            # --- Bệnh nền ---
            benh_nen_match = re.search(r"bệnh nền\s*(?:là|:)?\s*([a-zA-ZÀ-ỹ\s]+?)(?:,|\.|và|nhập viện|tình trạng|$)", user_msg)
            benh_nen = benh_nen_match.group(1).strip() if benh_nen_match else "Không"
            if "không" in benh_nen or "chưa có" in benh_nen:
                benh_nen = "Không"

            # ===== SỬA LỖI REGEX (ĐỂ BẮT "3 LẦN" VÀ "NGHIÊM TRỌNG") =====
            
            # --- Số lần nhập viện (Logic mới, tìm cả "3 lần" và "nhập viện 3") ---
            so_lan = 1 # Mặc định
            so_lan_match_1 = re.search(r"(\d+)\s*lần", user_msg) # Tìm "3 lần"
            so_lan_match_2 = re.search(r"nhập viện\s*(\d+)", user_msg) # Tìm "nhập viện 3"
            
            if so_lan_match_1:
                so_lan = int(so_lan_match_1.group(1))
            elif so_lan_match_2:
                so_lan = int(so_lan_match_2.group(1))

            # --- Mức độ nghiêm trọng (Đã xóa code lặp) ---
            nghiem_trong = 0
            # Tìm từ khóa khẳng định "nghiêm trọng"
            # Thêm "được đánh giá nghiêm trọng"
            positive_check = re.search(r'\b(nặng|trầm trọng|nguy kịch|được đánh giá nghiêm trọng)\b|(?<!không\s)nghiêm trọng', user_msg)
            if positive_check:
                nghiem_trong = 1
            
            # Tìm từ khóa phủ định "không nghiêm trọng"
            negative_check = re.search(r'\b(không nặng|không trầm trọng|không nghiêm trọng|chưa nghiêm trọng|nhẹ|bình thường|không đáng kể)\b', user_msg)
            if negative_check:
                nghiem_trong = 0
                
            # ===== KẾT THÚC SỬA LỖI REGEX =====
            
            # --- Gọi hàm dự đoán ---
            info = du_doan_benh_nhan_moi(ten, tuoi, benh, benh_nen, so_lan, nghiem_trong)

            # ===== CẬP NHẬT: PROMPT MỚI YÊU CẦU PHÂN TÍCH SÂU + ĐỊNH DẠNG 1. 2. =====
            prompt = (
                "Thông tin bệnh nhân:\n\n"
                + format_patient_info(info)
                + "\n\n"
                + "Hãy thực hiện các yêu cầu sau:\n"
                + "Đầu tiên, hãy viết một đoạn phân tích sâu và chi tiết về tình trạng của bệnh nhân, giải thích tại sao các yếu tố (như tuổi, bệnh nền, số lần nhập viện) lại quan trọng và ảnh hưởng lẫn nhau.\n"
                + "Sau đó, trả lời 2 mục sau theo định dạng số:\n"
                + "1. Tư vấn sức khỏe: Cung cấp các lời khuyên chi tiết về chế độ ăn uống, sinh hoạt, và những dấu hiệu cụ thể cần theo dõi ngay tại nhà. (Sử dụng gạch đầu dòng - cho các ý nhỏ).\n"
                + f"2. Gợi ý bệnh viện: Gợi ý 2-3 bệnh viện chuyên khoa hàng đầu tại Việt Nam (ví dụ: Bệnh viện Chợ Rẫy, Bệnh viện Bạch Mai) có thể điều trị tốt bệnh chính là {info['Bệnh']}. (Sử dụng gạch đầu dòng - cho các bệnh viện).\n"
                + "Hãy trả lời một cách tự nhiên và thân thiện."
            )
            # ===== KẾT THÚC CẬP NHẬT PROMPT =====
            
            return jsonify({"reply": call_gemini(prompt), "data": sanitize(info)})

        except Exception as e:
            return jsonify({"reply": f"⚠️ Không thể nhận diện thông tin bệnh nhân mới. Lỗi: {e}. Vui lòng cung cấp lại thông tin theo cấu trúc ví dụ: 'Bệnh nhân [Tên], [Tuổi] tuổi, mắc bệnh [Tên bệnh], bệnh nền [Có/Không], nhập viện [Số lần] lần, [nghiêm trọng/không nghiêm trọng]'."})

    # ================== 2. TRA CỨU THEO ID (TỰ NHIÊN VÀ GỢI Ý RIÊNG) ==================
    # Khối này đang bị comment (''') trong code bạn gửi,
    # Nếu bạn mở nó ra, nó cũng đã được cập nhật prompt mới
    
    if re.search(r"\b(id|bệnh nhân|hồ sơ|tra|xem|coi|check|nguy cơ)\b", user_msg):
        numbers = re.findall(r"\d+", user_msg)

        # ===== TRA MỘT ID =====
        if len(numbers) == 1:
            pid = int(numbers[0])
            info, err = du_doan_benh_nhan_cu(pid)
            if err:
                return jsonify({"reply": err})

            # ===== CẬP NHẬT: PROMPT MỚI YÊU CẦU PHÂN TÍCH SÂU + ĐỊNH DẠNG 1. 2. =====
            prompt = (
                "Thông tin bệnh nhân:\n\n"
                + format_patient_info(info)
                + "\n\n"
                + "Hãy thực hiện các yêu cầu sau:\n"
                + "Đầu tiên, hãy viết một đoạn phân tích sâu và chi tiết về tình trạng của bệnh nhân, giải thích tại sao các yếu tố (như tuổi, bệnh nền, số lần nhập viện) lại quan trọng và ảnh hưởng lẫn nhau.\n"
                + "Sau đó, trả lời 2 mục sau theo định dạng số:\n"
                + "1. Tư vấn sức khỏe: Cung cấp các lời khuyên chi tiết về chế độ ăn uống, sinh hoạt, và những dấu hiệu cụ thể cần theo dõi ngay tại nhà. (Sử dụng gạch đầu dòng - cho các ý nhỏ).\n"
                + f"2. Gợi ý bệnh viện: Gợi ý 2-3 bệnh viện chuyên khoa hàng đầu tại Việt Nam (ví dụ: Bệnh viện Chợ Rẫy, Bệnh viện Bạch Mai) có thể điều trị tốt bệnh chính là {info['Bệnh']}. (Sử dụng gạch đầu dòng - cho các bệnh viện).\n"
                + "Hãy trả lời một cách tự nhiên và thân thiện."
            )
            # ===== KẾT THÚC CẬP NHẬT PROMPT =====
            
            return jsonify({"reply": call_gemini(prompt), "data": sanitize(info)})

        # ===== TRA NHIỀU ID (DÙNG LOGIC CŨ, KHÔNG GỌI GEMINI ĐỂ TRÁNH SPAM) =====
        elif len(numbers) > 1:
            results = []
            all_text = "📋 Kết quả tra cứu nhiều bệnh nhân:\n\n"
            for n in numbers:
                pid = int(n)
                info, err = du_doan_benh_nhan_cu(pid)
                if err:
                    all_text += f"- ID {pid}: ❌ Không tìm thấy hồ sơ.\n\n"
                else:
                    results.append(info)
                    muc, goi_y = tao_goi_y_tu_nhien(info) 
                    all_text += (
                        f"🧍‍♂️ {info['Tên']} ({info['Tuổi']} tuổi) ID {pid}\n"
                        f"→ Nguy cơ tái nhập viện: {info['Nguy cơ']}% – Mức {muc.upper()} ({info['summary']})\n\n"
                        f"{goi_y}\n\n"
                    )

            return jsonify({"reply": all_text, "data": sanitize(results)})

    # ================== 4. CHAT THƯỜNG ==================
    return jsonify({"reply": call_gemini(user_msg)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)