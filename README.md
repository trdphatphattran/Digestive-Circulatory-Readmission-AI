# DỰ ĐOÁN KHẢ NĂNG TÁI NHẬP VIỆN BỆNH TIÊU HÓA VÀ TUẦN HOÀN  

## 📌 Tổng quan dự án  
Hệ thống dự báo tái nhập viện (Tiêu hóa & Tuần hoàn) sử dụng thuật toán Random Forest. Dự án kết hợp Học máy và Web Chatbot để phân tích hồ sơ bệnh lý, dữ liệu lâm sàng và thói quen sinh hoạt. Bằng cách tìm kiếm sự tương đồng giữa các ca bệnh thông qua Random Forest, hệ thống đưa ra cảnh báo rủi ro sớm, giúp tối ưu hóa quản lý bệnh trạng và hỗ trợ ra quyết định y tế chủ động.  

## 🚀 Điểm Nổi Bật  
* **Quy mô dữ liệu:** Khoảng 700 dữ liệu bệnh với hơn 10 loại bệnh tiêu hóa và 10 loại bệnh tuần hoàn khác nhau, bên cạnh đó là hơn 10 loại bệnh nền.    
* **Dự đoán:** Phân loại nhị phân khả năng tái nhập viện với mô hình Random Forest (100 Decision Trees), đảm bảo độ chính xác và tính ổn định cao trên dữ liệu bệnh lý.  
* **Trợ lý Y tế AI:** Tích hợp LLM để giải thích kết quả dự đoán và tư vấn chuyên sâu về chế độ dinh dưỡng, sinh hoạt thông qua giao diện Chatbot tự nhiên.  
* **Giao diện:** Hỗ trợ nhập liệu linh hoạt qua ngôn ngữ tự nhiên hoặc tra cứu nhanh hệ thống hồ sơ bệnh án qua ID bệnh nhân.

## 🛠 Công Nghệ Sử Dụng  
* **Backend & Logic:** Python.  
* **Web Framework:** Flask.  
* **Machine Learning:** Scikit-learn (Random Forest Classifier), Pandas, Numpy.  
* **Generative AI:** Google Gemini API (Model Gemini 2.5 Flash Lite).  
* **Frontend:** HTML5, CSS, JavaScript.

## Demo  
### 1. Nhập vào câu hỏi:  
<img width="947" height="157" alt="image" src="https://github.com/user-attachments/assets/0187ae9e-b602-43e4-99eb-65021eaa874a" />  

### 2. Câu trả lời của hệ thống:  
<img width="572" height="526" alt="image" src="https://github.com/user-attachments/assets/4c8f70d4-0a25-4c91-9bf9-59e4652cb462" />  

## 📂 Cấu trúc thư mục  
```text
├── image                       
├── static
|   └── script.js
|   └── style.css
├── templates
|   ├── index.html
├── app1.py
├── nhadat.csv
├── requirements.txt  
└── README.md
```

## 💻 Hướng dẫn sử dụng  
### 1. Clone Repository  
```python
git clone https://github.com/trdphatphattran/Digestive-Circulatory-Readmission-AI.git
cd Digestive-Circulatory-Readmission-AI
```
### 2. Cài thư viện  
```python
pip install -r requirements.txt
```
### 3. Chạy Streamlit  
```python
python3 app1.py
```
## 📬 Thông tin liên hệ

Nếu bạn có bất kỳ câu hỏi nào về dự án hoặc muốn hợp tác, vui lòng liên hệ với mình qua:

* **Họ và tên:** Trần Đại Phát
* **LinkedIn:** [Phat Tran](https://www.linkedin.com/in/phat-tran-9ba42a341/)
* **GitHub:** [trdphatphattran](https://github.com/trdphatphattran)
* **Email:** phattrandai15062005@gmail.com
* **Phone:** 0908647977 




