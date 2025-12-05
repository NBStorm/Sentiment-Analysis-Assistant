# Vietnamese Sentiment Analysis Assistant

**(Trợ lý Phân loại Cảm xúc Tiếng Việt sử dụng Transformer)**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Transformers](https://img.shields.io/badge/NLP-HuggingFace-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Giới thiệu (Introduction)

Đây là đồ án môn học **Seminar Chuyên đề**, tập trung xây dựng một ứng dụng Desktop/Web đơn giản để phân tích cảm xúc của văn bản tiếng Việt.

Dự án sử dụng kỹ thuật **Transfer Learning** với các mô hình **Transformer Pre-trained** (như PhoBERT/DistilBERT) để phân loại câu văn thành 3 nhãn cảm xúc:

- 😄 **Tích cực (Positive)**
- 😐 **Trung tính (Neutral)**
- 😡 **Tiêu cực (Negative)**

## 🚀 Tính năng chính (Key Features)

- **Phân loại cảm xúc:** Nhận diện cảm xúc câu tiếng Việt với độ chính xác cao.
- **Xử lý ngôn ngữ tự nhiên:**
  - Hỗ trợ tiếng Việt có dấu và không dấu.
  - Tự động chuẩn hóa và xử lý các từ viết tắt (Teencode) thông dụng (vd: ko, dc, bt...).
- **Lưu trữ lịch sử:** Tự động lưu lại các câu đã nhập và kết quả phân tích vào cơ sở dữ liệu SQLite cục bộ.
- **Giao diện trực quan:** Giao diện Web App thân thiện, dễ sử dụng được xây dựng bằng Streamlit.

## 🛠️ Công nghệ sử dụng (Tech Stack)

- **Ngôn ngữ:** Python
- **Giao diện (Frontend):** Streamlit
- **AI/NLP Core:** Hugging Face Transformers, PyTorch
- **Database:** SQLite3
- **Data Processing:** Pandas

## 📂 Cấu trúc thư mục (Project Structure)

```text
Sentiment-Analysis-Assistant/
├── venv/                   # Môi trường ảo (Virtual Environment)
├── app.py                  # Mã nguồn chính của ứng dụng
├── requirements.txt        # Danh sách thư viện cần cài đặt
├── sentiment_history.db    # Database (Tự tạo khi chạy app)
└── README.md               # Tài liệu hướng dẫn
```

## ⚙️ Hướng dẫn Cài đặt (Installation)

Vui lòng thực hiện lần lượt theo các bước sau:

### Bước 1: Tải mã nguồn

Tải thư mục dự án về máy tính và giải nén (nếu có). Mở **Terminal** (hoặc CMD/PowerShell) tại thư mục dự án.

### Bước 2: Tạo môi trường ảo (Khuyên dùng)

Việc này giúp tránh xung đột thư viện với hệ thống.

- ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

### Bước 3: Cài đặt thư viện

Chạy lệnh sau để cài đặt toàn bộ các gói cần thiết:

- ```bash
  pip install streamlit transformers torch pandas
  ```

## 🚀 Hướng dẫn Sử dụng (Usage)

Khởi chạy ứng dụng
Tại terminal (đang kích hoạt môi trường ảo), gõ lệnh:

- ```bash
  streamlit run app.py
  ```

```

```
