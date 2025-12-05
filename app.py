import streamlit as st
from transformers import pipeline
import sqlite3
from datetime import datetime
import pandas as pd

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Trợ lý Phân loại Cảm xúc", page_icon="🤖")

# --- PHẦN 1: DATABASE (SQLite) ---
# Tạo hoặc kết nối đến database
def init_db():
    conn = sqlite3.connect('sentiment_history.db')
    c = conn.cursor()
    # Tạo bảng nếu chưa tồn tại
    c.execute('''
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment TEXT,
            score REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Hàm lưu kết quả
def save_to_db(text, sentiment, score):
    conn = sqlite3.connect('sentiment_history.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO sentiments (text, sentiment, score, timestamp) VALUES (?, ?, ?, ?)',
              (text, sentiment, score, timestamp))
    conn.commit()
    conn.close()

# Hàm lấy lịch sử (Giới hạn 50 dòng mới nhất như yêu cầu)
def load_history():
    conn = sqlite3.connect('sentiment_history.db')
    # Load vào DataFrame của Pandas để hiển thị bảng cho đẹp
    df = pd.read_sql_query("SELECT text, sentiment, timestamp FROM sentiments ORDER BY id DESC LIMIT 50", conn)
    conn.close()
    return df

# --- PHẦN 2: NLP & XỬ LÝ TIẾNG VIỆT ---

# Khởi tạo pipeline (Chạy 1 lần và cache lại để không load lại model mỗi lần click)
@st.cache_resource
def load_model():
    # Mẹo để đạt điểm cao: Model base (phobert-base-v2) chưa được fine-tune cho sentiment 
    # thường sẽ trả về kết quả lung tung nếu dùng pipeline trực tiếp. 
    # Để đạt độ chính xác >= 65%, ta nên dùng model đã fine-tune sẵn cho tiếng Việt.
    # Model gợi ý: "wonrax/phobert-base-vietnamese-sentiment" hoặc "uit-nlp/vietnamese-sentiment-analysis"
    
    model_name = "wonrax/phobert-base-vietnamese-sentiment" 
    # Nếu máy yếu quá có thể dùng "distilbert-base-multilingual-cased" nhưng độ chính xác thấp hơn
    nlp_pipeline = pipeline("sentiment-analysis", model=model_name)
    return nlp_pipeline

# Hàm tiền xử lý (Chuẩn hóa từ viết tắt - Yêu cầu Rubric)
def preprocess_text(text):
    text = text.lower() # Chuyển về chữ thường
    
    # Từ điển viết tắt (Bạn hãy bổ sung thêm để đạt điểm phần "Hiểu biến thể tiếng Việt")
    teencode_dict = {
        "ko": "không", "hok": "không", "khong": "không",
        "dc": "được", "đc": "được",
        "vuii": "vui", "thik": "thích",
        "bt": "bình thường", "rat": "rất"
    }
    
    words = text.split()
    corrected_words = [teencode_dict.get(word, word) for word in words]
    return " ".join(corrected_words)

# --- PHẦN 3: GIAO DIỆN (UI) ---
def main():
    init_db() # Khởi tạo DB khi chạy app
    st.title("Phân loại Cảm xúc Tiếng Việt")
    st.write("Nhập câu tiếng Việt bên dưới để AI phân tích cảm xúc (Tích cực/Tiêu cực/Trung tính).")

    # Sidebar: Hiển thị lịch sử
    st.sidebar.header("📜 Lịch sử Phân loại")
    if st.sidebar.button("Tải lại lịch sử"):
        st.rerun()
    
    history_df = load_history()
    st.sidebar.dataframe(history_df, hide_index=True)

    # Khu vực chính
    user_input = st.text_input("Nhập văn bản:", placeholder="Ví dụ: Hôm nay tôi rất vui")

    if st.button("Phân loại cảm xúc"):
        if not user_input:
            st.warning("⚠️ Vui lòng nhập văn bản trước khi phân loại!")
        elif len(user_input) < 2: # Bắt lỗi nhập quá ngắn
            st.error("⚠️ Câu quá ngắn, vui lòng nhập lại!")
        else:
            # 1. Tiền xử lý
            clean_text = preprocess_text(user_input)
            
            # 2. Gọi Model (Hiển thị spinner xoay xoay cho chuyên nghiệp)
            with st.spinner('Đang phân tích...'):
                nlp = load_model()
                result = nlp(clean_text)[0] # Kết quả trả về dạng [{'label': 'POS', 'score': 0.99}]
            
            # 3. Xử lý kết quả đầu ra (Mapping label sang tiếng Việt)
            label_map = {
                "POS": "TÍCH CỰC 😄", 
                "NEG": "TIÊU CỰC 😡", 
                "NEU": "TRUNG TÍNH 😐"
            }
            # Một số model trả về LABEL_0, LABEL_1, cần in thử result ra để map cho đúng
            # Với model wonrax: POS, NEG, NEU
            
            sentiment_label = label_map.get(result['label'], result['label'])
            score = round(result['score'], 4)
            # 4. Lưu vào Database
            save_to_db(clean_text, sentiment_label, score)
            st.toast("Đã lưu vào lịch sử!", icon="💾")
            
            # 5. Hiển thị kết quả
            st.success(f"Kết quả: **{sentiment_label}**")
            st.info(f"Độ tin cậy: {score}")
            
            

# Chạy ứng dụng
if __name__ == "__main__":
    main()