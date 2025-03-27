import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model với cache (cho phép output mutation)
@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load dữ liệu sản phẩm từ file Excel
@st.cache_data
def load_data():
    return pd.read_excel("data.xlsx")  # Đảm bảo file Excel nằm cùng thư mục

# Tính embeddings cho danh sách mô tả sản phẩm, bỏ qua việc hash đối số _model
@st.cache_data
def compute_embeddings(descriptions, _model):
    return _model.encode(descriptions)

# Load model và dữ liệu
model = load_model()
df = load_data()
descriptions = df["Description"].tolist()

# Tính embeddings cho toàn bộ mô tả sản phẩm
embeddings = compute_embeddings(descriptions, model)

st.title("🤖 Chatbot Tìm Kiếm Sản Phẩm - Mecsu.vn (Vector Search - Top 10 Grid)")
user_input = st.text_input("Nhập từ khóa sản phẩm cần tìm:")

if user_input:
    with st.spinner("💬 Đang xử lý truy vấn..."):
        # Tính embedding cho truy vấn người dùng
        query_embedding = model.encode([user_input])
        # Tính cosine similarity giữa query và embeddings của sản phẩm
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        # Lấy chỉ số của 10 sản phẩm có độ tương đồng cao nhất
        top_n = 10
        top_indices = np.argsort(similarities)[::-1][:top_n]
    
    st.markdown("### Kết quả tìm kiếm:")
    # Hiển thị kết quả theo grid: 5 sản phẩm trên 1 hàng (2 hàng tổng cộng 10 sản phẩm)
    for i in range(0, len(top_indices), 5):
        cols = st.columns(5)
        for j, idx in enumerate(top_indices[i:i+5]):
            sp = df.iloc[idx]
            score = similarities[idx]
            # Đóng gói kết quả trong một div với font nhỏ
            html = f"""
            <div style="font-size:12px; margin:5px;">
                <strong>Mô tả:</strong> {sp['Description']}<br>
                <strong>Mã hàng:</strong> {sp['Part Number']}<br>
                <strong>Link:</strong> <a href="{sp['Link']}" target="_blank">Xem sản phẩm</a><br>
                <strong>Độ tương đồng:</strong> {score:.2f}
            </div>
            """
            with cols[j]:
                st.markdown(html, unsafe_allow_html=True)
