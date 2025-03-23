import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime

# Thiết lập tiêu đề và CSS tùy chỉnh - PHẢI ĐẶT ĐẦU TIÊN
st.set_page_config(page_title="Chatbot AI", page_icon="🤖", layout="wide")

# CSS tùy chỉnh cho giao diện chat
st.markdown("""
<style>
/* Thiết lập layout tổng thể */
.main {
    display: flex;
    flex-direction: column;
    height: 100vh;
    padding: 0 !important;
    max-width: 100% !important;
}

/* Loại bỏ padding mặc định của Streamlit */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Định dạng header */
.header {
    background-color: #f8f9fa;
    padding: 1rem;
    border-bottom: 1px solid #e9ecef;
    position: sticky;
    top: 0;
    z-index: 100;
    width: 100%;
}

/* Định dạng khu vực chat */
.chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

/* Định dạng footer */
.footer {
    background-color: #f8f9fa;
    padding: 1rem;
    border-top: 1px solid #e9ecef;
    position: sticky;
    bottom: 0;
    width: 100%;
}

/* Định dạng tin nhắn */
.user-message {
    background-color: #e6f7ff;
    padding: 10px;
    border-radius: 15px 15px 15px 0;
    margin: 5px 0;
    align-self: flex-start;
    max-width: 80%;
}

.bot-message {
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 15px 15px 0 15px;
    margin: 5px 0;
    align-self: flex-end;
    max-width: 80%;
}

.message-content {
    word-wrap: break-word;
}

.timestamp {
    font-size: 0.8em;
    color: gray;
    margin-top: 5px;
}

/* Ẩn các phần tử không cần thiết */
.stDeployButton, .viewerBadge, .css-1dp5vir, .css-1n76uvr {
    display: none !important;
}

/* Định dạng nút và form */
.stButton button {
    width: 100%;
}

/* Ẩn footer của Streamlit */
footer {
    display: none !important;
}

/* Loại bỏ padding của các container */
.element-container {
    margin-bottom: 0 !important;
}

/* Loại bỏ border của input */
.stTextInput input {
    border: 1px solid #e9ecef;
    border-radius: 20px;
    padding: 10px 15px;
}

/* Định dạng nút xóa lịch sử */
.clear-button {
    text-align: center;
    margin-top: 0.5rem;
}
.clear-button button {
    background-color: transparent;
    color: #6c757d;
    border: 1px solid #6c757d;
    border-radius: 20px;
    padding: 5px 10px;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# Tải tokenizer và mô hình trước khi khởi động ứng dụng
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

tokenizer, model = load_model()

def chatbot(user_input, chat_history_ids, tokenizer, model):
    # Mã hóa đầu vào của người dùng
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Thêm đầu vào của người dùng vào lịch sử trò chuyện
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    # Tạo phản hồi của chatbot
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )

    # Giải mã phản hồi
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response, chat_history_ids

# Khởi tạo lịch sử chat trong session state nếu chưa có
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

# Header - Tiêu đề cố định ở đầu trang
st.markdown('<div class="header"><h1>Chatbot AI 🤖</h1></div>', unsafe_allow_html=True)

# Khu vực hiển thị lịch sử chat
st.markdown('<div class="chat-area">', unsafe_allow_html=True)
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <div class="message-content">
                <b>Bạn:</b> {message["content"]}
                <div class="timestamp">{message["time"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bot-message">
            <div class="message-content">
                <b>Chatbot:</b> {message["content"]}
                <div class="timestamp">{message["time"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Footer - Khung nhập tin nhắn cố định ở dưới cùng
st.markdown('<div class="footer">', unsafe_allow_html=True)
with st.form(key="message_form", clear_on_submit=True):
    cols = st.columns([4, 1])
    with cols[0]:
        user_input = st.text_input("", placeholder="Nhập tin nhắn của bạn...", label_visibility="collapsed")
    with cols[1]:
        submit_button = st.form_submit_button("Gửi")
    
    # Nút xóa lịch sử chat
    if st.form_submit_button("Xóa lịch sử chat", type="secondary"):
        st.session_state.chat_history = []
        st.session_state.chat_history_ids = None
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Xử lý khi người dùng gửi tin nhắn
if submit_button and user_input:
    # Thêm tin nhắn của người dùng vào lịch sử
    current_time = datetime.now().strftime("%H:%M")
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "time": current_time
    })
    
    # Lấy phản hồi từ chatbot
    response, st.session_state.chat_history_ids = chatbot(
        user_input, 
        st.session_state.chat_history_ids, 
        tokenizer, 
        model
    )
    
    # Thêm phản hồi của chatbot vào lịch sử
    st.session_state.chat_history.append({
        "role": "bot",
        "content": response,
        "time": current_time
    })
    
    # Tự động rerun để cập nhật giao diện
    st.rerun()

# JavaScript để cuộn xuống cuối cùng của container chat
st.markdown("""
<script>
    function scrollToBottom() {
        var chatArea = document.querySelector('.chat-area');
        if (chatArea) {
            chatArea.scrollTop = chatArea.scrollHeight;
        }
    }
    window.onload = scrollToBottom;
</script>
""", unsafe_allow_html=True)

