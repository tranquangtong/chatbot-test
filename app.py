import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime

# Thiết lập tiêu đề và CSS tùy chỉnh - PHẢI ĐẶT ĐẦU TIÊN
st.set_page_config(page_title="Chatbot AI", page_icon="🤖")

# CSS tùy chỉnh cho giao diện chat và loại bỏ khoảng trắng
st.markdown("""
<style>
/* Loại bỏ padding mặc định của Streamlit */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0rem !important;
    margin-top: -2rem;
}

/* Loại bỏ khoảng trắng dưới tiêu đề */
.stTitle {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

/* Loại bỏ khoảng trắng giữa các phần tử */
.element-container {
    margin-bottom: 0 !important;
}

/* Định dạng tin nhắn */
.user-message {
    background-color: #e6f7ff;
    padding: 10px;
    border-radius: 15px 15px 15px 0;
    margin: 5px 0;
    display: flex;
    justify-content: flex-start;
}
.bot-message {
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 15px 15px 0 15px;
    margin: 5px 0;
    display: flex;
    justify-content: flex-end;
}
.message-container {
    max-width: 80%;
    word-wrap: break-word;
}
.timestamp {
    font-size: 0.8em;
    color: gray;
    margin-top: 5px;
}
.chat-container {
    min-height: 50px;
    max-height: 400px;
    overflow-y: auto;
    padding: 5px;
    border: 1px solid #e6e6e6;
    border-radius: 5px;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}
.main-container {
    display: flex;
    flex-direction: column;
    max-width: 800px;
    margin: 0 auto;
}

/* Giảm khoảng cách giữa các phần tử form */
.stForm {
    margin-top: 0.5rem !important;
}
.stForm > div {
    margin-bottom: 0 !important;
}
.stButton {
    margin-top: 0 !important;
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

# Tiêu đề với padding nhỏ hơn
st.markdown("<h1 style='margin-bottom:0; padding-bottom:0;'>Chatbot AI 🤖</h1>", unsafe_allow_html=True)

# Khởi tạo lịch sử chat trong session state nếu chưa có
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

# Hiển thị lịch sử chat
st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <div class="message-container">
                <b>Bạn:</b> {message["content"]}
                <div class="timestamp">{message["time"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bot-message">
            <div class="message-container">
                <b>Chatbot:</b> {message["content"]}
                <div class="timestamp">{message["time"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Tạo form nhập tin nhắn với padding nhỏ hơn
with st.form(key="message_form", clear_on_submit=True):
    user_input = st.text_input("Nhập tin nhắn của bạn:", key="user_input")
    cols = st.columns([4, 1])
    with cols[1]:
        submit_button = st.form_submit_button("Gửi")

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
        var chatContainer = document.getElementById('chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
    window.onload = scrollToBottom;
</script>
""", unsafe_allow_html=True)

# Thêm nút để xóa lịch sử chat - đặt ở cuối và nhỏ hơn
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Xóa lịch sử chat", type="secondary", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.chat_history_ids = None
        st.rerun()

