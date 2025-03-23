import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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

st.title("Chatbot AI")

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

user_input = st.text_input("Bạn:")

if user_input:
    response, st.session_state.chat_history_ids = chatbot(user_input, st.session_state.chat_history_ids, tokenizer, model)
    st.write("Chatbot:", response)
