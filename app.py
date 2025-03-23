import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Tải tokenizer và mô hình trước khi khởi động ứng dụng
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

def chatbot(user_input, tokenizer, model):
    # Mã hóa đầu vào của người dùng
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Tạo phản hồi của chatbot
    chat_history_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    # Giải mã phản hồi
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response

st.title("Chatbot AI")

user_input = st.text_input("Bạn:")

if user_input:
    response = chatbot(user_input, tokenizer, model)
    st.write("Chatbot:", response)