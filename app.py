import streamlit as st
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


st.title("🤖💬 Extractive Question Answering Bot")
st.text("Hello! It's nice to meet you.\nPlease provide me your text(paragraphs, sentences,..) and your question!")


with st.spinner("Loading Model Into memory....."):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForQuestionAnswering.from_pretrained("huynguyen61098/Bert-Base-Cased-Squad-Extractive-QA").to(
        device)

context = st.text_input("Enter your context here..")
print(f"Context: {context}")
question = st.text_input("Enter your question here..")
print(f"Question: {question}")
if context:
    if question:
        st.write("Response :")
        with st.spinner("Searching for answers....."):
            tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            inputs = tokenizer(question, context, return_tensors="pt")
            outputs = model(**inputs)

            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax()
            predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
            answers = tokenizer.decode(predict_answer_tokens)
            st.write(f"answer: {answers}")
    st.write("")