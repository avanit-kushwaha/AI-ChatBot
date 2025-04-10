import streamlit as st
import nltk
from transformers import pipeline
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Optional: If using a Hugging Face token ---
# from huggingface_hub import login
# login("your_huggingface_token")

# Download NLTK data safely
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

# Load chatbot model with fallback
@st.cache_resource
def load_chatbot_model():
    try:
        return pipeline("text-generation", model="distilgpt2")
    except Exception as e:
        st.warning("⚠️ Falling back to a smaller model due to loading issues.")
        return pipeline("text-generation", model="sshleifer/tiny-gpt2")

chatbot = load_chatbot_model()

# Chatbot response logic
def healthcare_chatbot(user_input):
    user_input_lower = user_input.lower()
    if "symptom" in user_input_lower:
        return "Please consult a Doctor for accurate advice."
    elif "appointment" in user_input_lower:
        return "Would you like to schedule an appointment with the Doctor?"
    elif "medication" in user_input_lower or "medicine" in user_input_lower:
        return "It's important to take prescribed medicines regularly. If you have concerns, consult your doctor."
    else:
        response = chatbot(user_input, max_length=100, num_return_sequences=1)
        return response[0]['generated_text']

# Streamlit App
def main():
    st.set_page_config(page_title="Healthcare Assistant", page_icon="🩺")
    st.title("🩺 Healthcare Assistant Chatbot")
    user_input = st.text_input("How can I assist you today?")
    
    if st.button("Submit"):
        if user_input:
            st.write("**User:**", user_input)
            with st.spinner("Processing your query. Please wait ..."):
                response = healthcare_chatbot(user_input)
            st.success("**Healthcare Assistant:**")
            st.write(response)
        else:
            st.warning("Please enter a message to get a response.")

if __name__ == "__main__":
    main()
