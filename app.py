import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Set Streamlit page configuration (MUST BE FIRST)
st.set_page_config(page_title="Healthcare Assistant", page_icon="🩺")

# Load the model pipeline
generator = pipeline("text-generation", model="distilgpt2")

# Function to clean input text
def clean_input(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

# Chatbot response generator
def get_bot_response(user_input):
    user_input = user_input.strip()
    if not user_input:
        return "Please enter a message."
    
    # Predefined responses
    predefined_responses = {
        "what are the symptoms of covid": "Common symptoms include fever, dry cough, tiredness, and loss of taste or smell.",
        "how can i book an appointment": "To book an appointment, please call our helpline or use the online portal.",
        "what are the side effects of paracetamol": "Common side effects include nausea, rash, and liver issues if overdosed."
    }

    cleaned = clean_input(user_input.lower())

    for key, response in predefined_responses.items():
        if key in cleaned:
            return response

    # Fallback to language model
    response = generator(user_input, max_length=100, do_sample=True, temperature=0.7)
    return response[0]['generated_text']

# Streamlit App
def main():
    st.title("🩺 Healthcare Assistant Chatbot")
    user_input = st.text_input("How can I assist you today?")

    if st.button("Send"):
        if user_input:
            response = get_bot_response(user_input)
            st.markdown(f"**Bot:** {response}")
        else:
            st.warning("Please enter a message to get a response.")

if __name__ == "__main__":
    main()
