import streamlit as st
import torch
import pickle
from transformers import AutoModelForSequenceClassification

# Load the fine-tuned model and tokenizer from the pkl files
@st.cache_resource
def load_model_and_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('fine_tuned_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model, tokenizer

# Load model and tokenizer only once to speed up subsequent predictions
model, tokenizer = load_model_and_tokenizer()

# Set the device (GPU or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)  # Move the model to the correct device

# Streamlit web app setup
st.markdown("<h1><span style='color: red;'>Abusive Language</span> Detection using <span style='color: red;'>HateBERT LLM</span></h1>", unsafe_allow_html=True)
# Input text box for user input
user_input = st.text_area("Enter a sentence or paragraph to analyze:", "")

# Detect button to run the model on input
if st.button("Detect Abusive Content"):
    if not user_input:
        st.warning("Please enter some text to analyze.")  # Warn if no input is provided
    else:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = inputs.to(device)  # Move input tensors to the device

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1)

        # Interpret the prediction
        if prediction.item() == 1:
            st.error("Abusive content detected!")  # Abusive content detected
        else:
            st.success("Non-abusive content.")  # Non-abusive content detected

# Footer
st.write("This app uses a fine-tuned <span style='color: red;'>HateBERT model</span> for detecting abusive language in text.", unsafe_allow_html=True)