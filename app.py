import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the trained model and tokenizer
model_path = "saved_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Streamlit UI
st.title("🔥 Emotion Detector")

# User input
user_input = st.text_area("Enter a sentence:", "")

# Predict function
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()

    labels = ["sadness", "love", "anger", "joy","fear","surprise"] 
    return labels[predicted_class]

if st.button("Predict"):
    if user_input.strip():
        prediction = predict_emotion(user_input)
        st.success(f"Detected Emotion: **{prediction}**")
    else:
        st.warning("Please enter a sentence.")

