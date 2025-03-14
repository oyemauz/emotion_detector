import streamlit as st
import torch
import re
import nltk
import torch.nn as nn
import contractions
from nltk.corpus import stopwords
import torch.nn.functional as F
from nltk.stem import WordNetLemmatizer
from transformers import DistilBertTokenizer, DistilBertPreTrainedModel, DistilBertModel

# Define your custom model class
class CustomModel(DistilBertPreTrainedModel):
    def __init__(self, config):  # Fix: Use double underscores
        super().__init__(config)  # Fix: Use double underscores
        self.num_labels = config.num_labels
        
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and tokenizer
model_path = "bert_classification_model"
try:
    model = CustomModel.from_pretrained(model_path).to(device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) - {"no", "not", "never"}

# Define emotion labels with emojis
labels = {
    "sadness": "😢 Sadness",
    "love": "❤️ Love",
    "anger": "😡 Anger",
    "joy": "😊 Joy",
    "fear": "😨 Fear",
    "surprise": "😲 Surprise",
}

# Emotion descriptions
emotion_descriptions = {
    "sadness": "Feeling down or sorrowful.",
    "love": "Deep affection and strong feelings of connection.",
    "anger": "Strong displeasure or rage.",
    "joy": "A feeling of happiness and pleasure.",
    "fear": "A sense of danger or anxiety.",
    "surprise": "A reaction to something unexpected."
}

# Sidebar settings
st.sidebar.title("⚙️ Settings")
temperature = st.sidebar.slider("Adjust Temperature:", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# Main UI
st.title("🔥 Emotion Detector")
st.markdown("### 🤔 Enter a sentence to analyze its emotion")
user_input = st.text_area("", "", height=150)

def Pre_process(text):
    text = contractions.fix(text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def predict_emotion(text, temperature):
    cleaned_text = Pre_process(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs)
        probabilities = F.softmax(logits / temperature, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    probs_dict = {label: round(probabilities[i].item() * 100, 2) for i, label in enumerate(labels)}
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    return list(labels.keys())[predicted_class], probs_dict, sorted_probs

if st.button("🔍 Predict Emotion"):
    if user_input.strip():
        prediction, probs, sorted_probs = predict_emotion(user_input, temperature)
        st.success(f"**Detected Emotion: {labels[prediction]}**")
        st.write(f"📝 **Explanation:** {emotion_descriptions[prediction]}")
        
        # Display emotion probabilities
        st.subheader("🔍 Emotion Probabilities")
        cols = st.columns(len(labels))
        for col, (emotion, prob) in zip(cols, probs.items()):
            col.metric(label=f"**{labels[emotion]}**", value=f"{prob}%")
        
        # Show top 3 emotions
        st.subheader("🏆 Top 3 Detected Emotions")
        for emotion, prob in sorted_probs[:3]:
            st.write(f"🔹 **{labels[emotion]}**: {prob}%")
        
        # Display probability bars
        st.subheader("📊 Emotion Distribution")
        for emotion, prob in probs.items():
            st.progress(prob / 100)
            st.write(f"**{labels[emotion]}: {prob}%**")
    else:
        st.warning("⚠️ Please enter a sentence.")