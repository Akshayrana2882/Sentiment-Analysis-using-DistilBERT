from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).parent
MODEL_DIR = Path("/Users/akshayrana/Desktop/sentiment_model")
MODEL_NAME = "distilbert-base-uncased"
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")


@st.cache_resource
def load_model():
    if MODEL_DIR.exists():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=3
        )
    model.eval()
    return tokenizer, model


def predict_text(text):
    tokenizer, model = load_model()
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**encoding)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities


st.title("Sentiment Analysis")
st.write("Enter text to predict its sentiment.")

text = st.text_area("Enter Text", placeholder="Type your sentence here...", height=140)

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        predicted_class, probabilities = predict_text(text)
        st.success(f"Predicted Sentiment: {LABEL_MAP[predicted_class]}")
        st.subheader("Confidence Scores")
        st.write(f"Negative: {probabilities[0].item() * 100:.2f}%")
        st.write(f"Neutral: {probabilities[1].item() * 100:.2f}%")
        st.write(f"Positive: {probabilities[2].item() * 100:.2f}%")
        st.progress(float(probabilities[predicted_class].item()))
