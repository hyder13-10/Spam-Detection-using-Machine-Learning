import streamlit as st
import joblib, re, nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
sw = set(stopwords.words("english"))

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(w for w in text.split() if w not in sw)

@st.cache_resource
def load_model():
    return joblib.load("spam_model.pkl")

model = load_model()

st.title("📧 Spam Detector")
msg = st.text_area("Paste a message or email:")

if st.button("Check"):
    if msg.strip():
        cleaned = clean(msg)
        pred = model.predict([cleaned])[0]
        prob = model.predict_proba([cleaned])[0][1]
        if pred == 1:
            st.error(f"🚨 Spam ({prob:.0%} confidence)")
        else:
            st.success(f"✅ Not spam ({1 - prob:.0%} confidence)")
    else:
        st.warning("Type something first!")

st.caption("Trained on the SMS Spam Collection, so it works best on short, SMS-style messages.")