import streamlit as st
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

# 🎨 Page config and enhanced styling
st.set_page_config(
    page_title="🎧 Emotion Recognition",
    layout="centered",
    page_icon="🎵"
)

# Custom CSS for modern UI with sticky footer
st.markdown("""
    <style>
        :root {
            --primary: #3f51b5;
            --secondary: #4caf50;
            --bg: #f6f9fc;
        }
        body {
            background-color: var(--bg);
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        .main {
            flex: 1;
        }
        .title {
            font-size: 2.5rem;
            text-align: center;
            color: var(--primary);
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.2rem;
            text-align: center;
            color: #555;
            margin-bottom: 1.2rem;
        }
        .stButton>button {
            background-color: var(--primary);
            color: white;
            border-radius: 8px;
            font-size: 1.1rem;
            padding: 0.5rem 1.5rem;
        }
        .waveform-container {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .footer {
            text-align: center;
            padding: 1rem;
            color: #777;
            font-size: 0.9rem;
            background-color: var(--bg);
            border-top: 1px solid #eee;
            margin-top: 2rem;
        }
        .stFileUploader {
            background: #f0f2f6;
            border-radius: 8px;
            padding: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Main content container
st.markdown("<div class='main'>", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="title">🎵 Emotion Recognition from Audio</div>
    <div class="subtitle">Upload a <b>.wav</b> file to detect emotional content</div>
""", unsafe_allow_html=True)

# 🔁 Load model, encoder, and scaler
@st.cache_resource
def load_all():
    model = load_model("best_model.h5")
    encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, encoder, scaler

model, encoder, scaler = load_all()

# 🎼 Feature Extraction Function
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc, axis=1), y, sr

# Sidebar upload
st.sidebar.header("🔊 Upload Audio")
uploaded_file = st.sidebar.file_uploader("Choose WAV file", type=["wav"], label_visibility="collapsed")
st.sidebar.markdown("""
---
**How to use:**
- Upload a short, clear `.wav` file (speech or expressive sound)
- Make sure the audio is not silent and is in WAV format
- For best results, use files under 10 seconds
""")

# 🎯 Run prediction and visualization
if uploaded_file is not None:
    # Save uploaded file
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    # Audio player
    st.audio("temp.wav", format="audio/wav")
    
    with st.spinner("Analyzing audio..."):
        # Extract features and waveform
        features, y, sr = extract_features("temp.wav")
        
        # Create waveform visualization
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#3f51b5")
        ax.set_title("Audio Waveform", fontsize=14)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        # Display waveform
        st.markdown("### Audio Waveform")
        st.pyplot(fig)
        
        # Run prediction
        scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(scaled)
        label = encoder.inverse_transform([np.argmax(prediction)])
        confidence = np.max(prediction) * 100

    # Results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Emotion", f"{label[0]}", help="Detected emotional state")
    with col2:
        st.metric("Confidence", f"{confidence:.2f}%", delta_color="off")

    # Clean up temp file
    os.remove("temp.wav")
else:
    st.info("⬅️ Upload a .wav file to begin analysis")

# Footer
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
    <div class="footer">
        Emotion Recognition System • Using Deep Learning • 
        <span style="color: var(--primary);">Streamlit</span>
    </div>
""", unsafe_allow_html=True)
