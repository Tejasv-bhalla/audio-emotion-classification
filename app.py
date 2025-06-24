import streamlit as st
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import tempfile
import matplotlib.pyplot as plt
# from pydub import AudioSegment

# 🎨 Page config and styling
st.set_page_config(
    page_title="🎧 Emotion Recognition",
    layout="centered",
    page_icon="🎵"
)

# Custom CSS
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
        .main { flex: 1; }
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
        .footer {
            text-align: center;
            padding: 1rem;
            color: #777;
            font-size: 0.9rem;
            background-color: var(--bg);
            border-top: 1px solid #eee;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Main container
st.markdown("<div class='main'>", unsafe_allow_html=True)

# Title & subtitle
st.markdown("""
    <div class="title">🎵 Emotion Recognition from Audio</div>
    <div class="subtitle">Upload an <b>audio file</b> (wav, mp3, flac, etc.) to detect emotional content</div>
""", unsafe_allow_html=True)

# 🔁 Load model and helpers
@st.cache_resource
def load_all():
    model = load_model("emotion_model.h5")
    scaler = joblib.load("s_scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_all()

# 🔄 Convert any format to WAV
# def convert_to_wav(file):
#     suffix = os.path.splitext(file.name)[-1]
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
#         temp_audio.write(file.read())
#         temp_audio_path = temp_audio.name

#     sound = AudioSegment.from_file(temp_audio_path)
#     temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#     sound.export(temp_wav.name, format="wav")
#     return temp_wav.name


#🎼 Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc, axis=1), y, sr

# Sidebar upload
st.sidebar.header("🔊 Upload Audio")
uploaded_file = st.sidebar.file_uploader(
    "Choose an audio file", 
    type=["wav", "mp3", "ogg", "flac", "m4a"], 
    label_visibility="collapsed"
)
st.sidebar.markdown("""
---
**Instructions:**
- Upload a clear speech or expressive `.wav`, `.mp3`, etc.
- Keep duration under 10 seconds for best accuracy
- Make sure it's not silent or corrupted
""")

# 🎯 Process file
if uploaded_file is not None:
    # Save to temporary file
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # Play audio
    st.audio(temp_path)

    with st.spinner("🔍 Analyzing audio..."):
        features, y, sr = extract_features(temp_path)

        # Plot waveform
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#3f51b5")
        ax.set_title("Audio Waveform", fontsize=14)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.3)
        st.markdown("### 📈 Audio Waveform")
        st.pyplot(fig)

        # Scale and predict
        scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(scaled)
        label = encoder.inverse_transform([np.argmax(prediction)])
        confidence = np.max(prediction) * 100

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Emotion", f"{label[0]}")
    with col2:
        st.metric("Confidence", f"{confidence:.2f}%")

    # Clean up
    os.remove(temp_path)
else:
    st.info("⬅️ Upload a supported audio file to begin")

# Footer
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
    <div class="footer">
        Emotion Recognition System • Built with ❤️ using Streamlit
    </div>
""", unsafe_allow_html=True)
