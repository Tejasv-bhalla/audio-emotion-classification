import librosa
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfcc, axis=1) # Average across time frames
    return features


# Load tools
model = load_model("emotion_model.h5")
encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("s_scaler.pkl")

# Test
file = "sample_audio/YAF_back_disgust.wav" 
features = extract_features(file).reshape(1, -1)
scaled = scaler.transform(features)
prediction = model.predict(scaled)
label = encoder.inverse_transform([np.argmax(prediction)])
print("Predicted Emotion:", label[0])

