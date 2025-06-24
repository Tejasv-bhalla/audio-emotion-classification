
# 🎧 Audio Emotion Classification

Predict human emotions from audio using a deep learning model deployed via Streamlit. This project allows users to upload audio files and get real-time emotion predictions based on extracted audio features.

---

## 📚 Table of Contents

- [🎥 Demo](#-demo)
- [📌 Overview](#-overview)
- [✅ Features](#-features)
- [🧱 Tech Stack](#-tech-stack)
- [🗂 Project Structure](#-project-structure)
- [⚙️ Setup Instructions](#-setup-instructions)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. (Optional) Create a virtual environment](#2-optional-create-a-virtual-environment)
  - [3. Install dependencies](#3-install-dependencies)
  - [4. Run the Streamlit app](#4-run-the-streamlit-app)
- [🎯 How to Use](#-how-to-use)
- [🧠 Model Details](#-model-details)
- [🧩 Requirements (see requirements.txt)](#-requirements-see-requirementstxt)
- [🔐 Notes](#-notes)
- [🚀 Deployment](#-deployment)
- [🙌 Acknowledgements](#-acknowledgements)
- [📬 Contact](#-contact)

---

## 🎥 Demo

🎥 [Watch Demo Video](demo.mp4)


---

## 📌 Overview

This project detects emotions such as **happy**, **sad**, **angry**, **neutral**, etc., from short audio clips using a pre-trained neural network. It uses `librosa` for feature extraction and `tensorflow` for prediction.

---

## ✅ Features

- 🎙 Upload `.wav` audio files  
- ⚙️ Extract MFCC features  
- 🧠 Run emotion classification using a trained Keras model  
- 📊 Visualize waveform and prediction confidence  
- 🌐 Deployed with Streamlit for interactive web-based inference

---

## 🧱 Tech Stack

- **Frontend**: Streamlit  
- **Backend**: TensorFlow / Keras, NumPy  
- **Audio Processing**: Librosa  
- **Model Files**: `best_model.h5`, `scaler.pkl`, `label_encoder.pkl`

---

## 🗂 Project Structure

```
├── app.py                # Streamlit application
├── demo.mp4              # Demo video (recorded run)
├── best_model.h5         # Trained TensorFlow model
├── scaler.pkl            # Scaler used on training data
├── label_encoder.pkl     # LabelEncoder used for emotions
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Tejasv-bhalla/audio-emotion-classification.git
cd audio-emotion-classification
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🎯 How to Use

1. Launch the app locally or via Streamlit Cloud.  
2. Upload a `.wav` audio file under 10 seconds.  
3. View:
   - Audio waveform
   - Predicted emotion
   - Confidence score  
4. Delete or upload new audio to test again.

---

## 🧠 Model Details

- Model type: 1D CNN  
- Input: 40 MFCC features  
- Output: One-hot encoded emotion label  
- Trained on: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset with 7 emotion classes

---

## 🧩 Requirements (see `requirements.txt`)

```txt
streamlit==1.35.0
numpy==1.24.4
librosa==0.10.1
tensorflow==2.12.0
scikit-learn==1.2.2
joblib==1.3.2
matplotlib==3.7.1
soundfile==0.12.1
```

---

## 🔐 Notes

- Audio must be in a supported format: `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`.  
- Ideal duration: 2–10 seconds of clean, expressive speech.  
- Works best with emotional speech, not background noise.

---

## 🚀 Deployment

You can deploy the app on [Streamlit Cloud](https://streamlit.io/cloud) by pushing all files (including model files and `requirements.txt`) to GitHub and connecting the repo to Streamlit.

If you want to add a demo link to the public app, you can edit this section:

```markdown
### 🌐 Live App

[Click here to try it live](https://your-streamlit-app-link)
```

---

## 🙌 Acknowledgements

- Built by **Tejasv Bhalla**  
- Inspired by public emotion recognition datasets and models  
- Libraries used: `librosa`, `streamlit`, `tensorflow`, `joblib`, `matplotlib`, etc.

---

## 📬 Contact

Got suggestions or want to collaborate?

- GitHub: [@Tejasv-bhalla](https://github.com/Tejasv-bhalla)
- Email: [Add your email here if you want]

---

> 🎵 *“Where words fail, emotions speak. Let’s decode them.”*
