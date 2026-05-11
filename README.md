# Speech Emotion Recognition (SER) System

A deep learning solution for detecting human emotions from speech audio. This system combines **Convolutional Neural Networks (CNN)** for extracting spectral features and **Bidirectional Long Short-Term Memory (BiLSTM)** networks for capturing temporal dependencies in speech patterns. The model is deployed as an interactive web application using **Streamlit**.

## 🌟 Features

- **Hybrid Architecture**: Leverages CNNs for feature extraction and BiLSTM for sequence modeling to achieve high accuracy.
- **Real-time Inference**: Upload audio files or record via microphone for instant emotion detection.
- **Visual Insights**: Displays the predicted emotion, confidence scores, and the audio waveform.
- **Multi-Emotion Support**: Capable of classifying emotions such as Happy, Sad, Angry, Neutral, Fearful, Disgusted, and Surprise.
- **User-Friendly Interface**: Built with Streamlit for a seamless web-based experience.

## 🏗️ Architecture Overview

The system utilizes a hybrid deep learning model:

1.  **Input**: Audio files are preprocessed and converted into **Mel-Frequency Cepstral Coefficients (MFCCs)** or **Spectrograms**.
2.  **CNN Layer**: Extracts local spatial features from the spectrogram/MFCC data.
3.  **BiLSTM Layer**: Processes the sequence of features extracted by the CNN to understand temporal context (past and future dependencies).
4.  **Dense Layer**: Maps the extracted features to the final emotion classes.

```mermaid
graph LR
    A[Audio Input] --> B(MFCC/Spectrogram)
    B --> C[CNN Feature Extractor]
    C --> D[BiLSTM Sequence Model]
    D --> E[Softmax Classification]
    E --> F[Predicted Emotion]
