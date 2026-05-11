"""
Speech Emotion Recognition — Gradio Deployment App
====================================================
Usage:
    pip install gradio tensorflow librosa Pillow scipy scikit-learn
    python app.py

HuggingFace Spaces:
    Upload this file + cnn_bilstm_ser_final.keras + scaler_data.pkl
    The app will auto-launch on the Space's URL.

Files expected in the same directory:
    cnn_bilstm_ser_final.keras   ← trained model (from Kaggle output)
    scaler_data.pkl              ← scalers + label classes (from Kaggle output)
"""

import os
import pickle
import numpy as np
import gradio as gr
import librosa
from PIL import Image
import tensorflow as tf

# ── Paths (edit if your files are elsewhere) ──────────────────────────────────
MODEL_PATH  = os.environ.get("SER_MODEL_PATH",  "cnn_bilstm_ser_final.keras")
SCALER_PATH = os.environ.get("SER_SCALER_PATH", "scaler_data.pkl")

# ── Audio config (must match training CFG) ────────────────────────────────────
SAMPLE_RATE  = 22050
DURATION     = 3.0
TRIM_TOP_DB  = 20
N_MFCC       = 40
N_FFT        = 512
HOP_LENGTH   = 256
WIN_LENGTH   = 512
N_MELS       = 128
MEL_IMG_SIZE = 128

# Emotion display config
EMOTION_EMOJI = {
    "angry":     "😠",
    "calm":      "😌",
    "disgust":   "🤢",
    "fearful":   "😨",
    "happy":     "😄",
    "neutral":   "😐",
    "sad":       "😢",
    "surprised": "😲",
}


# ── Load model & scalers ──────────────────────────────────────────────────────
print("[INFO] Loading model …")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[OK]   Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"[WARN] Could not load model: {e}")
    model = None

print("[INFO] Loading scalers …")
try:
    with open(SCALER_PATH, "rb") as f:
        scaler_data = pickle.load(f)
    mfcc_scaler   = scaler_data["mfcc_scaler"]
    mel_mean      = scaler_data["mel_mean"]
    mel_std       = scaler_data["mel_std"]
    label_classes = scaler_data["label_classes"]
    time_steps    = scaler_data["time_steps"]
    print(f"[OK]   Scalers loaded. Classes: {label_classes}")
except Exception as e:
    print(f"[WARN] Could not load scalers: {e}")
    mfcc_scaler = mel_mean = mel_std = label_classes = time_steps = None


# ── Feature extraction helpers ────────────────────────────────────────────────
def load_and_preprocess(path: str):
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True,
                            duration=DURATION * 2)
        y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])
        target = int(SAMPLE_RATE * DURATION)
        y = np.pad(y, (0, max(0, target - len(y))))[:target]
        return y.astype(np.float32)
    except Exception as e:
        return None


def extract_mfcc(y: np.ndarray) -> np.ndarray:
    mfcc   = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                   n_fft=N_FFT, hop_length=HOP_LENGTH,
                                   win_length=WIN_LENGTH)
    delta  = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([mfcc, delta, delta2], axis=0).T.astype(np.float32)


def extract_mel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    img = Image.fromarray(log_mel).resize((MEL_IMG_SIZE, MEL_IMG_SIZE))
    return np.array(img, dtype=np.float32)[..., np.newaxis]


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(audio_path: str):
    if model is None or mfcc_scaler is None:
        return (
            "⚠️ Model or scalers not found. Please upload "
            "`cnn_bilstm_ser_final.keras` and `scaler_data.pkl`.",
            {}
        )

    y = load_and_preprocess(audio_path)
    if y is None:
        return "❌ Could not process audio. Try a .wav or .mp3 file.", {}

    mfcc = extract_mfcc(y)
    mel  = extract_mel(y)

    # Pad/trim MFCC
    if mfcc.shape[0] < time_steps:
        mfcc = np.pad(mfcc, ((0, time_steps - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:time_steps]

    # Normalise
    mfcc_n = mfcc_scaler.transform(mfcc).astype(np.float32)
    mel_n  = ((mel - mel_mean) / mel_std).astype(np.float32)

    proba = model.predict(
        {"mel_input": mel_n[np.newaxis], "mfcc_input": mfcc_n[np.newaxis]},
        verbose=0
    )[0]

    idx        = int(np.argmax(proba))
    emotion    = label_classes[idx]
    confidence = float(proba[idx])
    emoji      = EMOTION_EMOJI.get(emotion, "🎙️")

    result_str = (
        f"## {emoji} {emotion.upper()}\n"
        f"**Confidence:** {confidence*100:.1f}%"
    )

    bar_data = {
        f"{EMOTION_EMOJI.get(lbl,'🎙️')} {lbl}": round(float(p), 4)
        for lbl, p in zip(label_classes, proba)
    }
    return result_str, bar_data


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Speech Emotion Recognition",
    theme=gr.themes.Soft(),
    css=".result-box {font-size:1.4rem; text-align:center;}"
) as demo:

    gr.Markdown(
        """
        # 🎙️ Speech Emotion Recognition
        Upload or record a short speech clip (ideally 2–5 seconds).
        The CNN-BiLSTM model will predict the emotion from your voice.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="🎤 Audio Input (.wav / .mp3)",
            )
            predict_btn = gr.Button("🔍 Predict Emotion", variant="primary")

        with gr.Column(scale=1):
            result_md   = gr.Markdown(
                value="*Upload audio and click Predict*",
                elem_classes=["result-box"]
            )
            prob_chart  = gr.Label(
                label="📊 Class Probabilities",
                num_top_classes=8,
            )

    predict_btn.click(
        fn=predict,
        inputs=audio_input,
        outputs=[result_md, prob_chart],
    )

    gr.Markdown(
        """
        ---
        **Supported emotions:** angry · calm · disgust · fearful · happy · neutral · sad · surprised  
        **Model:** Dual-stream CNN (Mel-spectrogram) + BiLSTM (MFCC)  
        **Trained on:** RAVDESS + TESS datasets
        """
    )

    gr.Examples(
        examples=[],   # add sample .wav file paths here if you have them
        inputs=audio_input,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # change to "127.0.0.1" for local-only
        server_port=7860,
        share=True,              # creates a temporary public URL
        show_error=True,
    )
