import os
import io
import pickle
import numpy as np
import streamlit as st
import librosa
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
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

EMOTION_COLOR = {
    "angry":     "#e74c3c",
    "calm":      "#1abc9c",
    "disgust":   "#8e44ad",
    "fearful":   "#e67e22",
    "happy":     "#f1c40f",
    "neutral":   "#95a5a6",
    "sad":       "#3498db",
    "surprised": "#2ecc71",
}


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import tensorflow as tf
    try:
        m = tf.keras.models.load_model(MODEL_PATH)
        return m
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        return None


@st.cache_resource(show_spinner="Loading scalers…")
def load_scalers():
    try:
        with open(SCALER_PATH, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        st.error(f"❌ Could not load scalers: {e}")
        return None


# ── Feature helpers ───────────────────────────────────────────────────────────
def load_and_preprocess(audio_bytes: bytes) -> np.ndarray | None:
    try:
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE,
                            mono=True, duration=DURATION * 2)
        y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])
        target = int(SAMPLE_RATE * DURATION)
        y = np.pad(y, (0, max(0, target - len(y))))[:target]
        return y.astype(np.float32)
    except Exception as e:
        st.error(f"Audio processing error: {e}")
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


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(audio_bytes: bytes, model, scalers: dict) -> dict | None:
    y = load_and_preprocess(audio_bytes)
    if y is None:
        return None

    mfcc_scaler   = scalers["mfcc_scaler"]
    mel_mean      = scalers["mel_mean"]
    mel_std       = scalers["mel_std"]
    label_classes = scalers["label_classes"]
    time_steps    = scalers["time_steps"]

    mfcc = extract_mfcc(y)
    mel  = extract_mel(y)

    if mfcc.shape[0] < time_steps:
        mfcc = np.pad(mfcc, ((0, time_steps - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:time_steps]

    mfcc_n = mfcc_scaler.transform(mfcc).astype(np.float32)
    mel_n  = ((mel - mel_mean) / mel_std).astype(np.float32)

    proba = model.predict(
        {"mel_input": mel_n[np.newaxis], "mfcc_input": mfcc_n[np.newaxis]},
        verbose=0
    )[0]

    idx = int(np.argmax(proba))
    return {
        "emotion":    label_classes[idx],
        "confidence": float(proba[idx]),
        "probs":      {lbl: float(p) for lbl, p in zip(label_classes, proba)},
    }


# ── Probability chart ─────────────────────────────────────────────────────────
def plot_probabilities(probs: dict, top_emotion: str) -> plt.Figure:
    labels  = list(probs.keys())
    values  = [probs[l] * 100 for l in labels]
    colors  = [EMOTION_COLOR.get(l, "#aaa") for l in labels]
    emojis  = [EMOTION_EMOJI.get(l, "🎙️") for l in labels]
    y_labels = [f"{e} {l}" for e, l in zip(emojis, labels)]

    sorted_pairs = sorted(zip(values, y_labels, colors), key=lambda x: x[0])
    values_s, y_labels_s, colors_s = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(y_labels_s, values_s, color=colors_s, edgecolor="white",
                   linewidth=0.6, height=0.65)

    for bar, val in zip(bars, values_s):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=9,
                color="#333")

    ax.set_xlim(0, 110)
    ax.set_xlabel("Probability (%)", fontsize=10)
    ax.set_title("Emotion Probabilities", fontsize=12, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    plt.tight_layout()
    return fig


# ── Waveform plot ─────────────────────────────────────────────────────────────
def plot_waveform(audio_bytes: bytes) -> plt.Figure:
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True,
                        duration=DURATION * 2)
    times = np.linspace(0, len(y) / SAMPLE_RATE, len(y))
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.plot(times, y, color="#3498db", linewidth=0.6, alpha=0.85)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title("Waveform", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════════════════════════════════════

st.title("🎙️ Speech Emotion Recognition")
st.markdown(
    "Upload a short speech clip (2–5 seconds) and the **CNN-BiLSTM** model "
    "will predict the emotion in your voice."
)
st.divider()

# Load model & scalers
model   = load_model()
scalers = load_scalers()

if model is None or scalers is None:
    st.warning(
        "Place `cnn_bilstm_ser_final.keras` and `scaler_data.pkl` "
        "in the same folder as this app, then restart."
    )
    st.stop()

# ── Upload section ────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📂 Upload audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    help="Supported formats: WAV, MP3, OGG, FLAC, M4A",
)

st.markdown("**or**")
recorded = st.audio_input("🎤 Record from microphone")

audio_bytes = None
source_label = ""

if recorded is not None:
    audio_bytes  = recorded.read()
    source_label = "🎤 Microphone recording"
elif uploaded is not None:
    audio_bytes  = uploaded.read()
    source_label = f"📂 {uploaded.name}"

# ── Process & display ─────────────────────────────────────────────────────────
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.caption(source_label)
    with col2:
        run_btn = st.button("🔍 Predict Emotion", type="primary",
                            use_container_width=True)

    if run_btn:
        with st.spinner("Analysing audio…"):
            result = predict(audio_bytes, model, scalers)

        if result:
            emotion    = result["emotion"]
            confidence = result["confidence"]
            emoji      = EMOTION_EMOJI.get(emotion, "🎙️")
            color      = EMOTION_COLOR.get(emotion, "#888")

            # Result card
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {color}22, {color}44);
                    border: 2px solid {color};
                    border-radius: 16px;
                    padding: 24px;
                    text-align: center;
                    margin: 16px 0;
                ">
                    <div style="font-size: 3.5rem;">{emoji}</div>
                    <div style="font-size: 2rem; font-weight: 700;
                                color: {color}; text-transform: uppercase;
                                letter-spacing: 2px;">
                        {emotion}
                    </div>
                    <div style="font-size: 1.1rem; color: #666; margin-top: 6px;">
                        Confidence: <strong>{confidence*100:.1f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Charts
            tab1, tab2 = st.tabs(["📊 All Probabilities", "🌊 Waveform"])
            with tab1:
                fig = plot_probabilities(result["probs"], emotion)
                st.pyplot(fig, use_container_width=True)
            with tab2:
                fig2 = plot_waveform(audio_bytes)
                st.pyplot(fig2, use_container_width=True)

else:
    st.info("👆 Upload a file or record from your microphone to get started.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<small>Model: CNN-BiLSTM · Trained on RAVDESS + TESS · "
    "Emotions: angry · calm · disgust · fearful · happy · neutral · sad · surprised</small>",
    unsafe_allow_html=True,
)
