"""
Speech Emotion Recognition — Streamlit Deployment App
======================================================
Usage (local):
    pip install streamlit tensorflow librosa Pillow scipy scikit-learn soundfile
    streamlit run streamlit_app.py

Streamlit Cloud:
    1. Push this file + cnn_bilstm_ser_final.keras + scaler_data.pkl to a GitHub repo
    2. Go to https://share.streamlit.io → New app → connect your repo
    3. Set main file to streamlit_app.py

Files expected in the same directory:
    cnn_bilstm_ser_final.keras   ← trained model (from Kaggle output)
    scaler_data.pkl              ← scalers + label classes (from Kaggle output)
"""

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
    page_title="VoiceSense · Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered",
)

# ── Global CSS (only Streamlit widget overrides — everything else is inline) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=DM+Sans:wght@300;400;500&display=swap');
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 0; }
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #553c9a) !important;
    color: #fff !important; border: none !important; border-radius: 12px !important;
    font-weight: 700 !important; font-size: 1.0rem !important;
    letter-spacing: 1px !important; padding: 14px 28px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(66,153,225,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(66,153,225,0.5) !important;
    background: linear-gradient(135deg, #3182ce, #6b46c1) !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.85rem !important; color: rgba(255,255,255,0.45) !important;
    padding: 10px 18px !important; border-radius: 8px 8px 0 0 !important;
    background: transparent !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #63b3ed !important; background: rgba(99,179,237,0.08) !important;
    border-bottom: 2px solid #63b3ed !important;
}
</style>
""", unsafe_allow_html=True)

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

# ── Hero Banner (all inline styles — Streamlit-safe) ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=DM+Sans:wght@300;400&display=swap');
@keyframes pulse-bg { from{transform:scale(1);opacity:.7} to{transform:scale(1.1);opacity:1} }
@keyframes shimmer  { to{background-position:200% center} }
@keyframes wave-dance { from{transform:scaleY(0.3)} to{transform:scaleY(1)} }
@keyframes card-in { from{opacity:0;transform:scale(.85) translateY(20px)} to{opacity:1;transform:scale(1) translateY(0)} }
.ser-wave-bar {
    width:5px;border-radius:3px;display:inline-block;
    background:linear-gradient(to top,#4299e1,#9f7aea);
    animation:wave-dance 1.3s ease-in-out infinite alternate;
}
.ser-hero-title-gradient {
    background:linear-gradient(90deg,#63b3ed,#9f7aea,#63b3ed);
    background-size:200% auto;
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    animation:shimmer 3s linear infinite;
}
.ser-result-card { animation:card-in 0.5s cubic-bezier(0.34,1.56,0.64,1) both; }
</style>

<div style="background:linear-gradient(135deg,#0d0d1a 0%,#0f1f3d 50%,#0d0d1a 100%);border-radius:24px;padding:48px 32px 40px;text-align:center;position:relative;overflow:hidden;margin-bottom:28px;border:1px solid rgba(99,179,237,0.18);box-shadow:0 0 60px rgba(66,153,225,0.12);">

  <div style="position:absolute;top:-60%;left:-20%;width:140%;height:200%;background:radial-gradient(ellipse at center,rgba(66,153,225,0.10) 0%,rgba(159,122,234,0.07) 40%,transparent 70%);animation:pulse-bg 6s ease-in-out infinite alternate;pointer-events:none;"></div>

  <div style="display:flex;justify-content:center;align-items:flex-end;gap:5px;height:38px;margin-bottom:18px;">
    <span class="ser-wave-bar" style="height:14px;animation-delay:0.0s;"></span>
    <span class="ser-wave-bar" style="height:26px;animation-delay:0.1s;"></span>
    <span class="ser-wave-bar" style="height:34px;animation-delay:0.2s;"></span>
    <span class="ser-wave-bar" style="height:22px;animation-delay:0.3s;"></span>
    <span class="ser-wave-bar" style="height:38px;animation-delay:0.4s;"></span>
    <span class="ser-wave-bar" style="height:28px;animation-delay:0.3s;"></span>
    <span class="ser-wave-bar" style="height:34px;animation-delay:0.2s;"></span>
    <span class="ser-wave-bar" style="height:20px;animation-delay:0.1s;"></span>
    <span class="ser-wave-bar" style="height:14px;animation-delay:0.0s;"></span>
  </div>

  <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:2.6rem;letter-spacing:-1px;color:#fff;margin:0 0 8px;line-height:1.1;">
    Voice<span class="ser-hero-title-gradient">Sense</span>
  </div>
  <div style="font-size:1.0rem;color:rgba(255,255,255,0.52);font-weight:300;letter-spacing:0.5px;">
    CNN-BiLSTM &middot; Speech Emotion Recognition<br>
    <span style="font-size:0.82rem;opacity:0.75;">Upload a 2&ndash;5 second clip &bull; Get instant emotion insights from your voice</span>
  </div>
</div>
""", unsafe_allow_html=True)

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
st.markdown('<div style="font-family:sans-serif;font-size:0.78rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:#63b3ed;margin-bottom:10px;">&#128193; Upload Audio File</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop your file here",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    help="Supported: WAV, MP3, OGG, FLAC, M4A — ideal length 2–5 seconds",
    label_visibility="collapsed",
)

st.markdown(
    '<div style="text-align:center;color:rgba(255,255,255,0.28);'
    'font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;'
    'margin:12px 0;">— or —</div>',
    unsafe_allow_html=True
)
st.markdown('<div style="font-family:sans-serif;font-size:0.78rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:#63b3ed;margin-bottom:10px;">&#127908; Record from Microphone</div>', unsafe_allow_html=True)
recorded = st.audio_input("Record", label_visibility="collapsed")

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
                <div class="ser-result-card" style="
                    border-radius:20px;padding:36px 24px;text-align:center;
                    margin:20px 0;position:relative;overflow:hidden;
                    background:linear-gradient(135deg,{color}18,{color}30);
                    border:2px solid {color}88;
                    box-shadow:0 0 40px {color}22,inset 0 0 30px {color}0a;
                ">
                    <div style="font-size:4rem;line-height:1;margin-bottom:10px;">{emoji}</div>
                    <div style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;
                                text-transform:uppercase;letter-spacing:3px;
                                margin-bottom:6px;color:{color};">{emotion}</div>
                    <div style="font-size:1.0rem;opacity:0.72;font-weight:300;">
                        Confidence: <strong style="font-weight:700;">{confidence*100:.1f}%</strong>
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
    st.markdown(
        """
        <div style="
            background: rgba(99,179,237,0.07);
            border: 1.5px dashed rgba(99,179,237,0.3);
            border-radius: 16px;
            padding: 28px;
            text-align: center;
            color: rgba(255,255,255,0.45);
            font-size: 0.95rem;
            letter-spacing: 0.3px;
        ">
            🎙️ &nbsp; Upload a file or record from your microphone to get started
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Footer (all inline styles — Streamlit-safe) ───────────────────────────────
st.markdown("""
<div style="margin-top:60px;padding:40px 24px 32px;background:linear-gradient(135deg,#0d0d1a,#0f1f3d);border-radius:20px 20px 0 0;border-top:1px solid rgba(99,179,237,0.2);text-align:center;position:relative;overflow:hidden;">

  <div style="position:absolute;top:0;left:50%;transform:translateX(-50%);width:60%;height:1px;background:linear-gradient(90deg,transparent,#63b3ed,#9f7aea,transparent);"></div>

  <div style="font-size:2rem;margin-bottom:6px;opacity:0.75;">&#127897;</div>

  <div style="font-size:0.72rem;letter-spacing:3px;text-transform:uppercase;color:rgba(255,255,255,0.28);margin-bottom:22px;">
    Developed with passion by
  </div>

  <div style="display:flex;justify-content:center;flex-wrap:wrap;gap:10px;margin-bottom:24px;">
    <span style="font-weight:700;font-size:0.88rem;letter-spacing:1.5px;text-transform:uppercase;padding:9px 22px;border-radius:999px;color:#63b3ed;border:1.5px solid rgba(99,179,237,0.5);background:rgba(99,179,237,0.10);display:inline-block;">
      &#10022; FATIMA
    </span>
    <span style="font-weight:700;font-size:0.88rem;letter-spacing:1.5px;text-transform:uppercase;padding:9px 22px;border-radius:999px;color:#b794f4;border:1.5px solid rgba(183,148,244,0.5);background:rgba(183,148,244,0.10);display:inline-block;">
      &#10022; MUSA
    </span>
    <span style="font-weight:700;font-size:0.88rem;letter-spacing:1.5px;text-transform:uppercase;padding:9px 22px;border-radius:999px;color:#68d391;border:1.5px solid rgba(104,211,145,0.5);background:rgba(104,211,145,0.10);display:inline-block;">
      &#10022; USMAN
    </span>
  </div>

  <div style="display:flex;justify-content:center;flex-wrap:wrap;gap:6px;margin-bottom:22px;">
    <span style="padding:4px 11px;border-radius:999px;font-size:0.7rem;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);color:rgba(255,255,255,0.45);">&#128096; angry</span>
    <span style="padding:4px 11px;border-radius:999px;font-size:0.7rem;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);color:rgba(255,255,255,0.45);">&#128524; calm</span>
    <span style="padding:4px 11px;border-radius:999px;font-size:0.7rem;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);color:rgba(255,255,255,0.45);">&#129314; disgust</span>
    <span style="padding:4px 11px;border-radius:999px;font-size:0.7rem;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);color:rgba(255,255,255,0.45);">&#128552; fearful</span>
    <span style="padding:4px 11px;border-radius:999px;font-size:0.7rem;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);color:rgba(255,255,255,0.45);">&#128516; happy</span>
    <span style="padding:4px 11px;border-radius:999px;font-size:0.7rem;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);color:rgba(255,255,255,0.45);">&#128528; neutral</span>
    <span style="padding:4px 11px;border-radius:999px;font-size:0.7rem;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);color:rgba(255,255,255,0.45);">&#128546; sad</span>
    <span style="padding:4px 11px;border-radius:999px;font-size:0.7rem;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);color:rgba(255,255,255,0.45);">&#128562; surprised</span>
  </div>

  <div style="width:40px;height:1px;background:rgba(255,255,255,0.12);margin:0 auto 16px;"></div>

  <div style="font-size:0.72rem;color:rgba(255,255,255,0.22);letter-spacing:0.5px;line-height:2.0;">
    Model: CNN-BiLSTM &nbsp;&bull;&nbsp; Trained on RAVDESS + TESS<br>
    Built with <span style="color:#fc8181;">&#9829;</span> using Streamlit &amp; TensorFlow
  </div>

</div>
""", unsafe_allow_html=True)
