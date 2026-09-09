import warnings
warnings.filterwarnings("ignore")

import nltk
import joblib
import streamlit as st
import streamlit.components.v1 as components

from English_model import predict_english
from Arabic_model import predict_arabic
from Preprocessing_pipeline import detect_language


# --------------------------------------------------------------------------
# Setup (cached so it only runs once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def ensure_nltk_data():
    for pkg in [
        "punkt", "punkt_tab", "stopwords", "wordnet",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    ]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


@st.cache_resource
def load_language_classifier():
    weights = joblib.load("Language_classifier_weights.pkl")
    return weights["model"], weights["vectorizer"]


ensure_nltk_data()
language_model, language_vectorizer = load_language_classifier()

st.set_page_config(page_title="Wavelength — Sentiment Reader", page_icon="◐", layout="centered")

# --------------------------------------------------------------------------
# Styling — powder purple, one bilingual typeface, one animated moment
# --------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');

:root {
    --canvas: #F1EAFA;
    --canvas-2: #E3D6F5;
    --panel: rgba(255,255,255,0.62);
    --panel-border: rgba(255,255,255,0.9);
    --powder: #C9B6E4;
    --powder-mid: #9B7FC7;
    --deep: #4E3585;
    --ink: #362A52;
    --ink-soft: #6B5C8A;
    --positive: #5E9C7C;
    --positive-bg: #E4F3EA;
    --negative: #C96B78;
    --negative-bg: #FBEAEC;
}

html, body, [class*="css"] { font-family: 'Vazirmatn', sans-serif !important; }

.stApp {
    background:
        radial-gradient(60% 50% at 12% 8%, rgba(201,182,228,0.55), transparent 60%),
        radial-gradient(55% 45% at 92% 18%, rgba(155,127,199,0.35), transparent 60%),
        linear-gradient(180deg, var(--canvas) 0%, var(--canvas-2) 100%);
    background-attachment: fixed;
}

.block-container { padding-top: 3rem; max-width: 680px; }

/* ---- Hero ---- */
.wl-hero { margin-bottom: 2.2rem; }
.wl-eyebrow {
    display: inline-flex; align-items: center; gap: .5rem;
    color: var(--powder-mid); font-weight: 600; font-size: .85rem;
    letter-spacing: .02em;
}
.wl-eyebrow .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--deep);
    animation: wl-pulse 2.4s ease-in-out infinite;
}
@keyframes wl-pulse {
    0%, 100% { opacity: .35; transform: scale(.85); }
    50% { opacity: 1; transform: scale(1.1); }
}
.wl-title {
    font-size: 2.6rem; font-weight: 800; color: var(--ink);
    line-height: 1.08; margin: .5rem 0 .6rem 0; letter-spacing: -0.01em;
}
.wl-title em { font-style: normal; color: var(--deep); }
.wl-sub { color: var(--ink-soft); font-size: 1.02rem; line-height: 1.55; max-width: 46ch; }

/* ---- Input card ---- */
.stTextArea textarea {
    background: var(--panel) !important;
    border: 1.5px solid var(--panel-border) !important;
    border-radius: 18px !important;
    padding: 1.1rem 1.2rem !important;
    color: var(--ink) !important;
    font-size: 1.02rem !important;
    box-shadow: 0 8px 28px rgba(78, 53, 133, 0.08) !important;
    transition: border-color .2s ease, box-shadow .2s ease;
}
.stTextArea textarea:focus {
    border-color: var(--powder-mid) !important;
    box-shadow: 0 0 0 4px rgba(155,127,199,0.18), 0 8px 28px rgba(78,53,133,0.1) !important;
    outline: none !important;
}
.stTextArea label { color: var(--ink) !important; font-weight: 600 !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--powder-mid), var(--deep)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.65rem 1.8rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    box-shadow: 0 10px 24px rgba(78, 53, 133, 0.28) !important;
    transition: transform .15s ease, box-shadow .15s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(78, 53, 133, 0.34) !important;
}
.stButton > button:active { transform: translateY(0); }

div[data-testid="stAlert"] { border-radius: 14px !important; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown("""
<div class="wl-hero">
    <div class="wl-eyebrow"><span class="dot"></span>English &amp; Arabic · reads both</div>
    <div class="wl-title">What does this<br><em>review</em> really say?</div>
    <div class="wl-sub">
        Paste a sentence in English or Arabic. Wavelength detects the language
        on its own and reads the sentiment underneath it — positive or negative,
        with how confident it is.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
text = st.text_area(
    "Your review",
    height=140,
    placeholder="e.g. \"The delivery was late but the food was amazing\" or \"التطبيق بطيء جدا ومزعج\"",
    label_visibility="collapsed",
)

analyze = st.button("Read the sentiment  →")

# --------------------------------------------------------------------------
# Result — rendered as a self-contained HTML/CSS/JS component with an
# animated confidence ring (counts up + sweeps in on every run)
# --------------------------------------------------------------------------
if analyze:
    if not text.strip():
        st.warning("Type something first — the box above is empty.")
    else:
        language = detect_language(text, language_model, language_vectorizer)
        if language == "English":
            sentiment, confidence = predict_english(text)
        else:
            sentiment, confidence = predict_arabic(text)

        pct = round(float(confidence) * 100)
        is_positive = sentiment == "Positive"
        ring_color = "#5E9C7C" if is_positive else "#C96B78"
        ring_bg = "#E4F3EA" if is_positive else "#FBEAEC"
        mood_word = "Positive" if is_positive else "Negative"
        mood_icon = "◠" if is_positive else "◡"
        dir_attr = "rtl" if language != "English" else "ltr"
        lang_label = "Arabic" if language != "English" else "English"

        components.html(f"""
        <div id="wl-result" style="font-family: 'Vazirmatn', sans-serif;">
          <style>
            .wl-card {{
              display: flex; align-items: center; gap: 1.6rem;
              background: rgba(255,255,255,0.75);
              border: 1.5px solid rgba(255,255,255,0.95);
              border-radius: 22px;
              padding: 1.4rem 1.6rem;
              box-shadow: 0 14px 34px rgba(78,53,133,0.14);
              opacity: 0;
              transform: translateY(8px);
              animation: wl-in .5s ease forwards;
            }}
            @keyframes wl-in {{ to {{ opacity: 1; transform: translateY(0); }} }}
            .wl-ring {{
              width: 92px; height: 92px; border-radius: 50%;
              display: flex; align-items: center; justify-content: center;
              background: conic-gradient({ring_color} 0deg, {ring_bg} 0deg);
              flex-shrink: 0;
              transition: background 0.05s linear;
            }}
            .wl-ring-inner {{
              width: 72px; height: 72px; border-radius: 50%;
              background: #fff;
              display: flex; flex-direction: column; align-items: center; justify-content: center;
            }}
            .wl-pct {{ font-weight: 800; font-size: 1.15rem; color: #362A52; }}
            .wl-pct-label {{ font-size: .62rem; color: #6B5C8A; letter-spacing: .04em; }}
            .wl-body {{ flex: 1; }}
            .wl-chips {{ display: flex; gap: .5rem; margin-bottom: .5rem; }}
            .wl-chip {{
              display: inline-flex; align-items: center; gap: .35rem;
              font-size: .74rem; font-weight: 700; padding: .28rem .65rem;
              border-radius: 999px; background: #EFE8FA; color: #4E3585;
            }}
            .wl-chip.mood {{ background: {ring_bg}; color: {ring_color}; }}
            .wl-quote {{
              color: #362A52; font-size: .96rem; line-height: 1.5;
              max-width: 40ch; overflow-wrap: anywhere;
            }}
          </style>
          <div class="wl-card">
            <div class="wl-ring" id="wl-ring">
              <div class="wl-ring-inner">
                <div class="wl-pct" id="wl-pct-text">0%</div>
                <div class="wl-pct-label">CONFIDENT</div>
              </div>
            </div>
            <div class="wl-body">
              <div class="wl-chips">
                <span class="wl-chip">{lang_label}</span>
                <span class="wl-chip mood">{mood_icon} {mood_word}</span>
              </div>
              <div class="wl-quote" dir="{dir_attr}">&ldquo;{text}&rdquo;</div>
            </div>
          </div>
          <script>
            (function() {{
              var target = {pct};
              var ring = document.getElementById('wl-ring');
              var label = document.getElementById('wl-pct-text');
              var color = '{ring_color}';
              var bg = '{ring_bg}';
              var current = 0;
              var start = null;
              var duration = 900;
              function frame(ts) {{
                if (!start) start = ts;
                var progress = Math.min((ts - start) / duration, 1);
                current = Math.round(progress * target);
                var deg = (current / 100) * 360;
                ring.style.background = 'conic-gradient(' + color + ' ' + deg + 'deg, ' + bg + ' 0deg)';
                label.textContent = current + '%';
                if (progress < 1) window.requestAnimationFrame(frame);
              }}
              window.requestAnimationFrame(frame);
            }})();
          </script>
        </div>
        """, height=160)
