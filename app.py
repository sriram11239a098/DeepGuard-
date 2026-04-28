"""
app.py — DeepGuard / Sach-AI  |  Streamlit Web Dashboard

Run with:
    streamlit run app.py
    python main.py ui
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import time
from datetime import datetime

# ── Page config — must be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title          = "DeepGuard / Sach-AI",
    page_icon           = "🛡️",
    layout              = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.dg-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px; padding: 36px 32px 28px;
    margin-bottom: 28px; text-align: center; color: white;
}
.dg-header h1 { font-size: 2.6rem; font-weight: 700; margin: 0 0 6px; }
.dg-header p  { font-size: 1.05rem; opacity: 0.75; margin: 0; }

.verdict-fake {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    border-radius: 12px; padding: 24px; color: white;
    text-align: center; font-size: 1.5rem; font-weight: 700;
    box-shadow: 0 8px 30px rgba(255,65,108,0.4);
}
.verdict-real {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    border-radius: 12px; padding: 24px; color: white;
    text-align: center; font-size: 1.5rem; font-weight: 700;
    box-shadow: 0 8px 30px rgba(17,153,142,0.4);
}
.verdict-unknown {
    background: linear-gradient(135deg, #4e54c8, #8f94fb);
    border-radius: 12px; padding: 24px; color: white;
    text-align: center; font-size: 1.5rem; font-weight: 700;
}
.score-card {
    background: #1e1e2e; border-radius: 10px;
    padding: 16px 20px; margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}
.score-label { font-size: 0.85rem; color: #a0a0b8; margin-bottom: 6px; }
.score-value { font-size: 1.4rem; font-weight: 600; color: white; }
.upload-hint { text-align: center; color: #6b7280; font-size: 0.9rem; margin-top: 8px; }

/* Timer & Performance Styles */
.timer-card {
    background: rgba(15, 12, 41, 0.6);
    border: 1px solid rgba(78, 84, 200, 0.3);
    border-radius: 12px; padding: 12px 16px;
    margin-bottom: 20px;
}
.timer-label { font-size: 0.75rem; color: #8f94fb; text-transform: uppercase; letter-spacing: 1px; }
.timer-value { font-size: 1.6rem; font-weight: 700; color: #ffffff; font-family: 'Courier New', monospace; }

footer { visibility: hidden; }
[data-testid="stMetricValue"] { font-size: 2.2rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

import config

# ── Lazy-load inference engine ────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading AI models…")
def load_engine():
    from utils.inference import DeepGuardInference
    return DeepGuardInference(load_pretrained_weights=True)


SUPPORTED_TYPES = {
    "video": ["mp4", "avi", "mov", "mkv", "webm"],
    "audio": ["wav", "mp3", "flac", "ogg", "m4a"],
    "image": ["jpg", "jpeg", "png", "bmp"],
}
ALL_TYPES = sum(SUPPORTED_TYPES.values(), [])


def render_score_bar(label: str, emoji: str, score, threshold: float = 0.5, extra: str = None):
    if score is None:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">{emoji} {label}</div>
            <div class="score-value" style="color:#4b5563;">Not analysed</div>
        </div>""", unsafe_allow_html=True)
        return
    pct    = int(score * 100)
    colour = "#ff416c" if score >= threshold else "#38ef7d"
    
    extra_html = f'<div style="font-size:0.75rem; color:#a0a0b8; margin-top:8px;">🆔 {extra}</div>' if extra else ""
    
    st.markdown(f"""
    <div class="score-card">
        <div class="score-label">{emoji} {label}</div>
        <div class="score-value" style="color:{colour};">
            {pct}%&nbsp;<span style="font-size:0.8rem;color:#a0a0b8;">probability fake</span>
        </div>
        {extra_html}
    </div>""", unsafe_allow_html=True)
    st.progress(pct)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dg-header">
    <h1>🛡️ DeepGuard / Sach-AI</h1>
    <p>Multimodal Deepfake Detection — Video · Audio · Image</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Session Timer Fragment
    @st.fragment(run_every=1)
    def render_live_timer():
        uptime = int(time.time() - st.session_state.start_time)
        hours, remainder = divmod(uptime, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        st.markdown(f"""
        <div class="timer-card">
            <div class="timer-label">⏱️ Session Uptime</div>
            <div class="timer-value">{hours:02d}:{minutes:02d}:{seconds:02d}</div>
            <div style="font-size:0.7rem; color:#6b7280; margin-top:4px;">
                System Clock: {datetime.now().strftime('%H:%M:%S')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_live_timer()

    st.markdown("---")
    device_icon = "🟢" if "cuda" in config.DEVICE else "🟡"
    st.markdown(f"**Device:** {device_icon} `{config.DEVICE.upper()}`")
    st.markdown(f"**Threshold:** `{config.DETECTION_THRESHOLD}`")

    st.markdown("**Fusion Weights:**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("🎬 Video", f"{int(config.FUSION_WEIGHT_VIDEO * 100)}%")
        st.metric("🎵 Audio", f"{int(config.FUSION_WEIGHT_AUDIO * 100)}%")
    with col_b:
        st.metric("🖼️ Image", f"{int(config.FUSION_WEIGHT_IMAGE * 100)}%")

    st.markdown("---")
    st.markdown("**Dataset Paths:**")
    for name, path in [
        ("Video", config.DATASET_VIDEO_DIR),
        ("Audio", config.DATASET_AUDIO_DIR),
        ("Image", config.DATASET_IMAGE_DIR),
    ]:
        icon = "✅" if Path(path).exists() else "❌"
        st.markdown(f"{icon} `{name}`: `{Path(path).name}`")

    st.markdown("---")
    st.caption("DeepGuard / Sach-AI  ·  CSE CIP 2026")

# ── Upload UI ─────────────────────────────────────────────────────────────────
col_upload, col_info = st.columns([3, 2], gap="large")

with col_upload:
    st.markdown("### 📤 Upload Media")
    uploaded = st.file_uploader(
        "Drop your file here",
        type            = ALL_TYPES,
        label_visibility= "collapsed",
        help            = "Supported: MP4, AVI, MOV, WAV, MP3, FLAC, JPG, PNG …",
    )
    st.markdown("""<div class="upload-hint">
        Supported: Video (MP4/AVI/MOV) · Audio (WAV/MP3/FLAC) · Image (JPG/PNG)
    </div>""", unsafe_allow_html=True)

with col_info:
    st.markdown("### 🔍 How it works")
    st.markdown("""
1. **Upload** any media file
2. Models analyse all modalities
3. Fusion layer combines scores
4. Get a verdict with confidence

| Branch | Model |
|--------|-------|
| 🎬 Video | ResNext50 + Stacked LSTM |
| 🎵 Audio | Custom CNN on Mel-Spec |
| 🖼️ Image | DenseNet121 |
""")

# ── Analysis ──────────────────────────────────────────────────────────────────
if uploaded is not None:
    ext = Path(uploaded.name).suffix.lower().lstrip(".")

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.markdown("---")
    preview_col, result_col = st.columns([1, 1], gap="large")

    with preview_col:
        st.markdown("### 👁️ Preview")
        if ext in SUPPORTED_TYPES["image"]:
            st.image(tmp_path, caption=uploaded.name, use_container_width=True)
        elif ext in SUPPORTED_TYPES["video"]:
            st.video(tmp_path)
        elif ext in SUPPORTED_TYPES["audio"]:
            st.audio(tmp_path)
        st.caption(f"📄 `{uploaded.name}`  ·  {uploaded.size / 1024:.1f} KB")

    with result_col:
        st.markdown("### 🤖 Detection Result")
        
        # Start timing immediately upon entering the analysis block
        if 'analysis_start' not in st.session_state or uploaded.name != st.session_state.get('last_file'):
            st.session_state.analysis_start = time.time()
            st.session_state.last_file = uploaded.name

        start_ts = datetime.fromtimestamp(st.session_state.analysis_start).strftime('%H:%M:%S.%f')[:-3]
        
        status_placeholder = st.empty()
        status_placeholder.markdown(f"""
        <div class="timer-card" style="border-color: #4e54c8;">
            <div class="timer-label">📡 Processing Monitor</div>
            <div style="font-size:0.85rem; color:#a0a0b8; margin-top:8px;">
                Started at: <span style="color:white; font-family:monospace;">{start_ts}</span><br>
                Loopback: <span style="color:#8f94fb; font-family:monospace;">{config.LOOPBACK_ADDR}</span><br>
                Status: <span style="color:#f1c40f;">Analysing Modalities...</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🔍 Analysing… please wait"):
            try:
                engine = load_engine()
                result = engine.predict(tmp_path)
                # Ensure we use the actual engine internal timer for precision
                elapsed = result.elapsed_seconds 
                
                # Calculate Loopback Time (Total UI time - Engine internal time)
                total_ui_time = time.time() - st.session_state.analysis_start
                loopback_time = max(0.001, total_ui_time - elapsed)
            except Exception as exc:
                st.error(f"❌ Analysis failed: {exc}")
                os.unlink(tmp_path)
                st.stop()

        status_placeholder.markdown(f"""
        <div class="timer-card" style="border-color: #38ef7d;">
            <div class="timer-label">✅ Analysis Complete</div>
            <div class="timer-value" style="color: #38ef7d;">{elapsed:.3f}s</div>
            <div style="font-size:0.75rem; color:#a0a0b8; margin-top:6px;">
                Total Processing Time (Inference + Post-proc)<br>
                <span style="color:#38ef7d; font-weight:600;">Loopback Time: {loopback_time:.3f}s</span> ({config.LOOPBACK_ADDR})
            </div>
        </div>
        """, unsafe_allow_html=True)

        if result.verdict == "DEEPFAKE":
            st.markdown('<div class="verdict-fake">🚨 DEEPFAKE DETECTED</div>', unsafe_allow_html=True)
        elif result.verdict == "AUTHENTIC":
            st.markdown('<div class="verdict-real">✅ AUTHENTIC</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict-unknown">❓ UNKNOWN</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(
            label       = "Overall Fake Probability",
            value       = f"{result.confidence_pct:.1f}%",
            delta       = (
                f"{'⚠ Above' if result.confidence_pct >= 50 else '✓ Below'} "
                f"{int(config.DETECTION_THRESHOLD * 100)}% threshold"
            ),
            delta_color = "inverse",
        )

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-modality scores ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Per-Modality Scores")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        render_score_bar("Video Branch", "🎬", result.video.score)
    with sc2:
        sig = result.audio_sig if result.audio_sig != "N/A" else None
        render_score_bar("Audio Branch", "🎵", result.audio.score, extra=sig)
    with sc3:
        render_score_bar("Image Branch", "🖼️", result.image.score)

    # ── Forensic details ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧬 Forensic Details")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        st.info(f"**Start Time:** `{result.start_time}`")
        st.info(f"**End Time:**   `{result.end_time}`")
        st.info(f"**⏱️ Timer:** `{result.elapsed_seconds:.2f} seconds`")
        st.info(f"**🔄 Loopback:** `{loopback_time:.3f} s` ({config.LOOPBACK_ADDR})")
    with fcol2:
        st.success(f"**Audio Signature:** `{result.audio_sig}`")
        st.success(f"**Device:** `{result.model_data.get('device', 'N/A').upper()}`")
    with fcol3:
        st.warning(f"**Gist Summary:**\n\n{result.gist}")

    # ── Heatmap Display ───────────────────────────────────────────────────────
    if result.heatmap_path and os.path.exists(result.heatmap_path):
        st.markdown("---")
        h_col1, h_col2 = st.columns([2, 1])
        with h_col1:
            st.markdown("### 🗺️ Forensic Heatmap")
            label = "Grad-CAM (Image Manipulation Localization)" if "heatmap" in result.heatmap_path else "EVM Temporal Variation Map"
            st.image(result.heatmap_path, caption=label, use_container_width=True)
        with h_col2:
            st.markdown("### 💡 Explanation")
            if "heatmap" in result.heatmap_path:
                st.write("""
                    The **Grad-CAM** heatmap highlights the specific facial regions where the model detected 
                    synthetic artifacts. Red/Yellow areas indicate high contribution to the 'DEEPFAKE' verdict.
                """)
            else:
                st.write("""
                    The **EVM Map** (Eulerian Video Magnification) visualizes temporal color variations. 
                    Deepfakes often show unnatural flickering or static regions in these frequency bands.
                """)

    # ── Full JSON report ──────────────────────────────────────────────────────
    with st.expander("📋 Full JSON Report"):
        st.json(result.to_dict())

    # Cleanup temp file
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
