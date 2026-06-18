import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from collections import defaultdict
import os
import pickle
import tempfile
import csv

# ─── CORE FUNCTIONS ───────────────────────────────────────────
def compute_spectrogram(audio_path, n_fft=4096, hop_length=512):
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db, sr, hop_length

def get_peaks(S_db, neighborhood=20, threshold_db=-60):
    local_max = maximum_filter(S_db, size=neighborhood) == S_db
    peaks = np.argwhere(local_max & (S_db > threshold_db))
    return peaks

def hash_peaks(peaks, fan_out=15, time_delta_max=200):
    peaks_sorted = sorted(peaks.tolist(), key=lambda x: x[1])
    hashes = []
    for i, (f1, t1) in enumerate(peaks_sorted):
        for j in range(1, fan_out + 1):
            if i + j >= len(peaks_sorted):
                break
            f2, t2 = peaks_sorted[i + j]
            dt = t2 - t1
            if dt <= 0 or dt > time_delta_max:
                continue
            h = hash((int(f1), int(f2), int(dt)))
            hashes.append((h, t1))
    return hashes

def build_database(songs_folder, progress_bar=None):
    db = defaultdict(list)
    files = [f for f in sorted(os.listdir(songs_folder)) if f.endswith('.mp3')]
    for i, fname in enumerate(files):
        song_name = os.path.splitext(fname)[0]
        path = os.path.join(songs_folder, fname)
        try:
            S_db, sr, hop = compute_spectrogram(path)
            peaks = get_peaks(S_db)
            hashes = hash_peaks(peaks)
            for (h, t) in hashes:
                db[h].append((song_name, t))
        except Exception as e:
            pass
        if progress_bar:
            progress_bar.progress((i+1)/len(files), text=f"✨ Indexing: {song_name}")
    return dict(db)

def identify_song(query_path, db):
    S_db, sr, hop = compute_spectrogram(query_path)
    peaks = get_peaks(S_db)
    hashes = hash_peaks(peaks)
    scores = defaultdict(list)
    for (h, t_query) in hashes:
        if h in db:
            for (song_name, t_db) in db[h]:
                offset = t_db - t_query
                scores[song_name].append(offset)
    best_song = None
    best_count = 0
    best_offsets = []
    for song, offsets in scores.items():
        counts = defaultdict(int)
        for o in offsets:
            counts[o] += 1
        top = max(counts.values())
        if top > best_count:
            best_count = top
            best_song = song
            best_offsets = offsets
    return best_song, best_count, scores, S_db, peaks, best_offsets

# ─── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(page_title="Zapptain America 🎵", layout="wide", page_icon="🎵")

# ─── STYLING ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* Background */
.stApp {
    background: linear-gradient(160deg, #0d0b1a 0%, #130d2b 40%, #1a0f2e 70%, #0d0b1a 100%);
    background-attachment: fixed;
}

/* Floating stars effect */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image: 
        radial-gradient(1px 1px at 20% 30%, rgba(200,180,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 80% 10%, rgba(180,160,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 50% 80%, rgba(220,200,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 10% 60%, rgba(200,180,255,0.2) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 50%, rgba(180,160,255,0.2) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* Main title */
h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-style: italic !important;
    font-size: 3.5rem !important;
    background: linear-gradient(135deg, #c8b4ff, #e0d0ff, #a990ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    letter-spacing: 2px;
    margin-bottom: 0 !important;
}

/* Subtitles */
h4, h3, h2 {
    font-family: 'DM Sans', sans-serif !important;
    color: #b8a0e8 !important;
    font-weight: 300 !important;
    letter-spacing: 1px;
}

/* All text */
p, label, .stMarkdown {
    font-family: 'DM Sans', sans-serif !important;
    color: #d4c8f0 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c5cbf, #9b7fd4, #7c5cbf) !important;
    background-size: 200% auto !important;
    color: #f0eaff !important;
    border: 1px solid rgba(200,180,255,0.3) !important;
    border-radius: 30px !important;
    padding: 0.6rem 2.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    letter-spacing: 1.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(124,92,191,0.4), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background-position: right center !important;
    box-shadow: 0 6px 30px rgba(155,127,212,0.6), 0 0 20px rgba(155,127,212,0.3), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    transform: translateY(-2px) !important;
    border-color: rgba(200,180,255,0.6) !important;
}

/* Text inputs */
.stTextInput > div > div > input {
    background: rgba(124,92,191,0.1) !important;
    border: 1px solid rgba(200,180,255,0.3) !important;
    border-radius: 12px !important;
    color: #e0d0ff !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.3s ease !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(200,180,255,0.7) !important;
    box-shadow: 0 0 15px rgba(155,127,212,0.3) !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(155,127,212,0.5) !important;
    border-radius: 20px !important;
    background: rgba(124,92,191,0.05) !important;
    padding: 1.5rem !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(200,180,255,0.8) !important;
    background: rgba(124,92,191,0.1) !important;
    box-shadow: 0 0 25px rgba(155,127,212,0.2) !important;
}

/* Success box */
div[data-testid="stAlert"] {
    background: rgba(124,92,191,0.15) !important;
    border: 1px solid rgba(200,180,255,0.3) !important;
    border-radius: 15px !important;
    backdrop-filter: blur(10px) !important;
}

/* Radio buttons */
.stRadio > div {
    background: rgba(124,92,191,0.1) !important;
    border-radius: 15px !important;
    padding: 0.5rem 1rem !important;
    border: 1px solid rgba(200,180,255,0.2) !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #7c5cbf, #c8b4ff) !important;
    border-radius: 10px !important;
}

/* Divider */
hr {
    border-color: rgba(155,127,212,0.3) !important;
    margin: 1.5rem 0 !important;
}

/* Table */
.stTable {
    background: rgba(124,92,191,0.1) !important;
    border-radius: 15px !important;
}

/* Sidebar if any */
.css-1d391kg {
    background: rgba(13,11,26,0.9) !important;
}

/* Download button */
.stDownloadButton > button {
    background: rgba(124,92,191,0.2) !important;
    border: 1px solid rgba(200,180,255,0.4) !important;
    border-radius: 20px !important;
    color: #c8b4ff !important;
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.title("𝄞 Zapptain America")
st.markdown("#### *whisper a melody — the stars will name it for you*")
st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ─── DATABASE ─────────────────────────────────────────────────
if 'db' not in st.session_state:
    st.session_state.db = None

st.markdown("### 🔮 &nbsp; Index Your Library")
col1, col2 = st.columns([3, 1])
with col1:
    songs_folder = st.text_input(
        "Songs folder path",
        value=os.path.expanduser("~/EE200_Q3/songs"),
        label_visibility="collapsed",
        placeholder="Path to your songs folder..."
    )
with col2:
    build_btn = st.button("✦ Build Database")

if build_btn:
    if os.path.exists(songs_folder):
        pb = st.progress(0, text="Starting...")
        st.session_state.db = build_database(songs_folder, pb)
        pb.empty()
        st.success(f"✦ Database ready — {len(st.session_state.db):,} audio fingerprints indexed")
    else:
        st.error("✦ Folder not found — check the path")

if st.session_state.db:
    st.success(f"✦ {len(st.session_state.db):,} fingerprints loaded and ready")
    st.divider()

    # ─── MODE ─────────────────────────────────────────────────
    st.markdown("### 🎧 &nbsp; Identify a Song")
    mode = st.radio("", ["✦ Single Clip", "✦ Batch Mode"], horizontal=True, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    if mode == "✦ Single Clip":
        uploaded = st.file_uploader("Drop your audio clip here", type=["mp3"], label_visibility="collapsed")

        if uploaded:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner("🔮 Reading the frequencies..."):
                matched, count, scores, S_db, peaks, offsets = identify_song(tmp_path, st.session_state.db)

            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, rgba(124,92,191,0.2), rgba(155,127,212,0.1));
                border: 1px solid rgba(200,180,255,0.4);
                border-radius: 20px;
                padding: 1.5rem 2rem;
                margin: 1rem 0;
                text-align: center;
            '>
                <p style='color: #b8a0e8; font-size: 0.9rem; margin:0; letter-spacing:2px; text-transform:uppercase;'>identified as</p>
                <h2 style='color: #e0d0ff !important; font-size: 2rem; margin: 0.3rem 0; font-family: Cormorant Garamond, serif !important; font-style: italic !important;'>{matched}</h2>
                <p style='color: #9b7fd4; font-size: 0.85rem; margin:0;'>confidence score: {count:,}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)

            plot_style = {
                'facecolor': '#0d0b1a',
                'title_color': '#c8b4ff',
                'label_color': '#9b7fd4',
                'tick_color': '#7c6ba0'
            }

            with col1:
                st.markdown("**✦ Spectrogram**")
                fig, ax = plt.subplots(figsize=(6, 4), facecolor=plot_style['facecolor'])
                ax.set_facecolor('#0d0b1a')
                ax.imshow(S_db, origin='lower', aspect='auto', cmap='twilight')
                ax.set_title('frequency over time', color=plot_style['title_color'], fontsize=10, pad=10)
                ax.set_xlabel('time bins', color=plot_style['label_color'], fontsize=9)
                ax.set_ylabel('frequency bins', color=plot_style['label_color'], fontsize=9)
                ax.tick_params(colors=plot_style['tick_color'])
                for spine in ax.spines.values():
                    spine.set_edgecolor(rgba := 'rgba(124,92,191,0.3)')
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("**✦ Constellation Map**")
                fig, ax = plt.subplots(figsize=(6, 4), facecolor=plot_style['facecolor'])
                ax.set_facecolor('#0d0b1a')
                if len(peaks) > 0:
                    ax.scatter(peaks[:, 1], peaks[:, 0], s=0.5, c='#c8b4ff', alpha=0.4)
                ax.set_title('peak fingerprints', color=plot_style['title_color'], fontsize=10, pad=10)
                ax.set_xlabel('time bins', color=plot_style['label_color'], fontsize=9)
                ax.set_ylabel('frequency bins', color=plot_style['label_color'], fontsize=9)
                ax.tick_params(colors=plot_style['tick_color'])
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col3:
                st.markdown("**✦ Offset Histogram**")
                fig, ax = plt.subplots(figsize=(6, 4), facecolor=plot_style['facecolor'])
                ax.set_facecolor('#0d0b1a')
                if offsets:
                    ax.hist(offsets, bins=100, color='#9b7fd4', edgecolor='#0d0b1a', alpha=0.8)
                ax.set_title('time alignment', color=plot_style['title_color'], fontsize=10, pad=10)
                ax.set_xlabel('offset', color=plot_style['label_color'], fontsize=9)
                ax.set_ylabel('matches', color=plot_style['label_color'], fontsize=9)
                ax.tick_params(colors=plot_style['tick_color'])
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

    else:
        uploaded_files = st.file_uploader(
            "Drop multiple audio clips here",
            type=["mp3"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            st.markdown(f"*{len(uploaded_files)} files ready*")
            if st.button("✦ Identify All"):
                results = []
                pb = st.progress(0, text="Identifying...")
                for i, f in enumerate(uploaded_files):
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    matched, _, _, _, _, _ = identify_song(tmp_path, st.session_state.db)
                    results.append({"filename": f.name, "prediction": matched})
                    pb.progress((i+1)/len(uploaded_files), text=f"✦ {f.name} → {matched}")
                pb.empty()
                st.markdown("<br>", unsafe_allow_html=True)
                st.table(results)
                csv_lines = "filename,prediction\n" + "\n".join(
                    f"{r['filename']},{r['prediction']}" for r in results
                )
                st.download_button("✦ Download results.csv", csv_lines, "results.csv", mime="text/csv")

else:
    st.markdown("""
    <div style='
        text-align: center;
        padding: 3rem;
        border: 1px dashed rgba(155,127,212,0.3);
        border-radius: 20px;
        margin: 2rem 0;
    '>
        <p style='color: #7c6ba0; font-size: 1.1rem; letter-spacing: 1px;'>
            ✦ &nbsp; enter your songs path above and build the database to begin &nbsp; ✦
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color: #4a3d6b; font-size: 0.8rem; letter-spacing: 2px;'>
    ✦ &nbsp; EE200 · Signals, Systems & Networks · Zapptain America &nbsp; ✦
</p>
""", unsafe_allow_html=True)