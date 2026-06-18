import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from collections import defaultdict
import os
import tempfile

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
        except:
            pass
        if progress_bar:
            progress_bar.progress((i+1)/len(files), text=f"INDEXING: {song_name.upper()}")
    return dict(db)

def identify_song(query_path, db):
    S_db, sr, hop = compute_spectrogram(query_path)
    peaks = get_peaks(S_db)
    hashes = hash_peaks(peaks)
    scores = defaultdict(list)
    for (h, t_query) in hashes:
        if h in db:
            for (song_name, t_db) in db[h]:
                scores[song_name].append(t_db - t_query)
    best_song, best_count, best_offsets = None, 0, []
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
st.set_page_config(
    page_title="ZAPPTAIN AMERICA",
    layout="wide",
    page_icon="⬡",
    initial_sidebar_state="expanded"
)

# ─── MASTER CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500&display=swap');

/* ── RESET & BASE ── */
* { box-sizing: border-box; }

.stApp {
    background: #080808;
    font-family: 'Inter', sans-serif;
}

/* Animated background grid */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(180,0,0,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(180,0,0,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

/* Red glow orb top right */
.stApp::after {
    content: '';
    position: fixed;
    top: -200px;
    right: -200px;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(180,0,0,0.15) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d0d 0%, #111111 100%) !important;
    border-right: 1px solid rgba(180,0,0,0.3) !important;
    box-shadow: 4px 0 30px rgba(180,0,0,0.1) !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

/* ── TYPOGRAPHY ── */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 3px !important;
}

h1 {
    font-size: 2.2rem !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, #ff0000, #cc0000, #ff4444, #cc0000);
    background-size: 300% auto;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    animation: shimmer 4s linear infinite;
    text-transform: uppercase;
    margin-bottom: 0 !important;
}

@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 300% center; }
}

h2 {
    color: #cc0000 !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 4px !important;
}

h3 {
    color: #888 !important;
    font-size: 0.75rem !important;
    font-weight: 400 !important;
    text-transform: uppercase;
    letter-spacing: 3px !important;
}

p, label, .stMarkdown p {
    font-family: 'Inter', sans-serif !important;
    color: #aaa !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(180,0,0,0.6) !important;
    border-radius: 2px !important;
    color: #ff3333 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    padding: 0.7rem 2rem !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button::before {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(135deg, rgba(180,0,0,0) 0%, rgba(180,0,0,0.15) 100%) !important;
    opacity: 0 !important;
    transition: opacity 0.3s ease !important;
}

.stButton > button:hover {
    border-color: #ff0000 !important;
    color: #ffffff !important;
    box-shadow: 0 0 20px rgba(180,0,0,0.5), inset 0 0 20px rgba(180,0,0,0.1) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:hover::before {
    opacity: 1 !important;
}

/* ── TEXT INPUT ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(180,0,0,0.3) !important;
    border-radius: 2px !important;
    color: #e0e0e0 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    letter-spacing: 1px !important;
    padding: 0.7rem 1rem !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus {
    border-color: #cc0000 !important;
    box-shadow: 0 0 15px rgba(180,0,0,0.3), 0 0 0 1px rgba(180,0,0,0.2) !important;
    background: rgba(180,0,0,0.03) !important;
}

/* ── FILE UPLOADER ── */
div[data-testid="stFileUploader"] {
    border: 1px solid rgba(180,0,0,0.3) !important;
    border-radius: 2px !important;
    background: rgba(180,0,0,0.02) !important;
    padding: 2rem !important;
    transition: all 0.3s ease !important;
    position: relative !important;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #cc0000 !important;
    background: rgba(180,0,0,0.05) !important;
    box-shadow: 0 0 30px rgba(180,0,0,0.15) !important;
}

/* ── PROGRESS BAR ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #660000, #cc0000, #ff4444) !important;
    box-shadow: 0 0 10px rgba(204,0,0,0.5) !important;
}

.stProgress > div {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 0 !important;
    height: 3px !important;
}

/* ── ALERTS ── */
div[data-testid="stAlert"] {
    background: rgba(180,0,0,0.08) !important;
    border: 1px solid rgba(180,0,0,0.3) !important;
    border-left: 3px solid #cc0000 !important;
    border-radius: 2px !important;
    backdrop-filter: blur(10px) !important;
}

/* ── RADIO ── */
.stRadio > div {
    gap: 1rem !important;
}

.stRadio label {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(180,0,0,0.2) !important;
    border-radius: 2px !important;
    padding: 0.5rem 1.5rem !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 2px !important;
    color: #888 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}

.stRadio label:hover {
    border-color: #cc0000 !important;
    color: #cc0000 !important;
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(180,0,0,0.2) !important;
    margin: 2rem 0 !important;
}

/* ── SELECTBOX ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(180,0,0,0.3) !important;
    border-radius: 2px !important;
    color: #e0e0e0 !important;
}

/* ── TABLE ── */
.stTable {
    border: 1px solid rgba(180,0,0,0.2) !important;
}

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton > button {
    background: rgba(180,0,0,0.1) !important;
    border: 1px solid rgba(180,0,0,0.4) !important;
    border-radius: 2px !important;
    color: #ff3333 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 2px !important;
}

/* ── METRIC ── */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(180,0,0,0.2) !important;
    border-radius: 2px !important;
    padding: 1rem !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    color: #cc0000 !important;
    font-size: 1.5rem !important;
}

/* ── SPINNER ── */
.stSpinner > div {
    border-top-color: #cc0000 !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #330000; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #cc0000; }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 2rem 0;'>
        <div style='
            font-family: Orbitron, monospace;
            font-size: 1.4rem;
            font-weight: 900;
            color: #cc0000;
            letter-spacing: 4px;
            text-shadow: 0 0 20px rgba(204,0,0,0.5);
        '>⬡ Z·A</div>
        <div style='
            font-family: Rajdhani, sans-serif;
            font-size: 0.65rem;
            color: #444;
            letter-spacing: 4px;
            margin-top: 0.3rem;
            text-transform: uppercase;
        '>Stark Audio Systems</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <div style='font-family: Orbitron, monospace; font-size: 0.55rem; color: #333; letter-spacing: 3px; margin-bottom: 1rem; padding-left: 0.5rem;'>NAVIGATION</div>
    </div>
    """, unsafe_allow_html=True)

    nav = st.radio("", [
        "⬡  IDENTIFY",
        "⬡  BATCH SCAN",
        "⬡  DATABASE",
    ], label_visibility="collapsed")

    st.markdown("""
    <div style='position: absolute; bottom: 2rem; left: 0; right: 0; padding: 0 1rem;'>
        <div style='
            border: 1px solid rgba(180,0,0,0.2);
            border-radius: 2px;
            padding: 1rem;
            background: rgba(180,0,0,0.03);
        '>
            <div style='font-family: Orbitron, monospace; font-size: 0.55rem; color: #cc0000; letter-spacing: 2px;'>SYSTEM STATUS</div>
            <div style='font-family: Rajdhani, sans-serif; font-size: 0.85rem; color: #666; margin-top: 0.5rem;'>EE200 · SSN PROJECT</div>
            <div style='font-family: Rajdhani, sans-serif; font-size: 0.75rem; color: #444;'>Q3B · AUDIO FINGERPRINT</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN HEADER ──────────────────────────────────────────────
st.markdown("""
<div style='
    border-bottom: 1px solid rgba(180,0,0,0.2);
    padding-bottom: 2rem;
    margin-bottom: 2rem;
'>
    <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;'>
        <div style='
            width: 3px;
            height: 3rem;
            background: linear-gradient(180deg, #cc0000, transparent);
            border-radius: 2px;
        '></div>
        <div>
            <div style='
                font-family: Orbitron, monospace;
                font-size: 2rem;
                font-weight: 900;
                background: linear-gradient(135deg, #ff2222, #cc0000, #ff4444);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                letter-spacing: 6px;
                text-transform: uppercase;
            '>ZAPPTAIN AMERICA</div>
            <div style='
                font-family: Rajdhani, sans-serif;
                font-size: 0.8rem;
                color: #444;
                letter-spacing: 4px;
                text-transform: uppercase;
                margin-top: 0.2rem;
            '>Audio Fingerprint Recognition System · Stark Industries</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── DATABASE STATE ───────────────────────────────────────────
if 'db' not in st.session_state:
    st.session_state.db = None
if 'db_size' not in st.session_state:
    st.session_state.db_size = 0

# ─── STATUS BAR ───────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div style='border:1px solid rgba(180,0,0,0.2); padding:1rem; background:rgba(180,0,0,0.03);'>
        <div style='font-family:Orbitron,monospace; font-size:0.55rem; color:#444; letter-spacing:2px;'>DATABASE</div>
        <div style='font-family:Orbitron,monospace; font-size:1.3rem; color:#cc0000; margin-top:0.3rem;'>
            {"ONLINE" if st.session_state.db else "OFFLINE"}
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div style='border:1px solid rgba(180,0,0,0.2); padding:1rem; background:rgba(180,0,0,0.03);'>
        <div style='font-family:Orbitron,monospace; font-size:0.55rem; color:#444; letter-spacing:2px;'>FINGERPRINTS</div>
        <div style='font-family:Orbitron,monospace; font-size:1.3rem; color:#cc0000; margin-top:0.3rem;'>
            {f"{st.session_state.db_size:,}" if st.session_state.db else "——"}
        </div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div style='border:1px solid rgba(180,0,0,0.2); padding:1rem; background:rgba(180,0,0,0.03);'>
        <div style='font-family:Orbitron,monospace; font-size:0.55rem; color:#444; letter-spacing:2px;'>ALGORITHM</div>
        <div style='font-family:Orbitron,monospace; font-size:1.3rem; color:#cc0000; margin-top:0.3rem;'>SHA·FP</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div style='border:1px solid rgba(180,0,0,0.2); padding:1rem; background:rgba(180,0,0,0.03);'>
        <div style='font-family:Orbitron,monospace; font-size:0.55rem; color:#444; letter-spacing:2px;'>VERSION</div>
        <div style='font-family:Orbitron,monospace; font-size:1.3rem; color:#cc0000; margin-top:0.3rem;'>V·3.0</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── DATABASE PAGE ────────────────────────────────────────────
if nav == "⬡  DATABASE":
    st.markdown("""
    <div style='font-family:Orbitron,monospace; font-size:0.7rem; color:#cc0000; letter-spacing:4px; margin-bottom:1.5rem;'>
        ⬡ &nbsp; DATABASE CONFIGURATION
    </div>
    """, unsafe_allow_html=True)

    songs_folder = st.text_input(
        "SONGS DIRECTORY PATH",
        value=os.path.expanduser("~/EE200_Q3/songs"),
        placeholder="/path/to/your/songs"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        build_btn = st.button("⬡ INITIALIZE DATABASE")

    if build_btn:
        if os.path.exists(songs_folder):
            st.markdown("""
            <div style='font-family:Rajdhani,sans-serif; font-size:0.8rem; color:#666; letter-spacing:2px; margin-bottom:0.5rem;'>
                SCANNING AUDIO FILES...
            </div>
            """, unsafe_allow_html=True)
            pb = st.progress(0)
            st.session_state.db = build_database(songs_folder, pb)
            st.session_state.db_size = len(st.session_state.db)
            pb.empty()
            st.success(f"DATABASE INITIALIZED · {st.session_state.db_size:,} FINGERPRINTS INDEXED")
        else:
            st.error("DIRECTORY NOT FOUND · CHECK PATH")

# ─── IDENTIFY PAGE ────────────────────────────────────────────
elif nav == "⬡  IDENTIFY":
    st.markdown("""
    <div style='font-family:Orbitron,monospace; font-size:0.7rem; color:#cc0000; letter-spacing:4px; margin-bottom:1.5rem;'>
        ⬡ &nbsp; SINGLE TRACK IDENTIFICATION
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.db:
        st.markdown("""
        <div style='
            border: 1px solid rgba(180,0,0,0.2);
            border-left: 3px solid #cc0000;
            padding: 1.5rem;
            background: rgba(180,0,0,0.03);
            font-family: Rajdhani, sans-serif;
            color: #666;
            letter-spacing: 2px;
            font-size: 0.9rem;
        '>
            DATABASE OFFLINE · Navigate to DATABASE to initialize
        </div>
        """, unsafe_allow_html=True)
    else:
        uploaded = st.file_uploader(
            "UPLOAD AUDIO FILE",
            type=["mp3"],
            help="Upload an MP3 file to identify"
        )

        if uploaded:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner("ANALYZING AUDIO FINGERPRINT..."):
                matched, count, scores, S_db, peaks, offsets = identify_song(tmp_path, st.session_state.db)

            # Result card
            st.markdown(f"""
            <div style='
                border: 1px solid rgba(180,0,0,0.4);
                background: linear-gradient(135deg, rgba(180,0,0,0.08), rgba(0,0,0,0.5));
                padding: 2.5rem;
                margin: 1.5rem 0;
                position: relative;
                overflow: hidden;
            '>
                <div style='
                    position: absolute;
                    top: 0; left: 0;
                    width: 100%; height: 3px;
                    background: linear-gradient(90deg, transparent, #cc0000, transparent);
                '></div>
                <div style='font-family:Orbitron,monospace; font-size:0.55rem; color:#cc0000; letter-spacing:4px; margin-bottom:1rem;'>
                    ⬡ MATCH IDENTIFIED
                </div>
                <div style='font-family:Orbitron,monospace; font-size:2rem; font-weight:900; color:#ffffff; letter-spacing:3px; margin-bottom:0.5rem;'>
                    {matched.upper() if matched else "NO MATCH"}
                </div>
                <div style='font-family:Rajdhani,sans-serif; font-size:1rem; color:#666; letter-spacing:2px;'>
                    CONFIDENCE SCORE: <span style='color:#cc0000;'>{count:,}</span>
                </div>
                <div style='
                    position: absolute;
                    bottom: 0; right: 0;
                    width: 200px; height: 200px;
                    background: radial-gradient(circle, rgba(180,0,0,0.08) 0%, transparent 70%);
                '></div>
            </div>
            """, unsafe_allow_html=True)

            # Plots
            st.markdown("""
            <div style='font-family:Orbitron,monospace; font-size:0.7rem; color:#333; letter-spacing:4px; margin: 1.5rem 0 1rem 0;'>
                ⬡ &nbsp; SIGNAL ANALYSIS
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            bg = '#080808'

            with col1:
                st.markdown("<div style='font-family:Orbitron,monospace; font-size:0.6rem; color:#444; letter-spacing:3px; margin-bottom:0.5rem;'>SPECTROGRAM</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 4), facecolor=bg)
                ax.set_facecolor(bg)
                ax.imshow(S_db, origin='lower', aspect='auto', cmap='inferno')
                ax.set_xlabel('TIME', color='#333', fontsize=7, labelpad=8, fontfamily='monospace')
                ax.set_ylabel('FREQUENCY', color='#333', fontsize=7, labelpad=8, fontfamily='monospace')
                ax.tick_params(colors='#222', labelsize=6)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#1a1a1a')
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("<div style='font-family:Orbitron,monospace; font-size:0.6rem; color:#444; letter-spacing:3px; margin-bottom:0.5rem;'>CONSTELLATION</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 4), facecolor=bg)
                ax.set_facecolor(bg)
                if len(peaks) > 0:
                    ax.scatter(peaks[:, 1], peaks[:, 0], s=0.5, c='#cc0000', alpha=0.6)
                ax.set_xlabel('TIME', color='#333', fontsize=7, labelpad=8, fontfamily='monospace')
                ax.set_ylabel('FREQUENCY', color='#333', fontsize=7, labelpad=8, fontfamily='monospace')
                ax.tick_params(colors='#222', labelsize=6)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#1a1a1a')
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col3:
                st.markdown("<div style='font-family:Orbitron,monospace; font-size:0.6rem; color:#444; letter-spacing:3px; margin-bottom:0.5rem;'>OFFSET HISTOGRAM</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 4), facecolor=bg)
                ax.set_facecolor(bg)
                if offsets:
                    ax.hist(offsets, bins=100, color='#cc0000', edgecolor='#080808', alpha=0.8)
                ax.set_xlabel('OFFSET', color='#333', fontsize=7, labelpad=8, fontfamily='monospace')
                ax.set_ylabel('COUNT', color='#333', fontsize=7, labelpad=8, fontfamily='monospace')
                ax.tick_params(colors='#222', labelsize=6)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#1a1a1a')
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

# ─── BATCH PAGE ───────────────────────────────────────────────
elif nav == "⬡  BATCH SCAN":
    st.markdown("""
    <div style='font-family:Orbitron,monospace; font-size:0.7rem; color:#cc0000; letter-spacing:4px; margin-bottom:1.5rem;'>
        ⬡ &nbsp; BATCH SCAN PROTOCOL
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.db:
        st.markdown("""
        <div style='
            border: 1px solid rgba(180,0,0,0.2);
            border-left: 3px solid #cc0000;
            padding: 1.5rem;
            background: rgba(180,0,0,0.03);
            font-family: Rajdhani, sans-serif;
            color: #666;
            letter-spacing: 2px;
        '>
            DATABASE OFFLINE · Navigate to DATABASE to initialize
        </div>
        """, unsafe_allow_html=True)
    else:
        uploaded_files = st.file_uploader(
            "UPLOAD MULTIPLE AUDIO FILES",
            type=["mp3"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.markdown(f"""
            <div style='font-family:Rajdhani,sans-serif; color:#444; letter-spacing:2px; font-size:0.85rem; margin:1rem 0;'>
                {len(uploaded_files)} FILES QUEUED FOR ANALYSIS
            </div>
            """, unsafe_allow_html=True)

            if st.button("⬡ EXECUTE BATCH SCAN"):
                results = []
                pb = st.progress(0)
                for i, f in enumerate(uploaded_files):
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    matched, _, _, _, _, _ = identify_song(tmp_path, st.session_state.db)
                    results.append({"FILENAME": f.name, "IDENTIFIED AS": matched or "UNKNOWN"})
                    pb.progress((i+1)/len(uploaded_files), text=f"SCANNING: {f.name}")
                pb.empty()

                st.markdown("""
                <div style='font-family:Orbitron,monospace; font-size:0.7rem; color:#cc0000; letter-spacing:4px; margin: 1.5rem 0 1rem 0;'>
                    ⬡ &nbsp; SCAN RESULTS
                </div>
                """, unsafe_allow_html=True)

                st.table(results)

                csv_lines = "filename,prediction\n" + "\n".join(
                    f"{r['FILENAME']},{r['IDENTIFIED AS']}" for r in results
                )
                st.download_button(
                    "⬡ EXPORT results.csv",
                    csv_lines,
                    "results.csv",
                    mime="text/csv"
                )

# ─── FOOTER ───────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='
    border-top: 1px solid rgba(180,0,0,0.1);
    padding-top: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
'>
    <div style='font-family:Orbitron,monospace; font-size:0.55rem; color:#222; letter-spacing:3px;'>
        ⬡ ZAPPTAIN AMERICA · STARK AUDIO SYSTEMS
    </div>
    <div style='font-family:Rajdhani,sans-serif; font-size:0.75rem; color:#222; letter-spacing:2px;'>
        EE200 · SIGNALS SYSTEMS & NETWORKS · Q3B
    </div>
</div>
""", unsafe_allow_html=True)