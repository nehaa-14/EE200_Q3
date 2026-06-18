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

def build_database_from_files(uploaded_files, progress_bar=None):
    db = defaultdict(list)
    for i, uf in enumerate(uploaded_files):
        song_name = os.path.splitext(uf.name)[0]
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name
            S_db, sr, hop = compute_spectrogram(tmp_path)
            peaks = get_peaks(S_db)
            hashes = hash_peaks(peaks)
            for (h, t) in hashes:
                db[h].append((song_name, t))
        except:
            pass
        if progress_bar:
            progress_bar.progress((i+1)/len(uploaded_files), text=f"INDEXING: {song_name.upper()}")
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

# ─── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

.stApp {
    background: #080808;
    font-family: 'Inter', sans-serif;
}

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(180,0,0,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(180,0,0,0.025) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

.stApp::after {
    content: '';
    position: fixed;
    top: -200px; right: -200px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(180,0,0,0.12) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a0a 0%, #0d0d0d 100%) !important;
    border-right: 1px solid rgba(180,0,0,0.25) !important;
    min-width: 220px !important;
    max-width: 220px !important;
}

section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem !important;
}

/* BUTTONS */
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(180,0,0,0.5) !important;
    border-radius: 2px !important;
    color: #cc0000 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    padding: 0.65rem 1.5rem !important;
    text-transform: uppercase !important;
    transition: all 0.25s ease !important;
    width: auto !important;
}

.stButton > button:hover {
    border-color: #ff0000 !important;
    color: #ffffff !important;
    box-shadow: 0 0 25px rgba(180,0,0,0.4), inset 0 0 15px rgba(180,0,0,0.08) !important;
    transform: translateY(-1px) !important;
}

/* TEXT INPUT */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(180,0,0,0.25) !important;
    border-radius: 2px !important;
    color: #e0e0e0 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.95rem !important;
    letter-spacing: 1px !important;
    padding: 0.65rem 1rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: #cc0000 !important;
    box-shadow: 0 0 15px rgba(180,0,0,0.25) !important;
}

/* FILE UPLOADER */
div[data-testid="stFileUploader"] {
    border: 1px solid rgba(180,0,0,0.25) !important;
    border-radius: 2px !important;
    background: rgba(180,0,0,0.02) !important;
    padding: 1.5rem !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #990000 !important;
    box-shadow: 0 0 25px rgba(180,0,0,0.12) !important;
}

div[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* PROGRESS */
.stProgress > div > div {
    background: linear-gradient(90deg, #660000, #cc0000, #ff3333) !important;
    box-shadow: 0 0 8px rgba(204,0,0,0.4) !important;
}
.stProgress > div {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 0 !important;
    height: 2px !important;
}

/* ALERTS */
div[data-testid="stAlert"] {
    background: rgba(180,0,0,0.06) !important;
    border: 1px solid rgba(180,0,0,0.25) !important;
    border-left: 3px solid #cc0000 !important;
    border-radius: 2px !important;
}

/* RADIO */
div[role="radiogroup"] {
    gap: 0.5rem !important;
    flex-direction: column !important;
}

div[role="radiogroup"] label {
    background: rgba(255,255,255,0.01) !important;
    border: 1px solid rgba(180,0,0,0.15) !important;
    border-radius: 2px !important;
    padding: 0.5rem 1rem !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 2px !important;
    color: #555 !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

div[role="radiogroup"] label:hover {
    border-color: #cc0000 !important;
    color: #cc0000 !important;
}

/* DIVIDER */
hr {
    border: none !important;
    border-top: 1px solid rgba(180,0,0,0.15) !important;
    margin: 1.5rem 0 !important;
}

/* DOWNLOAD */
.stDownloadButton > button {
    background: rgba(180,0,0,0.08) !important;
    border: 1px solid rgba(180,0,0,0.35) !important;
    border-radius: 2px !important;
    color: #cc0000 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 2px !important;
}

/* SPINNER */
.stSpinner > div { border-top-color: #cc0000 !important; }

/* SCROLLBAR */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #330000; }
::-webkit-scrollbar-thumb:hover { background: #cc0000; }

/* TABLE */
table {
    font-family: 'Rajdhani', sans-serif !important;
    border-collapse: collapse !important;
    width: 100% !important;
}
th {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 2px !important;
    color: #cc0000 !important;
    border-bottom: 1px solid rgba(180,0,0,0.3) !important;
    padding: 0.75rem !important;
    background: rgba(180,0,0,0.05) !important;
}
td {
    color: #888 !important;
    padding: 0.75rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.03) !important;
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────
if 'db' not in st.session_state:
    st.session_state.db = None
if 'db_size' not in st.session_state:
    st.session_state.db_size = 0
if 'db_source' not in st.session_state:
    st.session_state.db_source = ""

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:0.5rem 0 2rem 0; border-bottom:1px solid rgba(180,0,0,0.15); margin-bottom:1.5rem;'>
        <div style='font-family:Orbitron,monospace; font-size:1.3rem; font-weight:900; color:#cc0000;
                    text-shadow:0 0 20px rgba(204,0,0,0.4); letter-spacing:4px;'>⬡ Z·A</div>
        <div style='font-family:Rajdhani,sans-serif; font-size:0.6rem; color:#333;
                    letter-spacing:3px; margin-top:0.3rem;'>STARK AUDIO SYSTEMS</div>
    </div>
    <div style='font-family:Orbitron,monospace; font-size:0.5rem; color:#333;
                letter-spacing:3px; margin-bottom:0.75rem;'>NAVIGATION</div>
    """, unsafe_allow_html=True)

    nav = st.radio("NAV", [
        "⬡  IDENTIFY",
        "⬡  BATCH SCAN",
        "⬡  DATABASE",
    ], label_visibility="collapsed")

    st.markdown("<br><br>", unsafe_allow_html=True)

    db_status = "ONLINE" if st.session_state.db else "OFFLINE"
    db_color = "#cc0000" if st.session_state.db else "#333"
    st.markdown(f"""
    <div style='border:1px solid rgba(180,0,0,0.15); border-radius:2px;
                padding:0.75rem; background:rgba(180,0,0,0.02); margin-top:1rem;'>
        <div style='font-family:Orbitron,monospace; font-size:0.5rem;
                    color:#333; letter-spacing:2px; margin-bottom:0.4rem;'>SYSTEM STATUS</div>
        <div style='font-family:Orbitron,monospace; font-size:0.75rem; color:{db_color};'>
            DB: {db_status}
        </div>
        <div style='font-family:Rajdhani,sans-serif; font-size:0.7rem; color:#333; margin-top:0.2rem;'>
            {f"{st.session_state.db_size:,} fingerprints" if st.session_state.db else "No data loaded"}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN HEADER ──────────────────────────────────────────────
st.markdown("""
<div style='border-bottom:1px solid rgba(180,0,0,0.15); padding-bottom:1.5rem; margin-bottom:1.5rem;'>
    <div style='display:flex; align-items:center; gap:1rem;'>
        <div style='width:3px; height:2.5rem;
                    background:linear-gradient(180deg,#cc0000,transparent); border-radius:2px;'></div>
        <div>
            <div style='font-family:Orbitron,monospace; font-size:1.8rem; font-weight:900;
                        background:linear-gradient(135deg,#ff2222,#cc0000,#ff4444);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text; letter-spacing:5px; text-transform:uppercase;'>
                ZAPPTAIN AMERICA
            </div>
            <div style='font-family:Rajdhani,sans-serif; font-size:0.7rem; color:#333;
                        letter-spacing:4px; text-transform:uppercase; margin-top:0.2rem;'>
                Audio Fingerprint Recognition System
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── STATUS CARDS ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
cards = [
    ("DATABASE", "ONLINE" if st.session_state.db else "OFFLINE"),
    ("FINGERPRINTS", f"{st.session_state.db_size:,}" if st.session_state.db else "——"),
    ("ALGORITHM", "SHA·FP"),
    ("VERSION", "V·3.0"),
]
for col, (label, val) in zip([c1,c2,c3,c4], cards):
    with col:
        st.markdown(f"""
        <div style='border:1px solid rgba(180,0,0,0.18); padding:1rem;
                    background:rgba(180,0,0,0.025); border-radius:2px;'>
            <div style='font-family:Orbitron,monospace; font-size:0.5rem;
                        color:#333; letter-spacing:2px;'>{label}</div>
            <div style='font-family:Orbitron,monospace; font-size:1.2rem;
                        color:#cc0000; margin-top:0.3rem;'>{val}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# DATABASE PAGE
# ═══════════════════════════════════════════════════════════════
if nav == "⬡  DATABASE":
    st.markdown("""
    <div style='font-family:Orbitron,monospace; font-size:0.65rem; color:#cc0000;
                letter-spacing:4px; margin-bottom:1.5rem;'>⬡ &nbsp; DATABASE CONFIGURATION</div>
    """, unsafe_allow_html=True)

    method = st.radio("Build method", [
        "📁  Upload song files directly",
        "🗂️  Use folder path (local only)"
    ], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    if method == "📁  Upload song files directly":
        st.markdown("""
        <div style='font-family:Orbitron,monospace; font-size:0.55rem; color:#555;
                    letter-spacing:3px; margin-bottom:0.75rem;'>UPLOAD YOUR SONG LIBRARY</div>
        """, unsafe_allow_html=True)

        song_files = st.file_uploader(
            "Browse and select all your MP3 files",
            type=["mp3"],
            accept_multiple_files=True,
            help="Select all 50 songs at once — Cmd+A to select all"
        )

        if song_files:
            st.markdown(f"""
            <div style='font-family:Rajdhani,sans-serif; color:#555;
                        letter-spacing:2px; font-size:0.85rem; margin:0.75rem 0;'>
                {len(song_files)} FILES SELECTED
            </div>
            """, unsafe_allow_html=True)

            if st.button("⬡ INITIALIZE DATABASE"):
                pb = st.progress(0)
                st.session_state.db = build_database_from_files(song_files, pb)
                st.session_state.db_size = len(st.session_state.db)
                st.session_state.db_source = "uploaded"
                pb.empty()
                st.success(f"DATABASE INITIALIZED · {st.session_state.db_size:,} FINGERPRINTS INDEXED")

    else:
        st.markdown("""
        <div style='font-family:Orbitron,monospace; font-size:0.55rem; color:#555;
                    letter-spacing:3px; margin-bottom:0.75rem;'>SONGS DIRECTORY PATH</div>
        """, unsafe_allow_html=True)

        songs_folder = st.text_input(
            "PATH",
            value=os.path.expanduser("~/EE200_Q3/songs"),
            label_visibility="collapsed"
        )

        if st.button("⬡ INITIALIZE DATABASE"):
            if os.path.exists(songs_folder):
                pb = st.progress(0)
                st.session_state.db = build_database(songs_folder, pb)
                st.session_state.db_size = len(st.session_state.db)
                st.session_state.db_source = "folder"
                pb.empty()
                st.success(f"DATABASE INITIALIZED · {st.session_state.db_size:,} FINGERPRINTS INDEXED")
            else:
                st.error("DIRECTORY NOT FOUND")

# ═══════════════════════════════════════════════════════════════
# IDENTIFY PAGE
# ═══════════════════════════════════════════════════════════════
elif nav == "⬡  IDENTIFY":
    st.markdown("""
    <div style='font-family:Orbitron,monospace; font-size:0.65rem; color:#cc0000;
                letter-spacing:4px; margin-bottom:1.5rem;'>⬡ &nbsp; SINGLE TRACK IDENTIFICATION</div>
    """, unsafe_allow_html=True)

    if not st.session_state.db:
        st.markdown("""
        <div style='border:1px solid rgba(180,0,0,0.2); border-left:3px solid #cc0000;
                    padding:1.5rem; background:rgba(180,0,0,0.03); border-radius:2px;
                    font-family:Rajdhani,sans-serif; color:#555; letter-spacing:2px; font-size:0.9rem;'>
            DATABASE OFFLINE · Go to DATABASE tab to initialize
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='font-family:Orbitron,monospace; font-size:0.55rem; color:#555;
                    letter-spacing:3px; margin-bottom:0.75rem;'>UPLOAD QUERY TRACK</div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Browse and select an MP3 file to identify",
            type=["mp3"]
        )

        if uploaded:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner("ANALYZING AUDIO FINGERPRINT..."):
                matched, count, scores, S_db, peaks, offsets = identify_song(tmp_path, st.session_state.db)

            st.markdown(f"""
            <div style='
                border:1px solid rgba(180,0,0,0.35);
                background:linear-gradient(135deg, rgba(180,0,0,0.07), rgba(0,0,0,0.4));
                padding:2rem 2.5rem; margin:1.5rem 0; border-radius:2px; position:relative; overflow:hidden;
            '>
                <div style='position:absolute; top:0; left:0; width:100%; height:2px;
                            background:linear-gradient(90deg,transparent,#cc0000,transparent);'></div>
                <div style='font-family:Orbitron,monospace; font-size:0.5rem; color:#cc0000;
                            letter-spacing:4px; margin-bottom:0.75rem;'>⬡ MATCH IDENTIFIED</div>
                <div style='font-family:Orbitron,monospace; font-size:1.8rem; font-weight:900;
                            color:#ffffff; letter-spacing:3px; margin-bottom:0.4rem;'>
                    {matched.upper() if matched else "NO MATCH FOUND"}
                </div>
                <div style='font-family:Rajdhani,sans-serif; font-size:0.95rem; color:#555; letter-spacing:2px;'>
                    CONFIDENCE SCORE: <span style='color:#cc0000;'>{count:,}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style='font-family:Orbitron,monospace; font-size:0.6rem; color:#333;
                        letter-spacing:4px; margin:1.5rem 0 1rem 0;'>⬡ &nbsp; SIGNAL ANALYSIS</div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            bg = '#080808'

            def styled_plot(ax, fig, title):
                ax.set_facecolor(bg)
                ax.set_xlabel('TIME', color='#2a2a2a', fontsize=7, labelpad=6, fontfamily='monospace')
                ax.set_ylabel('FREQUENCY', color='#2a2a2a', fontsize=7, labelpad=6, fontfamily='monospace')
                ax.tick_params(colors='#1a1a1a', labelsize=6)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#141414')
                fig.tight_layout()

            with col1:
                st.markdown("<div style='font-family:Orbitron,monospace;font-size:0.55rem;color:#333;letter-spacing:3px;margin-bottom:0.5rem;'>SPECTROGRAM</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6,4), facecolor=bg)
                ax.imshow(S_db, origin='lower', aspect='auto', cmap='inferno')
                styled_plot(ax, fig, "SPECTROGRAM")
                st.pyplot(fig); plt.close()

            with col2:
                st.markdown("<div style='font-family:Orbitron,monospace;font-size:0.55rem;color:#333;letter-spacing:3px;margin-bottom:0.5rem;'>CONSTELLATION</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6,4), facecolor=bg)
                if len(peaks) > 0:
                    ax.scatter(peaks[:,1], peaks[:,0], s=0.5, c='#cc0000', alpha=0.6)
                styled_plot(ax, fig, "CONSTELLATION")
                st.pyplot(fig); plt.close()

            with col3:
                st.markdown("<div style='font-family:Orbitron,monospace;font-size:0.55rem;color:#333;letter-spacing:3px;margin-bottom:0.5rem;'>OFFSET HISTOGRAM</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6,4), facecolor=bg)
                if offsets:
                    ax.hist(offsets, bins=100, color='#cc0000', edgecolor='#080808', alpha=0.85)
                styled_plot(ax, fig, "OFFSET")
                st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════
# BATCH PAGE
# ═══════════════════════════════════════════════════════════════
elif nav == "⬡  BATCH SCAN":
    st.markdown("""
    <div style='font-family:Orbitron,monospace; font-size:0.65rem; color:#cc0000;
                letter-spacing:4px; margin-bottom:1.5rem;'>⬡ &nbsp; BATCH SCAN PROTOCOL</div>
    """, unsafe_allow_html=True)

    if not st.session_state.db:
        st.markdown("""
        <div style='border:1px solid rgba(180,0,0,0.2); border-left:3px solid #cc0000;
                    padding:1.5rem; background:rgba(180,0,0,0.03); border-radius:2px;
                    font-family:Rajdhani,sans-serif; color:#555; letter-spacing:2px; font-size:0.9rem;'>
            DATABASE OFFLINE · Go to DATABASE tab to initialize
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='font-family:Orbitron,monospace; font-size:0.55rem; color:#555;
                    letter-spacing:3px; margin-bottom:0.75rem;'>UPLOAD QUERY FILES</div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Browse and select multiple MP3 files",
            type=["mp3"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.markdown(f"""
            <div style='font-family:Rajdhani,sans-serif; color:#444;
                        letter-spacing:2px; font-size:0.85rem; margin:0.75rem 0;'>
                {len(uploaded_files)} FILES QUEUED
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
                <div style='font-family:Orbitron,monospace; font-size:0.6rem; color:#333;
                            letter-spacing:4px; margin:1.5rem 0 1rem 0;'>⬡ &nbsp; SCAN RESULTS</div>
                """, unsafe_allow_html=True)

                st.table(results)

                csv_lines = "filename,prediction\n" + "\n".join(
                    f"{r['FILENAME']},{r['IDENTIFIED AS']}" for r in results
                )
                st.download_button("⬡ EXPORT results.csv", csv_lines, "results.csv", mime="text/csv")