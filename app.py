import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from collections import defaultdict
import os
import tempfile
import time

# ─── CORE FUNCTIONS ───────────────────────────────────────────
def compute_spectrogram(audio_path, n_fft=4096, hop_length=512):
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db, sr, hop_length, y

def get_peaks(S_db, neighborhood=20, threshold_db=-60):
    local_max = maximum_filter(S_db, size=neighborhood) == S_db
    return np.argwhere(local_max & (S_db > threshold_db))

def hash_peaks(peaks, fan_out=15, time_delta_max=200):
    peaks_sorted = sorted(peaks.tolist(), key=lambda x: x[1])
    hashes = []
    for i, (f1, t1) in enumerate(peaks_sorted):
        for j in range(1, fan_out + 1):
            if i + j >= len(peaks_sorted): break
            f2, t2 = peaks_sorted[i + j]
            dt = t2 - t1
            if dt <= 0 or dt > time_delta_max: continue
            hashes.append((hash((int(f1), int(f2), int(dt))), t1))
    return hashes

def build_database_from_files(uploaded_files, progress_bar=None):
    db = defaultdict(list)
    song_peaks = {}
    song_hashes_count = {}
    for i, uf in enumerate(uploaded_files):
        song_name = os.path.splitext(uf.name)[0]
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name
            S_db, _, _, _ = compute_spectrogram(tmp_path)
            peaks = get_peaks(S_db)
            hashes = hash_peaks(peaks)
            song_peaks[song_name] = peaks
            song_hashes_count[song_name] = len(hashes)
            for (h, t) in hashes:
                db[h].append((song_name, t))
        except:
            pass
        if progress_bar:
            progress_bar.progress((i+1)/len(uploaded_files),
                                   text=f"INDEXING: {song_name.upper()}")
    return dict(db), song_peaks, song_hashes_count

def identify_song(query_path, db):
    t0 = time.time()
    S_db, sr, hop, y = compute_spectrogram(query_path)
    t1 = time.time()
    peaks = get_peaks(S_db)
    t2 = time.time()
    hashes = hash_peaks(peaks)
    t3 = time.time()
    scores = defaultdict(list)
    for (h, t_query) in hashes:
        if h in db:
            for (song_name, t_db) in db[h]:
                scores[song_name].append(t_db - t_query)
    t4 = time.time()
    best_song, best_count, best_offsets = None, 0, []
    all_scores = {}
    for song, offsets in scores.items():
        counts = defaultdict(int)
        for o in offsets: counts[o] += 1
        top = max(counts.values())
        all_scores[song] = top
        if top > best_count:
            best_count = top
            best_song = song
            best_offsets = offsets
    t5 = time.time()
    timing = {
        'spectrogram': int((t1-t0)*1000),
        'constellation': int((t2-t1)*1000),
        'hashing': int((t3-t2)*1000),
        'db_lookup': int((t4-t3)*1000),
        'scoring': int((t5-t4)*1000),
        'total': int((t5-t0)*1000),
        'n_peaks': len(peaks),
        'n_hashes': len(hashes),
        'n_tracks': len(scores),
        'best_offset': best_offsets[0] if best_offsets else 0
    }
    return best_song, best_count, all_scores, S_db, peaks, best_offsets, y, timing

def make_fig(w, h, bg='#0d0d0d', dpi=180):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=bg, dpi=dpi)
    ax.set_facecolor(bg)
    for sp in ax.spines.values(): sp.set_edgecolor('#1a1a1a')
    ax.tick_params(colors='#2a2a2a', labelsize=7)
    return fig, ax

# ─── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="ZAPPTAIN AMERICA",
    layout="wide",
    page_icon="⬡",
    initial_sidebar_state="collapsed"
)

# ─── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

.stApp {
    background: #080808 !important;
    font-family: 'Inter', sans-serif;
}

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(180,0,0,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(180,0,0,0.02) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

.stApp::after {
    content: '';
    position: fixed;
    top: -300px; right: -300px;
    width: 800px; height: 800px;
    background: radial-gradient(circle, rgba(180,0,0,0.08) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(180,0,0,0.2) !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #444 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 3px !important;
    padding: 0.75rem 2rem !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #cc0000 !important;
}

.stTabs [aria-selected="true"] {
    color: #cc0000 !important;
    border-bottom: 2px solid #cc0000 !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding: 2rem 0 !important;
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
    padding: 0.7rem 2rem !important;
    text-transform: uppercase !important;
    transition: all 0.25s ease !important;
}

.stButton > button:hover {
    border-color: #ff0000 !important;
    color: #ffffff !important;
    box-shadow: 0 0 25px rgba(180,0,0,0.4) !important;
    transform: translateY(-1px) !important;
}

/* FILE UPLOADER */
div[data-testid="stFileUploader"] {
    border: 1px dashed rgba(180,0,0,0.3) !important;
    border-radius: 2px !important;
    background: rgba(180,0,0,0.02) !important;
    padding: 1.5rem !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #990000 !important;
    box-shadow: 0 0 20px rgba(180,0,0,0.1) !important;
}

/* PROGRESS */
.stProgress > div > div {
    background: linear-gradient(90deg, #660000, #cc0000, #ff3333) !important;
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

/* DOWNLOAD */
.stDownloadButton > button {
    background: rgba(180,0,0,0.08) !important;
    border: 1px solid rgba(180,0,0,0.35) !important;
    color: #cc0000 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 2px !important;
    border-radius: 2px !important;
}

/* AUDIO PLAYER */
audio {
    width: 100% !important;
    height: 36px !important;
    border-radius: 2px !important;
    filter: invert(1) hue-rotate(180deg) !important;
    opacity: 0.7 !important;
}

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
    font-size: 0.55rem !important;
    letter-spacing: 2px !important;
    color: #cc0000 !important;
    border-bottom: 1px solid rgba(180,0,0,0.3) !important;
    padding: 0.75rem 1rem !important;
    background: rgba(180,0,0,0.04) !important;
    text-align: left !important;
}
td {
    color: #888 !important;
    padding: 0.65rem 1rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.03) !important;
    font-size: 0.9rem !important;
}

hr {
    border: none !important;
    border-top: 1px solid rgba(180,0,0,0.1) !important;
    margin: 1.5rem 0 !important;
}

.stSpinner > div { border-top-color: #cc0000 !important; }

p, label { color: #888 !important; font-family: 'Inter', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────
for key, default in [
    ('db', None), ('db_size', 0),
    ('song_peaks', {}), ('song_hashes', {}),
    ('song_files_data', {})
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Auto-load prebuilt database
import pickle, gzip
if st.session_state.db is None:
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'song_db.pkl.gz')
    if os.path.exists(db_path):
        try:
            with gzip.open(db_path, 'rb') as f:
                data = pickle.load(f)
            st.session_state.db = data['db']
            st.session_state.db_size = len(data['db'])
            st.session_state.song_peaks = data.get('song_peaks', {})
            st.session_state.song_hashes = data.get('song_hashes_count', {})
        except Exception as e:
            st.error(f"Failed to load database: {e}")
# ─── HEADER ───────────────────────────────────────────────────
db_status = "ONLINE" if st.session_state.db else "OFFLINE"
db_color = "#cc0000" if st.session_state.db else "#2a2a2a"
st.markdown(f"""
<div style='padding:2rem 0 1.5rem 0; border-bottom:1px solid rgba(180,0,0,0.15);
            margin-bottom:0;'>
    <div style='display:flex; align-items:center; gap:1.2rem;'>
        <div style='width:42px; height:42px; border:1px solid rgba(180,0,0,0.5);
                    border-radius:50%; display:flex; align-items:center;
                    justify-content:center; background:rgba(180,0,0,0.08);
                    box-shadow:0 0 20px rgba(180,0,0,0.2); font-size:1.2rem;'>⬡</div>
        <div>
            <div style='font-family:Orbitron,monospace; font-weight:900; font-size:1.5rem;
                        background:linear-gradient(135deg,#ff2222,#cc0000,#ff4444);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text; letter-spacing:5px; text-transform:uppercase;
                        line-height:1;'>ZAPPTAIN AMERICA</div>
            <div style='font-family:Rajdhani,sans-serif; font-size:0.7rem; color:#333;
                        letter-spacing:4px; text-transform:uppercase; margin-top:0.3rem;'>
                Signals, Systems & Networks · Audio Fingerprinting
            </div>
        </div>
        <div style='margin-left:auto; text-align:right;'>
            <div style='font-family:Orbitron,monospace; font-size:0.5rem;
                        color:#333; letter-spacing:2px;'>DATABASE</div>
            <div style='font-family:Orbitron,monospace; font-size:0.85rem;
                        color:{db_color};'>{db_status}</div>
            <div style='font-family:Rajdhani,sans-serif; font-size:0.7rem; color:#2a2a2a;'>
                {f"{st.session_state.db_size:,} fingerprints" if st.session_state.db else "no data"}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⬡  LIBRARY", "⬡  IDENTIFY", "⬡  BATCH"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — LIBRARY
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div style='font-family:Orbitron,monospace; font-size:0.6rem; color:#333;
                letter-spacing:4px; margin-bottom:1.5rem;'>LIBRARY</div>
    """, unsafe_allow_html=True)

    if not st.session_state.db:
        st.markdown("""
        <div style='border:1px dashed rgba(180,0,0,0.15); border-radius:2px;
                    padding:3rem; text-align:center; margin-bottom:2rem;'>
            <div style='font-family:Orbitron,monospace; font-size:0.6rem;
                        color:#2a2a2a; letter-spacing:3px; margin-bottom:0.5rem;'>
                SONG INDEXING REQUIRED
            </div>
            <div style='font-family:Rajdhani,sans-serif; font-size:0.85rem; color:#2a2a2a;'>
                Upload your song library below to index it
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:Orbitron,monospace; font-size:0.55rem; color:#555;
                letter-spacing:3px; margin-bottom:0.75rem;'>
        INDEX SONG LIBRARY · SELECT ALL MP3 FILES (CMD+A)
    </div>
    """, unsafe_allow_html=True)

    song_files = st.file_uploader(
        "Upload MP3 files",
        type=["mp3"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if song_files:
        st.markdown(f"""
        <div style='font-family:Rajdhani,sans-serif; color:#555;
                    letter-spacing:2px; font-size:0.85rem; margin:0.75rem 0;'>
            {len(song_files)} FILES SELECTED
        </div>
        """, unsafe_allow_html=True)

        if st.button("⬡ INDEX LIBRARY"):
            pb = st.progress(0)
            # Save file data for playback
            files_data = {}
            for uf in song_files:
                files_data[os.path.splitext(uf.name)[0]] = uf.read()

            # Reset file positions
            for uf in song_files:
                uf.seek(0)

            db, song_peaks, song_hashes = build_database_from_files(song_files, pb)
            st.session_state.db = db
            st.session_state.db_size = len(db)
            st.session_state.song_peaks = song_peaks
            st.session_state.song_hashes = song_hashes
            st.session_state.song_files_data = files_data
            pb.empty()
            st.success(f"⬡ LIBRARY INDEXED · {len(db):,} FINGERPRINTS · {len(song_files)} SONGS")

    # Song cards grid
    if st.session_state.db and st.session_state.song_peaks:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:Orbitron,monospace; font-size:0.55rem; color:#333;
                    letter-spacing:4px; margin-bottom:1rem;'>IN THE DATABASE</div>
        """, unsafe_allow_html=True)

        songs = sorted(st.session_state.song_peaks.keys())
        cols_per_row = 4
        for row_start in range(0, len(songs), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, song in enumerate(songs[row_start:row_start+cols_per_row]):
                with cols[col_idx]:
                    peaks = st.session_state.song_peaks[song]
                    n_hashes = st.session_state.song_hashes.get(song, 0)

                    fig, ax = make_fig(3, 2, dpi=120)
                    if len(peaks) > 0:
                        colors = plt.cm.plasma(peaks[:,0] / (peaks[:,0].max()+1))
                        ax.scatter(peaks[:,1], peaks[:,0], s=0.3,
                                   c=colors, alpha=0.8, linewidths=0)
                    ax.set_xlim(0, peaks[:,1].max() if len(peaks)>0 else 1)
                    ax.set_ylim(0, peaks[:,0].max() if len(peaks)>0 else 1)
                    ax.axis('off')
                    fig.tight_layout(pad=0)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                    st.markdown(f"""
                    <div style='margin-top:0.3rem; padding-bottom:1rem;'>
                        <div style='font-family:Rajdhani,sans-serif; font-size:0.85rem;
                                    color:#ccc; font-weight:600; white-space:nowrap;
                                    overflow:hidden; text-overflow:ellipsis;'>{song}</div>
                        <div style='font-family:Orbitron,monospace; font-size:0.5rem;
                                    color:#444; letter-spacing:1px;
                                    margin-top:0.2rem;'>{n_hashes:,} hashes</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Audio playback
                    if song in st.session_state.song_files_data:
                        st.audio(st.session_state.song_files_data[song],
                                format="audio/mp3")

# ═══════════════════════════════════════════════════════════════
# TAB 2 — IDENTIFY
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style='font-family:Orbitron,monospace; font-size:0.6rem; color:#333;
                letter-spacing:4px; margin-bottom:0.5rem;'>SEARCH</div>
    <div style='font-family:Rajdhani,sans-serif; font-size:1.6rem; color:#e0e0e0;
                font-weight:600; margin-bottom:1.5rem;'>Identify a clip</div>
    """, unsafe_allow_html=True)

    if not st.session_state.db:
        st.markdown("""
        <div style='border:1px solid rgba(180,0,0,0.2); border-left:3px solid #cc0000;
                    padding:1.5rem; background:rgba(180,0,0,0.03); border-radius:2px;
                    font-family:Rajdhani,sans-serif; color:#555; letter-spacing:2px;'>
            ⬡ DATABASE OFFLINE · Go to LIBRARY tab to index songs first
        </div>
        """, unsafe_allow_html=True)
    else:
        uploaded = st.file_uploader(
            "Upload clip",
            type=["mp3", "wav", "flac", "m4a"],
            label_visibility="collapsed"
        )

        if uploaded:
            audio_bytes = uploaded.read()
            uploaded.seek(0)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            with st.spinner("ANALYZING..."):
                matched, count, all_scores, S_db, peaks, offsets, y, timing = identify_song(
                    tmp_path, st.session_state.db)

            # ── TIMING STATS ──
            st.markdown(f"""
            <div style='display:flex; gap:0; padding:1rem 0;
                        border-top:1px solid rgba(180,0,0,0.1);
                        border-bottom:1px solid rgba(180,0,0,0.1);
                        margin-bottom:1.5rem; flex-wrap:wrap;'>
                {''.join([
                    f"""<div style='text-align:center; flex:1; min-width:80px;
                                    padding:0 1rem; border-right:1px solid rgba(180,0,0,0.1);'>
                        <div style='font-family:Orbitron,monospace; font-size:0.42rem;
                                    color:#2a2a2a; letter-spacing:2px; margin-bottom:0.3rem;'>{label}</div>
                        <div style='font-family:Orbitron,monospace; font-size:1rem;
                                    color:#cc0000;'>{val}</div>
                        <div style='font-family:Rajdhani,sans-serif; font-size:0.65rem;
                                    color:#2a2a2a; margin-top:0.2rem;'>{sub}</div>
                    </div>"""
                    for label, val, sub in [
                        ("SPECTROGRAM", f"{timing['spectrogram']} ms", f"{S_db.shape[0]}×{S_db.shape[1]}"),
                        ("CONSTELLATION", f"{timing['constellation']} ms", f"{timing['n_peaks']} peaks"),
                        ("HASHING", f"{timing['hashing']} ms", f"{timing['n_hashes']:,} hashes"),
                        ("DB LOOKUP", f"{timing['db_lookup']} ms", f"{timing['n_tracks']} tracks"),
                        ("SCORING", f"{timing['scoring']} ms", f"offset {timing['best_offset']}"),
                    ]
                ])}
                <div style='flex:1; display:flex; align-items:center;
                            justify-content:flex-end; padding:0 1rem;'>
                    <div style='font-family:Rajdhani,sans-serif; font-size:0.7rem;
                                color:#2a2a2a;'>total {timing['total']} ms</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── MATCH RESULT ──
            runner_up = sorted(all_scores.values(), reverse=True)
            ratio = round(count / max(1, runner_up[1] if len(runner_up) > 1 else 1))
            st.markdown(f"""
            <div style='border:1px solid rgba(180,0,0,0.3);
                        background:linear-gradient(135deg,rgba(180,0,0,0.06),rgba(0,0,0,0.5));
                        padding:2rem 2.5rem; margin-bottom:1.5rem; position:relative; overflow:hidden;'>
                <div style='position:absolute; top:0; left:0; width:100%; height:2px;
                            background:linear-gradient(90deg,transparent,#cc0000,transparent);'></div>
                <div style='font-family:Orbitron,monospace; font-size:0.5rem; color:#cc0000;
                            letter-spacing:4px; margin-bottom:0.5rem;'>MATCH FOUND</div>
                <div style='font-family:Rajdhani,sans-serif; font-size:2.5rem; font-weight:700;
                            color:#ffffff; line-height:1; margin-bottom:0.5rem;'>
                    {matched if matched else "NO MATCH"}
                </div>
                <div style='font-family:Orbitron,monospace; font-size:0.55rem; color:#444;
                            letter-spacing:2px;'>
                    cluster score <span style='color:#cc0000;'>{count:,}</span>
                    &nbsp;·&nbsp;
                    <span style='color:#555;'>{ratio}× the runner-up</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Audio playback of uploaded clip
            st.markdown("""
            <div style='font-family:Orbitron,monospace; font-size:0.5rem; color:#333;
                        letter-spacing:3px; margin-bottom:0.5rem;'>QUERY AUDIO</div>
            """, unsafe_allow_html=True)
            st.audio(audio_bytes, format="audio/mp3")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── CANDIDATE SCORES ──
            st.markdown("""
            <div style='font-family:Orbitron,monospace; font-size:0.55rem; color:#333;
                        letter-spacing:3px; margin-bottom:1rem;'>CANDIDATE SCORES</div>
            """, unsafe_allow_html=True)

            top5 = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            max_score = top5[0][1] if top5 else 1
            for song_name, score in top5:
                bar_pct = int((score / max_score) * 100)
                is_match = song_name == matched
                st.markdown(f"""
                <div style='display:flex; align-items:center; gap:1rem; padding:0.5rem 0;
                            border-bottom:1px solid rgba(255,255,255,0.03);'>
                    <div style='font-family:Rajdhani,sans-serif; font-size:0.95rem;
                                color:{"#e0e0e0" if is_match else "#333"};
                                min-width:250px; white-space:nowrap; overflow:hidden;
                                text-overflow:ellipsis;'>{song_name}</div>
                    <div style='flex:1; height:4px; background:rgba(255,255,255,0.04);'>
                        <div style='height:100%; width:{bar_pct}%;
                                    background:{"#cc0000" if is_match else "#1a1a1a"};'></div>
                    </div>
                    <div style='font-family:Orbitron,monospace; font-size:0.7rem;
                                color:{"#cc0000" if is_match else "#2a2a2a"};
                                min-width:50px; text-align:right;'>{score}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br><br>", unsafe_allow_html=True)

            # ── STEP 1: SPECTROGRAM + CONSTELLATION ──
            st.markdown(f"""
            <div style='margin-bottom:1rem;'>
                <div style='font-family:Orbitron,monospace; font-size:0.5rem; color:#cc0000;
                            letter-spacing:3px;'>STEP 1 · FEATURE EXTRACTION</div>
                <div style='font-family:Rajdhani,sans-serif; font-size:1.3rem; color:#e0e0e0;
                            font-weight:600; margin:0.3rem 0;'>From spectrogram to constellation</div>
                <div style='font-family:Inter,sans-serif; font-size:0.8rem; color:#444;
                            line-height:1.7; margin-bottom:1rem;'>
                    The clip was converted into a time-frequency map (left); brighter means
                    louder at that frequency and moment. From that rich image, only the
                    <span style='color:#cc0000;'>{len(peaks)} most prominent peaks</span>
                    were kept (right). Discarding amplitude and phase makes the fingerprint
                    robust to EQ, volume changes, and mild noise.
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            bg = '#0d0d0d'

            with col1:
                fig, ax = make_fig(10, 5, bg, dpi=180)
                ax.imshow(S_db, origin='lower', aspect='auto',
                          cmap='inferno', vmin=-80, vmax=0)
                ax.set_xlabel('time (s)', color='#444', fontsize=9)
                ax.set_ylabel('freq (Hz)', color='#444', fontsize=9)
                ax.tick_params(colors='#333', labelsize=8)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col2:
                fig, ax = make_fig(10, 5, bg, dpi=180)
                if len(peaks) > 0:
                    ax.scatter(peaks[:,1], peaks[:,0], s=6,
                               c='#00ffcc', alpha=0.8, linewidths=0)
                ax.set_xlim(0, S_db.shape[1])
                ax.set_ylim(0, S_db.shape[0])
                ax.set_xlabel('time (s)', color='#444', fontsize=9)
                ax.set_ylabel('freq (Hz)', color='#444', fontsize=9)
                ax.tick_params(colors='#333', labelsize=8)
                ax.text(0.98, 0.02, f'{len(peaks)} peaks',
                        transform=ax.transAxes, color='#444',
                        fontsize=8, ha='right', va='bottom', fontfamily='monospace')
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            st.markdown("<br>", unsafe_allow_html=True)

            # ── STEP 2: DATABASE SEARCH ──
            if matched and matched in st.session_state.song_peaks:
                st.markdown(f"""
                <div style='margin-bottom:1rem;'>
                    <div style='font-family:Orbitron,monospace; font-size:0.5rem; color:#cc0000;
                                letter-spacing:3px;'>STEP 2 · DATABASE SEARCH</div>
                    <div style='font-family:Rajdhani,sans-serif; font-size:1.3rem; color:#e0e0e0;
                                font-weight:600; margin:0.3rem 0;'>Where in the song?</div>
                    <div style='font-family:Inter,sans-serif; font-size:0.8rem; color:#444;
                                line-height:1.7; margin-bottom:1rem;'>
                        The <span style='color:#cc0000;'>{timing['n_hashes']:,} fingerprint hashes</span>
                        were looked up against every indexed track. Below is the full fingerprint
                        of <em style='color:#888;'>{matched}</em> reconstructed from the database.
                        The highlighted window is exactly where the query clip sits inside the full song.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                db_peaks = st.session_state.song_peaks[matched]
                fig, ax = make_fig(16, 5, bg, dpi=150)
                if len(db_peaks) > 0:
                    ax.scatter(db_peaks[:,1], db_peaks[:,0], s=0.8,
                               c='#00aaff', alpha=0.5, linewidths=0)
                if offsets:
                    best_off = max(set(offsets), key=offsets.count)
                    q_end = best_off + S_db.shape[1]
                    ax.axvspan(best_off, q_end, alpha=0.12,
                               color='#cc0000')
                    ax.axvline(best_off, color='#cc0000', linewidth=1.5, alpha=0.7)
                    ax.axvline(q_end, color='#cc0000', linewidth=1.5, alpha=0.7)
                ax.set_xlabel('time (frames)', color='#444', fontsize=9)
                ax.set_ylabel('freq bin', color='#444', fontsize=9)
                ax.tick_params(colors='#333', labelsize=8)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # Play matched song if available
                if matched in st.session_state.song_files_data:
                    st.markdown("""
                    <div style='font-family:Orbitron,monospace; font-size:0.5rem; color:#333;
                                letter-spacing:3px; margin:0.75rem 0 0.5rem 0;'>
                        MATCHED SONG · NOW PLAYING
                    </div>
                    """, unsafe_allow_html=True)
                    st.audio(st.session_state.song_files_data[matched],
                             format="audio/mp3")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── STEP 3: ALIGNMENT SPIKE ──
            st.markdown(f"""
            <div style='margin-bottom:1rem;'>
                <div style='font-family:Orbitron,monospace; font-size:0.5rem; color:#cc0000;
                            letter-spacing:3px;'>STEP 3 · THE PROOF</div>
                <div style='font-family:Rajdhani,sans-serif; font-size:1.3rem; color:#e0e0e0;
                            font-weight:600; margin:0.3rem 0;'>The alignment spike</div>
                <div style='font-family:Inter,sans-serif; font-size:0.8rem; color:#444;
                            line-height:1.7; margin-bottom:1rem;'>
                    Every matched hash votes for a time offset (database frame minus query frame).
                    Chance matches scatter votes randomly, forming a flat noise floor. A genuine
                    match makes them converge:
                    <span style='color:#cc0000; font-weight:600;'>{count:,} hashes agreed on a
                    single offset.</span> That spike cannot be a coincidence.
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig, ax = make_fig(16, 5, bg, dpi=150)
            if offsets:
                offset_counts = defaultdict(int)
                for o in offsets: offset_counts[o] += 1
                best_offset = max(offset_counts, key=offset_counts.get)
                peak_count = offset_counts[best_offset]
                xs = sorted(offset_counts.keys())
                ys = [offset_counts[x] for x in xs]
                ax.bar(xs, ys, width=2, color='#0d2a1a',
                       edgecolor='none', alpha=0.9)
                ax.bar([best_offset], [peak_count],
                       width=8, color='#cc8800', edgecolor='none')
                offset_range = max(xs) - min(xs) if xs else 1
                ax.annotate(
                    f'{peak_count:,} hashes\nalign here',
                    xy=(best_offset, peak_count),
                    xytext=(best_offset + offset_range*0.08, peak_count * 0.75),
                    color='#cc8800', fontsize=8, fontfamily='monospace',
                    arrowprops=dict(arrowstyle='->', color='#cc8800', lw=1.2)
                )
                ax.text(0.98, 0.08, 'chance matches\n(noise floor)',
                        transform=ax.transAxes, color='#333',
                        fontsize=7, ha='right', fontfamily='monospace')
            ax.set_xlabel('time offset  (database frame − query frame)',
                         color='#444', fontsize=9)
            ax.set_ylabel('# hashes', color='#444', fontsize=9)
            ax.tick_params(colors='#333', labelsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

# ═══════════════════════════════════════════════════════════════
# TAB 3 — BATCH
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style='font-family:Orbitron,monospace; font-size:0.6rem; color:#333;
                letter-spacing:4px; margin-bottom:0.5rem;'>BATCH</div>
    <div style='font-family:Rajdhani,sans-serif; font-size:1.6rem; color:#e0e0e0;
                font-weight:600; margin-bottom:0.5rem;'>Identify many clips at once</div>
    <div style='font-family:Inter,sans-serif; font-size:0.82rem; color:#444;
                line-height:1.6; margin-bottom:1.5rem;'>
        Upload a set of query clips. Each is identified against the
        <span style='color:#888;'>currently indexed library</span>, and the results
        are written to a standardised
        <code style='color:#cc0000; background:rgba(180,0,0,0.1);
                     padding:0.1rem 0.4rem;'>results.csv</code>
        with columns
        <code style='color:#cc0000; background:rgba(180,0,0,0.1);
                     padding:0.1rem 0.4rem;'>filename, prediction</code>.
        The prediction is the matched track's filename without its extension,
        or <code style='color:#555;'>none</code> when no candidate clears the threshold.
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.db:
        st.markdown("""
        <div style='border:1px solid rgba(180,0,0,0.2); border-left:3px solid #cc0000;
                    padding:1.5rem; background:rgba(180,0,0,0.03); border-radius:2px;
                    font-family:Rajdhani,sans-serif; color:#555; letter-spacing:2px;'>
            ⬡ DATABASE OFFLINE · Go to LIBRARY tab to index songs first
        </div>
        """, unsafe_allow_html=True)
    else:
        batch_files = st.file_uploader(
            "Upload query clips",
            type=["mp3", "wav", "flac", "m4a"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if st.button("⬡ RUN BATCH"):
            if not batch_files:
                st.warning("Upload some files first!")
            else:
                results = []
                pb = st.progress(0, text=f"Identifying... 0/{len(batch_files)}")
                for i, f in enumerate(batch_files):
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    matched, score, _, _, _, _, _, _ = identify_song(
                        tmp_path, st.session_state.db)
                    prediction = matched if (matched and score > 10) else "none"
                    results.append({"filename": f.name, "prediction": prediction})
                    pb.progress(
                        (i+1)/len(batch_files),
                        text=f"Identifying... {i+1}/{len(batch_files)}"
                    )
                pb.empty()

                matched_count = sum(1 for r in results if r['prediction'] != 'none')
                st.markdown("""
                <div style='font-family:Orbitron,monospace; font-size:0.55rem; color:#333;
                            letter-spacing:4px; margin:1.5rem 0 1rem 0;'>RESULTS</div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <table>
                <tr><th>FILE</th><th>PREDICTION</th></tr>
                """ + "\n".join(
                    f"<tr><td>{r['filename']}</td>"
                    f"<td style='color:{'#cc0000' if r['prediction']!='none' else '#333'};'>"
                    f"{r['prediction']}</td></tr>"
                    for r in results
                ) + "</table>", unsafe_allow_html=True)

                st.markdown(f"""
                <div style='font-family:Rajdhani,sans-serif; font-size:0.8rem; color:#444;
                            letter-spacing:1px; margin:1rem 0;'>
                    {matched_count} / {len(results)} clips matched
                    ({len(results)-matched_count} returned none).
                </div>
                """, unsafe_allow_html=True)

                csv_lines = "filename,prediction\n" + "\n".join(
                    f"{r['filename']},{r['prediction']}" for r in results
                )
                st.download_button(
                    "⬡ DOWNLOAD results.csv",
                    csv_lines,
                    "results.csv",
                    mime="text/csv"
                )