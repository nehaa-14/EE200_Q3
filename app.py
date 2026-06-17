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

SONGS_FOLDER = os.path.expanduser("~/EE200_Q3/songs")
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

def load_database():
    db_path = os.path.join(os.path.expanduser("~/EE200_Q3"), "database.pkl")
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            return pickle.load(f)
    return {}

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

# --- Streamlit UI ---
st.set_page_config(page_title="Zapptain America", layout="wide")
st.title("🎵 Zapptain America — Song Identifier")
st.markdown("Upload a song clip and identify it using audio fingerprinting!")

db = load_database()
if not db:
    st.error("Database not found! Run fingerprint.py first.")
else:
    st.success(f"Database loaded: {len(db)} hashes")

mode = st.radio("Select Mode", ["Single Clip", "Batch Mode"])

if mode == "Single Clip":
    uploaded = st.file_uploader("Upload a query clip (.mp3)", type=["mp3"])
    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Identifying..."):
            matched, count, scores, S_db, peaks, offsets = identify_song(tmp_path, db)

        st.success(f"🎯 Match: **{matched}** (score: {count})")

        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(S_db, origin='lower', aspect='auto', cmap='magma')
            ax.set_title('Spectrogram')
            ax.set_xlabel('Time bins')
            ax.set_ylabel('Frequency bins')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_facecolor('black')
            if len(peaks) > 0:
                ax.scatter(peaks[:, 1], peaks[:, 0], s=1, c='cyan', alpha=0.5)
            ax.set_title('Constellation Map')
            ax.set_xlabel('Time bins')
            ax.set_ylabel('Frequency bins')
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            if offsets:
                ax.hist(offsets, bins=100, color='orange', edgecolor='black')
            ax.set_title(f'Offset Histogram')
            ax.set_xlabel('Time Offset')
            ax.set_ylabel('Count')
            st.pyplot(fig)

else:
    uploaded_files = st.file_uploader("Upload multiple query clips", 
                                       type=["mp3"], 
                                       accept_multiple_files=True)
    if uploaded_files and st.button("Run Batch Identification"):
        results = []
        progress = st.progress(0)
        for i, f in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            matched, _, _, _, _, _ = identify_song(tmp_path, db)
            results.append((f.name, matched))
            progress.progress((i+1)/len(uploaded_files))

        st.success("Done!")
        st.table(results)

        csv_path = os.path.expanduser("~/EE200_Q3/results.csv")
        with open(csv_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["filename", "prediction"])
            for fname, pred in results:
                writer.writerow([fname, pred])
        st.info(f"results.csv saved!")