import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from collections import defaultdict
import os
import pickle

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

def build_database():
    db = defaultdict(list)
    print("Building database...")
    for fname in sorted(os.listdir(SONGS_FOLDER)):
        if fname.endswith('.mp3'):
            song_name = os.path.splitext(fname)[0]
            path = os.path.join(SONGS_FOLDER, fname)
            try:
                S_db, sr, hop = compute_spectrogram(path)
                peaks = get_peaks(S_db)
                hashes = hash_peaks(peaks)
                for (h, t) in hashes:
                    db[h].append((song_name, t))
                print(f"  Indexed: {song_name}")
            except Exception as e:
                print(f"  ERROR with {song_name}: {e}")
    with open("database.pkl", "wb") as f:
        pickle.dump(dict(db), f)
    print(f"\nDone! {len(db)} hashes stored.")
    return dict(db)

def load_database():
    if os.path.exists("database.pkl"):
        with open("database.pkl", "rb") as f:
            return pickle.load(f)
    return build_database()

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

def plot_result(S_db, peaks, scores, matched_song, best_offsets, query_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Query: {os.path.basename(query_path)}\nMatch: {matched_song}", fontsize=14)
    axes[0].imshow(S_db, origin='lower', aspect='auto', cmap='magma')
    axes[0].set_title('Spectrogram of Query')
    axes[0].set_xlabel('Time bins')
    axes[0].set_ylabel('Frequency bins')
    if len(peaks) > 0:
        axes[1].scatter(peaks[:, 1], peaks[:, 0], s=1, c='cyan', alpha=0.5)
    axes[1].set_title('Constellation Map')
    axes[1].set_xlabel('Time bins')
    axes[1].set_ylabel('Frequency bins')
    axes[1].set_facecolor('black')
    if best_offsets:
        axes[2].hist(best_offsets, bins=100, color='orange', edgecolor='black')
    axes[2].set_title(f'Offset Histogram\nMatch: {matched_song}')
    axes[2].set_xlabel('Time Offset')
    axes[2].set_ylabel('Count')
    plt.tight_layout()
    plt.savefig('result.png', dpi=150)
    plt.show()

def batch_identify(query_folder, db):
    import csv
    results = []
    for fname in sorted(os.listdir(query_folder)):
        if fname.endswith('.mp3'):
            path = os.path.join(query_folder, fname)
            matched, count, scores, _, _, _ = identify_song(path, db)
            results.append((fname, matched))
            print(f"{fname} → {matched}")
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prediction"])
        for fname, pred in results:
            writer.writerow([fname, pred])
    print("Saved results.csv")

if __name__ == "__main__":
    db = load_database()
    test_song = os.path.join(SONGS_FOLDER, "Yesterday.mp3")
    print(f"\nIdentifying: {test_song}")
    matched, count, scores, S_db, peaks, offsets = identify_song(test_song, db)
    print(f"Result: {matched} (score: {count})")
    plot_result(S_db, peaks, scores, matched, offsets, test_song)