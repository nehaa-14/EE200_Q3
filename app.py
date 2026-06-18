import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import maximum_filter
from collections import defaultdict
import os

# ─── CONFIG ───────────────────────────────────────────────────
SONGS_FOLDER = os.path.expanduser("~/EE200_Q3/songs")
OUTPUT_FOLDER = os.path.expanduser("~/EE200_Q3/report_plots")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Pick one song for analysis
SONG_NAME = "Yesterday.mp3"
SONG_PATH = os.path.join(SONGS_FOLDER, SONG_NAME)

STYLE = {
    'bg': '#080808',
    'red': '#cc0000',
    'dim': '#333333',
    'text': '#888888',
}

def setup_ax(ax, title=""):
    ax.set_facecolor(STYLE['bg'])
    if title:
        ax.set_title(title, color=STYLE['red'], fontsize=9,
                     fontfamily='monospace', pad=10, fontweight='bold')
    ax.tick_params(colors=STYLE['dim'], labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a1a1a')

def save(fig, name):
    path = os.path.join(OUTPUT_FOLDER, name)
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=STYLE['bg'], edgecolor='none')
    plt.close(fig)
    print(f"Saved: {name}")

print("Loading audio...")
y, sr = librosa.load(SONG_PATH, sr=22050, mono=True)
print(f"Loaded: {SONG_NAME} | sr={sr} | duration={len(y)/sr:.1f}s")

# ═══════════════════════════════════════════════════════════════
# PLOT 1 — DFT of entire song (why it fails)
# ═══════════════════════════════════════════════════════════════
print("\n[1/7] DFT magnitude plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=STYLE['bg'])
fig.suptitle("PLOT 1 · WHY A SINGLE DFT IS NOT ENOUGH",
             color=STYLE['red'], fontsize=10, fontfamily='monospace',
             fontweight='bold', y=1.02)

# Time domain
axes[0].plot(np.linspace(0, len(y)/sr, len(y)), y,
             color='#cc0000', linewidth=0.3, alpha=0.8)
setup_ax(axes[0], "TIME DOMAIN WAVEFORM")
axes[0].set_xlabel("Time (seconds)", color=STYLE['text'], fontsize=8)
axes[0].set_ylabel("Amplitude", color=STYLE['text'], fontsize=8)

# DFT magnitude
Y = np.fft.rfft(y)
freqs = np.fft.rfftfreq(len(y), 1/sr)
magnitude_db = 20 * np.log10(np.abs(Y) + 1e-10)
axes[1].plot(freqs, magnitude_db, color='#cc0000', linewidth=0.5, alpha=0.9)
setup_ax(axes[1], "DFT MAGNITUDE SPECTRUM (tells WHAT, not WHEN)")
axes[1].set_xlabel("Frequency (Hz)", color=STYLE['text'], fontsize=8)
axes[1].set_ylabel("Magnitude (dB)", color=STYLE['text'], fontsize=8)
axes[1].set_xlim(0, 8000)
axes[1].axvline(x=1000, color='#440000', linewidth=0.5, linestyle='--')

fig.tight_layout()
save(fig, "1_dft_magnitude.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 2 — Spectrogram: short vs long window
# ═══════════════════════════════════════════════════════════════
print("[2/7] Short vs long window spectrogram...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor=STYLE['bg'])
fig.suptitle("PLOT 2 · SPECTROGRAM: SHORT vs LONG WINDOW",
             color=STYLE['red'], fontsize=10, fontfamily='monospace',
             fontweight='bold', y=1.02)

configs = [
    (256,  64,  "SHORT WINDOW (n=256)\nGood time res, poor freq res"),
    (2048, 512, "MEDIUM WINDOW (n=2048)\nBalanced — used for fingerprinting"),
    (8192, 2048,"LONG WINDOW (n=8192)\nGood freq res, poor time res"),
]

clip = y[:sr*30]  # first 30 seconds
for ax, (n_fft, hop, title) in zip(axes, configs):
    S = np.abs(librosa.stft(clip, n_fft=n_fft, hop_length=hop))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    img = ax.imshow(S_db, origin='lower', aspect='auto',
                    cmap='inferno', vmin=-80, vmax=0,
                    extent=[0, 30, 0, sr/2/1000])
    setup_ax(ax, title)
    ax.set_xlabel("Time (s)", color=STYLE['text'], fontsize=8)
    ax.set_ylabel("Frequency (kHz)", color=STYLE['text'], fontsize=8)
    plt.colorbar(img, ax=ax, format='%+2.0f dB').ax.yaxis.set_tick_params(color=STYLE['dim'])

fig.tight_layout()
save(fig, "2_spectrogram_windows.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 3 — Constellation map
# ═══════════════════════════════════════════════════════════════
print("[3/7] Constellation map...")
n_fft, hop = 4096, 512
S = np.abs(librosa.stft(clip, n_fft=n_fft, hop_length=hop))
S_db = librosa.amplitude_to_db(S, ref=np.max)

local_max = maximum_filter(S_db, size=20) == S_db
peaks = np.argwhere(local_max & (S_db > -60))

fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=STYLE['bg'])
fig.suptitle("PLOT 3 · CONSTELLATION MAP — PEAK FINGERPRINTS",
             color=STYLE['red'], fontsize=10, fontfamily='monospace',
             fontweight='bold', y=1.02)

axes[0].imshow(S_db, origin='lower', aspect='auto', cmap='inferno')
setup_ax(axes[0], "SPECTROGRAM (n_fft=4096)")
axes[0].set_xlabel("Time bins", color=STYLE['text'], fontsize=8)
axes[0].set_ylabel("Frequency bins", color=STYLE['text'], fontsize=8)

axes[1].set_facecolor(STYLE['bg'])
if len(peaks) > 0:
    axes[1].scatter(peaks[:,1], peaks[:,0], s=1.5,
                    c='#cc0000', alpha=0.7, linewidths=0)
setup_ax(axes[1], f"CONSTELLATION MAP ({len(peaks)} peaks)")
axes[1].set_xlabel("Time bins", color=STYLE['text'], fontsize=8)
axes[1].set_ylabel("Frequency bins", color=STYLE['text'], fontsize=8)
axes[1].set_xlim(0, S_db.shape[1])
axes[1].set_ylim(0, S_db.shape[0])

fig.tight_layout()
save(fig, "3_constellation.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 4 — Paired hashes vs single peaks matching
# ═══════════════════════════════════════════════════════════════
print("[4/7] Hash matching comparison...")

def get_peaks(S_db, neighborhood=20, threshold=-60):
    local_max = maximum_filter(S_db, size=neighborhood) == S_db
    return np.argwhere(local_max & (S_db > threshold))

def hash_paired(peaks, fan_out=15, dt_max=200):
    ps = sorted(peaks.tolist(), key=lambda x: x[1])
    hashes = []
    for i, (f1,t1) in enumerate(ps):
        for j in range(1, fan_out+1):
            if i+j >= len(ps): break
            f2,t2 = ps[i+j]
            dt = t2-t1
            if dt<=0 or dt>dt_max: continue
            hashes.append((hash((int(f1),int(f2),int(dt))), t1))
    return hashes

def hash_single(peaks):
    return [(hash((int(f), int(t))), t) for f,t in peaks]

# Build mini db from 5 songs
print("  Building mini database...")
db_paired = defaultdict(list)
db_single = defaultdict(list)
song_files = [f for f in os.listdir(SONGS_FOLDER) if f.endswith('.mp3')][:5]

for fname in song_files:
    sname = os.path.splitext(fname)[0]
    ys, _ = librosa.load(os.path.join(SONGS_FOLDER, fname), sr=22050, mono=True)
    S = np.abs(librosa.stft(ys, n_fft=4096, hop_length=512))
    Sdb = librosa.amplitude_to_db(S, ref=np.max)
    pk = get_peaks(Sdb)
    for h,t in hash_paired(pk): db_paired[h].append((sname,t))
    for h,t in hash_single(pk): db_single[h].append((sname,t))

# Query = first song with noise
yq, _ = librosa.load(os.path.join(SONGS_FOLDER, song_files[0]), sr=22050, mono=True)
yq = yq + 0.01 * np.random.randn(len(yq))
Sq = np.abs(librosa.stft(yq, n_fft=4096, hop_length=512))
Sdbq = librosa.amplitude_to_db(Sq, ref=np.max)
pkq = get_peaks(Sdbq)

def get_offsets(hashes_q, db):
    scores = defaultdict(list)
    for h,tq in hashes_q:
        if h in db:
            for sname,tdb in db[h]:
                scores[sname].append(tdb-tq)
    return scores

scores_p = get_offsets(hash_paired(pkq), db_paired)
scores_s = get_offsets(hash_single(pkq), db_single)
true_name = os.path.splitext(song_files[0])[0]

fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=STYLE['bg'])
fig.suptitle("PLOT 4 · PAIRED HASHES vs SINGLE PEAKS MATCHING",
             color=STYLE['red'], fontsize=10, fontfamily='monospace',
             fontweight='bold', y=1.02)

for ax, scores, title, color in zip(
    axes,
    [scores_p, scores_s],
    ["PAIRED HASHES (decisive spike)", "SINGLE PEAKS (scattered, noisy)"],
    ['#cc0000', '#884400']
):
    if true_name in scores:
        ax.hist(scores[true_name], bins=200, color=color,
                edgecolor='none', alpha=0.85)
    setup_ax(ax, title)
    ax.set_xlabel("Time Offset", color=STYLE['text'], fontsize=8)
    ax.set_ylabel("Match Count", color=STYLE['text'], fontsize=8)

fig.tight_layout()
save(fig, "4_paired_vs_single.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 5 — Noise robustness
# ═══════════════════════════════════════════════════════════════
print("[5/7] Noise robustness test...")

def match_score(y_query, db, sr=22050):
    S = np.abs(librosa.stft(y_query, n_fft=4096, hop_length=512))
    Sdb = librosa.amplitude_to_db(S, ref=np.max)
    pk = get_peaks(Sdb)
    hashes = hash_paired(pk)
    scores = defaultdict(list)
    for h,tq in hashes:
        if h in db:
            for sname,tdb in db[h]: scores[sname].append(tdb-tq)
    best, best_c = None, 0
    for sname, offsets in scores.items():
        c = defaultdict(int)
        for o in offsets: c[o]+=1
        top = max(c.values()) if c else 0
        if top > best_c: best_c=top; best=sname
    return best, best_c

noise_levels = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
correct = []
scores_list = []
true = os.path.splitext(song_files[0])[0]
ybase, _ = librosa.load(os.path.join(SONGS_FOLDER, song_files[0]), sr=22050, mono=True)

for nl in noise_levels:
    yn = ybase + nl * np.random.randn(len(ybase))
    m, sc = match_score(yn, db_paired)
    correct.append(1 if m==true else 0)
    scores_list.append(sc)
    print(f"  noise={nl:.3f} → {m} (score={sc})")

fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=STYLE['bg'])
fig.suptitle("PLOT 5 · NOISE ROBUSTNESS TEST",
             color=STYLE['red'], fontsize=10, fontfamily='monospace',
             fontweight='bold', y=1.02)

axes[0].plot(noise_levels, correct, 'o-', color='#cc0000',
             linewidth=2, markersize=8, markerfacecolor='#ff3333')
axes[0].fill_between(noise_levels, correct, alpha=0.1, color='#cc0000')
setup_ax(axes[0], "CORRECT IDENTIFICATION vs NOISE LEVEL")
axes[0].set_xlabel("Noise Amplitude (σ)", color=STYLE['text'], fontsize=8)
axes[0].set_ylabel("Correct (1) / Failed (0)", color=STYLE['text'], fontsize=8)
axes[0].set_ylim(-0.1, 1.2)

axes[1].plot(noise_levels, scores_list, 's-', color='#cc0000',
             linewidth=2, markersize=8, markerfacecolor='#ff3333')
setup_ax(axes[1], "CONFIDENCE SCORE vs NOISE LEVEL")
axes[1].set_xlabel("Noise Amplitude (σ)", color=STYLE['text'], fontsize=8)
axes[1].set_ylabel("Peak Offset Count", color=STYLE['text'], fontsize=8)

fig.tight_layout()
save(fig, "5_noise_robustness.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 6 — Pitch shift robustness
# ═══════════════════════════════════════════════════════════════
print("[6/7] Pitch shift test...")
semitones = [0, 0.5, 1, 2, 3, 5]
correct_pitch = []
scores_pitch = []

for st_shift in semitones:
    if st_shift == 0:
        yp = ybase.copy()
    else:
        yp = librosa.effects.pitch_shift(ybase, sr=22050, n_steps=st_shift)
    m, sc = match_score(yp, db_paired)
    correct_pitch.append(1 if m==true else 0)
    scores_pitch.append(sc)
    print(f"  pitch_shift={st_shift} semitones → {m} (score={sc})")

fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=STYLE['bg'])
fig.suptitle("PLOT 6 · PITCH SHIFT ROBUSTNESS TEST",
             color=STYLE['red'], fontsize=10, fontfamily='monospace',
             fontweight='bold', y=1.02)

axes[0].plot(semitones, correct_pitch, 'o-', color='#cc0000',
             linewidth=2, markersize=8, markerfacecolor='#ff3333')
axes[0].fill_between(semitones, correct_pitch, alpha=0.1, color='#cc0000')
setup_ax(axes[0], "CORRECT ID vs PITCH SHIFT")
axes[0].set_xlabel("Pitch Shift (semitones)", color=STYLE['text'], fontsize=8)
axes[0].set_ylabel("Correct (1) / Failed (0)", color=STYLE['text'], fontsize=8)
axes[0].set_ylim(-0.1, 1.2)

axes[1].plot(semitones, scores_pitch, 's-', color='#cc0000',
             linewidth=2, markersize=8, markerfacecolor='#ff3333')
setup_ax(axes[1], "CONFIDENCE SCORE vs PITCH SHIFT")
axes[1].set_xlabel("Pitch Shift (semitones)", color=STYLE['text'], fontsize=8)
axes[1].set_ylabel("Peak Offset Count", color=STYLE['text'], fontsize=8)

fig.tight_layout()
save(fig, "6_pitch_shift.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 7 — Full pipeline summary
# ═══════════════════════════════════════════════════════════════
print("[7/7] Full pipeline summary...")
fig = plt.figure(figsize=(20, 10), facecolor=STYLE['bg'])
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

# Waveform
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(np.linspace(0,30,sr*30), ybase[:sr*30],
         color='#cc0000', linewidth=0.3, alpha=0.8)
setup_ax(ax1, "1. WAVEFORM")
ax1.set_xlabel("Time (s)", color=STYLE['text'], fontsize=7)

# DFT
ax2 = fig.add_subplot(gs[0,1])
Yc = np.fft.rfft(ybase[:sr*30])
fc = np.fft.rfftfreq(sr*30, 1/sr)
ax2.plot(fc, 20*np.log10(np.abs(Yc)+1e-10),
         color='#cc0000', linewidth=0.4, alpha=0.8)
setup_ax(ax2, "2. DFT MAGNITUDE")
ax2.set_xlabel("Frequency (Hz)", color=STYLE['text'], fontsize=7)
ax2.set_xlim(0,8000)

# Spectrogram
ax3 = fig.add_subplot(gs[0,2])
Sc = np.abs(librosa.stft(ybase[:sr*30], n_fft=4096, hop_length=512))
Sdbc = librosa.amplitude_to_db(Sc, ref=np.max)
ax3.imshow(Sdbc, origin='lower', aspect='auto', cmap='inferno')
setup_ax(ax3, "3. SPECTROGRAM")
ax3.set_xlabel("Time bins", color=STYLE['text'], fontsize=7)
ax3.set_ylabel("Freq bins", color=STYLE['text'], fontsize=7)

# Constellation
ax4 = fig.add_subplot(gs[1,0])
pkc = get_peaks(Sdbc)
ax4.scatter(pkc[:,1], pkc[:,0], s=0.8, c='#cc0000', alpha=0.6)
setup_ax(ax4, "4. CONSTELLATION MAP")
ax4.set_facecolor(STYLE['bg'])
ax4.set_xlabel("Time bins", color=STYLE['text'], fontsize=7)

# Offset histogram
ax5 = fig.add_subplot(gs[1,1])
sc_full = get_offsets(hash_paired(pkc), db_paired)
if true in sc_full:
    ax5.hist(sc_full[true], bins=200, color='#cc0000',
             edgecolor='none', alpha=0.85)
setup_ax(ax5, "5. OFFSET HISTOGRAM (match spike)")
ax5.set_xlabel("Offset", color=STYLE['text'], fontsize=7)

# Score bar chart
ax6 = fig.add_subplot(gs[1,2])
if sc_full:
    top5 = sorted(sc_full.items(),
                  key=lambda x: max(defaultdict(int, {o:1 for o in x[1]}).values()),
                  reverse=True)[:5]
    names = [x[0][:15] for x in top5]
    vals = [max(defaultdict(int, {o:1 for o in x[1]}).values()) for x in top5]
    bars = ax6.barh(names, vals, color=['#cc0000']+['#330000']*4)
    setup_ax(ax6, "6. TOP MATCHES")
    ax6.set_xlabel("Peak Count", color=STYLE['text'], fontsize=7)
    ax6.tick_params(axis='y', labelsize=6)

fig.suptitle("COMPLETE FINGERPRINTING PIPELINE",
             color=STYLE['red'], fontsize=12,
             fontfamily='monospace', fontweight='bold')
save(fig, "7_full_pipeline.png")

print(f"\n✓ All plots saved to: {OUTPUT_FOLDER}")
print("Open ~/EE200_Q3/report_plots/ to see all 7 plots!")