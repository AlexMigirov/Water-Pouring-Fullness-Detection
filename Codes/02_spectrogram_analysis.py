"""
02_spectrogram_analysis.py
==========================
Step 2 of the Water-Level Detection ML Pipeline.

Purpose:
  - Generate frequency-tracker spectrograms for each of the 4 cup classes
    using the mel-spectrogram approach from Spectrogram_Code.txt
  - Output one figure per class showing the average spectrogram + linear
    frequency track + valid/rejected Pathfinder points

Outputs (in outputs/figures/):
  - Tracker_Thick_Glass.png
  - Tracker_Tall_Thin_Glass.png
  - Tracker_Ceramic_Cup.png
  - Tracker_Plastic_Cup.png

Note:
  The MFCC features themselves are already extracted in master_mfcc_dataset.xlsx.
  This script only produces the *visual* spectrogram analysis.

Usage:
  python 02_spectrogram_analysis.py
"""

import matplotlib
matplotlib.use('Agg')

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import av
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = PROJECT_ROOT / "הקלטות"
OUT_DIR        = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset group rules — matches the original Spectrogram_Code.txt
group_rules = [
    {"range": range(1,  16), "type": "Thick_Glass",     "label": "Thick Glass (Class 1)"},
    {"range": range(16, 31), "type": "Tall_Thin_Glass",  "label": "Tall Thin Glass (Class 2)"},
    {"range": range(31, 46), "type": "Ceramic_Cup",      "label": "Ceramic Cup (Class 3)"},
    {"range": range(46, 61), "type": "Plastic_Cup",      "label": "Plastic Cup (Class 4)"},
]

plt.rcParams.update({
    "figure.dpi":   100,
    "savefig.dpi":  300,
    "font.family":  "DejaVu Sans",
})

# ─────────────────────────────────────────────────────────────────────────────
# Audio loading helper
# ─────────────────────────────────────────────────────────────────────────────
def load_audio_from_mp4(filepath: Path):
    """Decode audio from an .mp4 file using PyAV.  Returns (y, sr)."""
    container = av.open(str(filepath))
    if not container.streams.audio:
        container.close()
        return None, None
    audio_stream = container.streams.audio[0]
    sr = audio_stream.rate
    chunks = []
    for frame in container.decode(audio_stream):
        arr = frame.to_ndarray()
        if arr.ndim > 1:
            arr = np.mean(arr, axis=0)
        chunks.append(arr)
    container.close()
    y = np.concatenate(chunks).astype(np.float32)
    max_vol = np.max(np.abs(y))
    if max_vol > 0:
        y /= max_vol
    return y, sr


# ─────────────────────────────────────────────────────────────────────────────
# Main: process each group
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 2 – Spectrogram & Frequency-Tracker Analysis")
print("="*65)

sample_rate_global = None

for g in group_rules:
    print(f"\n→ Processing group: {g['label']}")
    group_specs   = []
    max_frames    = 0

    for file_num in g["range"]:
        mp4_path = RECORDINGS_DIR / f"{file_num}.mp4"
        if not mp4_path.exists():
            print(f"   [SKIP] {mp4_path} not found")
            continue

        try:
            y, sr = load_audio_from_mp4(mp4_path)
            if y is None:
                continue
            if sample_rate_global is None:
                sample_rate_global = sr

            S    = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
            S_dB = librosa.power_to_db(S, ref=np.max)
            group_specs.append(S_dB)
            if S_dB.shape[1] > max_frames:
                max_frames = S_dB.shape[1]
            print(f"   ✓ {mp4_path.name}  ({S_dB.shape[1]} frames, {len(y)/sr:.1f}s)")

        except Exception as exc:
            print(f"   [ERROR] {mp4_path.name}: {exc}")

    if not group_specs:
        print(f"   [SKIP] No valid spectrograms for {g['type']}")
        continue

    # ── Average spectrogram ───────────────────────────────────────────────
    padded = []
    for S in group_specs:
        pad = max_frames - S.shape[1]
        if pad > 0:
            S = np.pad(S, ((0,0),(0,pad)), mode='constant', constant_values=-100)
        padded.append(S)
    avg_spec = np.mean(padded, axis=0)

    # Background subtraction
    bg          = np.median(avg_spec, axis=1, keepdims=True)
    clean_spec  = avg_spec - bg

    # Mel frequency grid & time axis
    mel_freqs  = librosa.mel_frequencies(n_mels=avg_spec.shape[0],
                                         fmin=0.0, fmax=sample_rate_global / 2)
    times      = librosa.frames_to_time(np.arange(avg_spec.shape[1]),
                                        sr=sample_rate_global, hop_length=512)
    max_time_s = librosa.frames_to_time(max_frames,
                                        sr=sample_rate_global, hop_length=512)

    # ── Pathfinder frequency tracker ──────────────────────────────────────
    good_x, good_y     = [], []
    rejected_x, rejected_y = [], []
    current_f, current_t   = None, None
    step_plot = 15
    eval_indices = np.arange(0, avg_spec.shape[1], step_plot)

    for col_idx in eval_indices:
        t = times[col_idx]
        if t < 1.5 or t > max_time_s:
            continue

        column          = clean_spec[:, col_idx]
        global_peak_idx = np.argmax(column)
        global_peak_hz  = mel_freqs[global_peak_idx]

        if current_f is None:
            mask = (mel_freqs >= 512) & (mel_freqs <= 562)
        else:
            dt            = t - current_t
            max_allowed_f = min(current_f + 50.0 * (dt / 0.1), 2048)
            min_allowed_f = current_f
            mask = (mel_freqs >= min_allowed_f) & (mel_freqs <= max_allowed_f)

        if not np.any(mask):
            rejected_x.append(t)
            rejected_y.append(global_peak_hz)
            continue

        subset_idx = np.where(mask)[0]
        best_idx   = subset_idx[np.argmax(column[subset_idx])]
        f_best     = mel_freqs[best_idx]

        good_x.append(t)
        good_y.append(f_best)
        current_f = f_best
        current_t = t

        if mel_freqs[global_peak_idx] not in mel_freqs[subset_idx]:
            rejected_x.append(t)
            rejected_y.append(global_peak_hz)

    # ── Linear trend line ─────────────────────────────────────────────────
    smooth_x = np.linspace(1.5, max_time_s, 150)
    if len(good_x) > 2:
        coeffs    = np.polyfit(good_x, good_y, 1)
        smooth_y  = np.poly1d(coeffs)(smooth_x)
        slope_str = f"slope = {coeffs[0]:.1f} Hz/s"
    else:
        smooth_y  = np.zeros_like(smooth_x)
        slope_str = "slope = N/A"

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    img = librosa.display.specshow(
        avg_spec, sr=sample_rate_global, hop_length=512,
        x_axis='time', y_axis='mel', ax=ax,
    )
    plt.colorbar(img, ax=ax, format='%+2.0f dB', label='Power (dB)')

    if rejected_x:
        ax.scatter(rejected_x, rejected_y, color='red', marker='x',
                   alpha=0.5, s=40, label='Rejected (out-of-tunnel)')
    if good_x:
        ax.scatter(good_x, good_y, color='limegreen', edgecolor='white',
                   s=60, zorder=10, label='Pathfinder valid points')
    if len(good_x) > 2:
        ax.plot(smooth_x, smooth_y, color='white', linewidth=3,
                zorder=9, label=f'Linear track ({slope_str})')

    ax.axvline(x=max_time_s, color='cyan', linestyle='--', linewidth=2.5, zorder=5)
    ax.text(max(max_time_s * 0.72, 0.2), ax.get_ylim()[1] * 0.88,
            f'END  (~{max_time_s:.1f}s)',
            color='cyan', fontsize=11, fontweight='bold', zorder=6)

    ax.set_title(
        f"Average Mel-Spectrogram + Frequency Tracker\n{g['label']}  "
        f"({len(group_specs)} recordings)",
        fontsize=13, fontweight='bold',
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz, Mel scale)")
    ax.legend(loc='upper left', framealpha=0.85)
    fig.tight_layout()

    out_path = OUT_DIR / f"Tracker_{g['type']}.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved → {out_path}")

print("\n" + "="*65)
print("  Step 2 COMPLETE — spectrogram figures saved to figures/")
print("  Next: run  python Codes/03_model_training.py")
print("="*65 + "\n")
