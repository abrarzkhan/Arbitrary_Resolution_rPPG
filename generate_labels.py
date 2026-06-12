"""
generate_labels.py  (adaptive multi-method pseudo-labels)

Picks the cleanest pulse-shape pseudo-label PER CLIP from {POS, CHROM, green}
instead of always using POS. Because the Masimo average HR is known offline,
selection is data-driven: band-pass each method around the true HR and keep the
one with the highest SNR at that frequency. This auto-adapts to skin tone
(POS usually wins on darker skin, green on lighter -- the routing your groupmate
does with ITA), but optimally and per clip.

We also compute the Individual Typology Angle (ITA; Del Bino & Bernerd, BJD 2013)
per clip and log it next to the chosen method, so you can verify the
dark->POS / light->green pattern on your own recordings.

The waveform is only a light SHAPE prior (Pearson weight ~0.1 in train.py). The
authoritative HR target for the frequency loss is your measured Masimo value,
kept verbatim below.

All method waveforms are polarity-aligned to a common physiological convention
(positively correlated with -green, i.e. blood-volume up -> less reflected green)
so the shape prior is consistent across clips.

Requires: numpy, opencv-python, scipy
"""

import os
import glob
import json
import numpy as np
import cv2
from scipy.signal import butter, filtfilt, detrend

# --- measured ground-truth HR per clip (Masimo finger pulse-ox, 20 s average) ---
heart_rates = [
    67.0, 62.0, 61.0, 63.0, 64.0, 66.0, 70.0, 57.0, 54.0, 64.0,
    69.0, 69.0, 71.0, 71.0, 70.0, 73.0, 69.0, 69.0, 74.0, 74.0,
    64.0, 58.0, 66.0, 61.0, 63.0, 59.0, 55.0, 60.0, 65.0, 64.0,
    58.0, 55.0, 59.0, 60.0, 64.0, 60.0, 62.0, 65.0, 67.0, 64.0,
    69.0, 65.0, 68.0, 63.0, 64.0, 68.0, 65.0, 66.0, 65.0, 66.0,
    60.0, 59.0, 61.0, 64.0, 62.0, 61.0, 58.0, 58.0, 60.0, 61.0,
    103.0, 98.0, 103.0, 79.0, 99.0, 103.0, 94.0, 96.0, 99.0, 80.0, 88.0, 87.0, 96.0, 82.0, 96.0,
    47.0, 60.0, 52.0, 50.0, 52.0, 50.0, 49.0, 49.0, 63.0, 50.0,
    81.0, 81.0, 83.0, 92.0, 87.0, 86.0, 90.0, 95.0, 96.0, 96.0,
    93.0, 94.0, 83.0, 102.0, 97.0, 93.0, 97.0, 93.0, 95.0, 93.0, 106.0, 104.0, 107.0, 95.0, 101.0, 102.0, 99.0, 105.0, 104.0, 81.0,
    74.0, 79.0, 72.0, 87.0, 85.0, 88.0, 77.0, 80.0, 80.0, 84.0,
    78.0, 75.0, 76.0, 79.0, 65.0, 76.0, 80.0, 81.0, 79.0, 75.0,
    69.0, 74.0, 67.0, 55.0, 79.0, 69.0, 59.0, 64.0, 67.0, 65.0,
    # ---- clips 145-159: 15 new Advaith (darkest-skin) sessions ----
    83.0, 84.0, 81.0, 92.0, 87.0, 88.0, 81.0, 92.0, 88.0, 94.0, 86.0, 87.0, 87.0, 85.0, 88.0,
]

VIDEO_FPS = 24.0
DATASET_BASE = r"C:\Users\Acema\Random\MNI_Lab\ML_RPPG\Arbitrary_Resolution_rPPG\custom_dataset"
ITA_LIGHT_THRESHOLD = -45.0  # COSMETIC ONLY (affects the printed summary, NOT label/method
                             # selection -- that's by SNR below). Your warm/dim room shifts all
                             # ITA strongly negative (lightest subjects ~-30, darkest ~-68), so the
                             # standard +20 boundary tagged everyone 'dark'. -45 separates your dark
                             # cluster (Aditya/Advaith/Abrar/Adi) from the lighter group. Bridget
                             # straddles it -- absolute ITA can't cleanly tell her from Adi, so for
                             # boundary subjects trust identity (SUBJECT_TONE in test.py).


def skin_mean_rgb_series(frame_files):
    """Per-frame mean RGB over non-black (skin) pixels of each masked crop. (T,3)"""
    series = []
    for fp in frame_files:
        bgr = cv2.imread(fp)
        if bgr is None:
            series.append(series[-1] if series else np.array([0.0, 0.0, 0.0]))
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        mask = lum > 8.0
        if mask.sum() < 25:
            series.append(series[-1] if series else np.array([0.0, 0.0, 0.0]))
            continue
        series.append(rgb[mask].mean(axis=0))
    return np.asarray(series, dtype=np.float64)


def rgb_to_ita(r, g, b):
    """Individual Typology Angle (deg) from sRGB; higher ~ lighter (Del Bino 2013)."""
    rgb = np.array([r, g, b], dtype=np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    x = (0.4124564 * lin[0] + 0.3575761 * lin[1] + 0.1804375 * lin[2]) / 0.95047
    y = (0.2126729 * lin[0] + 0.7151522 * lin[1] + 0.0721750 * lin[2]) / 1.0
    z = (0.0193339 * lin[0] + 0.1191920 * lin[1] + 0.9503041 * lin[2]) / 1.08883
    f = lambda t: np.where(t > 0.008856, np.cbrt(t), (903.3 * t + 16.0) / 116.0)
    fy, fz = f(y), f(z)
    l_star = 116.0 * fy - 16.0
    b_star = 200.0 * (fy - fz)
    return float(np.degrees(np.arctan2(l_star - 50.0, b_star)))


def pos_waveform(rgb, fs=VIDEO_FPS):
    T = rgb.shape[0]
    H = np.zeros(T)
    l = max(8, int(1.6 * fs))
    for n in range(l - 1, T):
        m = n - l + 1
        win = rgb[m:n + 1]
        mu = win.mean(axis=0) + 1e-8
        Cn = win / mu
        s1 = Cn[:, 1] - Cn[:, 2]
        s2 = -2.0 * Cn[:, 0] + Cn[:, 1] + Cn[:, 2]
        sigma = np.std(s1) / (np.std(s2) + 1e-8)
        h = s1 + sigma * s2
        H[m:n + 1] += (h - h.mean())
    return H


def chrom_waveform(rgb, fs=VIDEO_FPS):
    """CHROM (de Haan & Jeanne 2013), global form."""
    mu = rgb.mean(axis=0) + 1e-8
    Rn, Gn, Bn = rgb[:, 0] / mu[0], rgb[:, 1] / mu[1], rgb[:, 2] / mu[2]
    Xs = 3.0 * Rn - 2.0 * Gn
    Ys = 1.5 * Rn + Gn - 1.5 * Bn
    Xf = _bp(Xs, fs); Yf = _bp(Ys, fs)
    alpha = np.std(Xf) / (np.std(Yf) + 1e-8)
    return Xf - alpha * Yf


def green_waveform(rgb, fs=VIDEO_FPS):
    """Inverted green channel (blood volume up -> green reflectance down)."""
    return -detrend(rgb[:, 1], type='linear')


def _bp(sig, fs, low_hz=0.7, high_hz=3.5, order=3):
    nyq = 0.5 * fs
    if len(sig) <= 3 * order + 1:
        return sig
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype='band')
    return filtfilt(b, a, sig)


def bandpass_hr(sig, hr_bpm, fs=VIDEO_FPS, half=15.0, order=3):
    nyq = 0.5 * fs
    low = max(0.5, (hr_bpm - half) / 60.0)
    high = min(nyq - 0.1, (hr_bpm + half) / 60.0)
    if high <= low or len(sig) <= 3 * order + 1:
        return sig
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)


def snr_at_hr(sig, hr_bpm, fs=VIDEO_FPS, nfft=4096):
    s = sig - np.mean(sig)
    if s.std() < 1e-8:
        return -np.inf
    w = (s / s.std()) * np.hanning(len(s))
    p = np.abs(np.fft.rfft(w, nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs) * 60.0
    band = (freqs >= 40) & (freqs <= 180)
    f, p = freqs[band], p[band]
    sigm = (f >= hr_bpm - 5) & (f <= hr_bpm + 5)
    sig_p = float(np.sum(p[sigm]))
    noise_p = max(float(np.sum(p) - sig_p), 1e-12)
    return sig_p / noise_p


def align_polarity(sig, anchor):
    a = anchor - anchor.mean()
    s = sig - sig.mean()
    if np.dot(a, s) < 0:
        return -sig
    return sig


def synthetic_fallback(num_frames, hr_bpm, fs=VIDEO_FPS):
    t = np.arange(num_frames) / fs
    return np.sin(2 * np.pi * (hr_bpm / 60.0) * t)


print(f"Generating adaptive pseudo-labels for {len(heart_rates)} clips...\n")
method_wins = {'POS': 0, 'CHROM': 0, 'green': 0, 'fallback': 0}
dark_methods, light_methods = [], []

for i, hr_val in enumerate(heart_rates):
    images_dir = clip_name = None
    for name in (f"clip_{i:02d}", f"clip_{i}"):
        p = os.path.join(DATASET_BASE, name, "pic", "1.0")
        if os.path.exists(p):
            images_dir, clip_name = p, name
            break
    if images_dir is None:
        print(f"Skipping index {i}: folder not found. Run extract_frames.py first.")
        continue

    frame_files = sorted(glob.glob(os.path.join(images_dir, "*.png")),
                         key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    n = len(frame_files)
    if n == 0:
        print(f"Skipping {clip_name}: no frames.")
        continue

    rgb = skin_mean_rgb_series(frame_files)
    med = np.median(rgb, axis=0)
    ita = rgb_to_ita(*med) if np.all(med > 0) else float('nan')

    anchor = bandpass_hr(green_waveform(rgb), hr_val)  # polarity reference (-green)
    candidates = {}
    try:
        candidates['POS'] = bandpass_hr(pos_waveform(rgb), hr_val)
        candidates['CHROM'] = bandpass_hr(chrom_waveform(rgb), hr_val)
        candidates['green'] = anchor
    except Exception as e:
        candidates = {}

    best_name, best_wave, best_snr = 'fallback', synthetic_fallback(n, hr_val), -np.inf
    for name, wave in candidates.items():
        if not np.isfinite(wave).all() or np.std(wave) < 1e-6:
            continue
        wave = align_polarity(wave, anchor)
        s = snr_at_hr(wave, hr_val)
        if s > best_snr:
            best_snr, best_name, best_wave = s, name, wave

    method_wins[best_name] += 1
    if np.isfinite(ita):
        (light_methods if ita >= ITA_LIGHT_THRESHOLD else dark_methods).append(best_name)

    wave = best_wave - np.mean(best_wave)
    wave = wave / (np.max(np.abs(wave)) + 1e-6)
    frame_data = [{"PulseRate": float(hr_val), "Wave": float(w)} for w in wave]
    with open(os.path.join(DATASET_BASE, clip_name, f"{clip_name}.json"), "w") as f:
        json.dump({"/ImageData/FrameData": frame_data}, f, indent=4)

    tone = ("light" if (np.isfinite(ita) and ita >= ITA_LIGHT_THRESHOLD)
            else "dark" if np.isfinite(ita) else "n/a")
    print(f"  {clip_name:<9} HR={hr_val:5.0f} ITA={ita:6.1f} ({tone:>5}) "
          f"-> {best_name:<8} (snr={best_snr:.2f})")

print(f"\nMethod chosen: {method_wins}")
if dark_methods or light_methods:
    from collections import Counter
    print(f"  dark-skin clips  (ITA<{ITA_LIGHT_THRESHOLD}): {dict(Counter(dark_methods))}")
    print(f"  light-skin clips (ITA>={ITA_LIGHT_THRESHOLD}): {dict(Counter(light_methods))}")
print("\nDone. Waveforms are video-aligned and per-clip best-of-{POS,CHROM,green}.")