import torch
import argparse
import os
import sys
import glob
import cv2
import json
import numpy as np
from collections import defaultdict

# Make sibling modules (heartRate.py, train.py, model/) importable no matter
# which directory the script is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.PhysNet_PFE_TFA_crcloss import PhysNet_padding_ED_peak
from heartRate import predict_heart_rate

# Keep the split definition in ONE place (same seed/params -> identical split).
try:
    from train import make_split, clip_subject
except Exception:
    raise ImportError("Run test.py from the repo root so it can import the split from train.py")


# Skin tone by SUBJECT identity -- you know these, and measuring ITA on the
# masked/filled crops proved unreliable (it read David lighter-than-Ben backwards).
# Tag each group 'dark' or 'light'; groups left out are excluded from the
# complexion summary (per-subject MAE below still covers them). Edit freely.
SUBJECT_TONE = {
    # dark (Indian; per you: Advaith darkest, Abrar darker, Adi lighter but still dark):
    'Advaith': 'dark', 'Abrar': 'dark', 'Adi': 'dark',
    'Aditya': 'dark',   # Indian, strongly negative ITA -- adjust if wrong
    # light:
    'David_a': 'light', 'David_b': 'light', 'David_c': 'light', 'David_d': 'light',
    'Ben': 'light', 'Alice': 'light', 'Bridget': 'light', 'Gigi': 'light',
}


def get_device(gpu0):
    if torch.cuda.is_available():
        d = torch.device(f'cuda:{gpu0}')
        print(f"[CUDA] Using {torch.cuda.get_device_name(d)}")
        return d
    print("[WARNING] CUDA not available -- running on CPU.")
    return torch.device('cpu')


def load_one(log, version, epoch, args, device):
    if str(epoch).upper() == "BEST":
        wp = os.path.join(log, f"{log}_con_{version}_BEST.pkl")
    elif str(epoch).upper() == "LATEST":
        wp = os.path.join(log, f"{log}_con_{version}_latest.pkl")
    else:
        wp = os.path.join(log, f"{log}_con_{version}_1_{epoch}.pkl")
    if not os.path.exists(wp):
        alt = os.path.join(log, f"{log}_con_{version}_latest.pkl")
        if str(epoch).upper() == "BEST" and os.path.exists(alt):
            print(f"[model] BEST checkpoint not found; falling back to LATEST.")
            wp = alt
        else:
            raise FileNotFoundError(wp)
    ckpt = torch.load(wp, map_location=device)
    # Build the model to MATCH the checkpoint (window length + diff are stored at train time),
    # so you can never silently test at the wrong window length again.
    ck_frames = int(ckpt.get('frames', args.frames))
    ck_diff = bool(ckpt.get('use_diff', bool(args.use_diff)))
    model = PhysNet_padding_ED_peak(
        frames=ck_frames, device_ids=args.gpu, hidden_layer=args.hidden_layer,
        use_tfa=bool(args.use_tfa), use_checkpoint=False, tfa_blocks=args.tfa_blocks,
        spynet_path=args.spynet_path, freeze_spynet=True).to(device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
    model.eval()
    if ck_frames != args.frames:
        print(f"[model] Using frames={ck_frames} from checkpoint (ignoring --frames {args.frames}).")
    if ck_diff != bool(args.use_diff):
        print(f"[model] NOTE: checkpoint trained with use_diff={ck_diff}; using that for inference.")
    print(f"[model] Loaded {wp}  (use_diff={ck_diff}, frames={ck_frames})")
    return model, ck_diff, ck_frames


def load_model(args, device):
    return load_one(args.log, args.version, args.epoch, args, device)


def parse_ensemble(spec, args, device):
    """spec like 'run_f224:1,run_b:1,run_c:1' (log:version, version optional -> args.version,
    epoch always BEST). Returns list of (model, frames, use_diff)."""
    models = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            log, ver = item.split(':', 1)
        else:
            log, ver = item, str(args.version)
        m, d, fr = load_one(log.strip(), ver.strip(), "BEST", args, device)
        models.append((m, fr, d))
    return models


def load_clip_yuv(pic_path, frames):
    files = sorted(glob.glob(os.path.join(pic_path, "*.png")),
                   key=lambda x: int(os.path.basename(x).split('.')[0]))
    if len(files) < frames:
        return None
    vid = np.zeros((len(files), 128, 128, 3), np.float32)
    for i, fp in enumerate(files):
        im = cv2.imread(fp)
        im = np.zeros((128, 128, 3), np.uint8) if im is None else cv2.resize(im, (128, 128))
        vid[i] = cv2.cvtColor(im, cv2.COLOR_BGR2YUV).astype(np.float32)
    return vid, len(files)            # RAW YUV; normalization happens per-window (matches training order)


def _to_diff_window(vid):
    """Same normalized-temporal-difference as training (LoadVideotrain_pure._to_diff),
    computed within the window and z-scored per channel."""
    d = np.zeros_like(vid)
    d[1:] = (vid[1:] - vid[:-1]) / (vid[1:] + vid[:-1] + 1e-6)
    for c in range(3):
        sd = d[..., c].std()
        if sd > 1e-6:
            d[..., c] = d[..., c] / sd
    return (d * 128.0 + 127.5).astype(np.float32)


def rgb_to_ita(*a, **k):
    raise NotImplementedError  # ITA on filled crops proved unreliable; using SUBJECT_TONE instead


ITA_DARK_THRESHOLD = -45.0   # room-calibrated: ITA < this -> 'dark'. Re-derived from the raw ITA
                             # in ita.json at read time, so changing it needs NO re-extract.


def load_clip_tone(base, clip):
    """Objective per-clip tone from extract_frames' ita.json (ITA measured on the RAW,
    unmasked face). Re-derives dark/light from the stored raw ITA using the room-
    calibrated threshold above -- so you can re-threshold without re-extracting.
    Returns 'dark'/'light' or None if absent. For the report bucket ONLY; inference is
    never routed by tone. Boundary subjects (e.g. Bridget) are better trusted to
    SUBJECT_TONE since absolute ITA can't separate them cleanly in this lighting."""
    try:
        with open(os.path.join(base, clip, "ita.json")) as f:
            ita = json.load(f).get("ita")
        if ita is None:
            return None
        return 'dark' if float(ita) < ITA_DARK_THRESHOLD else 'light'
    except Exception:
        return None


def bpm_from_windows(waves, fs=24.0, min_hr=40.0, max_hr=180.0, conf_weight=False):
    """Robust clip HR: incoherently average each window's power spectrum, then
    pick the peak of the AVERAGE. Consistent pulse energy stacks across windows;
    window-specific noise and stray subharmonics wash out. Far steadier than
    taking the median of per-window peaks when the signal is weak."""
    waves = [np.asarray(w, dtype=np.float64) for w in waves if w is not None and len(w) > 1]
    if not waves:
        return None
    L = max(len(w) for w in waves)
    nfft = int(2 ** np.ceil(np.log2(L * 8)))            # heavy zero-pad -> ~0.5 BPM bins
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs) * 60.0     # bin centers in BPM
    band = (freqs >= min_hr) & (freqs <= max_hr)
    fb = freqs[band]
    psd = np.zeros(int(band.sum()), dtype=np.float64)
    peak_bpms = []
    for w in waves:
        w = w - w.mean()
        sd = w.std()
        if sd < 1e-8:
            continue
        w = (w / sd) * np.hanning(len(w))
        sp = (np.abs(np.fft.rfft(w, n=nfft)) ** 2)[band]
        if conf_weight:
            # trust sharp-peaked windows over flat (noisy) ones: weight by peakedness
            conf = float(sp.max() / (sp.mean() + 1e-12))
            psd += conf * sp
        else:
            psd += sp
        peak_bpms.append(fb[int(np.argmax(sp))])
    if psd.sum() <= 0:
        return None
    k = int(np.argmax(psd))
    if 0 < k < len(psd) - 1:                              # parabolic sub-bin refine
        y0, y1, y2 = psd[k - 1], psd[k], psd[k + 1]
        den = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        bpm = float(fb[k] + delta * (fb[1] - fb[0]))
    else:
        bpm = float(fb[k])
    spread = float(np.std(peak_bpms)) if len(peak_bpms) > 1 else 0.0  # window agreement
    return bpm, spread


def clip_waves(model, pic_path, frames, device, batch_size=1, use_diff=False, tta_flip=False):
    """Return the list of per-window rPPG waveforms this model produces for the clip
    (the raw material bpm_from_windows averages). Pulled out of infer_clip so an
    ensemble can pool windows from several models into ONE spectral average."""
    loaded = load_clip_yuv(pic_path, frames)
    if loaded is None:
        return None
    vid, n = loaded                                          # raw YUV (N,128,128,3)
    if n < frames:
        return None
    step = max(1, frames // 4)                               # 75% overlap -> more windows to average, steadier
    starts = list(range(0, n - frames + 1, step))

    def prep(s, flip=False):
        w = vid[s:s + frames]                                # (F,128,128,3) raw YUV
        if use_diff:
            w = _to_diff_window(w)                            # match training diff
        w = (w - 127.5) / 128.0                               # match Normaliztion (last step in training)
        w = w.transpose((3, 0, 1, 2))                         # (3,F,128,128)
        if flip:
            w = w[:, :, :, ::-1].copy()                       # mirror L-R
        return w

    tasks = [(s, False) for s in starts]
    if tta_flip:
        tasks += [(s, True) for s in starts]

    waves = []
    with torch.no_grad():
        for i in range(0, len(tasks), batch_size):
            chunk = tasks[i:i + batch_size]
            batch = np.stack([prep(s, fl) for s, fl in chunk]).astype(np.float32)
            ct = torch.from_numpy(batch).to(device)
            peak, _, _, _ = model(ct, ct)                     # internal 2b; first b are the x-branch
            r = peak[:ct.shape[0], 0, :].float().cpu().numpy()
            waves.extend(list(r))
    return waves


def infer_clip(model, pic_path, frames, fs, device, batch_size=1, use_diff=False,
               conf_weight=False, tta_flip=False):
    waves = clip_waves(model, pic_path, frames, device, batch_size, use_diff, tta_flip)
    if not waves:
        return None
    return bpm_from_windows(waves, fs=fs, conf_weight=conf_weight)


def infer_clip_ensemble(models, pic_path, fs, device, batch_size=1, conf_weight=False, tta_flip=False):
    """Pool the per-window waveforms from every model into a single spectral average.
    Strictly a variance reducer -- can't make any one model worse. Models may have
    different window lengths; bpm_from_windows zero-pads each window independently."""
    all_waves = []
    for (model, frames, use_diff) in models:
        w = clip_waves(model, pic_path, frames, device, batch_size, use_diff, tta_flip)
        if w:
            all_waves.extend(w)
    if not all_waves:
        return None
    return bpm_from_windows(all_waves, fs=fs, conf_weight=conf_weight)




def test(args):
    if args.n_threads > 0:
        torch.set_num_threads(args.n_threads)
    device = get_device(args.gpu[0])

    ensemble = None
    if args.ensemble.strip():
        ensemble = parse_ensemble(args.ensemble, args, device)
        use_diff, frames = ensemble[0][2], ensemble[0][1]
        print(f"--- ENSEMBLE of {len(ensemble)} models (window spectra pooled) ---")
    else:
        model, use_diff, frames = load_model(args, device)

    base = "./custom_dataset/"
    videoList = sorted([d for d in os.listdir(base)
                        if os.path.isdir(os.path.join(base, d)) and d.startswith("clip_")])
    holdout = [s.strip() for s in args.holdout_subjects.split(',') if s.strip()] or None
    _, _, test_list = make_split(videoList, test_per_subject=args.test_per_subject,
                                 val_per_subject=args.val_per_subject,
                                 holdout_subjects=holdout, seed=args.seed)
    print(f"--- Reserved TEST set ({len(test_list)} clips, never trained) | batch_size={args.batch_size} ---")

    errs, per_subj, by_tone = [], defaultdict(list), defaultdict(list)
    for c in test_list:
        jf = os.path.join(base, c, f"{c}.json")
        if not os.path.exists(jf):
            continue
        with open(jf) as f:
            gt = json.load(f)["/ImageData/FrameData"][0]["PulseRate"]
        pic = os.path.join(base, c, "pic", "1.0")
        if ensemble is not None:
            res = infer_clip_ensemble(ensemble, pic, 24.0, device, args.batch_size,
                                      conf_weight=bool(args.conf_weight), tta_flip=bool(args.tta_flip))
        else:
            res = infer_clip(model, pic, frames, 24.0, device, args.batch_size,
                             use_diff=use_diff, conf_weight=bool(args.conf_weight),
                             tta_flip=bool(args.tta_flip))
        if res is None:
            print(f"{c:<10} | skipped (too few frames)")
            continue
        pred, spread = res
        err = abs(gt - pred)
        grp = clip_subject(c)
        tone = load_clip_tone(base, c) or SUBJECT_TONE.get(grp, 'untagged')
        errs.append(err)
        per_subj[grp].append(err)
        if tone in ('dark', 'light'):
            by_tone[tone].append(err)
        print(f"{c:<10} | {grp:<9} | {tone:<8} | "
              f"GT {gt:5.1f} | Pred {pred:5.1f} | Err {err:4.1f} | spread {spread:4.1f}")

    if errs:
        errs = np.array(errs)
        print("\n--- Per-group MAE (dark-skin = Advaith, your main benchmark) ---")
        for s in sorted(per_subj):
            e = np.array(per_subj[s])
            print(f"  {s:<10} MAE {e.mean():5.2f}  (n={len(e)})")
        if by_tone:
            print("\n--- By complexion (tagged subjects only) ---")
            for t in ('light', 'dark'):
                if by_tone[t]:
                    e = np.array(by_tone[t])
                    print(f"  {t:<5} MAE {e.mean():5.2f}  (n={len(e)})")
        print(f"\nOVERALL MAE  = {errs.mean():.2f} BPM")
        print(f"OVERALL RMSE = {np.sqrt((errs**2).mean()):.2f} BPM")
        print(f"within 5 BPM = {100.0*np.mean(errs<=5):.0f}% | within 10 BPM = {100.0*np.mean(errs<=10):.0f}%")
    else:
        print("No test clips evaluated.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--frames', type=int, default=160)
    p.add_argument('--hidden_layer', type=int, default=128)
    p.add_argument('--log', type=str, default="SSTTFinallog_Constrative")
    p.add_argument('--version', default=11)
    p.add_argument('--epoch', type=str, default='BEST')      # BEST | LATEST | <number>
    p.add_argument('--batch_size', type=int, default=2)      # set 2, drop to 1 if VRAM tight
    p.add_argument('--n_threads', type=int, default=0)
    p.add_argument('--use_tfa', type=int, default=0)
    p.add_argument('--use_diff', type=int, default=0)  # checkpoint's stored value overrides this
    p.add_argument('--conf_weight', type=int, default=0)  # opt-in: weight cleaner windows more in the spectral average
    p.add_argument('--tta_flip', type=int, default=0)     # OFF: A/B showed this model isn't flip-invariant; mirroring corrupts correct clips
    p.add_argument('--ensemble', type=str, default='')    # e.g. "run_f224:1,run_b:1,run_c:1" -> pool window spectra across models
    p.add_argument('--tfa_blocks', type=int, default=7)
    p.add_argument('--spynet_path', type=str, default='./weights/spynet_.pth')
    # must match train.py to reproduce the same split
    p.add_argument('--test_per_subject', type=int, default=2)
    p.add_argument('--val_per_subject', type=int, default=1)
    p.add_argument('--holdout_subjects', type=str, default='')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    args.gpu = [int(x) for x in args.gpu.split(',')]
    test(args)