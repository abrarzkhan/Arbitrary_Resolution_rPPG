import torch
import argparse
import warnings
import os
import glob
import cv2
import json
import numpy as np
from model.PhysNet_PFE_TFA_crcloss import PhysNet_padding_ED_peak

warnings.filterwarnings("ignore")


def get_device(gpu0):
    if torch.cuda.is_available():
        d = torch.device(f'cuda:{gpu0}')
        print(f"[CUDA] Using {torch.cuda.get_device_name(d)}")
        return d
    print("[WARNING] CUDA not available -- running on CPU.")
    return torch.device('cpu')


def _to_diff_window(vid):
    """Normalized temporal difference, z-scored per channel -- identical to training
    (LoadVideotrain_pure._to_diff) and to test.py, so a diff-trained model is fed
    the representation it learned on."""
    d = np.zeros_like(vid)
    d[1:] = (vid[1:] - vid[:-1]) / (vid[1:] + vid[:-1] + 1e-6)
    for c in range(3):
        sd = d[..., c].std()
        if sd > 1e-6:
            d[..., c] = d[..., c] / sd
    return (d * 128.0 + 127.5).astype(np.float32)


def bpm_from_windows(waves, fs=24.0, min_hr=40.0, max_hr=180.0):
    """Incoherently average each window's power spectrum, then pick the peak of the
    AVERAGE (same as test.py). Far steadier than the median of per-window peaks when
    the pulse is weak -- which is exactly the dark/high-HR case."""
    waves = [np.asarray(w, dtype=np.float64) for w in waves if w is not None and len(w) > 1]
    if not waves:
        return None
    L = max(len(w) for w in waves)
    nfft = int(2 ** np.ceil(np.log2(L * 8)))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs) * 60.0
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
        psd += sp
        peak_bpms.append(fb[int(np.argmax(sp))])
    if psd.sum() <= 0:
        return None
    k = int(np.argmax(psd))
    if 0 < k < len(psd) - 1:
        y0, y1, y2 = psd[k - 1], psd[k], psd[k + 1]
        den = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        bpm = float(fb[k] + delta * (fb[1] - fb[0]))
    else:
        bpm = float(fb[k])
    spread = float(np.std(peak_bpms)) if len(peak_bpms) > 1 else 0.0
    return bpm, spread


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
    return vid, len(files)            # RAW YUV; per-window normalization happens below


def infer_clip(model, pic_path, frames, fs, device, batch_size, use_diff, tta_flip=True):
    loaded = load_clip_yuv(pic_path, frames)
    if loaded is None:
        return None
    vid, n = loaded
    step = max(1, frames // 4)
    starts = list(range(0, n - frames + 1, step))

    def prep(s, flip=False):
        w = vid[s:s + frames]
        if use_diff:
            w = _to_diff_window(w)
        w = (w - 127.5) / 128.0
        w = w.transpose((3, 0, 1, 2))
        if flip:
            w = w[:, :, :, ::-1].copy()           # mirror L-R: same pulse, different pixel noise
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
            peak, _, _, _ = model(ct, ct)
            r = peak[:ct.shape[0], 0, :].float().cpu().numpy()
            waves.extend(list(r))
    return bpm_from_windows(waves, fs=fs)


def run_inference(args):
    if args.n_threads > 0:
        torch.set_num_threads(args.n_threads)
    device = get_device(args.gpu[0])

    if str(args.epoch).upper() == "BEST":
        wp = os.path.join(args.log, f"{args.log}_con_{args.version}_BEST.pkl")
    elif str(args.epoch).upper() == "LATEST":
        wp = os.path.join(args.log, f"{args.log}_con_{args.version}_latest.pkl")
    else:
        wp = os.path.join(args.log, f"{args.log}_con_{args.version}_1_{args.epoch}.pkl")
    print(f"-> Loading weights: {wp}")
    if not os.path.exists(wp):
        print(f"Error: weights not found at {wp}")
        return
    ckpt = torch.load(wp, map_location=device)
    # Build to match the checkpoint (window length + diff stored at train time).
    use_diff = bool(ckpt.get('use_diff', bool(args.use_diff)))
    frames = int(ckpt.get('frames', args.frames))
    model = PhysNet_padding_ED_peak(
        frames=frames, device_ids=args.gpu, hidden_layer=args.hidden_layer,
        use_tfa=bool(args.use_tfa), use_checkpoint=False, tfa_blocks=args.tfa_blocks,
        spynet_path=args.spynet_path, freeze_spynet=True).to(device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
    model.eval()
    print(f"[model] use_diff={use_diff}, frames={frames}")

    base = "./custom_dataset/"
    clips = sorted([d for d in os.listdir(base)
                    if os.path.isdir(os.path.join(base, d)) and d.startswith("clip_")],
                   key=lambda c: int(c.split('_')[1]))

    print(f"\n=== Full-video inference (spectral averaging over windows) | batch_size={args.batch_size} ===")
    errs = []
    for c in clips:
        pic = os.path.join(base, c, "pic", "1.0")
        jf = os.path.join(base, c, f"{c}.json")
        gt = None
        if os.path.exists(jf):
            with open(jf) as f:
                gt = json.load(f)["/ImageData/FrameData"][0]["PulseRate"]
        res = infer_clip(model, pic, frames, 24.0, device, args.batch_size, use_diff,
                         tta_flip=bool(args.tta_flip))
        if res is None:
            continue
        bpm, spread = res
        conf = "HIGH" if spread < 3.0 else "MEDIUM" if spread < 8.0 else "LOW"
        if gt is not None:
            err = abs(gt - bpm)
            errs.append(err)
            print(f"{c:<10} | GT {gt:5.1f} | Pred {bpm:5.1f} | err {err:4.1f} | spread {spread:5.1f} ({conf})")
        else:
            print(f"{c:<10} | Pred {bpm:5.1f} | spread {spread:5.1f} BPM ({conf})")
    if errs:
        errs = np.array(errs)
        print(f"\nMAE = {errs.mean():.2f} BPM | RMSE = {np.sqrt((errs**2).mean()):.2f} | "
              f"within 5 = {100*np.mean(errs<=5):.0f}% | within 10 = {100*np.mean(errs<=10):.0f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--frames', type=int, default=160)
    p.add_argument('--hidden_layer', type=int, default=128)
    p.add_argument('--log', type=str, default="SSTTFinallog_Constrative")
    p.add_argument('--version', type=int, default=11)
    p.add_argument('--epoch', type=str, default='BEST')
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--n_threads', type=int, default=0)
    p.add_argument('--use_tfa', type=int, default=0)
    p.add_argument('--use_diff', type=int, default=0)  # checkpoint value overrides this
    p.add_argument('--tta_flip', type=int, default=0)  # OFF: A/B showed mirroring hurts this model
    p.add_argument('--tfa_blocks', type=int, default=7)
    p.add_argument('--spynet_path', type=str, default='./weights/spynet_.pth')
    args = p.parse_args()
    args.gpu = [int(x) for x in args.gpu.split(',')]
    run_inference(args)