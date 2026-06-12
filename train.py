import torch
import argparse
import os
import sys
import json
import random
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloader.dataloader import MHDataLoader
from dataloader.LoadVideotrain_pure import PURE_train, Normaliztion, ToTensor, RandomHorizontalFlip
from model.PhysNet_PFE_TFA_crcloss import PhysNet_padding_ED_peak
from utils.TorchLossComputer import TorchLossComputer

cudnn.benchmark = True

# ==========================================================================
# SUBJECT / GROUP MAP
# David (0-29 and 40-49) is one person; David_a..d are motion/session groups so
# the test set draws 2 clips from each. AdityaK(30-39) and Aditya(115-124) are
# the SAME person -> both mapped to 'Aditya' (one group, 2 test clips total).
# Advaith (60-74 AND 145-159) is the darkest-skin subject -> all his clips share
# the 'Advaith' group; watch his per-group / 'dark' MAE as the cross-skin-tone
# benchmark. The extra 145-159 sessions mostly add dark-skin TRAINING data.
# >>> VERIFY these ranges match your recordings; edit freely. <<<
SUBJECT_RANGES = [
    (0, 9, 'David_a'), (10, 19, 'David_b'), (20, 29, 'David_c'),
    (30, 39, 'Aditya'), (40, 49, 'David_d'), (50, 59, 'Alice'),
    (60, 74, 'Advaith'), (75, 84, 'Ben'), (85, 94, 'Abrar'), (95, 114, 'Adi'),
    (115, 124, 'Aditya'), (125, 134, 'Bridget'), (135, 144, 'Gigi'),
    (145, 159, 'Advaith'),
]


def clip_subject(clip_name):
    """Fine group (David_a, Aditya, Alice, ...) used for the per-group test split."""
    try:
        idx = int(clip_name.split('_')[1])
    except Exception:
        return 'unknown'
    for lo, hi, name in SUBJECT_RANGES:
        if lo <= idx <= hi:
            return name
    return 'unknown'


def clip_person(clip_name):
    """Coarse identity used for true unseen-person holdout. David_* -> David."""
    g = clip_subject(clip_name)
    return 'David' if g.startswith('David') else g


def make_split(videoList, test_per_subject=2, val_per_subject=1,
               holdout_subjects=None, seed=42):
    """
    Deterministic split. Returns (train, val, test).

    Default (holdout_subjects is None): per-GROUP holdout -> `test_per_subject`
    clips of every group are reserved for the TEST set (never trained on, never
    used for model selection), `val_per_subject` more are used only to pick the
    best checkpoint, and the rest train. Always leaves >=1 train clip per group.

    If `holdout_subjects` is given: whole groups go to TEST (true unseen-person
    evaluation), val is drawn 1/group from the remaining groups.
    """
    rng = random.Random(seed)
    by_subj = {}
    for c in videoList:
        by_subj.setdefault(clip_subject(c), []).append(c)

    train, val, test = [], [], []
    if holdout_subjects:
        hs = set(holdout_subjects)
        for subj, clips in by_subj.items():
            clips_sorted = sorted(clips, key=lambda c: int(c.split('_')[1]))
            if clip_person(clips_sorted[0]) in hs:     # whole person -> TEST (unseen-person)
                test.extend(clips_sorted)
            else:
                k = min(val_per_subject, max(0, len(clips_sorted) - 1))
                vpick = set(rng.sample(clips_sorted, k)) if k > 0 else set()
                for c in clips_sorted:
                    (val if c in vpick else train).append(c)
    else:
        for subj, clips in by_subj.items():
            clips_sorted = sorted(clips, key=lambda c: int(c.split('_')[1]))
            n = len(clips_sorted)
            n_test = min(test_per_subject, max(0, n - 2))   # keep >=2 for train+val
            tpick = set(rng.sample(clips_sorted, n_test)) if n_test > 0 else set()
            rest = [c for c in clips_sorted if c not in tpick]
            n_val = min(val_per_subject, max(0, len(rest) - 1))  # keep >=1 for train
            vpick = set(rng.sample(rest, n_val)) if n_val > 0 else set()
            for c in clips_sorted:
                if c in tpick:
                    test.append(c)
                elif c in vpick:
                    val.append(c)
                else:
                    train.append(c)
    return sorted(train), sorted(val), sorted(test)


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = self.sum = self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt if self.cnt else 0.0


def get_clip_hr(dataset, idx):
    clip_name = dataset.videoList[idx]
    jf = os.path.join(dataset.path_data, clip_name, f"{clip_name}.json")
    if os.path.exists(jf):
        with open(jf) as f:
            return json.load(f)["/ImageData/FrameData"][0]["PulseRate"]
    return 60.0


def build_sampler(dataset):
    """Balance training draws by HR *and* by person, so the subject with the most
    clips (David) can't dominate and the model is pushed toward generalising to
    everyone -- which also keeps the darkest-skin subject (Advaith) well sampled."""
    from collections import Counter
    n = len(dataset)
    hrs = np.array([get_clip_hr(dataset, i) for i in range(n)])
    persons = [clip_person(dataset.videoList[i]) for i in range(n)]

    bins = np.clip(((hrs - 40) // 10).astype(int), 0, 13)
    counts = np.bincount(bins, minlength=14).astype(float)
    hr_w = np.array([1.0 / np.sqrt(counts[b] + 1e-6) for b in bins])

    pc = Counter(persons)
    person_w = np.array([1.0 / np.sqrt(pc[p]) for p in persons])

    w = hr_w * person_w
    w = w / w.mean()
    w = np.clip(w, 0.4, 2.5)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=n, replacement=True)


def latest_path(args):
    return os.path.join(args.log, f"{args.log}_con_{args.version}_latest.pkl")


def best_path(args):
    return os.path.join(args.log, f"{args.log}_con_{args.version}_BEST.pkl")


class EMA:
    """Exponential moving average of weights. A shadow copy is nudged toward the live
    weights after every optimizer step; we validate and save the SHADOW (smoother than
    any single noisy step), which is what reduces the val-selection jitter you saw.
    Floating params are averaged; integer buffers (e.g. BN num_batches_tracked) copied."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        """Load shadow weights into model; return a backup of the live weights."""
        msd = model.state_dict()
        backup = {k: msd[k].detach().clone() for k in msd}
        model.load_state_dict({k: self.shadow[k].to(msd[k].dtype) for k in msd}, strict=False)
        return backup

    @staticmethod
    def restore(model, backup):
        model.load_state_dict(backup, strict=False)


def train(args):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu[0]}')
        torch.cuda.set_device(device)
        print(f"[CUDA] Using {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("[WARNING] CUDA not available -- training on CPU will be very slow.")

    os.makedirs(args.log, exist_ok=True)
    log_file = open(os.path.join(args.log, f'{args.log}_con_{args.version}_log.txt'), 'a')

    print("[1] Building model...")
    sys.stdout.flush()
    model = PhysNet_padding_ED_peak(
        frames=args.frames, device_ids=args.gpu, hidden_layer=args.hidden_layer,
        use_tfa=bool(args.use_tfa), use_checkpoint=bool(args.use_checkpoint),
        tfa_blocks=args.tfa_blocks, spynet_path=args.spynet_path, freeze_spynet=True,
    ).to(device)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- resume ----
    start_epoch = 0
    best_val_mae = 999.0
    if args.resume and os.path.exists(latest_path(args)):
        ckpt = torch.load(latest_path(args), map_location=device)
        print(model.load_state_dict(ckpt['model'], strict=False))
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
            scheduler.load_state_dict(ckpt['scheduler'])
        except Exception as e:
            print(f"   optimizer/scheduler not restored ({e}); continuing with fresh ones")
        start_epoch = ckpt.get('epoch', -1) + 1
        best_val_mae = ckpt.get('best_val_mae', 999.0)
        print(f"[resume] Loaded {latest_path(args)} -> starting at epoch {start_epoch} "
              f"(best val MAE so far {best_val_mae:.2f})")
        if start_epoch >= args.epochs:
            print(f"[resume] Already trained {start_epoch} epochs. "
                  f"Increase --epochs above {start_epoch} to train more. Exiting.")
            log_file.close()
            return

    # ---- EMA of weights (smoother model selection; off by default) ----
    ema = EMA(model, args.ema_decay) if args.ema else None
    if ema is not None and args.resume and os.path.exists(latest_path(args)):
        try:
            _ck = torch.load(latest_path(args), map_location=device)
            if _ck.get('ema') is not None:
                ema.shadow = {k: v.to(device).float() for k, v in _ck['ema'].items()}
                print("[resume] EMA shadow restored.")
        except Exception:
            pass

    between_loss = nn.MSELoss()
    accumulation_steps = args.accum

    path_data = "./custom_dataset"
    if not os.path.exists(path_data):
        raise ValueError(f"Cannot find dataset path: {path_data}")
    videoList = sorted([d for d in os.listdir(path_data)
                        if os.path.isdir(os.path.join(path_data, d)) and d.startswith("clip_")])
    if not videoList:
        raise ValueError("No clips found! Run extract_frames.py and generate_labels.py first.")

    holdout = [s.strip() for s in args.holdout_subjects.split(',') if s.strip()] or None
    train_list, val_list, test_list = make_split(
        videoList, test_per_subject=args.test_per_subject,
        val_per_subject=args.val_per_subject, holdout_subjects=holdout, seed=args.seed)
    print(f"[2] Split -> train {len(train_list)} | val {len(val_list)} | test {len(test_list)} "
          f"(test reserved, never trained){' | holdout=' + str(holdout) if holdout else ''}")
    print(f"    TEST clips (run test.py on these): {test_list}")
    log_file.write(f"TEST={test_list}\nVAL={val_list}\n")

    train_ds = PURE_train(args, train_list, path_data, train=True,
                          transform=transforms.Compose([Normaliztion(), RandomHorizontalFlip(), ToTensor()]),
                          cache_frames=True, speed_min=args.speed_min, speed_max=args.speed_max,
                          use_diff=bool(args.use_diff),
                          tone_aug=bool(args.tone_aug), tone_aug_prob=args.tone_aug_prob)
    val_ds = PURE_train(args, val_list, path_data, train=False,
                        transform=transforms.Compose([Normaliztion(), ToTensor()]),
                        cache_frames=True, use_diff=bool(args.use_diff)) if val_list else None

    sampler = build_sampler(train_ds) if args.balance else None
    dataloader = MHDataLoader(args, train_ds, batch_size=args.batch_size,
                              shuffle=(sampler is None), sampler=sampler)
    val_dataloader = (MHDataLoader(args, val_ds, batch_size=args.batch_size, shuffle=False)
                      if val_ds is not None else None)

    amp_enabled = bool(args.use_amp) and torch.cuda.is_available()
    dev_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_dtype = torch.bfloat16
    print(f"[3] Train: epochs {start_epoch}->{args.epochs}, AMP={'bf16' if amp_enabled else 'off'}, "
          f"checkpoint={bool(args.use_checkpoint)}, TFA={bool(args.use_tfa)}, batch={args.batch_size}")
    sys.stdout.flush()

    last_done = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs):
            loss_fre_avg, loss_snr_avg, loss_mse_avg = AvgrageMeter(), AvgrageMeter(), AvgrageMeter()
            model.train()
            optimizer.zero_grad()
            nb = len(dataloader)
            echo_points = sorted({max(0, nb // 3 - 1), max(0, 2 * nb // 3 - 1), nb - 1})

            for i, batch in enumerate(dataloader):
                in1 = batch['video_x'].to(device, non_blocking=True)
                in2 = batch['video_y'].to(device, non_blocking=True)
                clip_hr = batch['clip_average_HR'].to(device, non_blocking=True)
                frame_rate = batch['frame_rate'].to(device, non_blocking=True)
                ecg_gt = batch['ecg'].to(device, non_blocking=True)

                with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=amp_enabled):
                    rPPG_peak, _, _, _ = model(in1, in2)

                rPPG = rPPG_peak[:, 0, :].float()
                rPPG = (rPPG - rPPG.mean(dim=-1, keepdim=True)) / (rPPG.std(dim=-1, keepdim=True) + 1e-6)

                bs = args.batch_size
                rPPG_first = rPPG[:bs]
                rPPG_second = rPPG[bs:] if rPPG.shape[0] > bs else rPPG_first
                loss_between = between_loss(rPPG_first, rPPG_second)

                if rPPG.shape[0] > clip_hr.shape[0]:
                    target_hr = torch.cat([clip_hr, clip_hr], 0)
                    target_fr = torch.cat([frame_rate, frame_rate], 0)
                    ecg_gt2 = torch.cat([ecg_gt, ecg_gt], 0)
                else:
                    target_hr, target_fr, ecg_gt2 = clip_hr, frame_rate, ecg_gt

                adjusted_HR = torch.clamp(target_hr - 40.0, 0.0, 139.0).long()

                fre_loss, valid = 0.0, 0
                for b in range(rPPG.shape[0]):
                    if 40.0 <= target_hr[b] <= 179.9:
                        fl, _ = TorchLossComputer.cross_entropy_power_spectrum_loss(
                            rPPG[b], adjusted_HR[b], target_fr[b], args.gpu)
                        fre_loss = fre_loss + fl
                        valid += 1
                fre_loss = fre_loss / valid if valid > 0 else rPPG.sum() * 0.0

                loss_snr = TorchLossComputer.compute_snr_loss(rPPG, target_hr, target_fr[0])
                pearson = TorchLossComputer.pearson_loss(rPPG, ecg_gt2.float())

                loss = fre_loss + 0.1 * loss_between + 0.5 * loss_snr + args.pearson_weight * pearson
                (loss / accumulation_steps).backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    if ema is not None:
                        ema.update(model)

                def _scalar(t):
                    return float(t.detach()) if torch.is_tensor(t) else float(t)
                n = in1.size(0)
                loss_fre_avg.update(_scalar(fre_loss), n)
                loss_snr_avg.update(_scalar(loss_snr), n)
                loss_mse_avg.update(_scalar(loss_between), n)

                if i in echo_points:
                    msg = (f'  Epoch {epoch+1} [{i+1}/{nb}]  CE={loss_fre_avg.avg:.4f} '
                           f'SNR={loss_snr_avg.avg:.4f} MSE={loss_mse_avg.avg:.4f}')
                    print(msg); log_file.write(msg + "\n"); log_file.flush()

            # ---- validation (model selection only; test clips are NOT here) ----
            val_mae = float('nan')
            ema_backup = ema.copy_to(model) if ema is not None else None   # validate/save the EMA weights
            if val_dataloader is not None:
                model.eval()
                ve = []
                with torch.no_grad():
                    for v in val_dataloader:
                        v1 = v['video_x'].to(device); v2 = v['video_y'].to(device)
                        vhr = v['clip_average_HR'].to(device); vfs = v['frame_rate'].to(device)
                        with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=amp_enabled):
                            vpeak, _, _, _ = model(v1, v2)
                        vr = vpeak[:, 0, :].float()
                        vr = (vr - vr.mean(-1, keepdim=True)) / (vr.std(-1, keepdim=True) + 1e-6)
                        if vr.shape[0] > vhr.shape[0]:
                            vhr = torch.cat([vhr, vhr], 0); vfs = torch.cat([vfs, vfs], 0)
                        for b in range(vr.shape[0]):
                            if 40.0 <= vhr[b] <= 179.9:
                                adj = torch.clamp(vhr[b] - 40.0, 0, 139).long()
                                _, err = TorchLossComputer.cross_entropy_power_spectrum_loss(vr[b], adj, vfs[b], args.gpu)
                                ve.append(err.item())
                val_mae = float(np.mean(ve)) if ve else float('nan')
                print(f"--> Epoch {epoch+1} Val MAE: {val_mae:.2f} BPM")
                log_file.write(f"Epoch {epoch+1} ValMAE {val_mae:.2f}\n"); log_file.flush()

                if not np.isnan(val_mae) and val_mae < best_val_mae:
                    best_val_mae = val_mae
                    torch.save({'model': model.state_dict(),
                                'use_diff': bool(args.use_diff), 'frames': args.frames}, best_path(args))
                    print(f"--> [*] New best (Val MAE {val_mae:.2f}) -> {os.path.basename(best_path(args))}")

            if ema is not None:
                EMA.restore(model, ema_backup)   # back to live weights for continued training

            scheduler.step()
            last_done = epoch
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'epoch': epoch,
                        'best_val_mae': best_val_mae,
                        'ema': (ema.shadow if ema is not None else None),
                        'use_diff': bool(args.use_diff), 'frames': args.frames}, latest_path(args))
            sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n[paused] Interrupted. Last completed epoch saved: {last_done+1}. "
              f"Resume with the same command (--resume 1).")
        log_file.close()
        return

    print(f'Finished Training. Best Val MAE: {best_val_mae:.2f}. '
          f'Now run: python test.py --epoch BEST  (evaluates the reserved 2-per-person test clips)')
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--frames', type=int, default=224)  # longer window = more cycles = better weak-signal SNR
    parser.add_argument('--hidden_layer', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--echo_batches', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--log', type=str, default="SSTTFinallog_Constrative")
    parser.add_argument('--version', default=11)
    parser.add_argument('--resume', type=int, default=1)         # auto-resume from *_latest.pkl
    parser.add_argument('--n_threads', type=int, default=0)
    parser.add_argument('--scale', type=str, default='1.0')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--use_amp', type=int, default=1)
    parser.add_argument('--use_checkpoint', type=int, default=1)
    parser.add_argument('--use_tfa', type=int, default=0)  # OFF by default: better unseen-person generalization on small data
    parser.add_argument('--tfa_blocks', type=int, default=7)
    parser.add_argument('--spynet_path', type=str, default='./weights/spynet_.pth')
    parser.add_argument('--pearson_weight', type=float, default=0.1)
    parser.add_argument('--balance', type=int, default=1)
    parser.add_argument('--speed_min', type=float, default=0.5)  # 1/speed_max -> log-uniform is centered on 1.0 (no HR bias)
    parser.add_argument('--speed_max', type=float, default=2.0)  # wide -> clean clips generate high-HR examples
    parser.add_argument('--tone_aug', type=int, default=0)        # OFF: A/B showed it regresses badly (9.51->18.48, light 6.0->13.6); kept only for reproducibility
    parser.add_argument('--ema', type=int, default=0)             # exponential moving average of weights: smoother model selection
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--tone_aug_prob', type=float, default=0.35)
    parser.add_argument('--use_diff', type=int, default=0)
    # split controls
    parser.add_argument('--test_per_subject', type=int, default=2)
    parser.add_argument('--val_per_subject', type=int, default=1)
    parser.add_argument('--holdout_subjects', type=str, default='')  # e.g. "Alice,Ben" for unseen-person test
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    args.scale = [1.0] if args.scale == '' else list(map(float, args.scale.split('+')))
    args.gpu = [int(x) for x in args.gpu.split(',')]
    print("[0] Imports OK. Starting...")
    train(args)