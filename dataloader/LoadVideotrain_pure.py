import os
import torch
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
import json
import glob


class Normaliztion(object):
    def __call__(self, sample):
        sample['video_x'] = (sample['video_x'] - 127.5) / 128.0
        sample['video_y'] = (sample['video_y'] - 127.5) / 128.0
        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if random.random() > 0.5:
            sample['video_x'] = np.flip(sample['video_x'], axis=2).copy()
            sample['video_y'] = np.flip(sample['video_y'], axis=2).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        video_x = sample['video_x'].transpose((3, 0, 1, 2))
        video_y = sample['video_y'].transpose((3, 0, 1, 2))
        return {
            'video_x': torch.from_numpy(video_x.astype(np.float32)),
            'video_y': torch.from_numpy(video_y.astype(np.float32)),
            'clip_average_HR': torch.tensor(sample['clip_average_HR'], dtype=torch.float32),
            'ecg': torch.from_numpy(sample['ecg'].astype(np.float32)),
            'frame_rate': torch.tensor(sample['frame_rate'], dtype=torch.float32),
            'clip_name': sample['clip_name'],
        }


class PURE_train(Dataset):
    """
    Changes vs original:
      * RAM cache: each clip's frames (uint8) + POS wave are decoded once and
        kept in memory. With num_workers=0 (recommended on Windows) this removes
        all per-epoch disk/decoding cost. 145 clips x 480 x 128x128x3 ~ 3.4 GB.
      * Temporal-resampling augmentation: a per-clip `speed` in [speed_min,
        speed_max] resamples the window so the APPARENT HR = clip_hr * speed.
        The label and the POS wave are scaled identically. This forces the net
        to read frequency (instead of memorising subject identity), expands HR
        coverage well beyond the recorded range, and directly fights the
        collapse-to-one-band failure mode. Both augmented views share the same
        speed so the consistency (MSE) loss stays valid.
      * `ecg` is now the POS-derived, video-aligned waveform (see
        generate_labels.py), so the Pearson loss is meaningful.
      * Optional `use_diff` difference-normalised input (EXPERIMENTAL: pair with
        model use_tfa=False, since optical flow on difference frames is invalid).
    """
    def __init__(self, args, videoList, path_data, train=True, transform=None,
                 cache_frames=True, speed_min=0.7, speed_max=1.4, use_diff=False,
                 tone_aug=False, tone_aug_prob=0.35):
        self.videoList = videoList
        self.path_data = path_data
        self.frames = args.frames
        self.train = train
        self.transform = transform
        self.cache_frames = cache_frames
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.use_diff = use_diff
        self.tone_aug = tone_aug
        self.tone_aug_prob = tone_aug_prob
        self._cache = {}

    def __len__(self):
        return len(self.videoList)

    def _load_clip(self, clip_name):
        if clip_name in self._cache:
            return self._cache[clip_name]
        clip_path = os.path.join(self.path_data, clip_name)
        pic_path = os.path.join(clip_path, "pic", "1.0")
        frame_files = sorted(glob.glob(os.path.join(pic_path, "*.png")),
                             key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        if len(frame_files) == 0:
            raise ValueError(f"No frames found for {clip_name}")
        imgs = np.zeros((len(frame_files), 128, 128, 3), dtype=np.uint8)
        for i, fp in enumerate(frame_files):
            im = cv2.imread(fp)
            imgs[i] = cv2.resize(im, (128, 128)) if im is not None else 0

        with open(os.path.join(clip_path, f"{clip_name}.json"), 'r') as f:
            data = json.load(f)
        fd = data["/ImageData/FrameData"]
        hr = float(fd[0]["PulseRate"])
        wave = np.array([it["Wave"] for it in fd], dtype=np.float32)
        T = len(frame_files)
        if len(wave) < T:
            wave = (np.pad(wave, (0, T - len(wave)), mode='edge')
                    if len(wave) > 0 else np.zeros(T, np.float32))
        wave = wave[:T]

        out = (imgs, wave, hr)
        if self.cache_frames:
            self._cache[clip_name] = out
        return out

    def _pick_speed(self, clip_hr, total_frames):
        if not self.train:
            return 1.0
        smin = max(self.speed_min, 42.0 / max(clip_hr, 1e-3))
        smax = min(self.speed_max, 178.0 / max(clip_hr, 1e-3))
        smax = min(smax, (total_frames - 1) / float(self.frames - 1))
        if smax <= smin:
            return float(np.clip(1.0, min(smin, smax), max(smin, smax)))
        # LOG-uniform: speeding up and slowing down are balanced, so the average
        # apparent HR is NOT inflated. Linear uniform skews HR upward and makes
        # the model over-predict (see the 0.9-2.0 run that collapsed high).
        return float(np.exp(random.uniform(np.log(smin), np.log(smax))))

    def __getitem__(self, idx):
        clip_name = self.videoList[idx]
        imgs, wave, clip_hr = self._load_clip(clip_name)
        total_frames = imgs.shape[0]
        if total_frames < self.frames:
            raise ValueError(f"Clip {clip_name} has {total_frames} frames, needs {self.frames}")

        speed = self._pick_speed(clip_hr, total_frames)
        span = speed * (self.frames - 1)
        max_start = total_frames - 1 - span
        if max_start < 0:                       # safety: shrink speed to fit
            speed = (total_frames - 1) / float(self.frames - 1)
            span = speed * (self.frames - 1)
            max_start = total_frames - 1 - span
        start = random.uniform(0, max(0.0, max_start)) if self.train else max(0.0, max_start) / 2.0

        src_idx = start + speed * np.arange(self.frames)
        nn_idx = np.clip(np.round(src_idx).astype(int), 0, total_frames - 1)
        target_hr = float(np.clip(clip_hr * speed, 40.0, 179.0))

        if self.train:
            gx, gy = random.uniform(0.6, 1.4), random.uniform(0.6, 1.4)
            ax, ay = random.uniform(0.6, 1.4), random.uniform(0.6, 1.4)
            bx, by = random.randint(-20, 20), random.randint(-20, 20)
            cbx, cby = random.randint(-20, 20), random.randint(-20, 20)
            crx, cry = random.randint(-20, 20), random.randint(-20, 20)
        else:
            gx = gy = ax = ay = 1.0
            bx = by = cbx = cby = crx = cry = 0

        video_x = np.zeros((self.frames, 128, 128, 3), np.float32)
        video_y = np.zeros((self.frames, 128, 128, 3), np.float32)
        for i, fi in enumerate(nn_idx):
            yuv = cv2.cvtColor(imgs[fi], cv2.COLOR_BGR2YUV).astype(np.float32)
            yx, yy = yuv.copy(), yuv.copy()
            if self.train:
                yx[:, :, 0] = np.clip(255.0 * (yx[:, :, 0] / 255.0 + 1e-6) ** gx, 0, 255)
                yy[:, :, 0] = np.clip(255.0 * (yy[:, :, 0] / 255.0 + 1e-6) ** gy, 0, 255)
                yx[:, :, 0] = np.clip(ax * yx[:, :, 0] + bx, 0, 255)
                yy[:, :, 0] = np.clip(ay * yy[:, :, 0] + by, 0, 255)
                yx[:, :, 1] = np.clip(yx[:, :, 1] + cbx, 0, 255)
                yx[:, :, 2] = np.clip(yx[:, :, 2] + crx, 0, 255)
                yy[:, :, 1] = np.clip(yy[:, :, 1] + cby, 0, 255)
                yy[:, :, 2] = np.clip(yy[:, :, 2] + cry, 0, 255)
            video_x[i] = yx
            video_y[i] = yy

        if self.train and self.tone_aug:
            video_x, video_y = self._maybe_tone_darken(video_x, video_y)

        if self.use_diff:
            video_x = self._to_diff(video_x)
            video_y = self._to_diff(video_y)

        # POS wave resampled to the same fractional indices, then z-scored
        ecg = np.interp(src_idx, np.arange(total_frames), wave).astype(np.float32)
        ecg = ecg - ecg.mean()
        s = ecg.std()
        if s > 1e-6:
            ecg = ecg / s

        sample = {'video_x': video_x, 'video_y': video_y, 'clip_average_HR': target_hr,
                  'ecg': ecg, 'frame_rate': 24.0, 'clip_name': clip_name}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _maybe_tone_darken(self, video_x, video_y):
        """Skin-tone / dark-capture domain randomization.

        Your 4 dark subjects are ALL high-HR, so the net can cheat by tying a dark
        appearance to a high rate. Here we darken a fraction of the *bright* (light-
        subject) clips -- which span the full HR range, including low rates -- so the
        model sees dark-looking faces at every HR and can no longer use tone as an HR
        cue. We also (a) raise gamma to compress the pulsatile AC the way real dark
        skin does, and (b) add per-pixel sensor noise. The model averages over
        thousands of skin pixels, so independent pixel noise is suppressed ~sqrt(N)
        while the spatially-coherent pulse survives -- i.e. the *apparent* SNR drops
        to the dark-capture regime without erasing the signal. Bright clips have
        signal headroom to spare, which is why we only degrade those (an already-weak
        dark clip is left untouched so its label stays grounded).

        Same gamma on both views (consistent 'tone'); independent noise per view
        (teaches noise-invariance). Applied before the (x-127.5)/128 normalization.
        """
        meanY = float(video_x[:, :, :, 0].mean())
        if meanY < 95.0 or random.random() > self.tone_aug_prob:
            return video_x, video_y
        gamma_d = random.uniform(1.25, 1.9)          # darken + compress AC (dark-skin-like)
        nstd_y = random.uniform(2.0, 7.0)            # luma sensor noise (killed by spatial pooling)
        nstd_c = random.uniform(0.5, 2.0)            # mild chroma noise
        for v in (video_x, video_y):
            y = v[:, :, :, 0] / 255.0
            y = 255.0 * np.clip(y, 0, 1) ** gamma_d
            y = y + np.random.randn(*y.shape).astype(np.float32) * nstd_y
            v[:, :, :, 0] = np.clip(y, 0, 255)
            v[:, :, :, 1] = np.clip(v[:, :, :, 1] +
                                    np.random.randn(*v[:, :, :, 1].shape).astype(np.float32) * nstd_c, 0, 255)
            v[:, :, :, 2] = np.clip(v[:, :, :, 2] +
                                    np.random.randn(*v[:, :, :, 2].shape).astype(np.float32) * nstd_c, 0, 255)
        return video_x, video_y

    @staticmethod
    def _to_diff(vid):
        """Normalised temporal difference, z-scored per channel, remapped so the
        downstream (x-127.5)/128 normalisation yields the z-scored difference."""
        d = np.zeros_like(vid)
        d[1:] = (vid[1:] - vid[:-1]) / (vid[1:] + vid[:-1] + 1e-6)
        for c in range(3):
            sd = d[..., c].std()
            if sd > 1e-6:
                d[..., c] = d[..., c] / sd
        return (d * 128.0 + 127.5).astype(np.float32)