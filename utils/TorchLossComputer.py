import math
import torch
import torch.nn.functional as F


class TorchLossComputer(object):
    """
    Frequency-domain losses for rPPG.

    Key properties (unchanged from the original, which was correct):
      * `cross_entropy_power_spectrum_loss` is a *fine-grained* frequency
        classifier: it evaluates a direct DFT at every BPM in [40,179] via
        `compute_complex_absolute_given_k`, normalises those 140 values into a
        distribution, and does NLL against the target BPM bin. 1-BPM granularity.
      * Index contract: target index 0 == 40 BPM (see train.py `adjusted_HR`).

    Fixes vs the original:
      * `compute_snr_loss` now zero-pads the rFFT so its frequency resolution
        (~0.7 BPM) matches the CE loss instead of the old ~9 BPM bins.
      * The SNR noise floor now EXCLUDES the 2nd/3rd harmonics of the target,
        which are physiological and should not be penalised as noise.
      * All tensors stay on the input device; no hard-coded cuda:0 transfers.
    """

    # ---- time-domain shape loss (now meaningful: paired with POS pseudo-labels) ----
    @staticmethod
    def pearson_loss(rppg_pred, rppg_gt):
        pred = rppg_pred - rppg_pred.mean(dim=-1, keepdim=True)
        gt = rppg_gt - rppg_gt.mean(dim=-1, keepdim=True)
        corr = (pred * gt).sum(dim=-1) / (pred.norm(dim=-1) * gt.norm(dim=-1) + 1e-8)
        return (1.0 - corr).mean()

    # ---- direct DFT magnitude at fractional bins k (Goertzel/Chirp-Z style) ----
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N, device_ids=[0]):
        device = output.device
        two_pi_n_over_N = (2 * math.pi * torch.arange(0, N, dtype=torch.float, device=device)) / N
        hanning = torch.hann_window(N, periodic=False, device=device).view(1, -1)

        k = k.float().to(device)
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)

        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
            + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2
        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None, device_ids=[0]):
        if torch.is_tensor(Fs):
            Fs = float(Fs.item())
        output = output.view(1, -1)
        N = output.size()[1]
        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz
        complex_abs = TorchLossComputer.compute_complex_absolute_given_k(output, k, N, device_ids)
        return (1.0 / (complex_abs.sum() + 1e-8)) * complex_abs

    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs, device_ids):
        inputs = inputs.view(1, -1)
        device = inputs.device
        bpm_range = torch.arange(40, 180, dtype=torch.float, device=device)  # idx0 -> 40 BPM

        complex_abs = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range, device_ids)
        _, whole_max_idx = complex_abs.view(-1).max(0)
        whole_max_idx = whole_max_idx.float()

        # complex_abs already sums to 1, so log() + NLL is a proper categorical loss
        # (using F.cross_entropy here would softmax a distribution twice).
        log_probs = torch.log(complex_abs + 1e-8)
        target = target.view(1).long().to(device)
        ce = F.nll_loss(log_probs, target)
        # second return value is per-window |true - argmax| in BPM (used as val MAE)
        return ce, torch.abs(target.float()[0] - whole_max_idx)

    @staticmethod
    def compute_snr_loss(inputs, target_bpm, fs, device_id=0, n_fft=2048):
        inputs = inputs.view(inputs.shape[0], -1)
        batch_size, n_samples = inputs.shape
        device = inputs.device

        inputs = (inputs - inputs.mean(dim=-1, keepdim=True)) / (inputs.std(dim=-1, keepdim=True) + 1e-6)
        hanning = torch.hann_window(n_samples, periodic=False, device=device).view(1, -1)
        windowed = inputs * hanning

        fs_val = fs.item() if torch.is_tensor(fs) else float(fs)
        rfft_out = torch.fft.rfft(windowed, n=n_fft, dim=-1)   # zero-padded -> fine bins
        fft_mag = torch.abs(rfft_out)
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fs_val).to(device) * 60.0  # in BPM

        total = inputs.sum() * 0.0  # gradable zero on the right device
        valid = 0
        band_mask = (freqs >= 40.0) & (freqs <= 180.0)
        for i in range(batch_size):
            target = target_bpm[i].item() if torch.is_tensor(target_bpm[i]) else float(target_bpm[i])
            if not (40.0 <= target <= 179.0):
                continue

            sig_mask = (freqs >= target - 6.0) & (freqs <= target + 6.0)
            if sig_mask.sum() == 0:
                continue
            # do not treat physiological harmonics as noise
            h2 = (freqs >= 2 * target - 6.0) & (freqs <= 2 * target + 6.0)
            h3 = (freqs >= 3 * target - 6.0) & (freqs <= 3 * target + 6.0)
            noise_mask = band_mask & (~sig_mask) & (~h2) & (~h3)

            signal_power = torch.sum(fft_mag[i][sig_mask] ** 2)
            noise_power = torch.sum(fft_mag[i][noise_mask] ** 2)
            if noise_power > 1e-8:
                snr = signal_power / (noise_power + 1e-8)
                total = total + (-10.0 * torch.log10(snr + 1e-8))
                valid += 1

        if valid > 0:
            return total / valid
        return torch.tensor(0.0, requires_grad=True, device=device)