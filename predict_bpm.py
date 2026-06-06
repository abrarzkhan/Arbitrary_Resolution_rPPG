import torch
import numpy as np
import argparse
import warnings
from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import CubicSpline
from model.PhysNet_PFE_TFA_crcloss import PhysNet_padding_ED_peak
from dataloader.dataloader import MHDataLoader
from dataloader.LoadVideotrain_pure import PURE_train, Normaliztion, ToTensor, RandomHorizontalFlip
from torchvision import transforms

# Suppress annoying PyTorch warnings for a clean terminal output
warnings.filterwarnings("ignore")

# FIX 1: Default target changed to 24.0 to match your true dataset
def calculate_bpm_and_snr(rppg_signal, fs=24.0):
    """
    Advanced FFT HR Calculation using Detrending, Bandpass, 
    Cubic Spline Upsampling, Hanning Windowing, and Parabolic Interpolation.
    """
    detrended_signal = detrend(rppg_signal, type='linear')
    
    lowcut = 0.75 / (0.5 * fs)
    highcut = 2.5 / (0.5 * fs)
    b, a = butter(6, [lowcut, highcut], btype='band')
    filtered_signal = filtfilt(b, a, detrended_signal)
    
    original_times = np.arange(len(filtered_signal)) / fs
    upsampled_fs = 256.0
    upsampled_times = np.arange(0, original_times[-1], 1.0 / upsampled_fs)
    
    cs = CubicSpline(original_times, filtered_signal)
    upsampled_signal = cs(upsampled_times)
    
    windowed_signal = upsampled_signal * np.hanning(len(upsampled_signal))
    
    N = 8192 
    freqs = np.fft.rfftfreq(N, 1/upsampled_fs)
    fft_magnitude = np.abs(np.fft.rfft(windowed_signal, n=N))
    
    valid_idx = np.where((freqs >= 0.75) & (freqs <= 2.5))[0]
    valid_freqs = freqs[valid_idx]
    valid_mag = fft_magnitude[valid_idx]
    
    max_idx = np.argmax(valid_mag)
    
    if 0 < max_idx < len(valid_mag) - 1:
        alpha = valid_mag[max_idx - 1]
        beta = valid_mag[max_idx]
        gamma = valid_mag[max_idx + 1]
        
        p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        peak_freq = valid_freqs[max_idx] + p * (valid_freqs[max_idx] - valid_freqs[max_idx - 1])
    else:
        peak_freq = valid_freqs[max_idx]
        
    bpm = peak_freq * 60.0
    
    signal_band = (valid_freqs >= peak_freq - 0.1) & (valid_freqs <= peak_freq + 0.1)
    peak_power = np.sum(valid_mag[signal_band]**2)
    noise_power = np.sum(valid_mag**2) - peak_power
    if noise_power < 1e-9: noise_power = 1e-9
    snr_db = 10 * np.log10(peak_power / noise_power)
    
    return bpm, snr_db


def run_inference(args):
    device_id = args.gpu[0]
    map_location = f'cuda:{device_id}'
    
    print("-> Initializing PhysNet Model Architecture...")
    model = PhysNet_padding_ED_peak(frames=args.frames, device_ids=args.gpu, hidden_layer=args.hidden_layer)
    model = torch.nn.DataParallel(model, device_ids=args.gpu)
    model = model.cuda(device=device_id)
    
    # FIX 2: Pointed the weights to load your fully-trained epoch 74 files
    weight_path = f"{args.log}/{args.log}_con_{args.version}_1_74.pkl"
    print(f"-> Loading trained weights from: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=map_location))
    model.eval()
    
    dataset = PURE_train(scale=args.scale, frames=args.frames, 
                         transform=transforms.Compose([Normaliztion(), RandomHorizontalFlip(), ToTensor()]), 
                         test=True)
    
    dataloader = MHDataLoader(args, dataset, batch_size=1, shuffle=False, pin_memory=False)
    
    print("\n=== Running Inference on Test Clips ===")
    with torch.no_grad():
        for i, sample_batched in enumerate(dataloader):
            inputs_1 = sample_batched['video_x'].cuda(device=device_id)
            inputs_2 = sample_batched['video_y'].cuda(device=device_id)
            ground_truth_hr = sample_batched['clip_average_HR'][0].item()
            
            clip_name = f"Test Clip Index {i}" 
            
            rPPG_peak, _, _, _ = model(inputs_1, inputs_2)
            rPPG = rPPG_peak[:, 0, :][0] 
            
            rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)
            rppg_np = rPPG.cpu().numpy()
            
            # FIX 1 (Continued): Explicitly passing fs=24.0 to correct the math
            predicted_bpm, snr = calculate_bpm_and_snr(rppg_np, fs=24.0)
            
            confidence_str = "HIGH" if snr > 3.0 else "MEDIUM" if snr > 1.0 else "LOW (Noisy/Movement)"
            
            print(f"{clip_name}")
            print(f"   Ground Truth:         {ground_truth_hr:.2f} BPM")
            print(f"   Predicted Heart Rate: {predicted_bpm:.2f} BPM")
            print(f"   Absolute Error:       {abs(ground_truth_hr - predicted_bpm):.2f} BPM")
            print(f"   Signal Quality/SNR:   {snr:.2f} dB -> {confidence_str}")
            print("-" * 45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--frames', type=int, default=160)
    parser.add_argument('--hidden_layer', type=int, default=128)
    parser.add_argument('--log', type=str, default="SSTTFinallog_Constrative")
    parser.add_argument('--version', type=int, default=3)
    parser.add_argument('--n_threads', type=int, default=0) # Kept at 0 to match safe dataloader threads
    args = parser.parse_args()
    
    args.gpu = [int(x) for x in args.gpu.split(',')]
    args.scale = [1.0] 
    
    run_inference(args)