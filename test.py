import torch
import argparse, os
import cv2
import numpy as np

from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import CubicSpline
from torchvision import transforms

from model.PhysNet_PFE_TFA_crcloss import PhysNet_padding_ED_peak
from dataloader.dataloader import MHDataLoader
from dataloader.LoadVideotrain_pure import PURE_train, Normaliztion, ToTensor

import torch.nn as nn

def calculate_stitched_bpm(chunks, fs=24.0):
    full_signal = np.concatenate(chunks)
    detrended_signal = detrend(full_signal, type='linear')
    
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
    fft_magnitude = np.abs(np.fft.rfft(windowed_signal, n=N))
    freqs = np.fft.rfftfreq(N, 1/upsampled_fs)
    
    valid_idx = np.where((freqs >= 0.75) & (freqs <= 2.5))[0]
    valid_freqs = freqs[valid_idx]
    valid_mag = fft_magnitude[valid_idx]
    
    max_idx = np.argmax(valid_mag)
    
    if 0 < max_idx < len(valid_mag) - 1:
        alpha = valid_mag[max_idx - 1]
        beta = valid_mag[max_idx]
        gamma = valid_mag[max_idx + 1]
        denom = (alpha - 2*beta + gamma)
        if denom != 0:
            p = 0.5 * (alpha - gamma) / denom
            peak_freq = valid_freqs[max_idx] + p * (valid_freqs[max_idx] - valid_freqs[max_idx - 1])
        else:
            peak_freq = valid_freqs[max_idx]
    else:
        peak_freq = valid_freqs[max_idx]
        
    final_bpm = peak_freq * 60.0
    
    signal_band = (valid_freqs >= peak_freq - 0.1) & (valid_freqs <= peak_freq + 0.1)
    peak_power = np.sum(valid_mag[signal_band]**2)
    noise_power = np.sum(valid_mag**2) - peak_power
    if noise_power < 1e-9: noise_power = 1e-9
    final_snr = 10 * np.log10(peak_power / noise_power)
    
    return final_bpm, final_snr

def print_clip_results(clip_idx, gt_val, pred_bpm, snr):
    confidence = "HIGH" if snr > 3.0 else "MEDIUM" if snr > 0.0 else "LOW"
    # Detect the fallback placeholder from your new clips
    if gt_val < 0:
        print(f"Clip {clip_idx:<2} | GT:  N/A      | Pred: {pred_bpm:5.1f} BPM | Err:  N/A  | SNR: {snr:5.1f} dB ({confidence})")
    else:
        err = abs(gt_val - pred_bpm)
        print(f"Clip {clip_idx:<2} | GT: {gt_val:5.1f} BPM | Pred: {pred_bpm:5.1f} BPM | Err: {err:5.1f} | SNR: {snr:5.1f} dB ({confidence})")

def test(condition, scale):
    condition = args.version
    device_ids = args.gpu
    frames = args.frames

    print(f'\n--- Starting Evaluation for Epoch {args.epoch} (Continuous Stitching) ---')
    print("--------------------------------------------------------------------------------")
    
    model = PhysNet_padding_ED_peak(frames = frames, device_ids = device_ids, hidden_layer = args.hidden_layer)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device=device_ids[0])
    map_location = 'cuda:' + str(device_ids[0])
    
    path_to_load = os.path.join(args.log, f"{args.log}_con_{condition}_1_{args.epoch}.pkl")
    model.load_state_dict(torch.load(path_to_load, map_location=map_location))
    model.eval()
    
    PURE_trainDL = PURE_train(args.scale, frames, transform=transforms.Compose([Normaliztion(), ToTensor()]), test=True)
    dataloader_train = MHDataLoader(args, PURE_trainDL, batch_size=1, shuffle=False, pin_memory=False)
    
    current_clip_idx = 1
    current_gt = None
    chunks = []
    
    with torch.no_grad():
        for i, sample_batched in enumerate(dataloader_train):
            inputs_1 = sample_batched['video_x'].cuda(device=device_ids[0])
            inputs_2 = sample_batched['video_y'].cuda(device=device_ids[0])
            clip_average_HR = sample_batched['clip_average_HR'].cuda(device=device_ids[0])
            frame_rate = sample_batched['frame_rate'].cuda(device=device_ids[0])

            rPPG_peak, _, _, _ = model(inputs_1, inputs_2)
            rppg_chunk = rPPG_peak[:, 0, :][0].cpu().numpy()
            
            fs = frame_rate.cpu().numpy().flatten()[0]
            if fs <= 0 or np.isnan(fs): fs = 24.0
            gt_hr = clip_average_HR.cpu().numpy().flatten()[0]
            
            if current_gt is None:
                current_gt = gt_hr
                
            # Group chunks into 4-segment sequential video clips (approx. 20-26 seconds total)
            # If the ground truth matches or is both -1.0, keep stitching
            is_same_clip = (abs(current_gt - gt_hr) < 0.1) or (current_gt < 0 and gt_hr < 0)
            
            if is_same_clip and len(chunks) < 4:
                chunks.append(rppg_chunk)
            else:
                pred_bpm, snr = calculate_stitched_bpm(chunks, fs=fs)
                print_clip_results(current_clip_idx, current_gt, pred_bpm, snr)
                
                del chunks
                chunks = [rppg_chunk]
                current_clip_idx += 1
                current_gt = gt_hr
                
        if chunks:
            pred_bpm, snr = calculate_stitched_bpm(chunks, fs=fs)
            print_clip_results(current_clip_idx, current_gt, pred_bpm, snr)

    print("--------------------------------------------------------------------------------")
    print('Finished Full Clip Evaluation Successfully!\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--frames',type=int,default=160)
    parser.add_argument('--hidden_layer',type=int,default=128)    
    parser.add_argument('--log', type=str, default="SSTTFinallog_Constrative")
    parser.add_argument('--version', default=3)
    parser.add_argument('--n_threads', type=int, default=6)
    parser.add_argument('--epoch', type=str, default='80')
    parser.add_argument('--scale', type=str, default='1.0')
    args = parser.parse_args()
    
    if args.scale=='': args.scale = [1.0]
    else: args.scale = list(map(lambda x: float(x), args.scale.split('+')))
        
    backup = args.scale 
    if args.gpu=='': args.gpu = [0]
    else: args.gpu = [int(x) for x in args.gpu.split(',')] 
        
    for scale in backup:
        args.scale = [scale]
        test(args.version, scale)