import math
import json
import os
import glob

# The TRUE 95 values: Original 30 clips + New 65 clips
heart_rates = [
    # --- ORIGINAL 30 CLIPS (clip_00 to clip_29) ---
    67.0, 62.0, 61.0, 63.0, 64.0, 66.0, 70.0, 57.0, 54.0, 64.0, # 0-9
    69.0, 69.0, 71.0, 71.0, 70.0, 73.0, 69.0, 69.0, 74.0, 74.0, # 10-19
    64.0, 58.0, 66.0, 61.0, 63.0, 59.0, 55.0, 60.0, 65.0, 64.0, # 20-29

    # --- NEW 65 CLIPS (clip_30 to clip_94) ---
    # Aditya Kumar 
    58.0, 55.0, 59.0, 60.0, 64.0, 60.0, 62.0, 65.0, 67.0, 64.0, # 30-39
    # David Morrow 
    69.0, 65.0, 68.0, 63.0, 64.0, 68.0, 65.0, 66.0, 65.0, 66.0, # 40-49
    # Alice 
    60.0, 59.0, 61.0, 64.0, 62.0, 61.0, 58.0, 58.0, 60.0, 61.0, # 50-59
    # Advaith (15 clips)
    103.0, 98.0, 103.0, 79.0, 99.0, 103.0, 94.0, 96.0, 99.0, 80.0, 88.0, 87.0, 96.0, 82.0, 96.0, # 60-74
    # Ben 
    47.0, 60.0, 52.0, 50.0, 52.0, 50.0, 49.0, 49.0, 63.0, 50.0, # 75-84
    # Abrar 
    81.0, 81.0, 83.0, 92.0, 87.0, 86.0, 90.0, 95.0, 96.0, 96.0  # 85-94
]

VIDEO_FPS = 24.0  
DATASET_BASE = "custom_dataset"

print("Starting ground-truth signal synthesis for 95 total clips...")

for i in range(len(heart_rates)):
    # Safely checks for BOTH padded (clip_00) and unpadded (clip_0) names
    possible_names = [f"clip_{i:02d}", f"clip_{i}"]
    
    images_dir = None
    clip_name = None
    
    for name in possible_names:
        test_path = os.path.join(DATASET_BASE, name, "pic", "1.0")
        if os.path.exists(test_path):
            images_dir = test_path
            clip_name = name
            break
            
    if images_dir is None:
        print(f"Skipping index {i}: Folder not found. Run extract_frames.py first!")
        continue
        
    num_frames = len(glob.glob(os.path.join(images_dir, "*.png")))
    if num_frames == 0:
        print(f"Skipping {clip_name}: No cropped face frames (.png) found.")
        continue
        
    hr_val = heart_rates[i]
    frequency = hr_val / 60.0  
    
    frame_data_list = []
    for frame_idx in range(num_frames):
        time_seconds = frame_idx / VIDEO_FPS
        wave_val = math.sin(2 * math.pi * frequency * time_seconds)
        
        frame_entry = {"PulseRate": float(hr_val), "Wave": float(wave_val)}
        frame_data_list.append(frame_entry)
        
    json_data = {"/ImageData/FrameData": frame_data_list}
    
    output_json_path = os.path.join(DATASET_BASE, clip_name, f"{clip_name}.json")
    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=4)
        
    print(f"  -> Generated {clip_name}.json with {num_frames} wave points matching {hr_val} BPM.")

print("\nSuccess! All 95 ground truth JSON maps are created and structurally synchronized.")