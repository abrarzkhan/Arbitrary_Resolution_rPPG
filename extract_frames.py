import cv2
import os
import glob
import re

# --- YOUR ACCURATE VIDEO PATH ---
video_folder = r"C:\Users\Acema\Random\MNI_Lab\Arbitrary_Resolution_rPPG\Arbitrary_Resolution_rPPG\my_videos"
dataset_base = "custom_dataset"
# -----------------------------

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Find all clip subfolders inside your video folder
unsorted_clips = glob.glob(os.path.join(video_folder, "clip_*"))

# Naturally sort folders so clip_0, clip_1 ... clip_10 sort correctly
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

clip_folders = sorted(unsorted_clips, key=natural_sort_key)

if len(clip_folders) == 0:
    print(f"No folders starting with 'clip_' found inside:\n'{video_folder}'\nPlease check your folder path!")

for clip_path in clip_folders:
    clip_name = os.path.basename(clip_path)
    output_folder = os.path.join(dataset_base, clip_name, "pic", "1.0")
    os.makedirs(output_folder, exist_ok=True)
    
    # Target BOTH 'frame_raw_*' and older 'frame_*' naming patterns
    raw_frames = glob.glob(os.path.join(clip_path, "frame_raw_*"))
    older_frames = glob.glob(os.path.join(clip_path, "frame_*"))
    
    # Combine both lists and drop any accidental duplicate paths
    combined_frames = list(set(raw_frames + older_frames))
    
    # Filter out non-image metadata files (like .json or .wav files) that might match 'frame_*'
    valid_images = [f for f in combined_frames if not f.lower().endswith(('.json', '.wav', '.txt', '.mat', '.bin'))]
    
    # Sort everything numerically by extracting digits from the filename
    frame_files = sorted(valid_images, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))
    
    print(f"Processing {clip_name} ({len(frame_files)} frames found) -> saving to {output_folder}...")
    
    count = 0
    last_face_box = None
    
    for frame_path in frame_files:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        if len(faces) > 0:
            last_face_box = faces[0]
            
        if last_face_box is not None:
            x, y, w, h = last_face_box
            face_crop = frame[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (128, 128))
            
            # Save sequentially as 0.png, 1.png, 2.png... as expected by the model
            out_path = os.path.join(output_folder, f"{count}.png")
            cv2.imwrite(out_path, face_crop)
            count += 1
            
    print(f"  -> Successfully extracted {count} face frames for {clip_name}")

print("\nAll image folders successfully processed!")