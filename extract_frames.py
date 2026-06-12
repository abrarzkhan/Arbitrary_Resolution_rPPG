import cv2
import os
import glob
import re
import json
import numpy as np
import mediapipe as mp


def _srgb_to_lab_ita(bgr):
    """bgr: length-3 mean skin color (B,G,R, 0-255). Returns (ITA_deg, L*).
    ITA = arctan((L*-50)/b*) in degrees -- the standard skin-tone measure used in
    rPPG-fairness work. Computed on the RAW, unmasked face skin (not the filled
    crop), which is why it's reliable here when ITA-on-the-crop was not."""
    rgb = np.array([bgr[2], bgr[1], bgr[0]], dtype=np.float64) / 255.0
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    R, G, B = lin
    X = (0.4124 * R + 0.3576 * G + 0.1805 * B) * 100 / 95.047
    Y = (0.2126 * R + 0.7152 * G + 0.0722 * B)
    Z = (0.0193 * R + 0.1192 * G + 0.9505 * B) * 100 / 108.883
    f = lambda t: np.cbrt(t) if t > 0.008856 else 7.787 * t + 16 / 116
    fy = f(Y)
    L = 116 * fy - 16
    b = 200 * (fy - f(Z))
    ita = float(np.degrees(np.arctan2(L - 50.0, b))) if abs(b) > 1e-9 else 0.0
    return ita, float(L)


# ITA < ITA_DARK_THRESH -> 'dark'. Calibrated for your warm/dim room: lightest
# subjects land ~-30, the 4 Indian subjects ~-48..-68, so -45 splits them. The raw
# ITA is also written to ita.json, and test.py re-derives the tone from that raw value
# at read time -- so changing this threshold needs NO re-extract.
ITA_DARK_THRESH = -45.0

# --- PATH CONFIGURATION ---
video_folder = r"C:\Users\Acema\Random\MNI_Lab\ML_RPPG\Arbitrary_Resolution_rPPG\my_videos"
dataset_base = r"C:\Users\Acema\Random\MNI_Lab\ML_RPPG\Arbitrary_Resolution_rPPG\custom_dataset"

# How to fill the masked-out region (eyes, mouth, background).
#   "skin_mean" -> fill with the per-frame mean skin colour (RECOMMENDED).
#                  Pure black injects zeros that drag the spatial average pooling
#                  toward 0; that weakens the already-faint pulse signal and hurts
#                  dark skin the most. Skin-mean fill keeps the DC level stable so
#                  the pulsatile AC component dominates -> better cross-skin-tone HR.
#   "black"     -> your original behaviour (eyes/mouth/background set to 0).
MASK_FILL = "skin_mean"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  refine_landmarks=True, min_detection_confidence=0.3)

SKIN_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109, 151, 9, 8, 168, 6, 197, 195, 5
]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


clip_folders = sorted(glob.glob(os.path.join(video_folder, "clip_*")), key=natural_sort_key)

for clip_path in clip_folders:
    clip_name = os.path.basename(clip_path)
    output_folder = os.path.join(dataset_base, clip_name, "pic", "1.0")
    os.makedirs(output_folder, exist_ok=True)

    raw_frames = glob.glob(os.path.join(clip_path, "frame_raw_*"))
    older_frames = glob.glob(os.path.join(clip_path, "frame_*"))
    combined = list(set(raw_frames + older_frames))
    valid_images = [f for f in combined if not f.lower().endswith(('.json', '.wav', '.txt', '.mat', '.bin'))]
    frame_files = sorted(valid_images,
                         key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))

    print(f"Processing {clip_name} ({len(frame_files)} frames found)...")
    count = 0
    ema_bbox = None
    skin_bgrs = []        # raw face-skin mean color per sampled frame (for ITA)

    for frame_path in frame_files:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        h_orig, w_orig, _ = frame.shape

        # CLAHE for DETECTION ONLY (never saved -- it would distort the pulse)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_bgr = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        results = face_mesh.process(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            continue
        landmarks = results.multi_face_landmarks[0].landmark

        mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        skin_pts = np.array([(int(landmarks[k].x * w_orig), int(landmarks[k].y * h_orig)) for k in SKIN_LANDMARKS])
        cv2.fillPoly(mask, [skin_pts], 255)
        for region in (LEFT_EYE, RIGHT_EYE, MOUTH):
            pts = np.array([(int(landmarks[k].x * w_orig), int(landmarks[k].y * h_orig)) for k in region])
            cv2.fillPoly(mask, [pts], 0)

        # Objective tone: mean of the RAW (unmasked) face-skin pixels, sampled
        # periodically. Median across frames -> one ITA per clip (written below).
        if count % 5 == 0:
            m = cv2.mean(frame, mask=mask)[:3]            # (B,G,R)
            if m[0] + m[1] + m[2] > 1:
                skin_bgrs.append(m)

        # --- mask out eyes / mouth / background (FaceMesh regions you liked) ---
        if MASK_FILL == "black":
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        else:  # "skin_mean": keep the same regions excluded, but fill stably
            skin_mean = cv2.mean(frame, mask=mask)[:3]
            masked_frame = frame.copy()
            masked_frame[mask == 0] = skin_mean

        coords = np.array([(int(l.x * w_orig), int(l.y * h_orig)) for l in landmarks])
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        margin = 20
        size = max(x_max - x_min, y_max - y_min) + 2 * margin
        half = size // 2
        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
        cx = max(half, min(cx, w_orig - half))
        cy = max(half, min(cy, h_orig - half))

        curr = np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)
        ema_bbox = curr if ema_bbox is None else 0.15 * curr + 0.85 * ema_bbox
        x1, y1, x2, y2 = [int(v) for v in ema_bbox]

        face_crop = masked_frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            face_crop = cv2.resize(face_crop, (128, 128))
            cv2.imwrite(os.path.join(output_folder, f"{count}.png"), face_crop)
            count += 1

    print(f"  -> Extracted {count} stabilized skin-mean-filled frames.")

    # Write objective skin tone for this clip (used by test.py's report bucket).
    if skin_bgrs:
        med_bgr = np.median(np.array(skin_bgrs), axis=0)
        ita, Lstar = _srgb_to_lab_ita(med_bgr)
        tone = 'dark' if ita < ITA_DARK_THRESH else 'light'
        with open(os.path.join(dataset_base, clip_name, "ita.json"), 'w') as f:
            json.dump({'ita': round(ita, 1), 'skin_L': round(Lstar, 1), 'tone': tone}, f)
        print(f"  -> ITA {ita:.1f} (skin L* {Lstar:.1f}) -> {tone}")

print("\nPreprocessing completed.")