import shutil
from pathlib import Path

# 1. Set your directories (Change these to your actual paths)
source_dir = Path('/path/to/your/old_video_folder')
clean_dir = Path('/path/to/your/new_clean_folder')

# 2. Search for the raw frames recursively
print(f"Scanning '{source_dir}' for raw frames...")
# This grabs anything starting with "frame_raw_" regardless of extension (.png, .jpg)
raw_frames = list(source_dir.rglob('frame_raw_*'))

# Filter out any accidental folders that might be named "frame_raw_something"
raw_frames = [f for f in raw_frames if f.is_file()]

print(f"Found {len(raw_frames)} frames. Starting copy process...")

# 3. Copy files while preserving the exact folder structure
for i, file_path in enumerate(raw_frames):
    
    # This finds the path relative to the source. 
    # Example: if file is "old_video_folder/clip_05/frame_raw_1.jpg", 
    # relative_path becomes "clip_05/frame_raw_1.jpg"
    relative_path = file_path.relative_to(source_dir)
    
    # We attach that relative path to the new clean directory
    dest_path = clean_dir / relative_path
    
    # Create the clip folder in the new directory if it doesn't exist yet
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy the file (shutil.copy2 also preserves file metadata like creation dates)
    shutil.copy2(file_path, dest_path)
    
    # Print progress every 1000 files
    if (i + 1) % 1000 == 0:
        print(f"Copied {i + 1} / {len(raw_frames)} files...")

print("Copy complete! Your new folder is clean and perfectly structured.")