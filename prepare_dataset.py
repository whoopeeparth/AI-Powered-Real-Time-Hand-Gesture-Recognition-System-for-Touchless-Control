import os
import shutil

source_dir = "."
target_dir = "dataset"

gestures = {
    "01_palm": "palm",
    "03_fist": "fist",
    "05_thumb": "thumb",
    "06_index": "index",
    "07_ok": "ok"
}

# create output folders
for gesture in gestures.values():
    os.makedirs(os.path.join(target_dir, gesture), exist_ok=True)

# loop through subject folders
for subject in os.listdir(source_dir):

    if not subject.isdigit():
        continue

    subject_path = os.path.join(source_dir, subject)

    for gesture_folder in os.listdir(subject_path):

        if gesture_folder not in gestures:
            continue

        gesture_name = gestures[gesture_folder]
        gesture_path = os.path.join(subject_path, gesture_folder)

        for img in os.listdir(gesture_path):

            src = os.path.join(gesture_path, img)
            dst = os.path.join(target_dir, gesture_name, subject + "_" + img)

            shutil.copy(src, dst)

print("Dataset prepared successfully!")