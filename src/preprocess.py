import cv2
import mediapipe as mp
import numpy as np
import os

DATASET_DIR = "data"      # your image dataset folder
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

X = []
y = []

# Loop through dataset folders (A, B, C, etc)
for label in os.listdir(DATASET_DIR):
    label_path = os.path.join(DATASET_DIR, label)

    if not os.path.isdir(label_path):
        continue

    print(f"Processing label: {label}")

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # If hand detected, extract landmarks
        if results.multi_hand_landmarks:
            pts = []
            for lm in results.multi_hand_landmarks[0].landmark:
                pts.extend([lm.x, lm.y, lm.z])

            X.append(pts)          # 63 values (21 landmarks x 3)
            y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Save processed dataset
np.save(os.path.join(OUT_DIR, "X.npy"), X)
np.save(os.path.join(OUT_DIR, "y.npy"), y)

print("✔ Preprocessing complete!")
print("Saved:", os.path.join(OUT_DIR, "X.npy"))
print("Saved:", os.path.join(OUT_DIR, "y.npy"))
print("Total samples:", len(X))
