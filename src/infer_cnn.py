import cv2
import numpy as np
import tensorflow as tf
import os
import pyttsx3
import time

MODEL_PATH = "models/sign_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels (A, B, C…)
DATASET_DIR = "data"  # change to "dataset" if needed
classes = sorted(os.listdir(DATASET_DIR))

IMG_SIZE = 64

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty("rate", 150)   # speed
engine.setProperty("volume", 1.0) # max volume

cap = cv2.VideoCapture(0)
last_spoken = ""
speak_delay = 1.2   # seconds between speaking the same letter

print("✔ Real-time Sign + Voice Started")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Crop center region
    h, w, _ = frame.shape
    size = min(h, w)
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    crop = frame[y1:y1+size, x1:x1+size]

    # Preprocess
    img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img, verbose=0)
    label = classes[np.argmax(pred)]
    conf = np.max(pred)

    # Display on screen
    cv2.putText(frame, f"{label} ({conf:.2f})", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Real-Time Sign Detection + Voice", frame)

    # Speak if new label detected
    if label != last_spoken and conf > 0.80:
        engine.say(label)
        engine.runAndWait()
        last_spoken = label
        time.sleep(speak_delay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
