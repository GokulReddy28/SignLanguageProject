import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

DATASET_DIR = "dataset"   # Folder containing A/B/C/... folders
IMG_SIZE = 64          # Resize images to 64x64
BATCH_SIZE = 32

# Image augmentation + scaling
train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,           # 80% train / 20% validation
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=False
)

train_data = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# CNN MODEL
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(256, activation="relu"),
    Dropout(0.4),

    Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# Save the model
os.makedirs("models", exist_ok=True)
model.save("models/sign_cnn.h5")

print("✔ CNN Model Training Complete!")
print("✔ Saved: models/sign_cnn.h5")
print("Classes:", train_data.class_indices)
