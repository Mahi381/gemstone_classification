import tensorflow as tf
from tensorflow.keras import layers, models
import json

# Dataset paths
train_dir = "dataset/train"
test_dir = "dataset/test"

# Image settings
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Get gemstone class names
class_names = train_ds.class_names
num_classes = len(class_names)

print("Number of gemstone classes:", num_classes)
print("Gemstone classes:", class_names)

# Improve loading speed
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(500).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# CNN model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Show model information
model.summary()

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Save trained model
model.save("model/gemstone_model.keras")

# Save class names
with open("model/class_names.json", "w") as f:
    json.dump(class_names, f)

print("\nTraining completed successfully!")
print("Model saved in model/gemstone_model.keras")
print("Class names saved in model/class_names.json")
