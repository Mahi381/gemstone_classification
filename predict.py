import tensorflow as tf
import json
import os

# Load trained model
model = tf.keras.models.load_model("model/gemstone_model.keras")

# Load class names
with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

# Test images folder
test_folder = "test_images"

# Check every image
for image_name in os.listdir(test_folder):

    image_path = os.path.join(test_folder, image_name)

    # Skip non-image files
    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # Load image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(128, 128)
    )

    # Convert image to array
    image_array = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    image_array = tf.expand_dims(image_array, 0)

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_index = tf.argmax(predictions[0]).numpy()
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    print("--------------------------------")
    print("Image:", image_name)
    print("Predicted Gemstone:", predicted_class)
    print("Confidence:", round(confidence, 2), "%")
