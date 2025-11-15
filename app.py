from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from gradcam_utils import generate_gradcam

# ----------------- LOAD MODEL -----------------
MODEL_PATH = "densenet_multiclass_best.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# 6 defect classes
CLASS_NAMES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]

app = Flask(__name__)

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ----------------- MULTI-CLASS PREDICTION -----------------
def predict_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(256, 256))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    preds = model.predict(img)[0]          # ex: [0.1, 0.3, 0.05, 0.45, 0.08, 0.02]
    class_index = np.argmax(preds)         # index of highest score
    confidence = float(np.max(preds))      # max softmax confidence
    label = CLASS_NAMES[class_index]       # convert index → class name

    return label, confidence, class_index


# ----------------- ROUTES -----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, "uploaded_img.jpg")
        file.save(filepath)

        # get prediction
        label, confidence, pred_index = predict_image(filepath)

        # generate Grad-CAM for this specific predicted class
        gradcam_path = generate_gradcam(model, filepath, pred_index)

        return render_template(
            "result.html",
            label=label,
            prob=round(confidence * 100, 2),           # convert to percentage
            img_path="static/uploaded_img.jpg",
            gradcam_path="static/gradcam_output.png"
        )

    return "No file uploaded!"


if __name__ == "__main__":
    app.run(debug=True)
