from flask import Flask, render_template, request
import tensorflow as tf
import os
from gradcam_utils import generate_gradcam

# Load model
model = tf.keras.models.load_model("densenet_binary_best.h5")

app = Flask(__name__)

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def predict_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(224,224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]
    if prob > 0.5:
        return "Faulty", prob
    else:
        return "Clean", prob


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, "uploaded_img.jpg")
        file.save(filepath)

        label, prob = predict_image(filepath)

        gradcam_path = generate_gradcam(model, filepath)

        return render_template(
            "result.html",
            label=label,
            prob=round(float(prob), 3),
            img_path="static/uploaded_img.jpg",
            gradcam_path="static/gradcam_output.png"
        )

    return "No file uploaded!"


if __name__ == "__main__":
    app.run(debug=True)
