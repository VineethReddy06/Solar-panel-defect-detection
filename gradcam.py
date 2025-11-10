# ---------------- GRAD-CAM for DenseNet Binary Model ----------------

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("densenet_binary_best.h5")
print("✅ Model Loaded for Grad-CAM")

IMG_SIZE = (224, 224)

# Select last convolution layer for DenseNet121
last_conv_layer_name = "conv5_block16_concat"

# ---------------- GRAD-CAM FUNCTION ----------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]   # Faulty = class 1

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()

# ---------------- GENERATE HEATMAP ----------------
def generate_gradcam(image_path):

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    cv2.imwrite("gradcam_output.png", overlay)
    print("🔥 Grad-CAM saved as gradcam_output.png")


# ---------------- RUN TEST ----------------
# Example test
# generate_gradcam("Faulty_solar_panel/Faulty/example_image.jpg")
