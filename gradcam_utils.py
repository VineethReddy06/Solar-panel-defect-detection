import tensorflow as tf
import numpy as np
import cv2

IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "conv5_block16_concat"   # DenseNet121 last conv layer

def load_and_prep_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img / 255.0, axis=0)
    return img

def generate_gradcam(model, image_path, output_path="static/gradcam_output.png"):
    img_array = load_and_prep_image(image_path)

    # Gradient model
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # for "Faulty" = class 1

    grads = tape.gradient(loss, conv_outputs)

    # Convert to numpy arrays safely
    conv_outputs = conv_outputs[0].numpy()
    grads = grads[0].numpy()

    # Global-average pool the gradients
    weights = np.mean(grads, axis=(0, 1))

    # Weighted sum
    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    # Normalize
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    # Resize heatmap to image size
    heatmap = cv2.resize(cam, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Original image
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)

    # Blend heatmap with original image
    output = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(output_path, output)
    return output_path
