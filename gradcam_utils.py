import tensorflow as tf
import numpy as np
import cv2

IMG_SIZE = (256, 256)
LAST_CONV_LAYER = "conv5_block16_concat"  # for DenseNet121


def generate_gradcam(model, image_path, class_index, output_path="static/gradcam_output.png"):
    # Preprocess input image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    # Build grad model
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    # Gradient calculation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]   # <--- MULTI-CLASS FIX

    grads = tape.gradient(loss, conv_outputs)
    conv_outputs = conv_outputs[0].numpy()
    grads = grads[0].numpy()

    # Apply GAP to gradients
    weights = np.mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)

    # Weighted sum of feature maps
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    # Normalize heatmap
    cam = np.maximum(cam, 0)
    cam /= cam.max() + 1e-8

    heatmap = cv2.resize(cam, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Original image
    original = cv2.imread(image_path)
    original = cv2.resize(original, IMG_SIZE)

    # Blend
    output = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, output)

    return output_path
