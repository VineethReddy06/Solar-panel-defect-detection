# ---------------- EVALUATION: DenseNet121 Binary Model ----------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- SETTINGS ----------------
DATASET_PATH = "Faulty_solar_panel"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = ["Clean", "Faulty"]   # Binary labels


# ---------------- LOAD TRAINED MODEL ----------------
model = tf.keras.models.load_model("densenet_binary_best.h5")
print("\n✅ DenseNet121 Binary Model Loaded Successfully!")


# ---------------- LOAD ENTIRE DATASET ----------------
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print(f"\n📸 Total Images Found: {test_generator.samples}\n")


# ---------------- PREDICT ----------------
y_prob = model.predict(test_generator)
y_pred = (y_prob > 0.5).astype("int32").flatten()
y_true = test_generator.classes


# ---------------- CLASSIFICATION REPORT ----------------
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - DenseNet121 Binary Model")
plt.tight_layout()
plt.savefig("densenet_binary_confusion_matrix.png")
plt.show()


# ---------------- ACCURACY & LOSS ----------------
loss, acc = model.evaluate(test_generator)
print(f"\n✅ Final Accuracy: {acc*100:.2f}%")
print(f"💡 Model Loss: {loss:.4f}\n")
