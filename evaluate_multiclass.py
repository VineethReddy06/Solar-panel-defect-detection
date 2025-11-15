import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ------------------- SETTINGS -------------------
MODEL_PATH = "densenet_multiclass_best.h5"     # or your MobileNet model
DATASET_PATH = "Faulty_solar_panel"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16

CLASS_NAMES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]

# ------------------- LOAD MODEL -------------------
print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# ------------------- DATA GENERATOR -------------------
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical'
)

print(f"\nTotal images for evaluation: {test_gen.samples}")

# ------------------- PREDICT -------------------
pred_probs = model.predict(test_gen)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = test_gen.classes

# ------------------- ACCURACY -------------------
loss, acc = model.evaluate(test_gen)
print(f"\n📌 Final Accuracy: {acc*100:.2f}%")
print(f"📌 Final Loss: {loss:.4f}")

# ------------------- CLASSIFICATION REPORT -------------------
print("\n📌 Classification Report:")
print(classification_report(true_classes, pred_classes, target_names=CLASS_NAMES))

# ------------------- CONFUSION MATRIX -------------------
cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.title("Confusion Matrix – 6-Class Solar Panel Defect Model")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.savefig("confusion_matrix_multiclass.png")
plt.show()

print("\nConfusion matrix saved as 'confusion_matrix_multiclass.png'")
