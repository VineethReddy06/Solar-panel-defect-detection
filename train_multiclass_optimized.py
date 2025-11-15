import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# --------------------- SETTINGS ---------------------
DATASET_PATH = "Faulty_solar_panel"
IMAGE_SIZE = (256, 256)      # Bigger size = better defect detection
BATCH_SIZE = 16              # Safe for most laptops
EPOCHS = 30
SEED = 42

CLASS_NAMES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]
NUM_CLASSES = len(CLASS_NAMES)

# --------------------- AUGMENTATION ---------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.25,
    shear_range=0.15,
    brightness_range=[0.6, 1.3],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
    validation_split=0.2
)

train = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

val = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

print(f"\nFound {train.samples} training images and {val.samples} validation images.")

# --------------------- CLASS WEIGHTS ---------------------
y = train.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = {i: w for i, w in enumerate(class_weights)}

print("Class Weights:", class_weights)

# --------------------- MODEL ---------------------
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

# Unfreeze last ~100 layers for deeper fine-tuning
for layer in base_model.layers[:-100]:
    layer.trainable = False
for layer in base_model.layers[-100:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(384, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(base_model.input, predictions)

model.compile(
    optimizer=Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------- CALLBACKS ---------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "densenet_multiclass_best.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=3,
        factor=0.3,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
]

# --------------------- TRAIN ---------------------
history = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# Save final model
model.save("densenet_multiclass_final.h5")
print("\nTraining complete!")
print("Saved: densenet_multiclass_best.h5  &  densenet_multiclass_final.h5")
