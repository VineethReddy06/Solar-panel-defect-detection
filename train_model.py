# ---------------- HIGH ACCURACY BINARY CNN (DenseNet121) ----------------
# Solar Panel Defect Detection - Clean vs Faulty
# Best model for small datasets with subtle defects

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------- SETTINGS ----------------
DATASET_PATH = "Faulty_solar_panel"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# ---------------- DATA AUGMENTATION ----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

print("\n📌 Data Loaded Successfully")


# ---------------- CLASS WEIGHTS ----------------
class_weights = {0: 2.0, 1: 1.0}  # Clean gets higher weight


# ---------------- CNN BASE ----------------
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze majority of layers, fine-tune last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True


# ---------------- TOP LAYERS ----------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n🚀 DenseNet121 CNN Model Created!")


# ---------------- CALLBACKS ----------------
checkpoint = ModelCheckpoint(
    "densenet_binary_best.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ---------------- TRAIN ----------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
    class_weight=class_weights
)

print("\n🎉 Training Completed — DenseNet Model Saved!")
model.save("densenet_binary_final.h5")
