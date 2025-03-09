# model_training.py
import json
import os
import pathlib
from typing import Any

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
DATA_DIR = pathlib.Path().resolve() / "data" / "processed"


def create_model(num_classes: int) -> tuple[Model, Model]:
    """Create a transfer learning model based on MobileNetV2."""
    # Load pre-trained MobileNetV2 without the classification layer
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def create_data_generators() -> tuple[Sequence, Sequence, Sequence]:
    """Create data generators with augmentation for training."""
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Validation and test data generators (no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    # Load validation data
    validation_generator = val_test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "validation"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    # Load test data
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, validation_generator, test_generator


def train_model() -> tuple[Model, Any]:
    """Train the dog breed classification model."""
    # Create data generators
    train_generator, validation_generator, test_generator = create_data_generators()

    # Get number of classes
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")

    # Create and compile the model
    model, base_model = create_model(num_classes)

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        "best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1
    )

    # Phase 1: Train only the top layers
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[early_stopping, checkpoint],
    )

    # Phase 2: Fine-tune the last few layers of the base model
    # Unfreeze the last 20 layers
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Continue training
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[early_stopping, checkpoint],
    )

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save the class indices for prediction
    class_indices = train_generator.class_indices
    # Invert the dictionary to map indices to class names
    label_map = {v: k for k, v in class_indices.items()}

    # Save the label map to a file
    with open("label_map.json", "w") as f:
        json.dump(label_map, f)

    # Save the model for TF Serving
    model.save("dog_breed_model.h5")

    # Convert to TensorFlow Lite for mobile deployment
    convert_to_tflite(model)

    return model, history


def convert_to_tflite(model: Model) -> None:
    """Convert Keras model to TensorFlow Lite format."""
    # Create a converter that's compatible with older TF versions
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Set the target ops to use only ops compatible with older versions
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Use only TFLite builtin ops
    ]

    # Avoid using newer op versions
    converter.allow_custom_ops = False

    # Convert the model
    tflite_model = converter.convert()

    # Save the model
    with open("dog_breed_model_quantized.tflite", "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    train_model()
