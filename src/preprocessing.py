import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(train_dir, val_dir, test_dir, image_size=(128, 128), batch_size=32):
    """
    Loads image data from train, validation, and test directories using Keras ImageDataGenerator.

    Args:
        train_dir (str): Path to training images.
        val_dir (str): Path to validation images.
        test_dir (str): Path to test images.
        image_size (tuple): Size to resize images to (default: (128, 128)).
        batch_size (int): Number of images per batch (default: 32).

    Returns:
        train_gen: Training data generator.
        val_gen: Validation data generator.
        test_gen: Test data generator.
        class_indices: Mapping of class names to indices.
    """
    # Basic augmentation for training, only rescaling for val/test
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )

    class_indices = train_gen.class_indices

    return train_gen, val_gen, test_gen, class_indices