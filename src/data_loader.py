"""
Veri Yükleyici Modülü
=====================
Brain Tumor MRI veri setini yükler ve train/val/test setlerine ayırır.
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Sabitler
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
SEED = 42


def get_data_generators(train_dir, test_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Eğitim, doğrulama ve test veri üreticilerini oluşturur.

    Args:
        train_dir (str): Eğitim verisinin yolu
        test_dir (str): Test verisinin yolu
        img_size (tuple): Hedef görüntü boyutu (h, w)
        batch_size (int): Batch boyutu

    Returns:
        tuple: (train_gen, val_gen, test_gen)
    """
    # Eğitim için augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        validation_split=0.15,
        fill_mode='nearest'
    )

    # Test için sadece normalizasyon
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=SEED,
        classes=CLASS_NAMES
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=SEED,
        classes=CLASS_NAMES
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=CLASS_NAMES
    )

    return train_gen, val_gen, test_gen


def get_class_distribution(generator):
    """Veri setindeki sınıf dağılımını döndürür."""
    import numpy as np
    classes = generator.classes
    unique, counts = np.unique(classes, return_counts=True)
    return dict(zip([CLASS_NAMES[i] for i in unique], counts))


if __name__ == "__main__":
    # Test amaçlı çalıştırma
    train_dir = "data/Training"
    test_dir = "data/Testing"
    if os.path.exists(train_dir):
        train_gen, val_gen, test_gen = get_data_generators(train_dir, test_dir)
        print(f"Train samples: {train_gen.samples}")
        print(f"Val samples: {val_gen.samples}")
        print(f"Test samples: {test_gen.samples}")
        print(f"Class distribution: {get_class_distribution(train_gen)}")
