"""
Ön İşleme Modülü
================
Görüntü yükleme ve normalizasyon yardımcı fonksiyonları.
"""

import numpy as np
import cv2
from tensorflow.keras.preprocessing import image


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Tek bir görüntüyü yükleyip ön işler."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return img_array


def preprocess_batch(image_paths, target_size=(224, 224)):
    """Birden fazla görüntüyü ön işler."""
    images = np.array([
        load_and_preprocess_image(p, target_size) for p in image_paths
    ])
    return images


if __name__ == "__main__":
    print("Preprocessing modülü hazır.")
