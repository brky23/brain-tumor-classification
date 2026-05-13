"""
Model Mimarileri
================
İki farklı sınıflandırma modeli tanımlar:
1. Custom CNN - baseline model
2. EfficientNetB0 - transfer learning tabanlı model
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

NUM_CLASSES = 4
INPUT_SHAPE = (224, 224, 3)


def build_custom_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Sıfırdan eğitilen özel CNN mimarisi.
    
    Mimari:
    - 4 adet Conv blok (Conv -> BatchNorm -> ReLU -> MaxPool)
    - GlobalAveragePooling2D
    - Dense katmanlar + Dropout
    
    Args:
        input_shape (tuple): Giriş görüntü boyutu
        num_classes (int): Sınıf sayısı
    
    Returns:
        tf.keras.Model: Derlenmiş Custom CNN modeli
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Blok 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Blok 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Blok 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Blok 4
        layers.Conv2D(256, (3, 3), padding='same', name='last_conv'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Sınıflandırma kafası
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='CustomCNN')
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_efficientnet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, fine_tune=True):
    """
    EfficientNetB0 tabanlı transfer learning modeli.
    
    Args:
        input_shape (tuple): Giriş görüntü boyutu
        num_classes (int): Sınıf sayısı
        fine_tune (bool): Üst katmanların ince ayar yapılıp yapılmayacağı
    
    Returns:
        tf.keras.Model: Derlenmiş EfficientNetB0 modeli
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # İlk eğitim aşamasında base modeli dondur
    base_model.trainable = False
    
    if fine_tune:
        # Son 30 katmanı eğitime aç
        for layer in base_model.layers[-30:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
    
    inputs = layers.Input(shape=input_shape)
    # EfficientNet'in kendi normalizasyonu var; biz [0,1] aralığında veriyoruz
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name='EfficientNetB0_TL')
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model_info(model):
    """Modelin temel bilgilerini döndürür."""
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    return {
        'name': model.name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layers': len(model.layers)
    }


if __name__ == "__main__":
    print("=== Custom CNN ===")
    cnn = build_custom_cnn()
    cnn.summary()
    print(get_model_info(cnn))
    
    print("\n=== EfficientNetB0 ===")
    eff = build_efficientnet()
    eff.summary()
    print(get_model_info(eff))
