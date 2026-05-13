"""
Eğitim Modülü
=============
Modelleri eğitir ve checkpoint olarak kaydeder.
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)


def get_callbacks(model_name, checkpoint_dir='models', patience=7):
    """Eğitim sırasında kullanılacak callback'leri döndürür."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'best_{model_name}.h5')
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks


def compute_class_weights(generator):
    """Dengesiz veri için sınıf ağırlıklarını hesaplar."""
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    return dict(enumerate(class_weights))


def train_model(model, train_gen, val_gen, epochs=20, model_name='model',
                use_class_weights=True, checkpoint_dir='models'):
    """
    Modeli eğitir.
    
    Args:
        model: Derlenmiş Keras modeli
        train_gen: Eğitim veri üreticisi
        val_gen: Doğrulama veri üreticisi
        epochs (int): Epoch sayısı
        model_name (str): Kaydetme için model adı
        use_class_weights (bool): Sınıf ağırlıklarını kullan
        checkpoint_dir (str): Kayıt klasörü
    
    Returns:
        tf.keras.callbacks.History: Eğitim geçmişi
    """
    callbacks = get_callbacks(model_name, checkpoint_dir)
    
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_gen)
        print(f"Class weights: {class_weights}")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return history


if __name__ == "__main__":
    print("Train modülü hazır.")
