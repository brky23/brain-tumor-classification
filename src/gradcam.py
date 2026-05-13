"""
Grad-CAM Modülü
===============
Modelin karar verirken görüntünün hangi bölgelerine odaklandığını görselleştirir.

Referans:
Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization. ICCV.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Bir görüntü için Grad-CAM ısı haritası üretir.
    
    Args:
        img_array (np.ndarray): (1, H, W, 3) şeklinde batch'lenmiş görüntü
        model: Eğitilmiş Keras modeli
        last_conv_layer_name (str): Son konvolüsyon katmanının adı
        pred_index (int, optional): Açıklanacak sınıfın indeksi. None ise tahmin edilen.
    
    Returns:
        np.ndarray: [0,1] aralığında normalize edilmiş ısı haritası
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Isı haritasını orijinal görüntü üzerine bindirir.
    
    Args:
        img (np.ndarray): Orijinal görüntü [0,1] veya [0,255]
        heatmap (np.ndarray): Grad-CAM ısı haritası
        alpha (float): Saydamlık (0-1)
        colormap: OpenCV renk haritası
    
    Returns:
        np.ndarray: Bindirilmiş görüntü
    """
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    overlayed = heatmap_color * alpha + img * (1 - alpha)
    overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)
    return overlayed


def visualize_gradcam_grid(images, true_labels, model, class_names,
                            last_conv_layer_name, n_samples=8, save_path=None):
    """
    Birden fazla örnek için Grad-CAM görselleştirmesini grid halinde gösterir.
    
    Args:
        images: Görüntü dizisi (n, H, W, 3)
        true_labels: Gerçek etiketler
        model: Eğitilmiş model
        class_names: Sınıf isimleri listesi
        last_conv_layer_name: Son konv katman adı
        n_samples: Görselleştirilecek örnek sayısı
        save_path: Kayıt yolu
    """
    n_samples = min(n_samples, len(images))
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        img = images[i]
        true_idx = true_labels[i] if isinstance(true_labels[i], (int, np.integer)) \
                                  else np.argmax(true_labels[i])
        img_batch = np.expand_dims(img, axis=0)
        
        preds = model.predict(img_batch, verbose=0)
        pred_idx = np.argmax(preds[0])
        confidence = preds[0][pred_idx]
        
        heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer_name)
        overlay = overlay_heatmap(img, heatmap, alpha=0.4)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Orijinal\nGerçek: {class_names[true_idx]}',
                             fontsize=11)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(heatmap, cmap='jet')
        axes[i, 1].set_title('Grad-CAM Isı Haritası', fontsize=11)
        axes[i, 1].axis('off')
        
        correct = "✓" if pred_idx == true_idx else "✗"
        color = 'green' if pred_idx == true_idx else 'red'
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(
            f'Bindirilmiş {correct}\n'
            f'Tahmin: {class_names[pred_idx]} ({confidence:.2%})',
            fontsize=11, color=color
        )
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def find_last_conv_layer(model):
    """Modelin son konvolüsyon katmanının adını otomatik bulur."""
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower() or isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Modelde konvolüsyon katmanı bulunamadı.")


if __name__ == "__main__":
    print("Grad-CAM modülü hazır.")
