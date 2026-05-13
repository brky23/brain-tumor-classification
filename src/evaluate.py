"""
Değerlendirme Modülü
====================
Eğitilmiş modellerin performansını ölçer, görselleştirir ve raporlar.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize


def evaluate_model(model, test_gen, class_names):
    """
    Test setinde modeli değerlendirir.
    
    Returns:
        dict: Tüm metrikleri içeren sözlük
    """
    test_gen.reset()
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_gen.classes
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names,
        zero_division=0, output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_p,
        'weighted_recall': weighted_r,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def plot_confusion_matrix(cm, class_names, title, save_path=None):
    """Confusion matrix grafiğini çizer."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_curves(history, model_name, save_path=None):
    """Eğitim sürecindeki accuracy ve loss eğrilerini çizer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Eğitim', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Doğrulama', linewidth=2)
    axes[0].set_title(f'{model_name} - Doğruluk Eğrisi', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Eğitim', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Doğrulama', linewidth=2)
    axes[1].set_title(f'{model_name} - Loss Eğrisi', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(y_true, y_pred_proba, class_names, model_name, save_path=None):
    """Her sınıf için ROC eğrisini çizer."""
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    plt.figure(figsize=(9, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, linewidth=2,
                 label=f'{cls} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} - ROC Eğrileri (One-vs-Rest)',
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def create_comparison_table(results_dict, class_names, save_path=None):
    """Birden fazla modelin karşılaştırma tablosunu oluşturur."""
    rows = []
    for model_name, metrics in results_dict.items():
        rows.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Macro Precision': f"{metrics['macro_precision']:.4f}",
            'Macro Recall': f"{metrics['macro_recall']:.4f}",
            'Macro F1': f"{metrics['macro_f1']:.4f}",
            'Weighted F1': f"{metrics['weighted_f1']:.4f}"
        })
    df = pd.DataFrame(rows)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def create_per_class_table(metrics, class_names, save_path=None):
    """Her sınıf için detaylı metrik tablosu oluşturur."""
    df = pd.DataFrame({
        'Sınıf': class_names,
        'Precision': [f"{p:.4f}" for p in metrics['precision_per_class']],
        'Recall': [f"{r:.4f}" for r in metrics['recall_per_class']],
        'F1-Score': [f"{f:.4f}" for f in metrics['f1_per_class']],
        'Support': metrics['support_per_class']
    })
    if save_path:
        df.to_csv(save_path, index=False)
    return df


if __name__ == "__main__":
    print("Evaluate modülü hazır.")
