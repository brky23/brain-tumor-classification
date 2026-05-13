"""
Brain Tumor MRI Classification - Ana Çalıştırma Scripti
========================================================
Tüm pipeline'ı tek komutla çalıştırır:
    python main.py

Notebook tabanlı bir çalışma için notebooks/brain_tumor_classification.ipynb kullanın.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

from src.data_loader import get_data_generators, CLASS_NAMES, IMG_SIZE, BATCH_SIZE
from src.models import build_custom_cnn, build_efficientnet
from src.train import train_model
from src.evaluate import (
    evaluate_model, plot_confusion_matrix, plot_training_curves,
    plot_roc_curves, create_comparison_table, create_per_class_table
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main(args):
    print(f"\n{'='*60}\nBrain Tumor MRI Classification\n{'='*60}\n")
    
    # 1. Veri yükle
    print("[1/5] Veri yükleniyor...")
    train_gen, val_gen, test_gen = get_data_generators(
        args.train_dir, args.test_dir, IMG_SIZE, BATCH_SIZE
    )
    print(f"  Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")
    
    # 2. Custom CNN
    print("\n[2/5] Custom CNN eğitiliyor...")
    cnn = build_custom_cnn()
    cnn_history = train_model(cnn, train_gen, val_gen,
                              epochs=args.epochs, model_name='cnn',
                              checkpoint_dir=args.model_dir)
    
    # 3. EfficientNet
    print("\n[3/5] EfficientNetB0 eğitiliyor...")
    eff = build_efficientnet()
    eff_history = train_model(eff, train_gen, val_gen,
                              epochs=args.epochs, model_name='efficientnet',
                              checkpoint_dir=args.model_dir)
    
    # 4. Değerlendirme
    print("\n[4/5] Modeller değerlendiriliyor...")
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'metrics'), exist_ok=True)
    
    cnn_results = evaluate_model(cnn, test_gen, CLASS_NAMES)
    eff_results = evaluate_model(eff, test_gen, CLASS_NAMES)
    
    plot_training_curves(cnn_history, 'Custom CNN',
                         os.path.join(args.results_dir, 'figures/training_cnn.png'))
    plot_training_curves(eff_history, 'EfficientNetB0',
                         os.path.join(args.results_dir, 'figures/training_efficientnet.png'))
    plot_confusion_matrix(cnn_results['confusion_matrix'], CLASS_NAMES,
                          'Custom CNN', os.path.join(args.results_dir, 'figures/cm_cnn.png'))
    plot_confusion_matrix(eff_results['confusion_matrix'], CLASS_NAMES,
                          'EfficientNetB0', os.path.join(args.results_dir, 'figures/cm_efficientnet.png'))
    plot_roc_curves(cnn_results['y_true'], cnn_results['y_pred_proba'],
                    CLASS_NAMES, 'Custom CNN',
                    os.path.join(args.results_dir, 'figures/roc_cnn.png'))
    plot_roc_curves(eff_results['y_true'], eff_results['y_pred_proba'],
                    CLASS_NAMES, 'EfficientNetB0',
                    os.path.join(args.results_dir, 'figures/roc_efficientnet.png'))
    
    comp = create_comparison_table(
        {'Custom CNN': cnn_results, 'EfficientNetB0': eff_results},
        CLASS_NAMES,
        os.path.join(args.results_dir, 'metrics/comparison.csv')
    )
    print("\n=== Sonuç Karşılaştırması ===")
    print(comp.to_string(index=False))
    
    # 5. Tamamlandı
    print(f"\n[5/5] Tamamlandı. Sonuçlar: {args.results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Brain Tumor MRI Classification')
    parser.add_argument('--train_dir', default='data/Training')
    parser.add_argument('--test_dir', default='data/Testing')
    parser.add_argument('--model_dir', default='models')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    main(args)
