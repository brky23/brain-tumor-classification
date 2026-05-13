# 🧠 Derin Öğrenme ile Beyin MR Görüntülerinden Tümör Sınıflandırması ve Grad-CAM ile Açıklanabilirlik Analizi

> Bu proje, beyin MR (Manyetik Rezonans) görüntülerini derin öğrenme yöntemleri ile dört farklı sınıfa ayırmayı ve modelin karar verme sürecini Grad-CAM (Gradient-weighted Class Activation Mapping) tekniği ile görselleştirmeyi amaçlamaktadır.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 İçindekiler

- [Proje Amacı](#-proje-amacı)
- [Problem Tanımı](#-problem-tanımı)
- [Veri Seti](#-veri-seti)
- [Klasör Yapısı](#-klasör-yapısı)
- [Kurulum](#%EF%B8%8F-kurulum)
- [Kullanım](#-kullanım)
- [Veri Ön İşleme](#-veri-ön-i̇şleme)
- [Kullanılan Modeller](#%EF%B8%8F-kullanılan-modeller)
- [Eğitim Yapılandırması](#-eğitim-yapılandırması)
- [Performans Metrikleri](#-performans-metrikleri)
- [Sonuçlar](#-sonuçlar)
- [Grad-CAM Analizi](#-grad-cam-analizi)
- [Rapor ve Broşür](#-rapor-ve-broşür)
- [Demo Video](#-demo-video)
- [Gelecek Çalışmalar](#-gelecek-çalışmalar)
- [Lisans](#-lisans)

---

## 🎯 Proje Amacı

Beyin tümörleri, erken teşhis edilmediğinde ölümcül olabilen ciddi sağlık problemleridir. Manuel olarak MR görüntülerinden tümör tipinin belirlenmesi, uzman radyolog gerektiren zaman alıcı bir işlemdir. Bu projenin temel amaçları:

1. **Otomatik sınıflandırma**: Beyin MR görüntülerini glioma, meningioma, pituitary tümörü ve sağlıklı (no tumor) olarak sınıflandıran derin öğrenme modelleri geliştirmek.
2. **Model karşılaştırması**: Sıfırdan eğitilen bir Custom CNN ile transfer learning tabanlı EfficientNetB0 modelinin performansını karşılaştırmak.
3. **Açıklanabilirlik**: Grad-CAM ile modelin karar verirken görüntünün hangi bölgelerine odaklandığını görselleştirmek — bu klinik kabul için kritiktir.

---

## 🧩 Problem Tanımı

**Görev tipi:** Multi-class image classification (4 sınıf)

**Giriş:** Beyin MR görüntüsü (JPG, 224×224×3 boyutuna yeniden boyutlandırılır)

**Çıkış:** Aşağıdaki 4 sınıftan birine ait olasılık vektörü:
- `glioma` — Glioma tümörü
- `meningioma` — Meningioma tümörü
- `notumor` — Tümör yok (sağlıklı)
- `pituitary` — Pituitary (hipofiz) tümörü

**Ek hedef:** En başarılı modelin sınıflandırma kararını Grad-CAM ile görsel olarak açıklamak.

---

## 📊 Veri Seti

### Kaynak
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — Masoud Nickparvar (Kaggle)

### Özellikler

| Özellik | Değer |
|---------|-------|
| Toplam görüntü | 7023 |
| Eğitim seti | 5712 görüntü |
| Test seti | 1311 görüntü |
| Sınıf sayısı | 4 |
| Görüntü formatı | JPG |
| Orijinal boyut | Değişken |
| Model giriş boyutu | **224 × 224 × 3** (RGB) |

### Sınıf Dağılımı (Eğitim)

| Sınıf | Görüntü Sayısı |
|-------|----------------|
| glioma | ~1321 |
| meningioma | ~1339 |
| notumor | ~1595 |
| pituitary | ~1457 |

> **Not:** Sınıflar arası dengesizliği telafi etmek için eğitim sırasında `class_weight='balanced'` kullanılmıştır.

### Veri Bölümlemesi

- **Eğitim:** Training klasörünün %85'i
- **Doğrulama:** Training klasörünün %15'i (`validation_split=0.15`)
- **Test:** Testing klasörünün tamamı (eğitim sırasında hiç görülmemiştir)

---

## 📁 Klasör Yapısı

```
brain-tumor-classification/
│
├── README.md                          # Bu dosya
├── LICENSE                            # MIT lisansı
├── .gitignore                         # Git tarafından izlenmeyecek dosyalar
├── requirements.txt                   # Python bağımlılıkları
├── main.py                            # Tüm pipeline'ı çalıştıran ana script
│
├── data/                              # Veri seti klasörü (içerik git'e dahil değil)
│   └── README.md                      # Veri seti indirme talimatları
│
├── notebooks/
│   └── brain_tumor_classification.ipynb  # Tüm projeyi çalıştıran ana notebook
│
├── src/                               # Modüler kaynak kod
│   ├── __init__.py
│   ├── data_loader.py                 # Veri yükleme ve augmentation
│   ├── preprocessing.py               # Görüntü ön işleme yardımcı fonksiyonları
│   ├── models.py                      # Custom CNN ve EfficientNetB0 mimarileri
│   ├── train.py                       # Eğitim döngüsü ve callbacks
│   ├── evaluate.py                    # Metrikler, confusion matrix, ROC
│   └── gradcam.py                     # Grad-CAM görselleştirme
│
├── models/                            # Eğitilmiş model dosyaları (.h5, git'e dahil değil)
│
├── results/                           # Sonuç çıktıları
│   ├── figures/                       # Tüm grafikler
│   │   ├── class_distribution.png
│   │   ├── sample_images.png
│   │   ├── training_curves_cnn.png
│   │   ├── training_curves_efficientnet.png
│   │   ├── cm_cnn.png
│   │   ├── cm_efficientnet.png
│   │   ├── roc_cnn.png
│   │   ├── roc_efficientnet.png
│   │   ├── gradcam_results.png
│   │   └── sample_predictions.png
│   └── metrics/                       # CSV formatında metrik tabloları
│       ├── model_comparison.csv
│       └── per_class_metrics.csv
│
└── docs/                              # Doküman ve sunum materyalleri
    ├── report.pdf                     # Detaylı proje raporu
    ├── brochure.pdf                   # Tanıtım broşürü
    └── demo_video_link.txt            # Demo video linki
```

### Dosya Açıklamaları

| Dosya / Klasör | Açıklama |
|----------------|----------|
| `notebooks/brain_tumor_classification.ipynb` | **Ana notebook** — projeyi baştan sona çalıştırır (Colab uyumlu) |
| `main.py` | Pipeline'ı tek komutla terminal üzerinden çalıştırır |
| `src/data_loader.py` | Veri setini Keras `ImageDataGenerator` ile yükler, augmentation uygular |
| `src/models.py` | İki model mimarisini tanımlar: Custom CNN ve EfficientNetB0 |
| `src/train.py` | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint callback'leri ile eğitim |
| `src/evaluate.py` | Accuracy, precision, recall, F1, confusion matrix, ROC eğrileri |
| `src/gradcam.py` | Grad-CAM ısı haritası hesaplama ve görselleştirme |
| `results/` | Tüm grafikler, metrik tabloları, sonuç dosyaları |
| `docs/` | Rapor PDF, broşür PDF, demo video linki |

---

## ⚙️ Kurulum

### 1. Repo'yu klonla
```bash
git clone https://github.com/<kullaniciadi>/brain-tumor-classification.git
cd brain-tumor-classification
```

### 2. Sanal ortam oluştur (önerilen)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Bağımlılıkları yükle
```bash
pip install -r requirements.txt
```

### 4. Veri setini indir
`data/README.md` dosyasındaki talimatları izleyerek veri setini `data/Training/` ve `data/Testing/` klasörlerine yerleştirin.

### Sistem Gereksinimleri

| Gereksinim | Minimum | Önerilen |
|------------|---------|----------|
| Python | 3.9 | 3.10+ |
| RAM | 8 GB | 16 GB |
| GPU | (CPU mümkün ama çok yavaş) | NVIDIA GPU (>= 6 GB VRAM) veya Google Colab GPU |
| Disk | 2 GB | 5 GB |

> ⚡ **Önerilen kullanım:** Google Colab (ücretsiz GPU). Notebook tamamen Colab uyumludur.

---

## 🚀 Kullanım

### Yöntem 1: Notebook ile (önerilen)

```bash
jupyter notebook notebooks/brain_tumor_classification.ipynb
```

veya Google Colab'a yükleyip "Run All" deyin.

### Yöntem 2: Komut satırından

```bash
python main.py --train_dir data/Training --test_dir data/Testing --epochs 20
```

#### Parametreler
| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `--train_dir` | `data/Training` | Eğitim verisi yolu |
| `--test_dir` | `data/Testing` | Test verisi yolu |
| `--model_dir` | `models` | Modellerin kaydedileceği klasör |
| `--results_dir` | `results` | Sonuçların kaydedileceği klasör |
| `--epochs` | `20` | Eğitim epoch sayısı (EarlyStopping ile erken durur) |

---

## 🧪 Veri Ön İşleme

Aşağıdaki adımlar her görüntüye uygulanmıştır:

1. **Yeniden boyutlandırma:** Tüm görüntüler 224×224 piksele dönüştürülür.
2. **Renk dönüşümü:** Grayscale görüntüler 3 kanallı RGB'ye genişletilir (transfer öğrenmesi için).
3. **Normalizasyon:** Piksel değerleri [0, 1] aralığına ölçeklenir (`rescale=1./255`).
4. **Veri artırma (sadece eğitim setinde):**
   - Rotation: ±15°
   - Width / Height shift: ±10%
   - Zoom: ±10%
   - Horizontal flip: Evet
   - Brightness: ±10%
   - Fill mode: `nearest`

Doğrulama ve test setlerinde augmentation **uygulanmaz**, yalnızca normalizasyon yapılır.

---

## 🏗️ Kullanılan Modeller

### Model 1: Custom CNN (Baseline)

Sıfırdan eğitilen 4-blok konvolüsyonel sinir ağı:

```
Input (224×224×3)
  ├── Conv2D(32) → BatchNorm → ReLU → MaxPool(2)
  ├── Conv2D(64) → BatchNorm → ReLU → MaxPool(2)
  ├── Conv2D(128) → BatchNorm → ReLU → MaxPool(2)
  ├── Conv2D(256) → BatchNorm → ReLU → MaxPool(2)   ← Grad-CAM hedef katmanı
  ├── GlobalAveragePooling2D
  ├── Dense(128) → ReLU → Dropout(0.5)
  └── Dense(4) → Softmax
```

**Toplam parametre sayısı:** ~470K (yaklaşık)

### Model 2: EfficientNetB0 (Transfer Learning)

ImageNet üzerinde önceden eğitilmiş EfficientNetB0 omurgası + özel sınıflandırma kafası:

```
Input (224×224×3)
  ├── EfficientNetB0 (ImageNet pretrained)
  │   └── Son 30 katman fine-tuning için açık (BatchNorm hariç)
  ├── GlobalAveragePooling2D
  ├── Dropout(0.3)
  ├── Dense(128) → ReLU → Dropout(0.3)
  └── Dense(4) → Softmax
```

**Toplam parametre sayısı:** ~4M (~5M ile birlikte)

---

## 🔧 Eğitim Yapılandırması

| Parametre | Custom CNN | EfficientNetB0 |
|-----------|------------|----------------|
| **Optimizer** | Adam | Adam |
| **Learning rate** | 1e-3 | 1e-4 |
| **Loss** | Categorical Crossentropy | Categorical Crossentropy |
| **Batch size** | 32 | 32 |
| **Epochs (max)** | 20 | 20 |
| **EarlyStopping patience** | 7 | 7 |
| **ReduceLROnPlateau** | factor=0.5, patience=3 | factor=0.5, patience=3 |
| **Class weights** | Balanced | Balanced |
| **Random seed** | 42 | 42 |

### Callback'ler
- **EarlyStopping**: Doğrulama loss 7 epoch boyunca iyileşmezse durur, en iyi ağırlıkları geri yükler.
- **ReduceLROnPlateau**: Doğrulama loss 3 epoch boyunca iyileşmezse learning rate'i yarıya indirir.
- **ModelCheckpoint**: Her epoch'ta en iyi doğrulama doğruluğuna sahip modeli kaydeder.

---

## 📈 Performans Metrikleri

Aşağıdaki metrikler **test seti** üzerinde hesaplanır:

- **Accuracy** — Genel doğruluk
- **Precision** (per-class + macro + weighted)
- **Recall** (per-class + macro + weighted)
- **F1-Score** (per-class + macro + weighted)
- **Confusion Matrix** — Her iki model için ayrı ayrı
- **ROC Eğrileri + AUC** — One-vs-Rest yaklaşımı ile her sınıf için
- **Eğitim eğrileri** — Train/val accuracy ve loss grafikleri

---

## 📊 Sonuçlar

### Model Karşılaştırması

| Model | Accuracy | Macro F1 | Weighted F1 | Mean AUC |
| --- | --- | --- | --- | --- |
| Custom CNN | 81.75% | 0.8082 | 0.8082 | 0.9498 |
| **EfficientNetB0** ⭐ | **93.19%** | **0.9305** | **0.9305** | **0.9878** |

> 💡 EfficientNetB0 transfer learning ile Custom CNN'i %11.44 farkla geçmiştir. Bu sonuç, tıbbi görüntü sınıflandırmasında transfer öğrenmenin gücünü kanıtlamaktadır.

### Per-Class Performansı (EfficientNetB0)

| Sınıf | Precision | Recall | F1-Score | AUC | Support |
| --- | --- | --- | --- | --- | --- |
| glioma | 0.9752 | 0.7875 | 0.8714 | 0.969 | 400 |
| meningioma | 0.8539 | 0.9500 | 0.8994 | 0.983 | 400 |
| notumor | 0.9545 | 0.9975 | 0.9756 | 1.000 | 400 |
| pituitary | 0.9589 | 0.9925 | 0.9754 | 0.999 | 400 |
---

## 🔍 Grad-CAM Analizi

En başarılı model üzerinde Grad-CAM uygulanarak modelin karar verirken görüntünün hangi bölgelerine odaklandığı görselleştirilmiştir. Her görsel üçlüsü içerir:

1. **Orijinal MR görüntüsü** (gerçek sınıf etiketi ile)
2. **Grad-CAM ısı haritası** (kırmızı = yüksek dikkat, mavi = düşük)
3. **Bindirilmiş görüntü** (tahmin edilen sınıf + güven değeri ile)

Her 4 sınıftan örnekler için: `results/figures/gradcam_results.png`

> 💡 **Klinik anlamı:** Modelin tümör bölgesine odaklanıp odaklanmadığını görmek, modelin gerçek anatomik özelliklere mi yoksa görüntüdeki gürültü/artefakt'lara mı bakarak karar verdiğini anlamamızı sağlar.

---

## 📄 Rapor ve Broşür

- 📑 **Detaylı Proje Raporu**: [`docs/report.pdf`](docs/report.pdf)
  - Literatür taraması, yöntem, deneysel sonuçlar, tartışma, sonuç ve gelecek çalışmalar
- 📰 **Tanıtım Broşürü**: [`docs/brochure.pdf`](docs/brochure.pdf)
  - Projenin özet sunumu (2 sayfa)

## 🎬 Demo Video

1 dakikalık demo video: [`docs/demo_video_link.txt`](docs/demo_video_link.txt) dosyasındaki YouTube/Drive linkinden izlenebilir.

---

## 🔮 Gelecek Çalışmalar

Bu proje aşağıdaki yönlerde geliştirilebilir:

1. **Ensemble modelleri** — Custom CNN + EfficientNetB0 + ResNet50 ensemble ile daha yüksek doğruluk.
2. **Segmentasyon** — Sadece sınıflandırma değil, tümörün tam konumunu U-Net ile segmente etmek.
3. **3D MRI analizi** — 2D yerine 3D CNN ile volumetrik MR taramalarını işlemek.
4. **Daha büyük veri seti** — BraTS gibi daha kapsamlı veri setleri ile genelleme yeteneğini artırmak.
5. **Vision Transformer (ViT)** — Modern transformer mimarileri ile karşılaştırmak.
6. **Klinik validasyon** — Gerçek hastane verisi ile modelin sağlamlığını test etmek.
7. **Mobil uygulama** — Eğitilmiş modeli TFLite'a dönüştürerek mobil tabanlı tanı aracı geliştirmek.

---

## 📜 Lisans

Bu proje [MIT lisansı](LICENSE) altında lisanslanmıştır.

---

## 👤 Yazar

**Ahmet Berkay Kantarcı**

- 🎓 **Üniversite:** OSTİM Teknik Üniversitesi
- 📚 **Bölüm:** Yapay Zeka Mühendisliği — 3. Sınıf
- 📖 **Ders:** Derin Öğrenme
- 👨‍🏫 **Danışman:** Murat Şimşek
- 📅 **Tarih:** Mayıs 2026
- 🔗 **GitHub:** [@brky23](https://github.com/brky23)
## 🙏 Teşekkürler

- Veri seti: Masoud Nickparvar (Kaggle)
- TensorFlow ve Keras geliştirici topluluğu
- EfficientNet: Tan & Le (2019)
- Grad-CAM: Selvaraju et al. (2017)
