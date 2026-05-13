# Veri Seti

Bu proje **Brain Tumor MRI Dataset** veri setini kullanmaktadır.

## Veri Seti Bilgileri

- **Kaynak:** [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Yazar:** Masoud Nickparvar
- **Toplam görüntü:** 7023
- **Format:** JPG
- **Sınıflar:** glioma, meningioma, notumor, pituitary

## İndirme

### Yöntem 1: Kaggle API
```bash
pip install kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/
```

### Yöntem 2: Manuel
1. Yukarıdaki Kaggle linkine gidin
2. "Download" butonuna tıklayın (Kaggle hesabı gerekli)
3. İndirilen zip dosyasını bu klasöre çıkarın

## Beklenen Klasör Yapısı

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## Not

Veri seti dosyaları git'e eklenmemiştir (`.gitignore` ile hariç tutulur). Çalıştırmadan önce yukarıdaki adımları izleyerek veriyi indirin.
