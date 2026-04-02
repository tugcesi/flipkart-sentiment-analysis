# Flipkart Ürün İncelemesi Duygu Analizi

Bu proje, Flipkart ürün incelemelerini **Negatif**, **Nötr** ve **Pozitif** olarak sınıflandıran bir makine öğrenmesi modeli ve Streamlit tabanlı bir web uygulaması içermektedir.

## Dosya Yapısı

```
flipkart-sentiment-analysis/
├── app_streamlit.py              # Streamlit web uygulaması
├── FlipkartReviewsSentimentAnalysis.ipynb  # Model eğitim notebook'u
├── sentiment_model.joblib        # Eğitilmiş model
├── vectorizer.joblib             # TF-IDF vektörleştirici
├── requirements.txt              # Bağımlılıklar
└── README.md
```

## Kurulum

### 1. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### 2. Streamlit Uygulamasını Çalıştırın

```bash
streamlit run app_streamlit.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde çalışacaktır.

## Özellikler

- 🔍 **Tek Yorum Analizi** – Tek bir ürün yorumunu analiz eder
- 📋 **Çoklu Yorum Analizi** – Birden fazla yorumu aynı anda analiz eder
- 📊 **Güven Skorları** – Tüm 3 sınıf için olasılık skorlarını gösterir
- 🌍 **Türkçe Arayüz** – Türkçe dil desteği ile kullanıcı dostu arayüz

## Sınıflar

| Sınıf | Etiket  | Açıklama         |
|-------|---------|------------------|
| 0     | Negatif | Olumsuz inceleme |
| 1     | Nötr    | Tarafsız inceleme |
| 2     | Pozitif | Olumlu inceleme  |

## Model Bilgisi

Model, `FlipkartReviewsSentimentAnalysis.ipynb` notebook'unda TF-IDF vektörleştirme ve makine öğrenmesi sınıflandırıcısı kullanılarak eğitilmiş ve `joblib` ile kaydedilmiştir:

```python
joblib.dump(best_model, 'sentiment_model.joblib')
joblib.dump(vect, 'vectorizer.joblib')
```
