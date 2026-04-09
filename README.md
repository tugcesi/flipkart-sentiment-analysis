# Flipkart Sentiment Analysis

Bu proje, Flipkart ürün yorumlarının analiz edilerek otomatik olarak olumlu/olumsuz duygu (sentiment) sınıflandırması yapılmasını amaçlar. Projede ön işleme ve makine öğrenmesi teknikleri kullanılmaktadır.

## Kullanım

1. Gerekli paketleri yükleyin:
    ```
    pip install -r requirements.txt
    ```
2. Ana Jupyter defterini (`FlipkartReviewsSentimentAnalysis.ipynb`) çalıştırarak modeli eğitin ve değerlendirin.
3. Alternatif olarak, `app.py` dosyası ile Python üzerinden temel bir uygulama başlatabilirsiniz.

## Dosya Açıklamaları

- `FlipkartReviewsSentimentAnalysis.ipynb` : Tüm veri hazırlama, modelleme ve test süreçlerini içeren Jupyter defteri
- `app.py` : Python tabanlı örnek uygulama arayüzü
- `cloud.png` : Projeyle ilişkili görsel veya WordCloud çıktısı
- `sentiment_model.joblib` : Kaydedilmiş model dosyası
- `vectorizer.joblib` : Kaydedilmiş vektörleştirici/model girdisi
- `requirements.txt` : Kullanılan temel Python paketleri listesi
- `.gitignore` : Versiyon kontrolde hariç tutulan dosyalar

## Lisans

MIT Lisansı