<div align="center">

# 🌍 Deprem AI

**Gerçek Zamanlı Sismik İzleme ve Derin Öğrenme Sınıflandırma Sistemi**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20%7C%20CPU-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

## 📖 Proje Hakkında

**Deprem AI**, AFAD ve USGS'in resmi deprem API'lerinden gerçek zamanlı veri çekerek bir PyTorch sinir ağıyla sismik olayları sınıflandıran tam yığın bir uygulamadır.

Web arayüzü üzerinden tek tıkla model eğitimi başlatılabilir, canlı deprem haritası görüntülenebilir ve AI tahmin sonuçları anlık olarak takip edilebilir.


## 🏗️ Proje Yapısı

```
Deprem-Ai/
├── main.py              # Komut satırı versiyon (orijinal)
├── api.py               # FastAPI backend — REST API + web sunucu
├── requirements.txt     # Python bağımlılıkları
├── static/
│   ├── index.html       # Ana dashboard sayfası
│   ├── style.css        # Koyu tema CSS
│   └── app.js           # Harita, grafikler, event yönetimi
└── README.md
```

---

## 🧠 Model Mimarisi

İkili sınıflandırma (Binary Classification) sinir ağı:

```
Girdi Özellikleri:
  1. underground_wave_energy   → 10^(magnitude/2) / (depth_km + 1)
  2. vibration_axis_variation  → Coğrafi yayılım (enlem-boylam sapması)

Ağ:
  Linear(2 → 10) → ReLU → Linear(10 → 10) → ReLU → Linear(10 → 1)

Kayıp Fonksiyonu : BCEWithLogitsLoss
Optimizer        : Adam (lr=0.001)
Epoch            : 300
Sınıflandırma    : Magnitude ≥ 4.0 → Önemli Sismik Olay (1), Diğer → Normal (0)
```

---

## 📡 Veri Kaynakları

| Kaynak | URL | Öncelik |
|---|---|---|
| 🇹🇷 AFAD | https://deprem.afad.gov.tr/apiv2/event/filter | 1. tercih |
| 🌎 USGS | https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson | Fallback |

---

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler

- Python 3.11+
- (Opsiyonel) NVIDIA GPU + CUDA

### 2. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

`requirements.txt` içeriği:
```
pandas
numpy
torch
matplotlib
scikit-learn
requests
fastapi
uvicorn[standard]
```

### 3. Web Dashboard'u Başlat

```bash
python api.py
```

Tarayıcıda **http://localhost:8000** adresini açın.

### 4. (Alternatif) Komut Satırı Versiyonu

```bash
python main.py
```

---

## 🌐 API Endpoints

| Method | Endpoint | Açıklama |
|---|---|---|
| `GET` | `/` | Dashboard arayüzü |
| `GET` | `/api/status` | Cihaz bilgisi, model durumu |
| `GET` | `/api/earthquakes` | Güncel deprem listesi + AI tahminleri |
| `POST` | `/api/train` | Model eğitimini başlat |
| `GET` | `/api/training-status` | Eğitim ilerlemesi ve metrikler |

### Örnek Yanıt — `/api/earthquakes`

```json
{
  "earthquakes": [
    {
      "event_id": "us7000abcd",
      "time": "2026-03-15T10:22:00+00:00",
      "latitude": 38.5,
      "longitude": 29.1,
      "depth_km": 12.4,
      "magnitude": 4.2,
      "place": "Denizli, Türkiye",
      "source": "AFAD",
      "prediction": 1,
      "probability": 87.3
    }
  ],
  "total": 142,
  "significant": 8,
  "max_magnitude": 5.1,
  "source": "AFAD"
}
```

---

## 📊 Dashboard Kullanımı

1. **Verileri Yenile** → AFAD/USGS'den anlık veri çek
2. **Modeli Eğit** → 300 epoch sinir ağı eğitimini başlat; ilerleme çubuğu ve metrikler canlı güncellenir
3. **Harita** → Deprem noktalarına tıklayarak detay popup'ı aç
4. **Otomatik Yenile** → Aktifken dashboard 60 saniyede bir güncellenir

### Renk Kodları

| Renk | Büyüklük | Anlam |
|---|---|---|
| 🟢 Yeşil | M < 2.5 | Düşük |
| 🟡 Sarı | M 2.5–3.9 | Orta |
| 🔴 Kırmızı | M ≥ 4.0 | Önemli |

---

## 🔧 Teknoloji Yığını

**Backend**
- Python 3.11
- PyTorch (CUDA / CPU)
- FastAPI + Uvicorn
- Scikit-learn, Pandas, NumPy
- Requests (AFAD & USGS API)

**Frontend**
- Saf HTML5 / CSS3 / JavaScript (framework yok)
- [Leaflet.js](https://leafletjs.com/) — interaktif harita
- [Chart.js](https://www.chartjs.org/) — eğitim grafikleri
- OpenStreetMap tiles

---

## 📄 Lisans

MIT © 2026 — Deprem AI
