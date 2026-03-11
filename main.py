import time
import math
import requests
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn

# ======================================================
# 1) CİHAZ
# ======================================================
device = torch.device("cuda")
print("Kullanılan cihaz:", device)

# ======================================================
# 2) AYARLAR
# ======================================================
AFAD_URL = (
    "https://deprem.afad.gov.tr/apiv2/event/filter?"
    "start=2026-03-10T00:00:00&end=2026-03-11T23:59:59&format=json"
)

USGS_URL = (
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
)

POLL_INTERVAL = 60   # saniye
EPOCHS = 300
LR = 0.001

# ======================================================
# 3) VERİ ÇEKME FONKSİYONLARI
# ======================================================
def fetch_afad():
    """
    AFAD'dan deprem verisi çekmeye çalışır.
    JSON beklenir.
    """
    try:
        r = requests.get(AFAD_URL, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data, "AFAD"
    except Exception as e:
        print("AFAD veri çekme hatası:", e)
        return None, "AFAD"


def fetch_usgs():
    """
    USGS GeoJSON feed'ini çeker.
    """
    try:
        r = requests.get(USGS_URL, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data, "USGS"
    except Exception as e:
        print("USGS veri çekme hatası:", e)
        return None, "USGS"


# ======================================================
# 4) AFAD / USGS VERİSİNİ ORTAK FORMATA ÇEVİRME
# ======================================================
def normalize_afad(data):
    """
    AFAD verisini ortak tabloya çevirir.
    AFAD'daki alan adları zaman zaman farklı olabilir;
    bu yüzden birkaç muhtemel alan adı kontrol edilir.
    """
    records = []

    if not data:
        return pd.DataFrame()

    if isinstance(data, dict):
        # AFAD bazen listeyi dict içinde verebilir
        if "items" in data and isinstance(data["items"], list):
            items = data["items"]
        elif "result" in data and isinstance(data["result"], list):
            items = data["result"]
        elif "data" in data and isinstance(data["data"], list):
            items = data["data"]
        else:
            # doğrudan dict içindeki tekil kayıt senaryosu
            items = [data]
    elif isinstance(data, list):
        items = data
    else:
        items = []

    for item in items:
        event_id = (
            item.get("eventID")
            or item.get("eventId")
            or item.get("id")
            or item.get("earthquake_id")
            or f"afad_{len(records)}"
        )

        lat = item.get("latitude") or item.get("lat") or item.get("enlem")
        lon = item.get("longitude") or item.get("lon") or item.get("boylam")
        depth = item.get("depth") or item.get("depthkm") or item.get("derinlik")
        mag = (
            item.get("magnitude")
            or item.get("ml")
            or item.get("mag")
            or item.get("m")
        )
        place = item.get("location") or item.get("place") or item.get("title") or "Bilinmiyor"
        event_time = item.get("date") or item.get("time") or item.get("eventDate")

        try:
            lat = float(lat) if lat is not None else np.nan
            lon = float(lon) if lon is not None else np.nan
            depth = float(depth) if depth is not None else np.nan
            mag = float(mag) if mag is not None else np.nan
        except:
            continue

        records.append({
            "event_id": str(event_id),
            "time": event_time,
            "latitude": lat,
            "longitude": lon,
            "depth_km": depth,
            "magnitude": mag,
            "place": place,
            "source": "AFAD"
        })

    return pd.DataFrame(records)


def normalize_usgs(data):
    """
    USGS GeoJSON verisini ortak tabloya çevirir.
    """
    records = []

    if not data or "features" not in data:
        return pd.DataFrame()

    for feat in data["features"]:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [np.nan, np.nan, np.nan])

        lon = coords[0] if len(coords) > 0 else np.nan
        lat = coords[1] if len(coords) > 1 else np.nan
        depth = coords[2] if len(coords) > 2 else np.nan

        event_id = feat.get("id", f"usgs_{len(records)}")
        mag = props.get("mag", np.nan)
        place = props.get("place", "Bilinmiyor")
        event_time_ms = props.get("time", None)

        if event_time_ms is not None:
            event_time = pd.to_datetime(event_time_ms, unit="ms", utc=True)
        else:
            event_time = None

        records.append({
            "event_id": str(event_id),
            "time": event_time,
            "latitude": float(lat) if lat is not None else np.nan,
            "longitude": float(lon) if lon is not None else np.nan,
            "depth_km": float(depth) if depth is not None else np.nan,
            "magnitude": float(mag) if mag is not None else np.nan,
            "place": place,
            "source": "USGS"
        })

    return pd.DataFrame(records)


def fetch_and_prepare():
    """
    Önce AFAD, olmazsa USGS.
    """
    data, source = fetch_afad()
    if data is not None:
        df = normalize_afad(data)
        if not df.empty:
            return df

    data, source = fetch_usgs()
    if data is not None:
        df = normalize_usgs(data)
        if not df.empty:
            return df

    return pd.DataFrame()


# ======================================================
# 5) FEATURE ENGINEERING
# ======================================================
def build_features(df):
    """
    Elindeki eski veri setindeki feature isimlerine benzer 2 adet türetilmiş giriş üretiyoruz.
    Bu kısım geçici demo amaçlıdır.
    Gerçek projede burada yeniden eğitim yapılmalı.
    """

    df = df.copy()

    # Feature-1: underground_wave_energy yerine geçici temsil
    # büyüklük ve derinlikten sentetik bir enerji benzeri ölçü
    df["underground_wave_energy"] = (10 ** (df["magnitude"].fillna(0) / 2.0)) / (df["depth_km"].fillna(1) + 1)

    # Feature-2: vibration_axis_variation yerine geçici temsil
    # enlem-boylam değişimi / konumsal yayılımın kaba bir türevi
    lat0 = df["latitude"].fillna(df["latitude"].mean())
    lon0 = df["longitude"].fillna(df["longitude"].mean())
    df["vibration_axis_variation"] = np.sqrt((lat0 - lat0.mean())**2 + (lon0 - lon0.mean())**2)

    # Hedef etiket:
    # Demo için magnitude >= 4.0 olanları 1 kabul ediyoruz
    # Gerçek projede bu etiketi bilimsel/uygulama amacına göre yeniden tanımlamalısın
    df["seismic_event_detected"] = (df["magnitude"].fillna(0) >= 4.0).astype(float)

    return df


# ======================================================
# 6) MODEL
# ======================================================
class ClassificationNonLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))


def calculate_acc(y_predic, y_true):
    correct = torch.eq(y_true, y_predic).sum().item()
    acc = (correct / len(y_predic)) * 100
    return acc


# ======================================================
# 7) EĞİTİM
# ======================================================
def train_model(df):
    df = build_features(df).dropna(subset=["underground_wave_energy", "vibration_axis_variation", "seismic_event_detected"])

    X = df[["underground_wave_energy", "vibration_axis_variation"]].values
    y = df["seismic_event_detected"].values

    if len(df) < 10:
        raise ValueError("Model eğitimi için yeterli veri yok.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test  = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_test  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    torch.manual_seed(42)
    model = ClassificationNonLinearModel().to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(EPOCHS):
        model.train()

        y_logits = model(X_train)
        y_pred = torch.round(torch.sigmoid(y_logits))

        loss = loss_fn(y_logits, y_train)
        acc = calculate_acc(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test)
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = calculate_acc(test_pred, y_test)

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        train_accs.append(acc)
        test_accs.append(test_acc)

        if epoch % 50 == 0:
            print(
                f"epoch={epoch} "
                f"train_loss={loss.item():.4f} train_acc={acc:.2f}% "
                f"test_loss={test_loss.item():.4f} test_acc={test_acc:.2f}%"
            )

    torch.save(model.state_dict(), "model_weights.pth")

    return model, train_losses, test_losses, train_accs, test_accs


# ======================================================
# 8) GRAFİK
# ======================================================
def plot_metrics(train_losses, test_losses, train_accs, test_accs):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.title("Loss Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(test_accs, label="Test Accuracy")
    plt.title("Accuracy Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ======================================================
# 9) CANLI TAHMİN
# ======================================================
def preprocess_live_row(row, ref_df):
    """
    Canlı gelen tek bir olayı model girişine dönüştürür.
    """
    mag = row["magnitude"] if pd.notna(row["magnitude"]) else 0.0
    depth = row["depth_km"] if pd.notna(row["depth_km"]) else 0.0

    # eğitimdeki feature engineering ile uyumlu olacak şekilde
    underground_wave_energy = (10 ** (mag / 2.0)) / (depth + 1)

    lat = row["latitude"] if pd.notna(row["latitude"]) else ref_df["latitude"].mean()
    lon = row["longitude"] if pd.notna(row["longitude"]) else ref_df["longitude"].mean()

    vibration_axis_variation = math.sqrt(
        (lat - ref_df["latitude"].mean()) ** 2 +
        (lon - ref_df["longitude"].mean()) ** 2
    )

    x = torch.tensor(
        [[underground_wave_energy, vibration_axis_variation]],
        dtype=torch.float32
    ).to(device)

    return x


def live_monitor(model, ref_df):
    seen_ids = set()

    print("\nCanlı izleme başladı...\n")

    while True:
        live_df = fetch_and_prepare()

        if live_df.empty:
            print("Canlı veri alınamadı, tekrar denenecek.")
            time.sleep(POLL_INTERVAL)
            continue

        for _, row in live_df.iterrows():
            event_id = row["event_id"]

            if event_id in seen_ids:
                continue

            seen_ids.add(event_id)

            x = preprocess_live_row(row, ref_df)

            model.eval()
            with torch.inference_mode():
                logits = model(x)
                prob = torch.sigmoid(logits).item()
                pred = round(prob)

            print("-" * 60)
            print("Kaynak      :", row["source"])
            print("Olay ID     :", row["event_id"])
            print("Zaman       :", row["time"])
            print("Konum       :", row["place"])
            print("Büyüklük    :", row["magnitude"])
            print("Derinlik km :", row["depth_km"])
            print("Tahmin      :", pred)
            print("Olasılık    :", f"{prob:.4f}")

        time.sleep(POLL_INTERVAL)


# ======================================================
# 10) ANA ÇALIŞMA
# ======================================================
if __name__ == "__main__":
    df = fetch_and_prepare()

    if df.empty:
        raise RuntimeError("AFAD/USGS üzerinden veri alınamadı.")

    print("İlk veri çekildi. Kayıt sayısı:", len(df))
    print(df.head())

    model, train_losses, test_losses, train_accs, test_accs = train_model(df)

    plot_metrics(train_losses, test_losses, train_accs, test_accs)

    live_monitor(model, build_features(df))