"""
Deprem AI — FastAPI Backend

Çalıştırma:
    pip install fastapi uvicorn httpx
    python api.py
Ardından tarayıcıda http://localhost:8000 adresini açın.
"""

from __future__ import annotations

import asyncio
import math
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sklearn.model_selection import train_test_split

# ── DEVICE ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── SABITLER ──────────────────────────────────────────────────────────────────
USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
EPOCHS   = 300
LR       = 0.001
STATIC   = Path(__file__).parent / "static"


def _afad_url() -> str:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    return (
        "https://deprem.afad.gov.tr/apiv2/event/filter?"
        f"start={start.strftime('%Y-%m-%dT%H:%M:%S')}"
        f"&end={end.strftime('%Y-%m-%dT%H:%M:%S')}&format=json"
    )


# ── GLOBAL STATE ───────────────────────────────────────────────────────────────
_lock = threading.Lock()

training_state: dict = {
    "status":       "idle",   # idle | running | done | error
    "progress":     0,
    "message":      "Henüz eğitim başlatılmadı.",
    "train_losses": [],
    "test_losses":  [],
    "train_accs":   [],
    "test_accs":    [],
}

model_registry: dict = {
    "model":  None,
    "ref_df": None,
}


# ── MODEL ──────────────────────────────────────────────────────────────────────
class DepremModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10), nn.ReLU(),
            nn.Linear(10, 10), nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── VERİ FONKSİYONLARI ────────────────────────────────────────────────────────
def _safe_float(val) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _normalize_afad(data) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = (
            data.get("items")
            or data.get("result")
            or data.get("data")
            or [data]
        )
    else:
        items = []

    records = []
    for item in items:
        eid   = (item.get("eventID") or item.get("eventId")
                 or item.get("id") or f"afad_{len(records)}")
        lat   = _safe_float(item.get("latitude") or item.get("lat") or item.get("enlem"))
        lon   = _safe_float(item.get("longitude") or item.get("lon") or item.get("boylam"))
        depth = _safe_float(item.get("depth") or item.get("depthkm") or item.get("derinlik"))
        mag   = _safe_float(item.get("magnitude") or item.get("ml")
                            or item.get("mag") or item.get("m"))
        place = (item.get("location") or item.get("place")
                 or item.get("title") or "Bilinmiyor")
        t     = str(item.get("date") or item.get("time") or item.get("eventDate") or "")
        records.append({
            "event_id": str(eid), "time": t,
            "latitude": lat,  "longitude": lon,
            "depth_km": depth, "magnitude": mag,
            "place": str(place), "source": "AFAD",
        })
    return pd.DataFrame(records)


def _normalize_usgs(data) -> pd.DataFrame:
    if not data or "features" not in data:
        return pd.DataFrame()

    records = []
    for feat in data["features"]:
        props  = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [])
        lon    = _safe_float(coords[0] if len(coords) > 0 else None)
        lat    = _safe_float(coords[1] if len(coords) > 1 else None)
        depth  = _safe_float(coords[2] if len(coords) > 2 else None)
        eid    = feat.get("id", f"usgs_{len(records)}")
        mag    = _safe_float(props.get("mag"))
        place  = props.get("place") or "Bilinmiyor"
        t_ms   = props.get("time")
        t      = str(pd.to_datetime(t_ms, unit="ms", utc=True)) if t_ms else ""
        records.append({
            "event_id": str(eid), "time": t,
            "latitude": lat,  "longitude": lon,
            "depth_km": depth, "magnitude": mag,
            "place": str(place), "source": "USGS",
        })
    return pd.DataFrame(records)


def _fetch_data_sync() -> pd.DataFrame:
    """AFAD verisini dene, başarısızsa USGS'e geç."""
    try:
        r = requests.get(_afad_url(), timeout=20)
        r.raise_for_status()
        df = _normalize_afad(r.json())
        if not df.empty:
            return df
    except Exception:
        pass

    try:
        r = requests.get(USGS_URL, timeout=20)
        r.raise_for_status()
        df = _normalize_usgs(r.json())
        if not df.empty:
            return df
    except Exception:
        pass

    return pd.DataFrame()


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["underground_wave_energy"] = (
        10 ** (df["magnitude"].fillna(0) / 2.0)
    ) / (df["depth_km"].fillna(1) + 1)
    lat0 = df["latitude"].fillna(df["latitude"].mean())
    lon0 = df["longitude"].fillna(df["longitude"].mean())
    df["vibration_axis_variation"] = np.sqrt(
        (lat0 - lat0.mean()) ** 2 + (lon0 - lon0.mean()) ** 2
    )
    df["seismic_event_detected"] = (df["magnitude"].fillna(0) >= 4.0).astype(float)
    return df


def _make_input(row: pd.Series, ref_df: pd.DataFrame) -> torch.Tensor:
    mag   = float(row["magnitude"]) if pd.notna(row["magnitude"]) else 0.0
    depth = float(row["depth_km"])  if pd.notna(row["depth_km"])  else 0.0
    uwe   = (10 ** (mag / 2.0)) / (depth + 1)
    lat   = float(row["latitude"])  if pd.notna(row["latitude"])  else float(ref_df["latitude"].mean())
    lon   = float(row["longitude"]) if pd.notna(row["longitude"]) else float(ref_df["longitude"].mean())
    vav   = math.sqrt(
        (lat - float(ref_df["latitude"].mean())) ** 2
        + (lon - float(ref_df["longitude"].mean())) ** 2
    )
    return torch.tensor([[uwe, vav]], dtype=torch.float32).to(device)


# ── EĞİTİM İŞ PARÇACIĞI ───────────────────────────────────────────────────────
def _train_worker() -> None:
    global training_state, model_registry

    with _lock:
        training_state.update({
            "status": "running", "progress": 0,
            "message": "Veri çekiliyor…",
            "train_losses": [], "test_losses": [],
            "train_accs":   [], "test_accs":   [],
        })

    try:
        df = _fetch_data_sync()
        if df.empty:
            with _lock:
                training_state.update({"status": "error", "message": "Veri alınamadı."})
            return

        df_feat = _build_features(df).dropna(
            subset=["underground_wave_energy", "vibration_axis_variation", "seismic_event_detected"]
        )
        if len(df_feat) < 10:
            with _lock:
                training_state.update({"status": "error", "message": "Eğitim için yeterli veri yok (en az 10 satır gerekli)."})
            return

        X = df_feat[["underground_wave_energy", "vibration_axis_variation"]].values
        y = df_feat["seismic_event_detected"].values

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        def to_t(arr):
            return torch.tensor(arr, dtype=torch.float32).to(device)

        X_tr, X_te = to_t(X_tr), to_t(X_te)
        y_tr = to_t(y_tr).unsqueeze(1)
        y_te = to_t(y_te).unsqueeze(1)

        torch.manual_seed(42)
        model   = DepremModel().to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        opt     = torch.optim.Adam(model.parameters(), lr=LR)

        tl, tel, ta, tea = [], [], [], []

        with _lock:
            training_state["message"] = "Model eğitiliyor…"

        for epoch in range(EPOCHS):
            model.train()
            logits = model(X_tr)
            loss   = loss_fn(logits, y_tr)
            preds  = torch.round(torch.sigmoid(logits))
            acc    = (torch.eq(y_tr, preds).sum().item() / len(y_tr)) * 100
            opt.zero_grad()
            loss.backward()
            opt.step()

            model.eval()
            with torch.inference_mode():
                te_logits = model(X_te)
                te_loss   = loss_fn(te_logits, y_te)
                te_preds  = torch.round(torch.sigmoid(te_logits))
                te_acc    = (torch.eq(y_te, te_preds).sum().item() / len(y_te)) * 100

            tl.append(round(loss.item(), 6))
            tel.append(round(te_loss.item(), 6))
            ta.append(round(acc, 2))
            tea.append(round(te_acc, 2))

            # Her 10 epoch'ta güncelle (performans için)
            if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
                with _lock:
                    training_state["progress"]     = int((epoch + 1) / EPOCHS * 100)
                    training_state["train_losses"] = tl[:]
                    training_state["test_losses"]  = tel[:]
                    training_state["train_accs"]   = ta[:]
                    training_state["test_accs"]    = tea[:]

        # Model ağırlıklarını kaydet
        weights_path = Path(__file__).parent.parent.parent / "model_weights.pth"
        torch.save(model.state_dict(), str(weights_path))

        with _lock:
            model_registry["model"]  = model
            model_registry["ref_df"] = _build_features(df)
            training_state.update({
                "status":   "done",
                "progress": 100,
                "message":  (
                    f"✅ Eğitim tamamlandı! "
                    f"Eğitim Doğruluğu: {ta[-1]:.1f}% | Test Doğruluğu: {tea[-1]:.1f}%"
                ),
            })

    except Exception as exc:
        with _lock:
            training_state.update({"status": "error", "message": f"Hata: {exc}"})


# ── FASTAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Deprem AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(STATIC / "index.html"))


@app.get("/api/status")
async def api_status():
    with _lock:
        ts = dict(training_state)
    return JSONResponse({
        "device":        str(device),
        "cuda_available": torch.cuda.is_available(),
        "model_trained": model_registry["model"] is not None,
        "training":      ts,
    })


@app.get("/api/earthquakes")
async def api_earthquakes():
    loop = asyncio.get_event_loop()
    df   = await loop.run_in_executor(None, _fetch_data_sync)

    if df.empty:
        return JSONResponse({
            "earthquakes": [], "total": 0,
            "significant": 0, "max_magnitude": 0,
            "source": "none", "error": "Veri alınamadı.",
        })

    with _lock:
        model  = model_registry["model"]
        ref_df = model_registry["ref_df"]

    rows = []
    for _, row in df.iterrows():
        mag = float(row["magnitude"]) if pd.notna(row["magnitude"]) else None
        eq  = {
            "event_id":   row["event_id"],
            "time":       str(row["time"]) if row["time"] else "",
            "latitude":   float(row["latitude"])  if pd.notna(row["latitude"])  else None,
            "longitude":  float(row["longitude"]) if pd.notna(row["longitude"]) else None,
            "depth_km":   float(row["depth_km"])  if pd.notna(row["depth_km"])  else None,
            "magnitude":  mag,
            "place":      str(row["place"]),
            "source":     row["source"],
            "prediction": None,
            "probability": None,
        }

        if model is not None and ref_df is not None:
            try:
                x = _make_input(row, ref_df)
                model.eval()
                with torch.inference_mode():
                    prob = torch.sigmoid(model(x)).item()
                eq["prediction"]  = int(round(prob))
                eq["probability"] = round(prob * 100, 1)
            except Exception:
                pass

        rows.append(eq)

    mags = [r["magnitude"] for r in rows if r["magnitude"] is not None]
    return JSONResponse({
        "earthquakes":   rows,
        "total":         len(rows),
        "significant":   sum(1 for r in rows if r["magnitude"] and r["magnitude"] >= 4.0),
        "max_magnitude": round(max(mags), 1) if mags else 0,
        "source":        str(df["source"].iloc[0]),
    })


@app.post("/api/train")
async def api_train():
    with _lock:
        status = training_state["status"]
    if status == "running":
        return JSONResponse({"success": False, "message": "Eğitim zaten devam ediyor."})
    threading.Thread(target=_train_worker, daemon=True).start()
    return JSONResponse({"success": True, "message": "Eğitim başlatıldı."})


@app.get("/api/training-status")
async def api_training_status():
    with _lock:
        return JSONResponse(dict(training_state))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
