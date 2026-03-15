/* ============================================================
   Deprem AI — Frontend JavaScript
   ============================================================ */

"use strict";

// ── Yapılandırma ──────────────────────────────────────────────
const API_BASE      = "";        // same-origin
const REFRESH_MS    = 60_000;    // 60 saniye

// ── Global durum ───────────────────────────────────────────────
let map         = null;
let markers     = [];
let lossChart   = null;
let accChart    = null;
let autoTimer   = null;
let trainingPoller = null;

// ── Harita başlatma ────────────────────────────────────────────
function initMap() {
  map = L.map("map", {
    center:           [39.0, 35.0],
    zoom:             5,
    zoomControl:      true,
    attributionControl: true,
  });

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    maxZoom:     18,
  }).addTo(map);
}

// ── Magnitude → renk / sınıf ───────────────────────────────────
function magColor(mag) {
  if (mag === null || mag === undefined) return "#64748b";
  if (mag < 2.5) return "#10b981";
  if (mag < 4.0) return "#f59e0b";
  return "#ef4444";
}

function magClass(mag) {
  if (mag === null || mag === undefined) return "mag-badge";
  if (mag < 2.5) return "mag-badge mag-low";
  if (mag < 4.0) return "mag-badge mag-mid";
  return "mag-badge mag-high";
}

function magRadius(mag) {
  if (!mag) return 6;
  return Math.min(5 + mag * 3, 22);
}

// ── Haritayı güncelle ──────────────────────────────────────────
function updateMap(earthquakes) {
  markers.forEach(m => map.removeLayer(m));
  markers = [];

  earthquakes.forEach(eq => {
    if (eq.latitude === null || eq.longitude === null) return;

    const color  = magColor(eq.magnitude);
    const radius = magRadius(eq.magnitude);

    const circle = L.circleMarker([eq.latitude, eq.longitude], {
      radius:      radius,
      color:       color,
      fillColor:   color,
      fillOpacity: 0.75,
      weight:      1.5,
      opacity:     0.95,
    });

    const mag    = eq.magnitude !== null ? eq.magnitude.toFixed(1) : "?";
    const depth  = eq.depth_km !== null ? eq.depth_km.toFixed(1) + " km" : "?";
    const aiLine = eq.prediction !== null
      ? `<div style="margin-top:6px;padding-top:6px;border-top:1px solid #334155">
           <b>AI:</b> ${eq.prediction === 1
             ? '<span style="color:#ef4444">⚠️ Önemli Olay</span>'
             : '<span style="color:#10b981">✅ Normal</span>'}
           &nbsp; (Güven: <b>${eq.probability}%</b>)
         </div>`
      : "";

    circle.bindPopup(`
      <div style="min-width:200px">
        <div style="font-weight:700;font-size:.95rem;margin-bottom:6px">${eq.place}</div>
        <div><b>Büyüklük:</b> <span style="color:${color};font-weight:700">M ${mag}</span></div>
        <div><b>Derinlik:</b> ${depth}</div>
        <div><b>Zaman:</b> ${formatTime(eq.time)}</div>
        <div><b>Kaynak:</b> ${eq.source}</div>
        ${aiLine}
      </div>
    `);

    circle.addTo(map);
    markers.push(circle);
  });

  document.getElementById("map-hint").textContent =
    `${earthquakes.length} olay gösteriliyor`;
}

// ── Tabloyu güncelle ───────────────────────────────────────────
function updateTable(earthquakes) {
  const tbody = document.getElementById("eq-tbody");

  if (!earthquakes.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty-row">Gösterilecek veri yok.</td></tr>';
    return;
  }

  // Büyüklüğe göre sırala (azalan)
  const sorted = [...earthquakes].sort(
    (a, b) => (b.magnitude ?? 0) - (a.magnitude ?? 0)
  );

  tbody.innerHTML = sorted.map(eq => {
    const mag      = eq.magnitude !== null ? eq.magnitude.toFixed(1) : "—";
    const depth    = eq.depth_km  !== null ? eq.depth_km.toFixed(1)  : "—";
    const srcClass = eq.source === "AFAD" ? "src-afad" : "src-usgs";

    let aiCell = '<span style="color:#64748b">—</span>';
    let probCell = '<span style="color:#64748b">—</span>';
    if (eq.prediction !== null) {
      aiCell = eq.prediction === 1
        ? '<span class="ai-tag sig">⚠️ Önemli</span>'
        : '<span class="ai-tag low">✅ Normal</span>';
      probCell = `<span style="color:${eq.probability > 60 ? '#ef4444' : '#10b981'};font-weight:700">${eq.probability}%</span>`;
    }

    return `
      <tr>
        <td><span class="${magClass(eq.magnitude)}">${mag}</span></td>
        <td style="max-width:220px;overflow:hidden;text-overflow:ellipsis" title="${eq.place}">${eq.place}</td>
        <td style="color:#94a3b8">${formatTime(eq.time)}</td>
        <td style="color:#94a3b8">${depth}</td>
        <td><span class="src-badge ${srcClass}">${eq.source}</span></td>
        <td>${aiCell}</td>
        <td>${probCell}</td>
      </tr>
    `;
  }).join("");

  document.getElementById("table-hint").textContent =
    `${sorted.length} kayıt — büyüklüğe göre sıralandı`;
}

// ── İstatistik kartlarını güncelle ─────────────────────────────
function updateStats({ total, significant, max_magnitude }) {
  document.getElementById("stat-total").textContent       = total ?? "—";
  document.getElementById("stat-significant").textContent = significant ?? "—";
  document.getElementById("stat-max-mag").textContent     =
    max_magnitude ? `M ${max_magnitude}` : "—";
}

// ── Zaman formatı ──────────────────────────────────────────────
function formatTime(t) {
  if (!t) return "—";
  try {
    return new Date(t).toLocaleString("tr-TR", {
      day: "2-digit", month: "2-digit", year: "numeric",
      hour: "2-digit", minute: "2-digit",
    });
  } catch {
    return t.slice(0, 16);
  }
}

// ── Son güncelleme saatini ayarla ──────────────────────────────
function setLastUpdate() {
  document.getElementById("last-update").textContent =
    "Son güncelleme: " + new Date().toLocaleTimeString("tr-TR");
}

// ── Kaynak rozeti ──────────────────────────────────────────────
function setSource(source) {
  const el   = document.getElementById("source-badge");
  const icon = source === "AFAD" ? "🇹🇷" : (source === "USGS" ? "🌎" : "—");
  el.textContent = `Kaynak: ${icon} ${source}`;
}

// ── Deprem verilerini yükle ────────────────────────────────────
async function loadEarthquakes() {
  const btn = document.getElementById("btn-refresh");
  btn.disabled = true;
  btn.textContent = "Yükleniyor…";

  try {
    const res  = await fetch(`${API_BASE}/api/earthquakes`);
    const data = await res.json();

    updateMap(data.earthquakes || []);
    updateTable(data.earthquakes || []);
    updateStats(data);
    setSource(data.source);
    setLastUpdate();

    if (data.error) {
      showToast("⚠️ " + data.error, "error");
    }
  } catch (err) {
    showToast("Bağlantı hatası: " + err.message, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
        <path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/>
        <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/>
      </svg>
      Verileri Yenile
    `;
  }
}

// ── Durum endpoint'ini sorgula ─────────────────────────────────
async function loadStatus() {
  try {
    const res  = await fetch(`${API_BASE}/api/status`);
    const data = await res.json();

    const deviceEl = document.getElementById("device-badge");
    const cuda     = data.cuda_available;
    deviceEl.textContent = `⚡ ${cuda ? "GPU (CUDA)" : "CPU"}`;
    deviceEl.style.background = cuda ? "#16a34a" : "#2563eb";

    const modelEl = document.getElementById("stat-model");
    modelEl.textContent = data.model_trained ? "✅ Eğitildi" : "⏸️ Eğitilmedi";
  } catch {
    // sessizce geç
  }
}

// ── Eğitimi başlat ─────────────────────────────────────────────
async function startTraining() {
  const btn = document.getElementById("btn-train");
  btn.disabled = true;

  try {
    const res  = await fetch(`${API_BASE}/api/train`, { method: "POST" });
    const data = await res.json();

    if (data.success) {
      showToast("🧠 Eğitim başlatıldı!", "success");
      startTrainingPoller();
    } else {
      showToast("⚠️ " + data.message, "error");
      btn.disabled = false;
    }
  } catch (err) {
    showToast("Eğitim başlatılamadı: " + err.message, "error");
    btn.disabled = false;
  }
}

// ── Eğitim durumunu periyodik sorgula ─────────────────────────
function startTrainingPoller() {
  if (trainingPoller) clearInterval(trainingPoller);
  trainingPoller = setInterval(pollTrainingStatus, 2_000);
  pollTrainingStatus();
}

async function pollTrainingStatus() {
  try {
    const res  = await fetch(`${API_BASE}/api/training-status`);
    const data = await res.json();
    renderTrainingPanel(data);

    if (data.status === "done" || data.status === "error") {
      clearInterval(trainingPoller);
      trainingPoller = null;
      document.getElementById("btn-train").disabled = false;

      if (data.status === "done") {
        showToast("✅ Model eğitimi tamamlandı!", "success");
        await loadEarthquakes();   // tahminleri güncelle
        await loadStatus();
      } else {
        showToast("❌ Eğitim hatası: " + data.message, "error");
      }
    }
  } catch {
    // sessizce geç
  }
}

// ── Eğitim panelini güncelle ───────────────────────────────────
function renderTrainingPanel(data) {
  const iconEl    = document.getElementById("training-icon");
  const statusEl  = document.getElementById("training-status-text");
  const msgEl     = document.getElementById("training-message");
  const wrapEl    = document.getElementById("progress-wrap");
  const fillEl    = document.getElementById("progress-bar-fill");
  const labelEl   = document.getElementById("progress-label");
  const finalEl   = document.getElementById("training-final");

  const icons = { idle: "⏸️", running: "⚙️", done: "✅", error: "❌" };
  const texts = { idle: "Bekliyor", running: "Eğitiliyor…", done: "Tamamlandı", error: "Hata" };

  iconEl.textContent   = icons[data.status]  ?? "❓";
  statusEl.textContent = texts[data.status]  ?? data.status;
  msgEl.textContent    = data.message        ?? "";

  if (data.status === "running") {
    wrapEl.style.display  = "block";
    fillEl.style.width    = data.progress + "%";
    labelEl.textContent   = data.progress + "%";
  } else {
    wrapEl.style.display  = "none";
  }

  // Son metrikler
  if (data.status === "done" && data.train_accs?.length) {
    finalEl.style.display = "flex";
    const last = data.train_accs.length - 1;
    document.getElementById("final-train-acc").textContent =
      data.train_accs[last].toFixed(1) + "%";
    document.getElementById("final-test-acc").textContent  =
      data.test_accs[last].toFixed(1)  + "%";
    renderCharts(data);
  } else {
    finalEl.style.display = "none";
  }
}

// ── Chart.js grafikleri ────────────────────────────────────────
function renderCharts(data) {
  const metricsCard = document.getElementById("metrics-card");
  metricsCard.style.display = "block";

  // Sadece her 5. epoch'u göster (300 epoch çok yoğun)
  const step   = 5;
  const labels = data.train_losses
    .map((_, i) => i + 1)
    .filter((_, i) => i % step === 0 || i === data.train_losses.length - 1);

  const thin = arr =>
    arr.filter((_, i) => i % step === 0 || i === arr.length - 1);

  const chartDefaults = {
    responsive:          true,
    maintainAspectRatio: true,
    animation:           { duration: 400 },
    plugins: {
      legend: {
        labels: { color: "#94a3b8", font: { size: 11 } },
      },
    },
    scales: {
      x: {
        ticks:  { color: "#64748b", font: { size: 10 } },
        grid:   { color: "rgba(51,65,85,.5)" },
        title:  { display: true, text: "Epoch", color: "#64748b" },
      },
      y: {
        ticks:  { color: "#64748b", font: { size: 10 } },
        grid:   { color: "rgba(51,65,85,.5)" },
      },
    },
  };

  // -- Kayıp grafiği
  const lossCtx = document.getElementById("loss-chart");
  if (lossChart) lossChart.destroy();
  lossChart = new Chart(lossCtx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label:          "Eğitim Kaybı",
          data:           thin(data.train_losses),
          borderColor:    "#3b82f6",
          backgroundColor:"rgba(59,130,246,.1)",
          tension:        0.35,
          pointRadius:    0,
          borderWidth:    2,
        },
        {
          label:          "Test Kaybı",
          data:           thin(data.test_losses),
          borderColor:    "#f59e0b",
          backgroundColor:"rgba(245,158,11,.1)",
          tension:        0.35,
          pointRadius:    0,
          borderWidth:    2,
          borderDash:     [5, 3],
        },
      ],
    },
    options: { ...chartDefaults },
  });

  // -- Doğruluk grafiği
  const accCtx = document.getElementById("acc-chart");
  if (accChart) accChart.destroy();
  accChart = new Chart(accCtx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label:          "Eğitim Doğruluğu",
          data:           thin(data.train_accs),
          borderColor:    "#10b981",
          backgroundColor:"rgba(16,185,129,.1)",
          tension:        0.35,
          pointRadius:    0,
          borderWidth:    2,
        },
        {
          label:          "Test Doğruluğu",
          data:           thin(data.test_accs),
          borderColor:    "#a78bfa",
          backgroundColor:"rgba(167,139,250,.1)",
          tension:        0.35,
          pointRadius:    0,
          borderWidth:    2,
          borderDash:     [5, 3],
        },
      ],
    },
    options: {
      ...chartDefaults,
      scales: {
        ...chartDefaults.scales,
        y: {
          ...chartDefaults.scales.y,
          min:   0,
          max:   100,
          title: { display: true, text: "%", color: "#64748b" },
        },
      },
    },
  });
}

// ── Toast bildirimi ────────────────────────────────────────────
function showToast(msg, type = "info") {
  let wrap = document.querySelector(".toast-wrap");
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.className = "toast-wrap";
    document.body.appendChild(wrap);
  }

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  wrap.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity   = "0";
    toast.style.transform = "translateX(10px)";
    toast.style.transition= "all .3s";
    setTimeout(() => toast.remove(), 300);
  }, 4_000);
}

// ── Otomatik yenile ────────────────────────────────────────────
function setupAutoRefresh() {
  const checkbox = document.getElementById("auto-refresh");

  function schedule() {
    if (autoTimer) clearInterval(autoTimer);
    if (checkbox.checked) {
      autoTimer = setInterval(loadEarthquakes, REFRESH_MS);
    }
  }

  checkbox.addEventListener("change", schedule);
  schedule();
}

// ── Düğme olay dinleyicileri ───────────────────────────────────
function setupButtons() {
  document.getElementById("btn-refresh").addEventListener("click", loadEarthquakes);
  document.getElementById("btn-train").addEventListener("click", startTraining);
}

// ── Uygulama başlatma ──────────────────────────────────────────
async function init() {
  initMap();
  setupButtons();
  setupAutoRefresh();

  // İlk yüklemeler
  await Promise.all([loadStatus(), loadEarthquakes()]);

  // Eğer sunucu zaten eğitiyorsa devam eden durumu göster
  const res  = await fetch(`${API_BASE}/api/training-status`);
  const data = await res.json();
  if (data.status === "running") {
    document.getElementById("btn-train").disabled = true;
    startTrainingPoller();
  }
  if (data.status === "done") {
    renderTrainingPanel(data);
  }
}

document.addEventListener("DOMContentLoaded", init);
