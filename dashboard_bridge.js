(function () {
  const statusBadges = Array.from(document.querySelectorAll(".status-badge"));
  const setText = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  };
  const setWidth = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.style.width = `${Math.max(0, Math.min(100, value))}%`;
  };

  function markLive(isLive) {
    if (statusBadges[0]) statusBadges[0].innerHTML = `<div class="dot${isLive ? "" : " warn"}"></div> ${isLive ? "CAMERA LIVE" : "WAITING FOR BACKEND"}`;
    if (statusBadges[1]) statusBadges[1].innerHTML = `<div class="dot${isLive ? "" : " warn"}"></div> ${isLive ? "DETECTION LIVE" : "WS RECONNECTING"}`;
  }

  function renderDetections(detections) {
    const list = document.getElementById("det-list");
    if (!list) return;
    list.innerHTML = "";
    if (!detections.length) {
      const empty = document.createElement("div");
      empty.className = "det-item";
      empty.textContent = "No objects detected";
      list.appendChild(empty);
      return;
    }

    detections.slice(0, 8).forEach((det) => {
      const item = document.createElement("div");
      item.className = "det-item";
      const distance = Number(det.distance_m || 0).toFixed(1);
      item.innerHTML = `
        <div class="det-name">${det.label || "object"}</div>
        <div class="det-meta">${distance}m · ${det.direction || "FRONT"} · ${Math.round((det.confidence || 0) * 100)}%</div>
      `;
      list.appendChild(item);
    });
  }

  function render(payload) {
    const fps = Number(payload.fps || 0);
    setText("fps-display", `FPS: ${fps.toFixed(1)}`);
    setText("det-fps", `${fps.toFixed(1)} FPS`);
    setText("pipe-cap", `${fps.toFixed(0)}fps`);
    setText("pipe-yolo", `${fps.toFixed(0)}fps`);
    setText("pipe-midas", `${Math.max(0, fps - 3).toFixed(0)}fps`);
    setText("stat-inf", `${fps > 0 ? Math.round(1000 / fps) : "--"}ms`);
    setText("stat-alerts", String(payload.alert_count || 0));
    setText("det-count", String(payload.det_count || (payload.detections || []).length));

    const vramGb = Number(payload.vram_gb);
    const gpuPct = Number(payload.gpu_pct);
    const tempC = Number(payload.temp_c);
    if (!Number.isNaN(vramGb)) {
      setText("vram-val", `${vramGb.toFixed(1)}G`);
      setWidth("vram-bar", (vramGb / 6) * 100);
    }
    if (!Number.isNaN(gpuPct)) {
      setText("gpu-val", `${gpuPct.toFixed(0)}%`);
      setWidth("gpu-bar", gpuPct);
    }
    if (!Number.isNaN(tempC)) {
      setText("temp-val", `${tempC.toFixed(0)}C`);
      setWidth("temp-bar", tempC);
    }

    const modelSuffix = payload.assistant_model ? ` (${payload.assistant_model})` : "";
    const detections = payload.detections || [];
    let fallbackLine = "Path clear. No objects detected ahead.";
    if (detections.length) {
      const nearest = detections[0];
      const distance = Number(nearest.distance_m || 0).toFixed(1);
      fallbackLine = nearest.direction === "FRONT"
        ? `${nearest.label || "Object"} ahead, ${distance} metres.`
        : `Nearest object: ${nearest.label || "object"} ${(nearest.direction || "").toLowerCase()}, ${distance} metres.`;
    }
    setText("voice-text", `${payload.voice_line || fallbackLine}${modelSuffix}`);
    renderDetections(payload.detections || []);
  }

  function connect() {
    const socket = new WebSocket("ws://localhost:8765");
    socket.addEventListener("open", () => markLive(true));
    socket.addEventListener("message", (event) => render(JSON.parse(event.data)));
    socket.addEventListener("close", () => {
      markLive(false);
      setTimeout(connect, 1500);
    });
    socket.addEventListener("error", () => socket.close());
  }

  markLive(false);
  connect();
})();
