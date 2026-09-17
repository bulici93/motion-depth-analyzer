const MAX_UPLOAD_BYTES = 500 * 1024 * 1024;

const els = {
  fileInput: document.getElementById("fileInput"),
  uploadHint: document.getElementById("uploadHint"),
  sourceSection: document.getElementById("sourceSection"),
  sourceVideo: document.getElementById("sourceVideo"),
  metricName: document.getElementById("metricName"),
  metricRes: document.getElementById("metricRes"),
  metricFps: document.getElementById("metricFps"),
  metricDuration: document.getElementById("metricDuration"),
  metricFrames: document.getElementById("metricFrames"),
  scale: document.getElementById("scale"),
  stride: document.getElementById("stride"),
  gamma: document.getElementById("gamma"),
  crush: document.getElementById("crush"),
  smooth: document.getElementById("smooth"),
  audio: document.getElementById("audio"),
  actionBtn: document.getElementById("actionBtn"),
  progressWrap: document.getElementById("progressWrap"),
  progressBar: document.getElementById("progressBar"),
  status: document.getElementById("status"),
  previewSlot: document.getElementById("previewSlot"),
  previewImage: document.getElementById("previewImage"),
  resultSection: document.getElementById("resultSection"),
  resultSource: document.getElementById("resultSource"),
  resultDepth: document.getElementById("resultDepth"),
  downloadBtn: document.getElementById("downloadBtn"),
  dropzone: document.getElementById("dropzone"),
  fileChip: document.getElementById("fileChip"),
  fileChipName: document.getElementById("fileChipName"),
  fileChipSize: document.getElementById("fileChipSize"),
  runStatus: document.getElementById("runStatus"),
  stopBtn: document.getElementById("stopBtn"),
  menuBtn: document.getElementById("menuBtn"),
  menuPanel: document.getElementById("menuPanel"),
  langBtn: document.getElementById("langBtn"),
  clearCacheBtn: document.getElementById("clearCacheBtn"),
};

const state = {
  sourcePath: null,
  sourceUrl: null,
  runId: null,
  running: false,
  stopping: false,
  stream: null,
  frameCount: null,
  statusKind: null,
  progress: null,
  errorMessage: null,
};

function bindValue(input, label) {
  const paint = () => {
    label.textContent = input.step && Number(input.step) < 1 ? Number(input.value).toFixed(2).replace(/0+$/, "").replace(/\.$/, "") : input.value;
  };
  input.addEventListener("input", paint);
  paint();
}

bindValue(els.scale, document.getElementById("scaleVal"));
bindValue(els.stride, document.getElementById("strideVal"));
bindValue(els.gamma, document.getElementById("gammaVal"));
bindValue(els.crush, document.getElementById("crushVal"));
bindValue(els.smooth, document.getElementById("smoothVal"));

function mediaFileUrl(path) {
  return `/api/media/file?path=${encodeURIComponent(path)}`;
}

function mediaOutputUrl(jobId) {
  return `/api/media/output/${jobId}`;
}

function formatEta(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  const secs = String(total % 60).padStart(2, "0");
  return `${minutes}:${secs}`;
}

function labelSep() {
  return currentLang() === "en" ? ": " : "：";
}

function renderFrames() {
  if (state.frameCount == null) {
    els.metricFrames.textContent = "";
    return;
  }
  els.metricFrames.textContent = `${t("frames")}${labelSep()}${state.frameCount || "—"}`;
}

function renderStatus() {
  if (!state.statusKind) {
    els.status.textContent = "";
    return;
  }
  if (state.statusKind === "processing") {
    els.status.textContent = t("processing");
    return;
  }
  if (state.statusKind === "progress" && state.progress) {
    const data = state.progress;
    const total = data.total_frames || "?";
    const sep = labelSep();
    els.status.textContent =
      `${t("processing")}\n` +
      `${t("progress_frame")}${sep}${data.frame_index} / ${total}\n` +
      `${t("progress_pct")}${sep}${((data.progress || 0) * 100).toFixed(1)}%\n` +
      `${t("progress_fps")}${sep}${Number(data.fps).toFixed(1)}\n` +
      `${t("progress_eta")}${sep}${formatEta(data.eta_seconds)}`;
    return;
  }
  if (state.statusKind === "done") {
    els.status.textContent = t("done");
    return;
  }
  if (state.statusKind === "cancelled") {
    els.status.textContent = t("cancelled");
    return;
  }
  if (state.statusKind === "error") {
    els.status.textContent = state.errorMessage || "error";
  }
}

function setStatus(kind, extra) {
  state.statusKind = kind;
  state.progress = kind === "progress" ? extra || null : null;
  state.errorMessage = kind === "error" ? extra || "error" : null;
  renderStatus();
}

function setActionButton() {
  els.actionBtn.classList.toggle("stop", state.running);
  els.actionBtn.disabled = state.stopping || !state.sourcePath;
  els.actionBtn.textContent = state.running ? t("stop") : t("process");
  els.runStatus.classList.toggle("hidden", !state.running);
  els.stopBtn.disabled = state.stopping || !state.runId;
  els.clearCacheBtn.disabled = state.running;
}

function currentTheme() {
  const value = localStorage.getItem("dc-theme");
  return value === "light" || value === "dark" || value === "system" ? value : "system";
}

function applyTheme(theme) {
  const dark =
    theme === "dark" ||
    (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", dark);
  document.documentElement.dataset.theme = theme;
}

function syncMenuState() {
  const theme = currentTheme();
  document.querySelectorAll("[data-theme]").forEach((el) => {
    el.classList.toggle("is-active", el.getAttribute("data-theme") === theme);
  });
}

function closeMenu() {
  els.menuPanel.classList.add("hidden");
  els.menuBtn.setAttribute("aria-expanded", "false");
}

function toggleMenu() {
  const open = els.menuPanel.classList.contains("hidden");
  els.menuPanel.classList.toggle("hidden", !open);
  els.menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
  if (open) syncMenuState();
}

function closeStream() {
  if (state.stream) {
    state.stream.close();
    state.stream = null;
  }
}

async function readError(res) {
  const payload = await res.json().catch(() => ({ detail: res.statusText }));
  return payload.detail || res.statusText;
}

function unloadMedia(el) {
  if (!el) return;
  el.pause?.();
  el.removeAttribute("src");
  if (typeof el.load === "function") el.load();
}

function resetWorkspace() {
  closeStream();
  state.sourcePath = null;
  state.sourceUrl = null;
  state.runId = null;
  state.running = false;
  state.stopping = false;
  state.frameCount = null;
  state.statusKind = null;
  state.progress = null;
  state.errorMessage = null;
  unloadMedia(els.sourceVideo);
  unloadMedia(els.resultSource);
  unloadMedia(els.resultDepth);
  els.previewImage.removeAttribute("src");
  els.fileChip.classList.add("hidden");
  els.fileChipName.textContent = "—";
  els.fileChipSize.textContent = "";
  els.uploadHint.classList.remove("hidden");
  els.sourceSection.classList.add("hidden");
  els.progressWrap.classList.add("hidden");
  els.previewSlot.classList.add("hidden");
  els.resultSection.classList.add("hidden");
  renderFrames();
  renderStatus();
  els.progressBar.style.width = "0%";
  setActionButton();
}

async function clearCache() {
  if (state.running) {
    window.alert(t("clear_cache_busy"));
    return;
  }
  if (!window.confirm(t("clear_cache_confirm"))) return;
  unloadMedia(els.sourceVideo);
  unloadMedia(els.resultSource);
  unloadMedia(els.resultDepth);
  els.previewImage.removeAttribute("src");
  try {
    const res = await fetch("/api/cache", { method: "DELETE" });
    if (res.status === 409) {
      window.alert(t("clear_cache_busy"));
      return;
    }
    if (!res.ok) throw new Error(await readError(res));
    resetWorkspace();
    closeMenu();
  } catch (err) {
    window.alert(err instanceof Error ? err.message : "clear cache failed");
  }
}

els.stopBtn.addEventListener("click", () => {
  void stopProcess();
});

els.menuBtn.addEventListener("click", (ev) => {
  ev.stopPropagation();
  toggleMenu();
});

els.menuPanel.addEventListener("click", (ev) => {
  const theme = ev.target.closest("[data-theme]");
  if (theme) {
    const next = theme.getAttribute("data-theme");
    localStorage.setItem("dc-theme", next);
    applyTheme(next);
    syncMenuState();
  }
});

els.clearCacheBtn.addEventListener("click", (ev) => {
  ev.stopPropagation();
  void clearCache();
});

els.langBtn.addEventListener("click", () => {
  localStorage.setItem("dc-lang", currentLang() === "zh" ? "en" : "zh");
  applyI18n();
  setActionButton();
  renderFrames();
  renderStatus();
});

document.querySelectorAll(".help-wrap").forEach((wrap) => {
  const btn = wrap.querySelector(".help");
  if (!btn) return;
  btn.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    const open = !wrap.classList.contains("is-open");
    document.querySelectorAll(".help-wrap.is-open").forEach((other) => {
      other.classList.remove("is-open");
      const otherBtn = other.querySelector(".help");
      if (otherBtn) otherBtn.setAttribute("aria-expanded", "false");
    });
    wrap.classList.toggle("is-open", open);
    btn.setAttribute("aria-expanded", open ? "true" : "false");
  });
});

document.addEventListener("click", (ev) => {
  if (!els.menuPanel.classList.contains("hidden") && !ev.target.closest(".menu")) {
    closeMenu();
  }
  if (!ev.target.closest(".help-wrap")) {
    document.querySelectorAll(".help-wrap.is-open").forEach((wrap) => {
      wrap.classList.remove("is-open");
      const btn = wrap.querySelector(".help");
      if (btn) btn.setAttribute("aria-expanded", "false");
    });
  }
});

window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (currentTheme() === "system") applyTheme("system");
});

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

els.fileInput.addEventListener("change", () => {
  const file = els.fileInput.files && els.fileInput.files[0];
  els.fileInput.value = "";
  if (file) void handleFile(file);
});

["dragenter", "dragover"].forEach((name) => {
  els.dropzone.addEventListener(name, (ev) => {
    ev.preventDefault();
    els.dropzone.classList.add("is-over");
  });
});
els.dropzone.addEventListener("dragleave", (ev) => {
  if (!els.dropzone.contains(ev.relatedTarget)) els.dropzone.classList.remove("is-over");
});
els.dropzone.addEventListener("drop", (ev) => {
  ev.preventDefault();
  els.dropzone.classList.remove("is-over");
  const file = ev.dataTransfer.files && ev.dataTransfer.files[0];
  if (file) void handleFile(file);
});

async function handleFile(file) {
  if (file.size > MAX_UPLOAD_BYTES) {
    window.alert(t("upload_too_large"));
    return;
  }
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body });
  if (!res.ok) {
    window.alert(await readError(res));
    return;
  }
  const data = await res.json();
  state.sourcePath = data.source_path;
  state.sourceUrl = mediaFileUrl(data.info.path);
  state.runId = null;
  closeStream();
  state.running = false;
  state.stopping = false;
  els.uploadHint.classList.add("hidden");
  els.fileChip.classList.remove("hidden");
  els.fileChipName.textContent = file.name;
  els.fileChipSize.textContent = formatSize(file.size);
  els.sourceSection.classList.remove("hidden");
  els.sourceVideo.src = state.sourceUrl;
  els.metricName.textContent = data.info.name;
  els.metricRes.textContent = `${data.info.width} × ${data.info.height}`;
  els.metricFps.textContent = Number(data.info.fps).toFixed(2);
  els.metricDuration.textContent = data.info.duration_label;
  state.frameCount = data.info.frame_count || "—";
  state.statusKind = null;
  state.progress = null;
  state.errorMessage = null;
  renderFrames();
  els.progressWrap.classList.add("hidden");
  els.previewSlot.classList.add("hidden");
  els.resultSection.classList.add("hidden");
  renderStatus();
  els.progressBar.style.width = "0%";
  setActionButton();
}

els.actionBtn.addEventListener("click", () => {
  if (state.running) {
    void stopProcess();
    return;
  }
  void startProcess();
});

async function startProcess() {
  if (!state.sourcePath || state.running) return;
  state.running = true;
  state.stopping = false;
  els.progressWrap.classList.remove("hidden");
  els.previewSlot.classList.add("hidden");
  els.resultSection.classList.add("hidden");
  els.progressBar.style.width = "0%";
  setStatus("processing");
  setActionButton();
  try {
    const res = await fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_path: state.sourcePath,
        scale: Number(els.scale.value),
        stride: Number(els.stride.value),
        gamma: Number(els.gamma.value),
        crush: Number(els.crush.value),
        smooth: Number(els.smooth.value),
        preserve_audio: els.audio.checked,
      }),
    });
    if (!res.ok) {
      if (res.status === 409) throw new Error(t("clear_cache_busy"));
      throw new Error(await readError(res));
    }
    const data = await res.json();
    state.runId = data.run_id;
    setActionButton();
    listen(data.run_id);
  } catch (err) {
    state.running = false;
    setActionButton();
    window.alert(err instanceof Error ? err.message : "process failed");
  }
}

async function stopProcess() {
  if (!state.runId || state.stopping) return;
  state.stopping = true;
  setActionButton();
  try {
    const res = await fetch(`/api/process/${state.runId}/cancel`, { method: "POST" });
    if (!res.ok) throw new Error(await readError(res));
  } catch (err) {
    state.stopping = false;
    setActionButton();
    window.alert(err instanceof Error ? err.message : "cancel failed");
  }
}

function listen(runId) {
  closeStream();
  const stream = new EventSource(`/api/process/${runId}/events`);
  state.stream = stream;
  stream.addEventListener("progress", (ev) => {
    const data = JSON.parse(ev.data);
    const pct = Math.round((data.progress || 0) * 100);
    els.progressBar.style.width = `${pct}%`;
    setStatus("progress", data);
    if (data.preview_path) {
      els.previewImage.src = mediaFileUrl(data.preview_path);
      els.previewSlot.classList.remove("hidden");
    }
  });
  stream.addEventListener("done", (ev) => {
    const data = JSON.parse(ev.data);
    finishRun();
    els.progressBar.style.width = "100%";
    setStatus("done");
    els.previewSlot.classList.add("hidden");
    els.resultSource.src = state.sourceUrl;
    els.resultDepth.src = mediaOutputUrl(data.job_id);
    els.downloadBtn.href = mediaOutputUrl(data.job_id);
    els.resultSection.classList.remove("hidden");
  });
  stream.addEventListener("failed", (ev) => {
    const data = JSON.parse(ev.data);
    finishRun();
    setStatus("error", data.message || "error");
  });
  stream.addEventListener("cancelled", () => {
    finishRun();
    els.progressBar.style.width = "0%";
    els.previewSlot.classList.add("hidden");
    setStatus("cancelled");
  });
}

function finishRun() {
  closeStream();
  state.running = false;
  state.stopping = false;
  state.runId = null;
  setActionButton();
}

async function hydrateActiveRun() {
  try {
    const res = await fetch("/api/process/active");
    if (!res.ok) return;
    const data = await res.json();
    if (!data.run_id) return;
    state.runId = data.run_id;
    state.running = true;
    setStatus("processing");
    setActionButton();
    listen(data.run_id);
  } catch {
    // ignore
  }
}

applyI18n();
applyTheme(currentTheme());
syncMenuState();
setActionButton();
void hydrateActiveRun();
