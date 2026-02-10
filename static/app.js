// ==============================================================
// Real-Time Invisible Watermark Encoder (Production v3)
// ==============================================================

const DEFAULT_CONFIG = {
  protocol: "spread_v3",
  bit_frames: 8,
  chips_per_bit: 4,
  patch_size: 28,
  delta: 7,
  delta_dc: 3,
  dc_weight: 0.6,
  dct_weight: 0.55,
  dct_qim_delta: 14,
  pattern_block: 7,
  preamble: "11001011100101101001110010110100",
  cell_anchors: [
    [0.18, 0.22],
    [0.5, 0.22],
    [0.82, 0.22],
    [0.18, 0.5],
    [0.5, 0.5],
    [0.82, 0.5],
    [0.18, 0.78],
    [0.5, 0.78],
    [0.82, 0.78],
  ],
  prn_seed: 915131,
  base_fps: 24,
  max_payload_bytes: 128,
  repeat_packet: true,
  packet_gap_bits: 0,
  interleaver_depth: 8,
  pilot_bits: "101011001101",
  visible_strip: true,
  strip_cap_bits: 512,
  strip_sync_bits: "111000101011110010100111",
  strip_margin: 8,
  strip_height: 10,
  strip_alpha: 0.8,
};

// DOM elements
const form = document.getElementById("encode-form");
const input = document.getElementById("user-id");
const statusEl = document.getElementById("status");
const encodeBtn = document.getElementById("encode-btn");
const sourceVideo = document.getElementById("source-video");
const canvas = document.getElementById("preview-canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const stagePanel = document.querySelector(".stage-panel");
const countdownOverlay = document.getElementById("countdown-overlay");
const playBtn = document.getElementById("play-btn");
const pauseBtn = document.getElementById("pause-btn");
const focusBtn = document.getElementById("focus-btn");
const timeDisplay = document.getElementById("time-display");
const downloadBtn = document.getElementById("download-btn");

// State
let config = { ...DEFAULT_CONFIG };
let preambleBits = [];
let bitDuration = 0;
let chipDuration = 0;
let bits = [];
let visibleStripBits = [];
let isEncoding = false;
let frameRequestId = null;
let usesVideoFrameCallback = false;
let mediaRecorder = null;
let recordedChunks = [];
let encodedBlobUrl = null;
let recordedMimeType = "video/webm";
let serverEncodedUrl = "";
let serverEncodedFile = "";
let serverEncodeState = "idle"; // idle | pending | ready | failed
let cellsCacheKey = "";
let cachedCells = [];
let autoStartTimeoutId = null;
let autoStartIntervalId = null;

const AUTO_START_DELAY_MS = 4500;

// ============================================================== 
// Helpers
// ============================================================== 

const setStatus = (message, tone = "info") => {
  statusEl.textContent = message;
  statusEl.style.color = tone === "error" ? "var(--danger)" : "var(--info)";
};

const setCaptureMode = (enabled) => {
  document.body.classList.toggle("capture-mode", enabled);
};

const formatTime = (seconds) => {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "0:00";
  }
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
};

const parseBitString = (value) => {
  if (Array.isArray(value)) {
    return value.map((bit) => (bit ? 1 : 0));
  }
  return String(value)
    .split("")
    .filter((ch) => ch === "0" || ch === "1")
    .map((ch) => Number(ch));
};

const updateTiming = () => {
  const baseFps = Number(config.base_fps) > 0 ? Number(config.base_fps) : DEFAULT_CONFIG.base_fps;
  const bitFrames = Number(config.bit_frames) > 0 ? Number(config.bit_frames) : DEFAULT_CONFIG.bit_frames;
  const chipsPerBit = Number(config.chips_per_bit) > 0 ? Number(config.chips_per_bit) : DEFAULT_CONFIG.chips_per_bit;
  bitDuration = bitFrames / baseFps;
  chipDuration = bitDuration / chipsPerBit;
  preambleBits = parseBitString(config.preamble);
};

const normalizeAnchors = (anchors) => {
  if (!Array.isArray(anchors) || anchors.length === 0) {
    return DEFAULT_CONFIG.cell_anchors;
  }
  const normalized = anchors
    .map((pair) => {
      if (!Array.isArray(pair) || pair.length < 2) {
        return null;
      }
      const x = Number(pair[0]);
      const y = Number(pair[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        return null;
      }
      return [Math.min(Math.max(x, 0), 1), Math.min(Math.max(y, 0), 1)];
    })
    .filter(Boolean);
  return normalized.length > 0 ? normalized : DEFAULT_CONFIG.cell_anchors;
};

const applyConfig = (incoming) => {
  config = {
    ...DEFAULT_CONFIG,
    ...incoming,
    cell_anchors: normalizeAnchors(incoming.cell_anchors),
  };
  updateTiming();
  cellsCacheKey = "";
};

const loadConfig = async () => {
  try {
    const response = await fetch("/config", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const serverConfig = await response.json();
    applyConfig(serverConfig);
    return true;
  } catch (error) {
    applyConfig(DEFAULT_CONFIG);
    setStatus("Config endpoint unavailable. Using defaults.", "error");
    console.warn("Failed to load /config:", error);
    return false;
  }
};

const requestServerEncode = async (userId) => {
  serverEncodeState = "pending";
  try {
    const response = await fetch("/encode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: userId }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || `HTTP ${response.status}`);
    }

    serverEncodedUrl = payload.video_url || "";
    serverEncodedFile = payload.file || "";
    if (serverEncodedUrl) {
      serverEncodeState = "ready";
      downloadBtn.disabled = false;
      downloadBtn.classList.remove("disabled");
      setStatus("Server-encoded file is ready for reliable download.");
    }
  } catch (error) {
    serverEncodeState = "failed";
    console.warn("Server encoding failed:", error);
    setStatus(`Server encode failed: ${error.message}`, "error");
  }
};

const bytesToBits = (bytes) => {
  const out = [];
  for (const byte of bytes) {
    for (let shift = 7; shift >= 0; shift -= 1) {
      out.push((byte >> shift) & 1);
    }
  }
  return out;
};

const crc16Ccitt = (bytes) => {
  let crc = 0xffff;
  const poly = 0x1021;
  for (const byte of bytes) {
    crc ^= byte << 8;
    for (let i = 0; i < 8; i += 1) {
      if (crc & 0x8000) {
        crc = ((crc << 1) ^ poly) & 0xffff;
      } else {
        crc = (crc << 1) & 0xffff;
      }
    }
  }
  return crc;
};

const crc32Ieee = (bytes) => {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let i = 0; i < 8; i += 1) {
      if (crc & 1) {
        crc = (crc >>> 1) ^ 0xedb88320;
      } else {
        crc >>>= 1;
      }
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
};

const hamming74Encode = (bitsIn) => {
  const bitsLocal = bitsIn.slice();
  if (bitsLocal.length % 4 !== 0) {
    const pad = 4 - (bitsLocal.length % 4);
    for (let i = 0; i < pad; i += 1) {
      bitsLocal.push(0);
    }
  }

  const out = [];
  for (let i = 0; i < bitsLocal.length; i += 4) {
    const d1 = bitsLocal[i] & 1;
    const d2 = bitsLocal[i + 1] & 1;
    const d3 = bitsLocal[i + 2] & 1;
    const d4 = bitsLocal[i + 3] & 1;
    const p1 = d1 ^ d2 ^ d4;
    const p2 = d1 ^ d3 ^ d4;
    const p3 = d2 ^ d3 ^ d4;
    out.push(p1, p2, d1, p3, d2, d3, d4);
  }
  return out;
};

const interleaveBits = (bitsIn, depth) => {
  const bitsLocal = bitsIn.slice();
  if (!Number.isFinite(depth) || depth <= 1 || bitsLocal.length === 0) {
    return bitsLocal;
  }
  const rows = Math.max(1, Math.floor(depth));
  const cols = Math.ceil(bitsLocal.length / rows);
  const grid = Array.from({ length: rows }, () => Array(cols).fill(null));

  let idx = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      if (idx >= bitsLocal.length) {
        break;
      }
      grid[row][col] = bitsLocal[idx] & 1;
      idx += 1;
    }
  }

  const out = [];
  for (let col = 0; col < cols; col += 1) {
    for (let row = 0; row < rows; row += 1) {
      const value = grid[row][col];
      if (value === null) {
        continue;
      }
      out.push(value);
    }
  }
  return out;
};

const repeatBits = (bitsIn, repeatFactor) => {
  const factor = Math.max(1, Math.floor(repeatFactor));
  if (factor <= 1) {
    return bitsIn.slice();
  }
  const out = [];
  for (const bit of bitsIn) {
    for (let i = 0; i < factor; i += 1) {
      out.push(bit & 1);
    }
  }
  return out;
};

const buildVisibleStripBits = (text) => {
  if (!config.visible_strip) {
    return [];
  }
  const encoder = new TextEncoder();
  const payload = encoder.encode(text);
  const maxPayload = Math.min(Number(config.max_payload_bytes), 255);
  if (payload.length > maxPayload) {
    throw new Error(`User ID too long (${payload.length} bytes). Max strip payload is ${maxPayload} bytes.`);
  }

  const syncBits = parseBitString(config.strip_sync_bits);
  const stripCapBits = Math.max(Number(config.strip_cap_bits) || 0, 1);
  const raw = new Uint8Array(1 + payload.length + 2);
  raw[0] = payload.length & 0xff;
  raw.set(payload, 1);
  const crc = crc16Ccitt(raw.subarray(0, 1 + payload.length));
  raw[1 + payload.length] = (crc >> 8) & 0xff;
  raw[2 + payload.length] = crc & 0xff;
  const bitsLocal = syncBits.concat(bytesToBits(raw));
  if (bitsLocal.length > stripCapBits) {
    throw new Error("User ID too long for visible strip capacity.");
  }
  while (bitsLocal.length < stripCapBits) {
    bitsLocal.push(0);
  }
  return bitsLocal;
};

const buildPacketBits = (text) => {
  const encoder = new TextEncoder();
  const payload = encoder.encode(text);

  if (payload.length > Number(config.max_payload_bytes)) {
    throw new Error(`User ID too long (${payload.length} bytes). Max is ${config.max_payload_bytes} bytes.`);
  }

  const repeatFactor = 1;
  const flags = ((3 & 0x0f) << 4) | (repeatFactor & 0x03) | ((Number(config.interleaver_depth) > 1) ? 0x04 : 0);
  const raw = new Uint8Array(3 + payload.length + 4);
  raw[0] = 0x57; // 'W'
  raw[1] = flags;
  raw[2] = payload.length & 0xff;
  raw.set(payload, 3);

  const crc = crc32Ieee(raw.subarray(0, 3 + payload.length));
  raw[3 + payload.length] = (crc >>> 24) & 0xff;
  raw[4 + payload.length] = (crc >>> 16) & 0xff;
  raw[5 + payload.length] = (crc >>> 8) & 0xff;
  raw[6 + payload.length] = crc & 0xff;

  const rawBits = bytesToBits(raw);
  let eccBits = hamming74Encode(rawBits);
  eccBits = interleaveBits(eccBits, Number(config.interleaver_depth) || 1);
  eccBits = repeatBits(eccBits, repeatFactor);
  return preambleBits.concat(eccBits);
};

const estimateMaxPayloadBytes = (bitCapacity) => {
  if (bitCapacity <= preambleBits.length) {
    return 0;
  }
  const eccBits = bitCapacity - preambleBits.length;
  const rawBits = Math.floor(eccBits / 7) * 4;
  const rawBytes = Math.floor(rawBits / 8);
  return Math.max(rawBytes - 7, 0);
};

const u32 = (value) => value >>> 0;
const mul32 = (a, b) => Math.imul(a, b) >>> 0;

const prnSign = (bitIndex, chipIndex, cellIndex) => {
  let x = u32(config.prn_seed);
  x ^= mul32(bitIndex + 1, 0x9e3779b1);
  x ^= mul32(chipIndex + 1, 0x85ebca77);
  x ^= mul32(cellIndex + 1, 0xc2b2ae3d);
  x = u32(x);
  x ^= x >>> 15;
  x = mul32(x, 0x2c1b3c6d);
  x ^= x >>> 12;
  return (x & 1) ? 1 : -1;
};

const resolveCells = (width, height) => {
  const patchSize = Number(config.patch_size);
  const size = Math.min(
    patchSize,
    Math.max(Math.floor(width / 10), 8),
    Math.max(Math.floor(height / 10), 8),
  );
  const key = `${width}x${height}:${size}:${JSON.stringify(config.cell_anchors)}`;
  if (key === cellsCacheKey && cachedCells.length > 0) {
    return cachedCells;
  }

  cachedCells = config.cell_anchors.map(([cx, cy]) => {
    const x0 = Math.min(
      width - size,
      Math.max(0, Math.round((cx * (width - 1)) - (size / 2))),
    );
    const y0 = Math.min(
      height - size,
      Math.max(0, Math.round((cy * (height - 1)) - (size / 2))),
    );
    return { x0, y0, size };
  });
  cellsCacheKey = key;
  return cachedCells;
};

const applyWatermark = (imageData, bitIndex, chipIndex, bitValue) => {
  const data = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  const delta = Number(config.delta);
  const deltaDc = Number(config.delta_dc || 0);
  const block = Math.max(1, Number(config.pattern_block) || 1);
  const bitPolarity = bitValue === 1 ? 1 : -1;
  const cells = resolveCells(width, height);

  for (let cellIndex = 0; cellIndex < cells.length; cellIndex += 1) {
    const cell = cells[cellIndex];
    const sign = prnSign(bitIndex, chipIndex, cellIndex) * bitPolarity;

    for (let y = 0; y < cell.size; y += 1) {
      for (let x = 0; x < cell.size; x += 1) {
        const px = cell.x0 + x;
        const py = cell.y0 + y;
        const idx = (py * width + px) * 4;
        const checker = ((Math.floor(x / block) + Math.floor(y / block)) & 1) === 0 ? -1 : 1;
        const modulation = (sign * checker * delta) + (sign * deltaDc);
        const red = data[idx];
        const blue = data[idx + 2];

        data[idx] = Math.max(0, Math.min(255, red - modulation));
        data[idx + 2] = Math.max(0, Math.min(255, blue + modulation));
      }
    }
  }

  return imageData;
};

const drawVisibleStrip = () => {
  if (!config.visible_strip || !visibleStripBits.length) {
    return;
  }
  const width = canvas.width;
  const height = canvas.height;
  const stripCapBits = Math.max(Number(config.strip_cap_bits) || 0, 1);
  const stripMargin = Math.max(Number(config.strip_margin) || 0, 0);
  const stripHeight = Math.max(Number(config.strip_height) || 0, 2);
  const stripAlpha = Math.min(Math.max(Number(config.strip_alpha), 0.0), 1.0);

  const cellWidth = Math.max(Math.floor((width - (2 * stripMargin)) / stripCapBits), 1);
  const usedWidth = cellWidth * stripCapBits;
  const x0 = Math.max(Math.floor((width - usedWidth) / 2), 0);
  const y0 = Math.max(height - stripMargin - stripHeight, 0);

  ctx.save();
  for (let i = 0; i < Math.min(stripCapBits, visibleStripBits.length); i += 1) {
    const bit = visibleStripBits[i] ? 1 : 0;
    const tone = bit ? 245 : 10;
    ctx.fillStyle = `rgba(${tone}, ${tone}, ${tone}, ${stripAlpha})`;
    ctx.fillRect(x0 + (i * cellWidth), y0, cellWidth, stripHeight);
  }
  ctx.restore();
};

const getBitIndexForTime = (currentTime) => Math.floor(currentTime / bitDuration);

const getChipIndexForTime = (currentTime, bitIndex) => {
  const local = currentTime - (bitIndex * bitDuration);
  const idx = Math.floor(local / chipDuration);
  const chipsPerBit = Number(config.chips_per_bit);
  return Math.min(Math.max(idx, 0), chipsPerBit - 1);
};

const fullscreenElement = () => document.fullscreenElement || document.webkitFullscreenElement || null;

const isStageFullscreen = () => {
  if (!stagePanel) {
    return false;
  }
  return fullscreenElement() === stagePanel || stagePanel.classList.contains("is-fullscreen");
};

const updateFocusButton = () => {
  if (!focusBtn || !stagePanel) {
    return;
  }
  const active = isStageFullscreen();
  focusBtn.textContent = active ? "Exit Focus" : "Focus Mode";
  stagePanel.classList.toggle("is-fullscreen", active);
};

const enterFocusMode = async () => {
  if (!stagePanel) {
    return;
  }

  try {
    if (stagePanel.requestFullscreen) {
      await stagePanel.requestFullscreen();
    } else if (stagePanel.webkitRequestFullscreen) {
      stagePanel.webkitRequestFullscreen();
    } else {
      stagePanel.classList.add("is-fullscreen");
    }
    setStatus("Focus mode enabled for cleaner screen recording.");
  } catch (error) {
    setStatus(`Focus mode failed: ${error.message}`, "error");
  } finally {
    updateFocusButton();
  }
};

const exitFocusMode = async () => {
  try {
    if (document.exitFullscreen && document.fullscreenElement) {
      await document.exitFullscreen();
    } else if (document.webkitExitFullscreen && document.webkitFullscreenElement) {
      document.webkitExitFullscreen();
    }
  } catch (error) {
    console.warn("Failed to exit fullscreen:", error);
  } finally {
    if (stagePanel) {
      stagePanel.classList.remove("is-fullscreen");
    }
    updateFocusButton();
  }
};

const toggleFocusMode = async () => {
  if (isStageFullscreen()) {
    await exitFocusMode();
  } else {
    await enterFocusMode();
  }
};

const updateCountdownOverlay = (text = "", visible = false) => {
  if (!countdownOverlay) {
    return;
  }
  countdownOverlay.textContent = text;
  countdownOverlay.classList.toggle("hidden", !visible);
};

const clearAutoStartTimers = () => {
  if (autoStartTimeoutId !== null) {
    clearTimeout(autoStartTimeoutId);
    autoStartTimeoutId = null;
  }
  if (autoStartIntervalId !== null) {
    clearInterval(autoStartIntervalId);
    autoStartIntervalId = null;
  }
  updateCountdownOverlay("", false);
};

window.addEventListener("beforeunload", () => {
  clearAutoStartTimers();
});

const scheduleAutoStart = (delayMs = AUTO_START_DELAY_MS) => {
  clearAutoStartTimers();

  const startAt = Date.now() + Math.max(delayMs, 0);
  playBtn.disabled = true;
  pauseBtn.disabled = true;
  setCaptureMode(true);

  const renderCountdown = () => {
    const remaining = Math.max(0, startAt - Date.now());
    const remainingSec = Math.max(1, Math.ceil(remaining / 1000));
    updateCountdownOverlay(String(remainingSec), true);
    setStatus(
      `Auto-starting in ${remainingSec}s. Start your screen recorder now; overlay will clear before encode.`,
    );
  };

  renderCountdown();
  autoStartIntervalId = setInterval(renderCountdown, 200);
  autoStartTimeoutId = setTimeout(async () => {
    clearAutoStartTimers();
    await startEncoding();
  }, Math.max(delayMs, 0));
};

// ============================================================== 
// Rendering and recording
// ============================================================== 

const updateTimeDisplay = () => {
  timeDisplay.textContent = `${formatTime(sourceVideo.currentTime)} / ${formatTime(sourceVideo.duration)}`;
};

const renderCurrentFrame = () => {
  ctx.drawImage(sourceVideo, 0, 0, canvas.width, canvas.height);

  const currentTime = sourceVideo.currentTime;
  const bitIndex = getBitIndexForTime(currentTime);
  if (bitIndex >= 0 && bits.length > 0) {
    const payloadBit = bits[bitIndex % bits.length];
    const chipIndex = getChipIndexForTime(currentTime, bitIndex);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    applyWatermark(imageData, bitIndex, chipIndex, payloadBit);
    ctx.putImageData(imageData, 0, 0);
  }

  drawVisibleStrip();

  updateTimeDisplay();
};

const cancelFrameLoop = () => {
  if (frameRequestId === null) {
    return;
  }
  if (usesVideoFrameCallback && typeof sourceVideo.cancelVideoFrameCallback === "function") {
    sourceVideo.cancelVideoFrameCallback(frameRequestId);
  } else {
    cancelAnimationFrame(frameRequestId);
  }
  frameRequestId = null;
};

const scheduleFrameLoop = () => {
  if (!isEncoding) {
    return;
  }

  if (typeof sourceVideo.requestVideoFrameCallback === "function") {
    usesVideoFrameCallback = true;
    frameRequestId = sourceVideo.requestVideoFrameCallback(() => {
      frameRequestId = null;
      renderLoop();
    });
  } else {
    usesVideoFrameCallback = false;
    frameRequestId = requestAnimationFrame(() => {
      frameRequestId = null;
      renderLoop();
    });
  }
};

const renderLoop = () => {
  if (!isEncoding) {
    return;
  }

  if (sourceVideo.ended) {
    finishEncoding();
    return;
  }

  if (!sourceVideo.paused) {
    renderCurrentFrame();
  }

  scheduleFrameLoop();
};

const startRecording = () => {
  recordedChunks = [];
  const stream = canvas.captureStream(30);

  const mimeCandidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];

  let selectedMime = "";
  for (const mime of mimeCandidates) {
    if (MediaRecorder.isTypeSupported(mime)) {
      selectedMime = mime;
      break;
    }
  }

  try {
    if (selectedMime) {
      mediaRecorder = new MediaRecorder(stream, {
        mimeType: selectedMime,
        videoBitsPerSecond: 8000000,
      });
    } else {
      mediaRecorder = new MediaRecorder(stream, { videoBitsPerSecond: 8000000 });
    }
  } catch (error) {
    mediaRecorder = new MediaRecorder(stream);
    console.warn("Failed to use selected recorder settings:", error);
  }

  recordedMimeType = mediaRecorder.mimeType || selectedMime || "video/webm";

  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, { type: recordedMimeType });
    if (encodedBlobUrl) {
      URL.revokeObjectURL(encodedBlobUrl);
    }
    encodedBlobUrl = URL.createObjectURL(blob);
    downloadBtn.disabled = false;
    downloadBtn.classList.remove("disabled");
  };

  mediaRecorder.start(250);
};

const pauseRecording = () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.pause();
  }
};

const resumeRecording = () => {
  if (mediaRecorder && mediaRecorder.state === "paused") {
    mediaRecorder.resume();
  }
};

const stopRecording = () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
};

const finishEncoding = () => {
  isEncoding = false;
  clearAutoStartTimers();
  cancelFrameLoop();
  stopRecording();
  setCaptureMode(false);
  playBtn.disabled = false;
  pauseBtn.disabled = true;
  setStatus("Encoding complete. Download your encoded video.");
};

const startEncoding = async () => {
  if (bits.length === 0) {
    setStatus("No packet bits available. Enter a User ID and prepare first.", "error");
    return;
  }

  if (!sourceVideo.src) {
    setStatus("Load a source video first.", "error");
    return;
  }

  isEncoding = true;
  clearAutoStartTimers();
  setCaptureMode(true);
  cancelFrameLoop();
  if (!serverEncodedUrl) {
    downloadBtn.disabled = true;
    downloadBtn.classList.add("disabled");
  }

  sourceVideo.currentTime = 0;
  updateTimeDisplay();

  try {
    await sourceVideo.play();
  } catch (error) {
    isEncoding = false;
    stopRecording();
    setCaptureMode(false);
    setStatus(`Failed to start playback: ${error.message}`, "error");
    return;
  }

  renderCurrentFrame();
  startRecording();
  renderLoop();
  playBtn.disabled = true;
  pauseBtn.disabled = false;
  setStatus("Encoding in progress. Keep this tab visible.");
};

const pauseEncoding = () => {
  if (!isEncoding) {
    return;
  }
  sourceVideo.pause();
  pauseRecording();
  cancelFrameLoop();
  playBtn.disabled = false;
  pauseBtn.disabled = true;
  setStatus("Paused. Click Play to continue.");
};

// ============================================================== 
// Event handlers
// ============================================================== 

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const userId = input.value.trim();
  if (!userId) {
    setStatus("Please enter a User ID.", "error");
    return;
  }

  encodeBtn.disabled = true;
  clearAutoStartTimers();
  setCaptureMode(true);
  setStatus("Preparing encoder...");
  serverEncodedUrl = "";
  serverEncodedFile = "";
  serverEncodeState = "idle";

  // Request fullscreen from this direct user gesture when possible.
  if (!isStageFullscreen()) {
    void enterFocusMode();
  }

  await loadConfig();

  try {
    bits = buildPacketBits(userId);
    visibleStripBits = buildVisibleStripBits(userId);
  } catch (error) {
    setStatus(error.message, "error");
    visibleStripBits = [];
    setCaptureMode(false);
    encodeBtn.disabled = false;
    return;
  }

  sourceVideo.src = `/static/base.mp4?t=${Date.now()}`;
  sourceVideo.load();

  // Start reliable server-side encoding in parallel.
  requestServerEncode(userId);

  sourceVideo.onloadedmetadata = () => {
    canvas.width = sourceVideo.videoWidth;
    canvas.height = sourceVideo.videoHeight;
    cellsCacheKey = "";

    const capacityBits = Math.floor(sourceVideo.duration / bitDuration);
    const maxPayloadBytes = estimateMaxPayloadBytes(capacityBits);

    if (bits.length > capacityBits) {
      setStatus(
        `ID too long for this video. Capacity is about ${maxPayloadBytes} UTF-8 bytes.`,
        "error",
      );
      encodeBtn.disabled = false;
      playBtn.disabled = true;
      setCaptureMode(false);
      return;
    }

    sourceVideo.currentTime = 0;
    ctx.drawImage(sourceVideo, 0, 0, canvas.width, canvas.height);
    drawVisibleStrip();
    updateTimeDisplay();

    playBtn.disabled = true;
    pauseBtn.disabled = true;
    encodeBtn.disabled = false;
    downloadBtn.disabled = !serverEncodedUrl;
    if (serverEncodedUrl) {
      downloadBtn.classList.remove("disabled");
    } else {
      downloadBtn.classList.add("disabled");
    }

    scheduleAutoStart();
  };

  sourceVideo.onerror = () => {
    setStatus("Failed to load /static/base.mp4.", "error");
    setCaptureMode(false);
    clearAutoStartTimers();
    encodeBtn.disabled = false;
  };
});

playBtn.addEventListener("click", async () => {
  clearAutoStartTimers();
  if (!isEncoding) {
    await startEncoding();
    return;
  }

  try {
    await sourceVideo.play();
    resumeRecording();
    renderLoop();
    playBtn.disabled = true;
    pauseBtn.disabled = false;
    setStatus("Encoding resumed.");
  } catch (error) {
    setStatus(`Resume failed: ${error.message}`, "error");
  }
});

pauseBtn.addEventListener("click", () => {
  clearAutoStartTimers();
  pauseEncoding();
});

if (focusBtn) {
  focusBtn.addEventListener("click", async () => {
    await toggleFocusMode();
  });
}

downloadBtn.addEventListener("click", () => {
  if (serverEncodedUrl) {
    const a = document.createElement("a");
    const userId = input.value.trim() || "encoded";
    a.href = serverEncodedUrl;
    a.download = serverEncodedFile || `${userId}_encoded.mp4`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    return;
  }

  if (serverEncodeState === "pending") {
    setStatus("Server encode is still running. Wait for the ready message before downloading.", "error");
    return;
  }

  if (serverEncodeState !== "failed") {
    setStatus("Reliable server file is not ready yet.", "error");
    return;
  }

  if (!encodedBlobUrl) {
    return;
  }

  const a = document.createElement("a");
  const userId = input.value.trim() || "encoded";
  const ext = recordedMimeType.includes("mp4") ? "mp4" : "webm";
  a.href = encodedBlobUrl;
  a.download = `${userId}_encoded.${ext}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

sourceVideo.addEventListener("ended", () => {
  if (isEncoding) {
    finishEncoding();
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden && isEncoding && !sourceVideo.paused) {
    pauseEncoding();
    setCaptureMode(false);
    setStatus("Tab hidden. Encoding paused to protect timing. Keep this tab visible.", "error");
  }
});

window.addEventListener("blur", () => {
  if (isEncoding && !sourceVideo.paused) {
    setStatus("Keep this tab in focus while encoding for best decode quality.", "error");
  }
});

document.addEventListener("fullscreenchange", updateFocusButton);
document.addEventListener("webkitfullscreenchange", updateFocusButton);

document.addEventListener("keydown", async (event) => {
  if (event.key.toLowerCase() !== "f" || event.ctrlKey || event.metaKey || event.altKey) {
    return;
  }

  const target = event.target;
  if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) {
    return;
  }

  event.preventDefault();
  await toggleFocusMode();
});

// Initial setup
applyConfig(DEFAULT_CONFIG);
updateFocusButton();
clearAutoStartTimers();
setCaptureMode(false);
setStatus("Enter a User ID and click Start Encoding.");
