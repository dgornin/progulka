/*
 * Voice input: a microphone button that records speech and puts the transcript into
 * a text box, at the cursor.
 *
 * The browser only records; the server transcribes (POST /api/transcribe, an
 * audio-capable LLM). The Web Speech API would be free, but Firefox and Yandex
 * Browser do not have it and Chrome sends the audio to Google; recording works in
 * every current browser.
 *
 * MediaRecorder writes whatever the browser encodes (WebM/Opus in Chrome, Ogg in
 * Firefox, MP4/AAC in Safari), so the recording is decoded with Web Audio, mixed down
 * and resampled to 16 kHz mono by an OfflineAudioContext and sent as a WAV of 16-bit
 * samples: one format for the server whatever the browser, 32 KB a second, at most a
 * minute (under 2 MB). Silence is caught here and never sent: a transcription costs.
 *
 * States: idle -> recording -> transcribing -> idle, or error (shown for a few seconds).
 */

import { ApiError, api } from "./api.js";
import { h, svg } from "./ui.js";

const RATE = 16_000;
const MAX_SECONDS = 60;
const MIN_SECONDS = 0.6;
const SILENCE = 0.02; // peak amplitude below which nobody spoke
const ERROR_MS = 6000;

const AudioContextClass = window.AudioContext ?? window.webkitAudioContext;

/** Recording needs a microphone API, MediaRecorder and Web Audio (and a secure page). */
const voiceSupported = () =>
  Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder && AudioContextClass && window.OfflineAudioContext);

let active = null; // one recording at a time on the page

/* ---------- audio ---------- */

/** Little-endian 16-bit PCM in a RIFF/WAVE container. */
function wav(samples) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const text = (offset, value) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, RATE, true);
  view.setUint32(28, RATE * 2, true); // bytes per second
  view.setUint16(32, 2, true); // bytes per frame
  view.setUint16(34, 16, true); // bits per sample
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, i) => {
    const s = Math.max(-1, Math.min(1, sample));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  });
  return new Blob([view], { type: "audio/wav" });
}

function base64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1]);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

/** Whatever the browser recorded -> 16 kHz mono samples, at most MAX_SECONDS long. */
async function resample(blob, context) {
  const decoded = await context.decodeAudioData(await blob.arrayBuffer());
  const frames = Math.min(Math.ceil(decoded.duration * RATE), MAX_SECONDS * RATE);
  const offline = new OfflineAudioContext(1, frames, RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination); // a mono destination mixes the channels down
  source.start();
  return (await offline.startRendering()).getChannelData(0);
}

/* ---------- text ---------- */

/** Put `text` at the box's cursor (or over its selection), with spaces as needed. */
function insert(box, text) {
  const { value, selectionStart: from, selectionEnd: to } = box;
  const before = from > 0 && !/\s/.test(value[from - 1]) ? " " : "";
  const after = to < value.length && !/\s/.test(value[to]) ? " " : "";
  const room = box.maxLength > 0 ? box.maxLength - (value.length - (to - from)) : Infinity;
  const piece = `${before}${text}${after}`.slice(0, Math.max(0, room));
  box.setRangeText(piece, from, to, "end");
  box.dispatchEvent(new Event("input", { bubbles: true }));
  box.focus();
}

function failure(error) {
  if (error instanceof ApiError) return error.message;
  if (error?.name === "NotAllowedError" || error?.name === "SecurityError") return "Нет доступа к микрофону: разрешите его в настройках браузера";
  if (error?.name === "NotFoundError" || error?.name === "OverconstrainedError") return "Микрофон не найден";
  if (error?.name === "NotReadableError") return "Микрофон занят другим приложением";
  if (error?.name === "EncodingError") return "Не удалось прочитать запись, попробуйте ещё раз";
  return error?.message || "Не удалось записать голос";
}

/* ---------- the control ---------- */

/** Whether a control from voiceInput() holds words that are not in its box yet. */
export const dictating = (control) => ["starting", "recording", "transcribing"].includes(control?.dataset.state);

const micIcon = () =>
  svg(
    "svg",
    { viewBox: "0 0 24 24", "aria-hidden": "true", class: "voice-icon" },
    svg("rect", { x: "9", y: "3", width: "6", height: "11", rx: "3" }),
    svg("path", { d: "M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M8.5 21h7" }),
  );

const stopIcon = () =>
  svg("svg", { viewBox: "0 0 24 24", "aria-hidden": "true", class: "voice-icon" }, svg("rect", { x: "7", y: "7", width: "10", height: "10", rx: "2" }));

/**
 * A microphone control for `box` (a textarea or input), or null where the browser
 * cannot record (no getUserMedia on a page without HTTPS, say): then there is no
 * button to fail. `label` names what is dictated, for screen readers.
 */
export function voiceInput(box, { label = "Надиктовать" } = {}) {
  if (!voiceSupported()) return null;
  const button = h("button", { class: "voice-btn", type: "button", "aria-label": label, title: `${label} (до минуты)` }, micIcon());
  const status = h("span", { class: "voice-status", role: "status", "aria-live": "polite" });
  const timer = h("span", { class: "voice-timer", "aria-hidden": "true" });
  const element = h("div", { class: "voice", "data-state": "idle" }, status, timer, button);
  let session = null;
  let errorTimer = 0;

  const setState = (state, text = "") => {
    element.dataset.state = state;
    status.textContent = text;
    timer.textContent = "";
    const recording = state === "recording";
    button.replaceChildren(recording ? stopIcon() : micIcon());
    button.setAttribute("aria-label", recording ? "Остановить запись и распознать" : label);
    button.setAttribute("aria-pressed", String(recording));
    button.toggleAttribute("aria-busy", state === "transcribing");
  };

  const showError = (error) => {
    setState("error", failure(error));
    clearTimeout(errorTimer);
    errorTimer = setTimeout(() => element.dataset.state === "error" && setState("idle"), ERROR_MS);
  };

  async function start() {
    clearTimeout(errorTimer);
    active?.cancel();
    setState("starting", "Разрешите доступ к микрофону…");
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    } catch (error) {
      showError(error);
      return;
    }
    let context;
    let analyser;
    let recorder;
    try {
      context = new AudioContextClass();
      context.resume(); // Safari starts a context created after an await suspended
      analyser = context.createAnalyser();
      analyser.fftSize = 1024;
      context.createMediaStreamSource(stream).connect(analyser);
      recorder = new MediaRecorder(stream);
    } catch (error) {
      stream.getTracks().forEach((track) => track.stop());
      context?.close();
      showError(error);
      return;
    }
    const chunks = [];
    recorder.addEventListener("dataavailable", (e) => e.data.size && chunks.push(e.data));
    const stopped = new Promise((resolve) => recorder.addEventListener("stop", resolve, { once: true }));
    const began = performance.now();
    const samples = new Float32Array(analyser.fftSize);
    const limit = setTimeout(() => finish(), MAX_SECONDS * 1000);
    let frame = 0;
    let level = 0;
    let shown = -1;
    const meter = () => {
      analyser.getFloatTimeDomainData(samples);
      const rms = Math.sqrt(samples.reduce((sum, s) => sum + s * s, 0) / samples.length);
      level = Math.max(Math.min(1, rms * 5), level * 0.85); // a quick rise and a slow fall
      element.style.setProperty("--level", level.toFixed(3));
      const seconds = Math.floor((performance.now() - began) / 1000);
      if (seconds !== shown) {
        shown = seconds;
        timer.textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")} / 1:00`;
      }
      frame = requestAnimationFrame(meter);
    };
    const release = () => {
      cancelAnimationFrame(frame);
      clearTimeout(limit);
      stream.getTracks().forEach((track) => track.stop());
      element.style.removeProperty("--level");
      active = null;
    };
    // A recorder whose microphone went away (unplugged, revoked) has stopped by itself,
    // and stop() would throw on it: the control would hang in "transcribing".
    const stop = () => recorder.state !== "inactive" && recorder.stop();
    session = {
      cancel: () => {
        stop();
        release();
        context.close();
        session = null;
        setState("idle");
      },
      finish: async () => {
        stop();
        await stopped;
        release();
        return { blob: new Blob(chunks, { type: recorder.mimeType }), seconds: (performance.now() - began) / 1000, context };
      },
    };
    active = session;
    recorder.start();
    setState("recording", "Идёт запись: говорите, затем нажмите ещё раз");
    meter();
  }

  async function finish() {
    const current = session;
    if (!current) return;
    session = null;
    if (active === current) active = null; // another microphone may start now; this one is stopping
    setState("transcribing", "Распознаю речь…");
    const { blob, seconds, context } = await current.finish();
    try {
      if (seconds < MIN_SECONDS) throw new Error("Запись слишком короткая — скажите хотя бы пару слов");
      const pcm = await resample(blob, context);
      if (!pcm.some((sample) => Math.abs(sample) > SILENCE)) throw new Error("Не слышно голоса — проверьте микрофон");
      const { text } = await api("/api/transcribe", { method: "POST", body: { audio: await base64(wav(pcm)) } });
      if (!text?.trim()) throw new Error("Не удалось разобрать речь, попробуйте ещё раз");
      insert(box, text.trim());
      setState("idle");
    } catch (error) {
      showError(error);
    } finally {
      context.close();
    }
  }

  button.addEventListener("click", () => {
    if (element.dataset.state === "recording") finish();
    else if (element.dataset.state === "idle" || element.dataset.state === "error") start();
  });
  // Escape drops the recording instead of sending it
  element.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && session) {
      e.preventDefault();
      session.cancel();
    }
  });
  return element;
}
