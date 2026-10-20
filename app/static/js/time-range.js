/*
 * The daily time window of a journey: one track from 06:00 to 24:00, two handles,
 * half-hour steps. Every day of the trip is planned inside the same window.
 *
 * Two native range inputs share the track. Each keeps the browser's own keyboard
 * handling, focus and screen-reader value (aria-valuetext reads «10:00», not «600»);
 * only their thumbs take the pointer, so both stay draggable where they meet. A press
 * on the bare track moves the nearer handle there and keeps dragging it. The handles
 * stay an hour apart: one pushes the other rather than stopping, so a window can be
 * moved in one gesture.
 */

import { h } from "./ui.js";

const DAY_END = 24 * 60;
const SCALE = [6, 9, 12, 15, 18, 21, 24]; // hour labels under the track

const pad = (n) => String(n).padStart(2, "0");

/** 630 -> "10:30"; 1440 -> "24:00". */
export const clock = (minutes) => `${pad(Math.floor(minutes / 60))}:${pad(minutes % 60)}`;

/** What the server gets: its time type has no 24:00, so midnight is the day's last minute. */
export const wireTime = (minutes) => (minutes >= DAY_END ? "23:59" : clock(minutes));

/** "10:00", "10:00:00" or "23:59" -> minutes, snapped to the slider's half hour. */
export function minutesOf(time, step = 30) {
  const [hours, minutes] = time.split(":").map(Number);
  return Math.round((hours * 60 + minutes) / step) * step;
}

/** 480 -> "8 ч", 450 -> "7 ч 30 мин". */
export function spanText(minutes) {
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  if (!hours) return `${rest} мин`;
  return rest ? `${hours} ч ${rest} мин` : `${hours} ч`;
}

export class TimeRange {
  /** `onChange({start, end})` gets minutes since midnight on every move. */
  constructor({ min = 6 * 60, max = DAY_END, step = 30, gap = 60, value = [10 * 60, 18 * 60], label = "Время", onChange = () => {} } = {}) {
    Object.assign(this, { min, max, step, gap, onChange });
    this.output = h("output", { class: "trange-value" });
    const input = (name, edge) => {
      const element = h("input", {
        class: `trange-input is-${edge}`,
        type: "range",
        min: String(min),
        max: String(max),
        step: String(step),
        "aria-label": name,
      });
      element.addEventListener("input", () => this.move(edge, Number(element.value)));
      return element;
    };
    this.startInput = input("Начало дня", "start");
    this.endInput = input("Конец дня", "end");
    this.track = h(
      "div",
      { class: "trange-track" },
      h("div", { class: "trange-rail" }),
      h("div", { class: "trange-fill" }),
      this.startInput,
      this.endInput,
    );
    this.track.addEventListener("pointerdown", (e) => this.grab(e));
    const scale = SCALE.map((hour) =>
      h("span", { style: `--at: ${(hour * 60 - min) / (max - min)}` }, pad(hour)),
    );
    this.element = h(
      "div",
      { class: "trange", role: "group", "aria-label": label },
      h("div", { class: "trange-head" }, h("span", { class: "trange-label" }, label), this.output),
      this.track,
      h("div", { class: "trange-scale", "aria-hidden": "true" }, scale),
    );
    this.set(...value);
  }

  get value() {
    return { start: this.start, end: this.end };
  }

  /** Show a window without firing onChange. */
  set(start, end) {
    this.start = Math.max(this.min, Math.min(start, this.max - this.gap));
    this.end = Math.min(this.max, Math.max(end, this.start + this.gap));
    this.paint();
  }

  move(edge, minutes) {
    if (edge === "start") {
      this.start = Math.min(minutes, this.max - this.gap);
      this.end = Math.max(this.end, this.start + this.gap);
    } else {
      this.end = Math.max(minutes, this.min + this.gap);
      this.start = Math.min(this.start, this.end - this.gap);
    }
    // the handle in use stays on top where the two overlap
    this.track.dataset.top = edge;
    this.paint();
    this.onChange(this.value);
  }

  /** A press on the track, not on a thumb: the nearer handle jumps there and follows the pointer. */
  grab(e) {
    if (e.button !== 0 || e.target instanceof HTMLInputElement) return; // a thumb drags natively
    const at = (event) => {
      const box = this.track.getBoundingClientRect();
      const thumb = this.startInput.offsetHeight; // the thumb is as wide as the input is high
      const share = (event.clientX - box.left - thumb / 2) / (box.width - thumb);
      const minutes = this.min + Math.min(1, Math.max(0, share)) * (this.max - this.min);
      return Math.round(minutes / this.step) * this.step;
    };
    const target = at(e);
    const edge = Math.abs(target - this.start) <= Math.abs(target - this.end) ? "start" : "end";
    const input = edge === "start" ? this.startInput : this.endInput;
    e.preventDefault();
    input.focus();
    this.move(edge, target);
    this.track.setPointerCapture(e.pointerId);
    const follow = (event) => this.move(edge, at(event));
    const release = () => {
      this.track.removeEventListener("pointermove", follow);
      this.track.removeEventListener("pointerup", release);
      this.track.removeEventListener("pointercancel", release);
    };
    this.track.addEventListener("pointermove", follow);
    this.track.addEventListener("pointerup", release);
    this.track.addEventListener("pointercancel", release);
  }

  paint() {
    const share = (minutes) => (minutes - this.min) / (this.max - this.min);
    this.startInput.value = String(this.start);
    this.endInput.value = String(this.end);
    this.startInput.setAttribute("aria-valuetext", clock(this.start));
    this.endInput.setAttribute("aria-valuetext", clock(this.end));
    this.track.style.setProperty("--a", String(share(this.start)));
    this.track.style.setProperty("--b", String(share(this.end)));
    this.output.replaceChildren(
      h("span", { class: "trange-clock" }, `${clock(this.start)} — ${clock(this.end)},`),
      h("span", { class: "trange-span" }, spanText(this.end - this.start)),
    );
  }
}
