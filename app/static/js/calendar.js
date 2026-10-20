/*
 * A date-range calendar for the journey composer, and the Russian date wording the
 * planner shows around it and the cabinet shows on its cards («24–28 октября · 5 дней»,
 * «сб 24 окт»).
 *
 * Why our own: <input type="date"> picks one day, and two of them neither show the
 * range nor stop a trip at seven days; the browsers' pickers also differ (Yandex
 * Browser, Firefox, Safari) and cannot be styled. Here one month is a grid: the first
 * click sets the first day, and the trip's length then comes from the request («5 дней»);
 * the second click sets the last, so clicking one day twice means a one-day trip. Days
 * past the seven-day limit and days in the past cannot be picked.
 *
 * Keyboard: the grid has one tab stop (roving tabindex, the WAI-ARIA grid pattern);
 * arrows move by a day or a week, Home/End to the week's ends, PageUp/PageDown by a
 * month, Enter or Space picks, Escape keeps the range as it is.
 *
 * Days are counted in UTC day numbers, not local Dates: a daylight-saving switch can
 * neither skip nor repeat a day, and an ISO string converts back without a timezone.
 */

import { h, plural } from "./ui.js";

const DAY_MS = 86_400_000;
const MONTHS = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"];
const MONTHS_OF = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"];
const MONTHS_SHORT = ["янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"];
// Monday first, as Russian calendars are
const WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"];
const WEEKDAYS_FULL = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"];
const WEEKDAYS_SHORT = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"];

/* ---------- day numbers ---------- */

const dayOf = (iso) => Date.UTC(+iso.slice(0, 4), +iso.slice(5, 7) - 1, +iso.slice(8, 10)) / DAY_MS;
const isoOf = (day) => new Date(day * DAY_MS).toISOString().slice(0, 10);
const parts = (day) => {
  const date = new Date(day * DAY_MS);
  return { year: date.getUTCFullYear(), month: date.getUTCMonth(), date: date.getUTCDate(), weekday: (date.getUTCDay() + 6) % 7 };
};
const firstOfMonth = (year, month) => Date.UTC(year, month, 1) / DAY_MS;

/** Today in the traveller's own timezone, as an ISO date. */
function todayIso() {
  const now = new Date();
  return isoOf(Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()) / DAY_MS);
}

/** Days in an inclusive range: 24..28 is 5. */
export const daysInRange = (from, to) => dayOf(to) - dayOf(from) + 1;

/** The ISO date `offset` days after `iso`. */
export const addDays = (iso, offset) => isoOf(dayOf(iso) + offset);

/** "2026-10-24" -> "сб 24 окт": a day tab has room for little more. */
export function shortDay(iso) {
  const { month, date, weekday } = parts(dayOf(iso));
  return `${WEEKDAYS_SHORT[weekday]} ${date} ${MONTHS_SHORT[month]}`;
}

/** "24–28 октября", "30 октября – 2 ноября", "24 октября"; the year only when it is not the current one. */
export function rangeText(from, to = from) {
  const a = parts(dayOf(from));
  const b = parts(dayOf(to));
  if (a.year !== b.year) return `${a.date} ${MONTHS_OF[a.month]} ${a.year} – ${b.date} ${MONTHS_OF[b.month]} ${b.year}`;
  const year = a.year === new Date().getFullYear() ? "" : ` ${a.year}`;
  if (from === to) return `${a.date} ${MONTHS_OF[a.month]}${year}`;
  if (a.month !== b.month) return `${a.date} ${MONTHS_OF[a.month]} – ${b.date} ${MONTHS_OF[b.month]}${year}`;
  return `${a.date}–${b.date} ${MONTHS_OF[b.month]}${year}`;
}

/** "24–28 октября · 5 дней". */
export function rangeLabel(from, to = from) {
  const n = daysInRange(from, to);
  return `${rangeText(from, to)} · ${n} ${plural(n, "день", "дня", "дней")}`;
}

/* ---------- the calendar ---------- */

let instances = 0;

export class RangeCalendar {
  /**
   * `onChange({from, to} | null)` fires on every pick: after the first click only the
   * first day is set (`to` equals it, and `complete` is false), after the second the
   * range is complete.
   */
  constructor({ maxDays = 7, onChange = () => {} } = {}) {
    this.maxDays = maxDays;
    this.onChange = onChange;
    this.today = dayOf(todayIso());
    this.from = null; // day numbers
    this.to = null;
    this.picking = false; // the first day is set and the next click sets the last
    this.hover = null; // the day under the pointer while picking: the range preview
    this.focusDay = this.today;
    const { year, month } = parts(this.today);
    this.month = { year, month };
    this.cells = new Map(); // day number -> {cell, button}

    const id = `cal-${(instances += 1)}`;
    this.title = h("h3", { class: "cal-title", id: `${id}-title`, "aria-live": "polite" });
    this.prev = h("button", { class: "cal-nav", type: "button", "aria-label": "Предыдущий месяц", onclick: () => this.shiftMonth(-1) }, "‹");
    this.next = h("button", { class: "cal-nav", type: "button", "aria-label": "Следующий месяц", onclick: () => this.shiftMonth(1) }, "›");
    this.body = h("tbody");
    this.summary = h("p", { class: "cal-summary", "aria-live": "polite" });
    this.clearButton = h("button", { class: "cal-clear", type: "button", onclick: () => this.clear() }, "Сбросить");
    this.grid = h(
      "table",
      { class: "cal-grid", role: "grid", "aria-labelledby": `${id}-title` },
      h("thead", {}, h("tr", {}, WEEKDAYS.map((day, i) => h("th", { scope: "col", abbr: WEEKDAYS_FULL[i], class: i > 4 ? "is-weekend" : null }, day)))),
      this.body,
    );
    this.grid.addEventListener("keydown", (e) => this.onKey(e));
    this.grid.addEventListener("mouseleave", () => this.preview(null));
    this.element = h(
      "div",
      { class: "cal" },
      h("div", { class: "cal-head" }, this.prev, this.title, this.next),
      this.grid,
      h("div", { class: "cal-foot" }, this.summary, this.clearButton),
    );
    this.build();
  }

  /** The chosen range as ISO dates, or null when no day is chosen. */
  get value() {
    return this.from === null ? null : { from: isoOf(this.from), to: isoOf(this.to) };
  }

  /** The last day is picked too, not only the first. */
  get complete() {
    return this.from !== null && !this.picking;
  }

  /** Show a range (a saved one, say) without firing onChange; null clears, and a range
   * without `to` has only its first day picked. */
  set(range) {
    this.picking = Boolean(range && !range.to);
    this.hover = null;
    if (!range) {
      this.from = this.to = null;
    } else {
      this.from = dayOf(range.from);
      this.to = dayOf(range.to ?? range.from);
      this.focusDay = this.from;
      const { year, month } = parts(this.from);
      this.month = { year, month };
    }
    this.build();
  }

  clear() {
    this.set(null);
    this.onChange(null);
  }

  /* ---------- picking ---------- */

  /** Why a day cannot be picked right now, or null. */
  blocked(day) {
    if (day < this.today) return "прошедший день";
    if (this.picking && day > this.from + this.maxDays - 1) return `больше ${this.maxDays} дней`;
    return null;
  }

  pick(day) {
    if (this.blocked(day)) return;
    if (!this.picking || day < this.from) {
      this.from = this.to = day; // a new range starts here
      this.picking = true;
    } else {
      this.to = day;
      this.picking = false;
    }
    this.hover = null;
    this.focusDay = day;
    this.paint();
    this.onChange(this.value);
  }

  preview(day) {
    if (!this.picking || this.hover === day) return;
    this.hover = day;
    this.paint();
  }

  /* ---------- rendering ---------- */

  shiftMonth(step) {
    // PageUp reaches here past the disabled button: a month before today's has no day to
    // pick, and its grid would have no tab stop left for the keyboard
    if (step < 0 && this.prev.disabled) return;
    const first = Date.UTC(this.month.year, this.month.month + step, 1) / DAY_MS;
    const { year, month } = parts(first);
    this.month = { year, month };
    // keep the keyboard on the same date of the new month, clamped to its length and to today
    const { date } = parts(this.focusDay);
    const last = Date.UTC(year, month + 1, 0) / DAY_MS;
    this.focusDay = Math.max(this.today, Math.min(first + date - 1, last));
    this.build();
  }

  build() {
    const { year, month } = this.month;
    const first = firstOfMonth(year, month);
    const last = Date.UTC(year, month + 1, 0) / DAY_MS;
    this.title.textContent = `${MONTHS[month]} ${year}`;
    this.prev.disabled = first <= this.today;
    if (this.focusDay < first || this.focusDay > last) this.focusDay = Math.max(first, this.today);
    this.cells.clear();
    const rows = [];
    let row = [];
    for (let i = 0; i < parts(first).weekday; i += 1) row.push(h("td", { role: "gridcell" }));
    for (let day = first; day <= last; day += 1) {
      const { date, weekday } = parts(day);
      const button = h(
        "button",
        {
          type: "button",
          class: "cal-day",
          tabindex: "-1",
          onclick: () => this.pick(day),
          onmouseenter: () => this.preview(day),
          onfocus: () => this.preview(day),
        },
        String(date),
      );
      const cell = h("td", { role: "gridcell", class: weekday > 4 ? "is-weekend" : null }, button);
      this.cells.set(day, { cell, button, label: `${WEEKDAYS_FULL[weekday]}, ${date} ${MONTHS_OF[month]} ${year}` });
      row.push(cell);
      if (weekday === 6) {
        rows.push(h("tr", {}, row));
        row = [];
      }
    }
    if (row.length) {
      while (row.length < 7) row.push(h("td", { role: "gridcell" }));
      rows.push(h("tr", {}, row));
    }
    this.body.replaceChildren(...rows);
    this.paint();
  }

  /** Classes and states only: rebuilding under the pointer would swallow the click. */
  paint() {
    const end = this.picking && this.hover !== null && this.hover >= this.from && !this.blocked(this.hover) ? this.hover : this.to;
    for (const [day, { cell, button, label }] of this.cells) {
      const inRange = this.from !== null && day >= this.from && day <= end;
      const blocked = this.blocked(day);
      cell.setAttribute("aria-selected", String(inRange));
      cell.classList.toggle("in-range", inRange);
      cell.classList.toggle("is-from", day === this.from);
      cell.classList.toggle("is-to", this.from !== null && day === end);
      cell.classList.toggle("is-preview", inRange && this.picking && day > this.to);
      button.classList.toggle("is-today", day === this.today);
      button.setAttribute("aria-disabled", String(Boolean(blocked)));
      button.setAttribute("aria-label", blocked ? `${label}, нельзя выбрать: ${blocked}` : label);
      button.tabIndex = day === this.focusDay ? 0 : -1;
    }
    this.summary.textContent = this.summaryText();
    this.clearButton.hidden = this.from === null;
  }

  summaryText() {
    if (this.from === null) return "Даты не выбраны";
    // only the first day yet: not «1 день», the request may still set the length
    if (this.picking) return `с ${rangeText(isoOf(this.from))} — теперь последний день`;
    return rangeLabel(isoOf(this.from), isoOf(this.to));
  }

  onKey(e) {
    const steps = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -7, ArrowDown: 7 };
    let day = this.focusDay;
    const { weekday } = parts(day);
    if (e.key in steps) day += steps[e.key];
    else if (e.key === "Home") day -= weekday;
    else if (e.key === "End") day += 6 - weekday;
    else if (e.key === "PageUp" || e.key === "PageDown") {
      e.preventDefault();
      this.shiftMonth(e.key === "PageUp" ? -1 : 1);
      this.cells.get(this.focusDay)?.button.focus();
      return;
    } else if (e.key === "Escape" && this.picking) {
      e.preventDefault();
      this.picking = false;
      this.hover = null;
      this.paint();
      this.onChange(this.value); // the range is complete now: one day
      return;
    } else return; // Enter and Space reach the focused button as a click
    e.preventDefault();
    this.moveFocus(Math.max(day, this.today));
  }

  moveFocus(day) {
    this.focusDay = day;
    const { year, month } = parts(day);
    if (year !== this.month.year || month !== this.month.month) {
      this.month = { year, month };
      this.build();
    } else {
      this.paint();
    }
    this.cells.get(day)?.button.focus();
  }
}
