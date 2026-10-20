/*
 * DOM and feedback helpers shared by every page: element builders, toasts, a confirm
 * dialog and the Russian formatting several pages need.
 *
 * Pages build markup with h() so that outside text (place names, user names, memory
 * facts) always lands in text nodes: nothing in the frontend assigns innerHTML.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

function fill(el, attrs, children) {
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else if (key === "class") el.setAttribute("class", value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child);
  }
  return el;
}

/** h("a", {href, class, onclick}, "text", node) — strings become text nodes; null/false are skipped. */
export function h(tag, attrs = {}, ...children) {
  return fill(document.createElement(tag), attrs, children);
}

/** The same for SVG elements, which need their namespace. */
export function svg(tag, attrs = {}, ...children) {
  return fill(document.createElementNS(SVG_NS, tag), attrs, children);
}

/* ---------- feedback ---------- */

let toastRegion = null;

/** A short message at the bottom of the screen; kind is "info", "ok" or "error". */
export function toast(message, kind = "info") {
  if (!toastRegion) {
    toastRegion = h("div", { class: "toasts", role: "status", "aria-live": "polite" });
    document.body.append(toastRegion);
  }
  const item = h("div", { class: `toast is-${kind}` }, message);
  toastRegion.append(item);
  setTimeout(() => {
    item.classList.add("is-leaving");
    setTimeout(() => item.remove(), 250);
  }, kind === "error" ? 6000 : 3200);
}

/** A modal "are you sure?"; resolves to true only when the action is confirmed. */
export function confirmDialog({ title, text = "", confirmLabel = "Удалить", danger = true }) {
  return new Promise((resolve) => {
    const confirm = h("button", { class: `btn ${danger ? "btn-danger" : "btn-primary"}`, value: "yes" }, confirmLabel);
    const dialog = h(
      "dialog",
      { class: "dialog", "aria-labelledby": "dialog-title" },
      h(
        "form",
        { method: "dialog" },
        h("h2", { id: "dialog-title" }, title),
        text ? h("p", {}, text) : null,
        h("div", { class: "dialog-actions" }, h("button", { class: "btn btn-ghost", value: "no" }, "Отмена"), confirm),
      ),
    );
    dialog.addEventListener("close", () => {
      resolve(dialog.returnValue === "yes");
      dialog.remove();
    });
    document.body.append(dialog);
    dialog.showModal();
    confirm.focus();
  });
}

/* ---------- Russian formatting ---------- */

export function plural(n, one, few, many) {
  const tens = n % 100;
  const units = n % 10;
  if (units === 1 && tens !== 11) return one;
  if (units >= 2 && units <= 4 && (tens < 12 || tens > 14)) return few;
  return many;
}

/** A calendar date at local noon, so no timezone can shift it to the day before. */
const noon = (isoDate) => new Date(`${isoDate.slice(0, 10)}T12:00:00`);

/** "2026-10-24" -> "сб, 24 октября". */
export function dayLabel(isoDate) {
  return noon(isoDate).toLocaleDateString("ru-RU", { weekday: "short", day: "numeric", month: "long" });
}

const relative = new Intl.RelativeTimeFormat("ru", { numeric: "auto" });

/** An ISO timestamp as "5 минут назад", "вчера" or, past a week, "12 октября". */
export function ago(iso) {
  const then = new Date(iso);
  const seconds = (then.getTime() - Date.now()) / 1000;
  const minutes = Math.round(seconds / 60);
  if (Math.abs(seconds) < 60) return "только что";
  if (Math.abs(minutes) < 60) return relative.format(minutes, "minute");
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24) return relative.format(hours, "hour");
  const days = Math.round(hours / 24);
  if (Math.abs(days) < 7) return relative.format(days, "day");
  return then.toLocaleDateString("ru-RU", { day: "numeric", month: "long" });
}

/** Dollars with two significant digits, so a cheap run reads "$0.00042" rather than "$0.00". */
export function usd(value) {
  if (!value) return "$0";
  return value >= 1 ? `$${value.toFixed(2)}` : `$${Number(value.toPrecision(2))}`;
}
