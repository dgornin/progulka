/*
 * The cabinet: every walk the traveller planned, most recently edited first. A card
 * opens the trip in the planner (/app/trips/{id}), where it can be edited further.
 *
 * Each card carries a small dotted-route sketch seeded by the trip id. It is not the
 * real route (the list does not load geometry, and should stay light), but a stable
 * picture that makes trips easy to tell apart at a glance.
 */

import { api, requireUser } from "./api.js";
import { renderNav } from "./nav.js";
import { ago, confirmDialog, dayLabel, h, plural, svg, toast } from "./ui.js";

const list = document.getElementById("trips");
const count = document.getElementById("trips-count");
let trips = [];

const km = (value) => value.toLocaleString("ru-RU", { maximumFractionDigits: 1 });
const titleOf = (trip) => trip.title || `Прогулка: ${trip.city}`;

/* ---------- the sketch ---------- */

/** FNV-1a of the id seeds a xorshift generator: the same trip always gets the same sketch. */
function seeded(text) {
  let state = 2166136261;
  for (const char of text) {
    state ^= char.codePointAt(0);
    state = Math.imul(state, 16777619);
  }
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
}

/** A smooth curve through the points (Catmull-Rom turned into cubic Béziers). */
function smoothPath(points) {
  let d = `M${points[0].join(" ")}`;
  for (let i = 0; i < points.length - 1; i++) {
    const [p0, p1, p2, p3] = [points[i - 1] ?? points[i], points[i], points[i + 1], points[i + 2] ?? points[i + 1]];
    const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
    const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
    d += ` C${c1.map(Math.round).join(" ")} ${c2.map(Math.round).join(" ")} ${p2.join(" ")}`;
  }
  return d;
}

function sketch(trip) {
  const random = seeded(trip.id);
  const [width, height, margin] = [320, 112, 22];
  const n = Math.min(Math.max(trip.stops, 3), 6) + 1; // the start plus a few stops
  const step = (width - 2 * margin) / (n - 1);
  const points = Array.from({ length: n }, (_, i) => [
    Math.round(margin + i * step + (i && i < n - 1 ? (random() - 0.5) * step * 0.5 : 0)),
    Math.round(margin + random() * (height - 2 * margin)),
  ]);
  const dot = ([cx, cy], cls, r) => svg("circle", { class: cls, cx, cy, r });
  return svg(
    "svg",
    { viewBox: `0 0 ${width} ${height}`, "aria-hidden": "true" },
    svg("path", { class: "trip-path", d: smoothPath(points) }),
    points.slice(1, -1).map((point) => dot(point, "trip-stop", 5)),
    dot(points[0], "trip-start", 7),
    dot(points.at(-1), "trip-end", 7),
  );
}

/* ---------- cards ---------- */

function tripCard(trip) {
  const title = titleOf(trip);
  const card = h(
    "article",
    { class: "trip-card" },
    h("div", { class: "trip-sketch" }, sketch(trip)),
    h(
      "div",
      { class: "trip-body" },
      h("p", { class: "trip-meta" }, [trip.city, dayLabel(trip.date)].join(" · ")),
      h("h2", { class: "trip-title" }, h("a", { href: `/app/trips/${encodeURIComponent(trip.id)}` }, title)),
      h(
        "p",
        { class: "trip-stats" },
        h("span", {}, `${trip.stops} ${plural(trip.stops, "остановка", "остановки", "остановок")}`),
        h("span", {}, `${km(trip.walk_km)} км пешком`),
      ),
    ),
    h(
      "footer",
      { class: "trip-foot" },
      h("span", { title: new Date(trip.updated_at).toLocaleString("ru-RU") }, `изменена ${ago(trip.updated_at)}`),
      h(
        "button",
        { class: "btn btn-ghost btn-sm trip-delete", type: "button", "aria-label": `Удалить прогулку «${title}»` },
        "Удалить",
      ),
    ),
  );
  card.querySelector(".trip-delete").addEventListener("click", () => remove(trip, card));
  return card;
}

function renderCount() {
  const n = trips.length;
  count.textContent = n ? `${n} ${plural(n, "прогулка", "прогулки", "прогулок")}` : "";
}

function showEmpty() {
  list.replaceChildren(
    h(
      "div",
      { class: "card empty trips-empty" },
      h("h2", {}, "Здесь будут ваши прогулки"),
      h(
        "p",
        {},
        "Опишите день словами или отметьте, чего хочется, — агент соберёт маршрут из реальных мест, и он сохранится здесь.",
      ),
      h("a", { class: "btn btn-primary btn-lg", href: "/app/new" }, "Спланировать первую прогулку"),
    ),
  );
}

function render() {
  renderCount();
  if (trips.length) list.replaceChildren(...trips.map(tripCard));
  else showEmpty();
}

async function remove(trip, card) {
  const confirmed = await confirmDialog({
    title: "Удалить прогулку?",
    text: `«${titleOf(trip)}» пропадёт из кабинета вместе с историей правок. Вернуть её не получится.`,
  });
  if (!confirmed) return;
  try {
    await api(`/api/trips/${encodeURIComponent(trip.id)}`, { method: "DELETE" });
  } catch (error) {
    toast(error.message, "error");
    return;
  }
  trips = trips.filter((item) => item.id !== trip.id);
  renderCount();
  card.classList.add("is-removing");
  // Let the card fade before the grid closes the gap; only this card goes, so the
  // others do not replay their entrance.
  setTimeout(() => (trips.length ? card.remove() : showEmpty()), 220);
  toast("Прогулка удалена", "ok");
}

async function load() {
  list.setAttribute("aria-busy", "true");
  try {
    trips = await api("/api/trips");
    trips.sort((a, b) => b.updated_at.localeCompare(a.updated_at));
    render();
  } catch (error) {
    count.textContent = "";
    const retry = h("button", { class: "btn btn-sm", type: "button", onclick: load }, "Повторить");
    list.replaceChildren(h("div", { class: "alert is-error trips-error" }, `Не удалось загрузить прогулки: ${error.message} `, retry));
  } finally {
    list.removeAttribute("aria-busy");
  }
}

try {
  renderNav(await requireUser(), "trips");
  await load();
} catch (error) {
  list.replaceChildren(h("div", { class: "alert is-error" }, error.message));
}
