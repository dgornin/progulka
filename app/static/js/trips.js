/*
 * The cabinet: every journey the traveller planned, most recently edited first. A card
 * opens the journey in the planner (/app/trips/{id}), where any day can be edited further.
 *
 * Each card carries a small sketch seeded by the journey id: one dotted route per day in
 * that day's colour, all leaving the same start, the way the days of a trip leave the
 * same hotel. It is not the real geometry (the list does not load it, and should stay
 * light), but a stable picture that tells journeys apart and shows how long each is.
 */

import { api, requireUser } from "./api.js";
import { rangeText } from "./calendar.js";
import { renderNav } from "./nav.js";
import { ago, confirmDialog, h, plural, svg, toast } from "./ui.js";

const list = document.getElementById("trips");
const count = document.getElementById("trips-count");
let journeys = [];

const km = (value) => value.toLocaleString("ru-RU", { maximumFractionDigits: 1 });
const titleOf = (journey) => journey.title || `Путешествие: ${journey.city}`;
const daysLabel = (n) => `${n} ${plural(n, "день", "дня", "дней")}`;
const dayColour = (index) => `--day: var(--day-${index + 1})`;

/* ---------- the sketch ---------- */

const [WIDTH, HEIGHT, MARGIN] = [320, 128, 18];

/** FNV-1a of the id seeds a xorshift generator: the same journey always gets the same sketch. */
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

const clamp = (value, low, high) => Math.round(Math.min(Math.max(value, low), high));
const inside = ([x, y]) => [clamp(x, MARGIN, WIDTH - MARGIN), clamp(y, MARGIN, HEIGHT - MARGIN)];

/**
 * One day's route from the start along `angle`. A single day crosses the whole card;
 * several days fan out, stretched sideways because the card is wide and low.
 */
function dayRoute(random, start, angle, stops, single) {
  const [dx, dy] = [Math.cos(angle), Math.sin(angle)];
  const length = single ? WIDTH - 2 * MARGIN : (HEIGHT / 2 - MARGIN) * (0.75 + random() * 0.25);
  const [stretch, wobble] = single ? [1, HEIGHT / 2 - MARGIN] : [3.2, 14];
  const points = Array.from({ length: stops }, (_, k) => {
    const along = ((k + 1) / stops) * length;
    const aside = (random() - 0.5) * 2 * wobble;
    return inside([start[0] + dx * along * stretch - dy * aside, start[1] + dy * along + dx * aside]);
  });
  return [start, ...points];
}

function sketch(journey) {
  const random = seeded(journey.id);
  const { days } = journey;
  const single = days === 1;
  const stops = Math.min(Math.max(Math.round(journey.stops / days), 2), single ? 6 : 4);
  const start = single
    ? [MARGIN, Math.round(MARGIN + random() * (HEIGHT - 2 * MARGIN))]
    : [Math.round(WIDTH * (0.4 + random() * 0.2)), Math.round(HEIGHT * (0.4 + random() * 0.2))];
  const turn = random() * 2 * Math.PI;
  const dot = ([cx, cy], cls, r) => svg("circle", { class: cls, cx, cy, r });
  const routes = Array.from({ length: days }, (_, i) => {
    const points = dayRoute(random, start, single ? 0 : turn + (i * 2 * Math.PI) / days, stops, single);
    return svg(
      "g",
      { style: dayColour(i) },
      svg("path", { class: "trip-path", d: smoothPath(points) }),
      points.slice(1, -1).map((point) => dot(point, "trip-stop", 4.5)),
      dot(points.at(-1), "trip-end", 6.5),
    );
  });
  // day 1 is drawn last, on top: it is the colour the journey is known by
  return svg("svg", { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, "aria-hidden": "true" }, routes.reverse(), dot(start, "trip-start", 7.5));
}

/* ---------- cards ---------- */

function journeyCard(journey) {
  const title = titleOf(journey);
  const card = h(
    "article",
    { class: "trip-card" },
    h("div", { class: "trip-sketch" }, sketch(journey)),
    h(
      "div",
      { class: "trip-body" },
      h("p", { class: "trip-meta" }, journey.city),
      h("h2", { class: "trip-title" }, h("a", { href: `/app/trips/${encodeURIComponent(journey.id)}` }, title)),
      h(
        "p",
        { class: "trip-dates" },
        h("span", {}, rangeText(journey.date_from, journey.date_to)),
        h(
          "span",
          { class: "trip-days" },
          h("span", { class: "trip-ticks", "aria-hidden": "true" }, Array.from({ length: journey.days }, (_, i) => h("i", { style: dayColour(i) }))),
          daysLabel(journey.days),
        ),
      ),
      h(
        "p",
        { class: "trip-stats" },
        h("span", {}, `${journey.stops} ${plural(journey.stops, "остановка", "остановки", "остановок")}`),
        h("span", {}, `${km(journey.walk_km)} км пешком`),
      ),
    ),
    h(
      "footer",
      { class: "trip-foot" },
      h("span", { title: new Date(journey.updated_at).toLocaleString("ru-RU") }, `изменено ${ago(journey.updated_at)}`),
      h(
        "button",
        { class: "btn btn-ghost btn-sm trip-delete", type: "button", "aria-label": `Удалить путешествие «${title}»` },
        "Удалить",
      ),
    ),
  );
  card.querySelector(".trip-delete").addEventListener("click", () => remove(journey, card));
  return card;
}

function renderCount() {
  const n = journeys.length;
  count.textContent = n ? `${n} ${plural(n, "путешествие", "путешествия", "путешествий")}` : "";
}

function showEmpty() {
  list.replaceChildren(
    h(
      "div",
      { class: "card empty trips-empty" },
      h("h2", {}, "Куда поедем?"),
      h(
        "p",
        {},
        "Назовите город и даты — агент соберёт маршрут на каждый день поездки из реальных мест, и путешествие сохранится здесь.",
      ),
      h("a", { class: "btn btn-primary btn-lg", href: "/app/new" }, "Спланировать первую поездку"),
    ),
  );
}

function render() {
  renderCount();
  if (journeys.length) list.replaceChildren(...journeys.map(journeyCard));
  else showEmpty();
}

async function remove(journey, card) {
  const confirmed = await confirmDialog({
    title: "Удалить путешествие?",
    text: `«${titleOf(journey)}» пропадёт из кабинета со всеми днями и историей правок. Вернуть его не получится.`,
  });
  if (!confirmed) return;
  try {
    await api(`/api/journeys/${encodeURIComponent(journey.id)}`, { method: "DELETE" });
  } catch (error) {
    toast(error.message, "error");
    return;
  }
  journeys = journeys.filter((item) => item.id !== journey.id);
  renderCount();
  card.classList.add("is-removing");
  // The focused delete button goes with its card: a keyboard user continues from the
  // neighbouring journey (or the empty state's button) instead of the top of the page.
  const neighbour = card.nextElementSibling ?? card.previousElementSibling;
  // Let the card fade before the grid closes the gap; only this card goes, so the
  // others do not replay their entrance.
  setTimeout(() => {
    if (journeys.length) {
      card.remove();
      neighbour?.querySelector(".trip-title a")?.focus();
    } else {
      showEmpty();
      list.querySelector(".btn")?.focus();
    }
  }, 220);
  toast("Путешествие удалено", "ok");
}

async function load() {
  list.setAttribute("aria-busy", "true");
  try {
    journeys = await api("/api/journeys");
    journeys.sort((a, b) => b.updated_at.localeCompare(a.updated_at));
    render();
  } catch (error) {
    count.textContent = "";
    const retry = h("button", { class: "btn btn-sm", type: "button", onclick: load }, "Повторить");
    list.replaceChildren(h("div", { class: "alert is-error trips-error" }, `Не удалось загрузить путешествия: ${error.message} `, retry));
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
