/*
 * Прогулка — the web UI.
 *
 * The server streams the agent's work as NDJSON: one AgentEvent {type, data} per line.
 * Every event type has one handler below that turns it into something visible: a stage
 * strip, a live trace of tool calls and shortlisted places, then the route on a Leaflet
 * map and as a timeline. The page keeps no plan logic of its own — the server owns the
 * plan, the page only remembers the last session id — so a feedback run is just another
 * stream rendered by the same handlers.
 *
 * Text from outside (place names, Wikipedia extracts, LLM output) only ever reaches the
 * DOM through text nodes, never innerHTML, so a place name cannot inject markup.
 */

const $ = (selector) => document.querySelector(selector);

const SOURCES = { osm: "OpenStreetMap", wikipedia: "Википедия", wikivoyage: "Wikivoyage" };

// Stage names in pipeline order with short labels for the strip; the server sends a longer
// human label with each stage, which goes into the trace and the status line.
const STAGES = [
  ["feedback", "Отзыв"],
  ["brief", "Запрос"],
  ["weather", "Погода"],
  ["research", "Поиск"],
  ["optimize", "Расписание"],
  ["route", "Пути"],
  ["validate", "Проверка"],
  ["repair", "Правка"],
  ["write", "Тексты"],
];
// Stages a fresh plan always goes through; "repair" and the stages of a feedback run
// appear in the strip only once the server actually starts them.
const PLAN_STAGES = ["brief", "weather", "research", "optimize", "route", "validate", "write"];

// Tool names are matched by keyword so the icons survive renames on the agent side.
const TOOL_KINDS = [
  [/voyage/i, "WV", "src-wikivoyage", "Wikivoyage"],
  [/wiki/i, "W", "src-wikipedia", "Википедия"],
  [/osm|overpass/i, "OSM", "src-osm", "OpenStreetMap"],
  [/detail/i, "i", "tool-details", "Подробности о месте"],
  [/shortlist/i, "★", "tool-shortlist", "В шорт-лист"],
  [/remove/i, "−", "tool-shortlist", "Из шорт-листа"],
  [/finish/i, "✓", "tool-finish", "Конец поиска"],
];

// Place.kind values the sources produce: OSM categories, "landmark" for Wikipedia
// articles and the Wikivoyage section kinds.
const KINDS = {
  museum: "музей", gallery: "галерея", artwork: "арт-объект", monument: "памятник",
  memorial: "мемориал", historic: "история", architecture: "архитектура", church: "храм",
  viewpoint: "смотровая", park: "парк", garden: "сад", attraction: "достопримечательность",
  theatre: "театр", cinema: "кино", library: "библиотека", bookshop: "книжный",
  market: "рынок", cafe: "кафе", restaurant: "ресторан", street_food: "уличная еда",
  bakery: "пекарня", bar: "бар", pub: "паб", ice_cream: "мороженое", zoo: "зоопарк",
  aquarium: "океанариум", playground: "детская площадка", fountain: "фонтан",
  landmark: "достопримечательность", sight: "достопримечательность", activity: "досуг",
  food: "еда", shop: "магазин",
};

const VIOLATIONS = {
  closed_at_visit: "Закрыто",
  over_time: "Не успеваем",
  unknown_hours: "Часы неизвестны",
  rain_outdoor: "Дождь",
  avoid_violation: "Против пожеланий",
  long_walk: "Долгий переход",
  uncovered_interest: "Интерес не покрыт",
  too_few_stops: "Мало остановок",
};

const PACE = { relaxed: "неспешно", active: "активный темп" };

const state = {
  busy: false,
  sessionId: null,
  run: null, // bookkeeping of the stream being rendered, see beginRun()
  placeNames: new Map(), // place_id -> name, to name the places that did not fit
  markers: new Map(), // place_id -> Leaflet marker
  candidates: new Map(), // place_id -> live marker of a shortlisted place (see map section)
  cards: new Map(), // place_id -> timeline item
};

const ui = {
  composer: $("#composer"),
  request: $("#request"),
  build: $("#build"),
  steps: $("#steps"),
  status: $("#status"),
  hint: $("#trace-hint"),
  log: $("#log"),
  plan: $("#plan"),
  feedback: $("#feedback"),
  feedbackText: $("#feedback-text"),
  reply: $("#reply"),
  health: $("#health"),
};

/* ---------- small helpers ---------- */

/** h("a", {href, class}, "text", node) — strings become text nodes; null/false are skipped. */
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key === "class") el.className = value;
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child);
  }
  return el;
}

const clip = (text, max) => (text.length > max ? `${text.slice(0, max - 1)}…` : text);
const hhmm = (iso) => iso.slice(11, 16); // "2026-10-24T11:12:00" -> "11:12", no timezone games
const clock = (time) => time.slice(0, 5); // "11:00:00" -> "11:00"
const visitTime = (stop) => `${hhmm(stop.arrive)}–${hhmm(stop.depart)}`;
const timeWindow = (brief) => `${clock(brief.start_time)}–${clock(brief.end_time)}`;
const km = (meters) => (meters / 1000).toLocaleString("ru-RU", { maximumFractionDigits: 1 });
const distance = (meters) => (meters < 1000 ? `${meters} м` : `${km(meters)} км`);
const seconds = (ms) => `${(ms / 1000).toLocaleString("ru-RU", { maximumFractionDigits: 1 })} с`;

function plural(n, one, few, many) {
  const tens = n % 100;
  const units = n % 10;
  if (units === 1 && tens !== 11) return one;
  if (units >= 2 && units <= 4 && (tens < 12 || tens > 14)) return few;
  return many;
}

function duration(minutes) {
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  if (!hours) return `${rest} мин`;
  return rest ? `${hours} ч ${rest} мин` : `${hours} ч`;
}

function dayLabel(isoDate) {
  const day = new Date(`${isoDate}T12:00:00`);
  return day.toLocaleDateString("ru-RU", { weekday: "short", day: "numeric", month: "long" });
}

function usageText(usage) {
  const calls = usage.llm_calls ?? 0;
  const tokens = (usage.prompt_tokens ?? 0) + (usage.completion_tokens ?? 0);
  const parts = [`${calls} ${plural(calls, "вызов", "вызова", "вызовов")} LLM`];
  if (usage.cached_calls) parts.push(`${usage.cached_calls} из кэша`);
  parts.push(`${tokens.toLocaleString("ru-RU")} токенов`);
  // two significant digits, so a cheap run reads "$0.00042" rather than "$0.000"
  if (usage.cost_usd != null) parts.push(`$${Number(usage.cost_usd.toPrecision(2))}`);
  return parts.join(" · ");
}

const kindLabel = (kind) => KINDS[kind] ?? kind.replaceAll("_", " ");
const tag = (text) => h("span", { class: "tag" }, text);
const badge = (glyph, cls = "", title = null) => h("span", { class: `badge ${cls}`, title }, glyph);

function sourceLink(place) {
  const label = SOURCES[place.source] ?? place.source;
  if (!place.url || !/^https?:\/\//.test(place.url)) return h("span", { class: "source" }, label);
  return h("a", { class: "source", href: place.url, target: "_blank", rel: "noopener noreferrer" }, `${label} ↗`);
}

function violationItem(v) {
  return h("li", { class: `violation is-${v.severity}` }, h("b", {}, VIOLATIONS[v.code] ?? v.code), " ", v.message);
}

/* ---------- streaming ---------- */

function parseLine(line) {
  try {
    return JSON.parse(line);
  } catch {
    return { type: "warning", data: { message: `Непонятная строка от сервера: ${clip(line, 120)}` } };
  }
}

/** FastAPI explains a refusal in {"detail": ...}: a string, or a list of validation errors. */
function errorDetail(text) {
  try {
    const { detail } = JSON.parse(text);
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((problem) => problem.msg).join("; ");
  } catch {
    // not JSON (a proxy's error page, plain text): shown as is
  }
  return clip(text, 300);
}

/** POST a JSON body and call onEvent for every NDJSON line as soon as it arrives. */
async function streamEvents(url, body, onEvent) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" },
    body: JSON.stringify(body),
  });
  if (!response.ok || !response.body) {
    throw new Error(`Сервер ответил ${response.status}: ${errorDetail(await response.text())}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // the tail may be half a line; it waits for the next chunk
    for (const line of lines) if (line.trim()) onEvent(parseLine(line));
    if (done) break;
  }
  if (buffer.trim()) onEvent(parseLine(buffer));
}

/** Render one stream from request to end; resolves to whether it delivered a plan. */
async function run(url, body, kind, title) {
  setBusy(true);
  beginRun(kind, title);
  try {
    await streamEvents(url, body, dispatch);
  } catch (error) {
    // fetch and the stream reader throw TypeError when the connection itself fails
    onError({ message: error instanceof TypeError ? `Нет связи с сервером: ${error.message}` : error.message });
  } finally {
    endRun();
    setBusy(false);
  }
  return state.run.gotPlan;
}

/**
 * Hand one event to its handler. Unknown types are skipped, so the server can add new
 * ones. A handler that trips over an unexpected shape becomes an error row instead of
 * ending the stream, which would also be misreported as a lost connection.
 */
function dispatch(event) {
  const type = event?.type;
  try {
    handlers[type]?.(event.data ?? {});
  } catch (error) {
    onError({ message: `Не удалось показать событие «${type}»: ${error.message}` });
  }
}

function setBusy(busy) {
  state.busy = busy;
  document.body.classList.toggle("is-busy", busy);
  for (const button of document.querySelectorAll("button")) button.disabled = busy;
  ui.build.textContent = busy ? "Агент работает…" : "Построить маршрут";
}

function beginRun(kind, title) {
  state.run = {
    kind,
    stages: new Map(), // name -> "active" | "done" | "failed"
    stageRows: new Map(), // name -> {row, startedAt}
    toolRows: new Map(), // tool call id -> row
    gotPlan: false,
    failed: false,
    over: false,
    elapsed: null,
  };
  ui.hint.hidden = true;
  document.body.classList.add("has-run");
  if (kind === "plan") {
    ui.log.replaceChildren();
    state.sessionId = null;
    state.placeNames.clear();
    ui.feedback.hidden = true;
    clearRoute();
    clearCandidates();
    showEmpty(
      "Агент работает",
      "Следите за ним слева: он ищет места, отбирает подходящие, а затем расписание и проверка соберут маршрут.",
    );
  }
  followLog = true;
  addRow("divider", null, kind === "plan" ? "Новый маршрут" : "Пожелание", " ", h("q", {}, clip(title, 160)));
  renderStages();
  setStatus("Отправляю запрос агенту…", true);
}

function endRun() {
  const current = state.run;
  current.over = true;
  if (current.gotPlan) {
    setStatus(current.elapsed ? `Маршрут готов за ${current.elapsed}` : "Маршрут готов");
  } else {
    for (const [name, status] of current.stages) if (status === "active") current.stages.set(name, "failed");
    for (const row of current.toolRows.values()) row.classList.remove("is-pending"); // no answer is coming
    if (!current.failed) {
      const text = "Агент закончил работу, но маршрута не прислал. Попробуйте ещё раз или уточните запрос.";
      addRow("error", badge("×"), text);
    }
    setStatus(current.kind === "plan" ? "Маршрут не построен" : "Маршрут не изменён");
    if (current.kind === "plan") {
      showEmpty("Маршрут не построен", "Сообщения агента слева подскажут, что пошло не так. Уточните запрос.");
    }
  }
  renderStages();
}

/* ---------- agent trace ---------- */

let followLog = true; // stop auto-scrolling once the user scrolls up to read
ui.log.addEventListener("scroll", () => {
  followLog = ui.log.scrollHeight - ui.log.scrollTop - ui.log.clientHeight < 40;
});

function follow() {
  if (followLog) ui.log.scrollTop = ui.log.scrollHeight;
}

function addRow(cls, icon, ...content) {
  const row = h(
    "li",
    { class: `row ${cls}` },
    icon ?? h("span", { class: "badge is-blank" }),
    h("div", { class: "row-body" }, ...content),
  );
  ui.log.append(row);
  follow();
  return row;
}

function setStatus(text, live = false) {
  ui.status.textContent = text;
  ui.status.classList.toggle("is-live", live);
}

function renderStages() {
  const { stages, kind, over } = state.run;
  const expected = kind === "plan" ? PLAN_STAGES : ["feedback"];
  const items = STAGES.filter(([name]) => expected.includes(name) || stages.has(name)).map(([name, label]) =>
    h("li", { class: `step is-${stages.get(name) ?? (over ? "skipped" : "pending")}` }, label),
  );
  ui.steps.replaceChildren(...items);
}

function onStage({ name, status, label, detail }) {
  const current = state.run;
  const short = STAGES.find(([stage]) => stage === name)?.[1] ?? name;
  if (status === "start") {
    current.stages.set(name, "active");
    const title = h("span", { class: "stage-name" }, short);
    const row = addRow("stage", null, title, h("span", { class: "stage-label" }, label));
    current.stageRows.set(name, { row, startedAt: performance.now() });
    setStatus(label, true);
  } else {
    current.stages.set(name, "done");
    const started = current.stageRows.get(name);
    if (started) {
      started.row.classList.add("is-done");
      // the done label is the outcome ("Погода: …", "Вошло мест: 6…", "по прямой: …"),
      // which says more than the intent shown while the stage ran
      if (label) started.row.querySelector(".stage-label").textContent = label;
      started.row.lastChild.append(h("span", { class: "stage-time" }, seconds(performance.now() - started.startedAt)));
    }
    showStageDetail(name, detail);
  }
  renderStages();
}

function showStageDetail(name, detail) {
  if (!detail) return;
  if (name === "brief") renderBrief(detail);
  if (name === "weather" && detail.start) showStart(detail.start);
  if (name === "research") {
    const candidates = detail.candidates ?? 0;
    addRow(
      "note",
      badge("∑", "tool-finish"),
      `Поиск закончен: ${detail.steps} ${plural(detail.steps, "шаг", "шага", "шагов")}, `,
      `${candidates} ${plural(candidates, "кандидат", "кандидата", "кандидатов")}`,
      detail.stop_reason ? h("code", { class: "stop-reason" }, detail.stop_reason) : null,
    );
  }
}

function renderBrief(brief) {
  const chip = (cls, ...content) => h("span", { class: `chip ${cls}` }, ...content);
  addRow(
    "note",
    badge("≡", "tool-finish", "Разобранный запрос"),
    h("p", { class: "brief-title" }, "Агент понял запрос так"),
    h(
      "div",
      { class: "chips" },
      chip("is-strong", [brief.city, dayLabel(brief.date)].join(", ")),
      chip("is-strong", timeWindow(brief)),
      brief.start_point ? chip("", `старт: ${brief.start_point}`) : null,
      PACE[brief.pace] ? chip("", PACE[brief.pace]) : null,
      brief.with_kids ? chip("", "с детьми") : null,
      brief.interests.map((interest) =>
        h("span", { class: "chip is-interest", title: "вес интереса: 1 — обязательно, меньше — по возможности" },
          interest.label, h("b", {}, interest.weight.toFixed(1))),
      ),
      brief.must_include.map((name) => chip("is-must", `обязательно: ${name}`)),
      brief.avoid.map((item) => chip("is-avoid", `без: ${item}`)),
    ),
  );
}

function onThought({ text }) {
  if (!text?.trim()) return;
  const toggle = (e) => e.currentTarget.classList.toggle("is-open"); // long thoughts are clamped to 4 lines
  addRow("thought", null, h("p", { title: "Нажмите, чтобы раскрыть", onclick: toggle }, text.trim()));
}

function toolBadge(name) {
  const [, glyph, cls, title] = TOOL_KINDS.find(([pattern]) => pattern.test(name)) ?? [null, "ƒ", "", "Инструмент"];
  return badge(glyph, cls, title);
}

function briefArgs(args) {
  if (typeof args === "string") {
    try {
      args = JSON.parse(args);
    } catch {
      return clip(args, 140);
    }
  }
  if (!args || typeof args !== "object") return "";
  return Object.entries(args)
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([key, value]) => {
      let text = String(value);
      if (Array.isArray(value)) text = value.join(", ");
      else if (typeof value === "object") text = JSON.stringify(value);
      return `${key}: ${clip(text, 48)}`;
    })
    .join(" · ");
}

function onToolCall({ id, name, args }) {
  const row = addRow(
    "tool is-pending",
    toolBadge(name),
    h(
      "div",
      { class: "tool-head" },
      h("code", { class: "tool-name" }, name),
      h("span", { class: "tool-args" }, briefArgs(args)),
    ),
  );
  state.run.toolRows.set(id, row);
  return row;
}

function onToolResult({ id, name, summary, count, error }) {
  const row = state.run.toolRows.get(id) ?? onToolCall({ id, name, args: null });
  row.classList.remove("is-pending");
  row.classList.toggle("is-error", Boolean(error));
  const text = typeof error === "string" ? error : summary;
  const counter = count != null ? h("span", { class: "count" }, count) : null;
  row.lastChild.append(h("div", { class: "tool-result" }, counter, text ?? ""));
  follow();
}

function onCandidate({ action, place_id: id, name, kind, lat, lon, score, interests = [], reason }) {
  state.placeNames.set(id, name);
  const added = action === "add";
  if (added && typeof lat === "number") showCandidate(id, name, kind, lat, lon);
  if (!added) hideCandidate(id);
  addRow(
    `candidate ${added ? "is-add" : "is-remove"}`,
    badge(added ? "+" : "−", "", added ? "В шорт-лист" : "Из шорт-листа"),
    h(
      "div",
      { class: "cand-head" },
      h("span", { class: "cand-name" }, name),
      typeof score === "number" ? h("span", { class: "score", style: `--score: ${score}` }, score.toFixed(2)) : null,
    ),
    interests.length ? h("div", { class: "tags" }, interests.map(tag)) : null,
    reason ? h("p", { class: "cand-reason" }, reason) : null,
  );
}

function onWarning({ message }) {
  addRow("warning", badge("!"), message);
}

function onError({ message }) {
  state.run.failed = true;
  addRow("error", badge("×"), message);
}

function onDone({ usage, elapsed_s: elapsed }) {
  state.run.elapsed = typeof elapsed === "number" ? seconds(elapsed * 1000) : null;
  const text = state.run.elapsed ? `Готово за ${state.run.elapsed}` : "Готово";
  addRow("note", badge("■", "tool-finish"), text, usage ? ` · ${usageText(usage)}` : "");
}

function onPlan({ session_id: sessionId, plan, usage, reply }) {
  state.sessionId = sessionId;
  for (const stop of plan.stops) state.placeNames.set(stop.place.id, stop.place.name);
  renderRoute(plan);
  renderPlan(plan, usage);
  state.run.gotPlan = true;
  ui.reply.replaceChildren(h("b", {}, "Агент: "), reply ?? "");
  ui.reply.hidden = !reply;
  ui.feedback.hidden = false;
  const count = plan.stops.length;
  const stops = `${count} ${plural(count, "остановка", "остановки", "остановок")}`;
  const text = `Маршрут собран: ${stops}, ${distance(plan.total_walk_m)} пешком`;
  addRow("note plan-ready", badge("✓", "tool-shortlist"), text);
}

const handlers = {
  stage: onStage,
  thought: onThought,
  tool_call: onToolCall,
  tool_result: onToolResult,
  candidate: onCandidate,
  warning: onWarning,
  error: onError,
  plan: onPlan,
  done: onDone,
};

/* ---------- map ---------- */

const map = L.map("map", { zoomControl: false }).setView([57.5, 34], 5);
L.control.zoom({ position: "bottomright" }).addTo(map);
// Standard OSM tiles: keyless (CARTO basemaps now answer "API key required"); style.css
// mutes their colours so the route stays the brightest thing on the map.
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">участники OpenStreetMap</a>',
}).addTo(map);
const routeLayer = L.layerGroup().addTo(map);
// Places appear here the moment the agent shortlists them, so the audience watches the
// route's raw material gather; after the plan, the ones that did not fit stay as ghosts.
const candidateLayer = L.layerGroup().addTo(map);
let startMarker = null;
let fitTimer = null;
new ResizeObserver(() => map.invalidateSize()).observe($("#map"));

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function pinIcon(kind, label, order) {
  // label is a stop number or empty, never outside text, so an HTML string is safe here
  return L.divIcon({
    className: `pin pin-${kind}`,
    html: `<span style="--i: ${order}">${label}</span>`,
    iconSize: [34, 34],
    iconAnchor: [17, 17],
    popupAnchor: [0, -16],
  });
}

function popupContent(stop, index) {
  const place = stop.place;
  return h(
    "div",
    { class: "popup" },
    h("p", { class: "popup-meta" }, `${index + 1} · ${visitTime(stop)} · ${kindLabel(place.kind)}`),
    h("h3", {}, place.name),
    h("p", {}, clip(stop.blurb || stop.reason, 240)),
    sourceLink(place),
  );
}

/** Animate the route drawing itself once; the dash trick is removed afterwards so zooming stays correct. */
function drawIn(line) {
  const path = line.getElement();
  const length = path.getTotalLength();
  path.style.strokeDasharray = `${length}`;
  path.style.strokeDashoffset = `${length}`;
  path.getBoundingClientRect(); // commit the hidden state so the transition has a start
  path.style.transition = "stroke-dashoffset 1.6s ease-in-out";
  path.style.strokeDashoffset = "0";
  path.addEventListener("transitionend", () => path.removeAttribute("style"), { once: true });
}

function clearRoute() {
  routeLayer.clearLayers();
  state.markers.clear();
}

function clearCandidates() {
  candidateLayer.clearLayers();
  state.candidates.clear();
  startMarker = null;
}

/** As soon as the city is geocoded: fly there and mark the start. */
function showStart({ lat, lon, label }) {
  startMarker?.remove();
  startMarker = L.marker([lat, lon], { icon: pinIcon("start", "", 0), title: label ?? "Старт" })
    .bindTooltip(label ?? "Старт", { direction: "top", offset: [0, -14] })
    .addTo(candidateLayer);
  map.flyTo([lat, lon], 15, { animate: !reduceMotion, duration: 1.2 });
}

function showCandidate(id, name, kind, lat, lon) {
  hideCandidate(id);
  const icon = L.divIcon({ className: "pin pin-cand", html: "<span></span>", iconSize: [16, 16], iconAnchor: [8, 8] });
  const marker = L.marker([lat, lon], { icon, title: name, keyboard: false })
    .bindTooltip(`${name} · ${kindLabel(kind)}`, { direction: "top", offset: [0, -8] })
    .addTo(candidateLayer);
  state.candidates.set(id, marker);
  // Batch the re-framing: a turn can shortlist ten places at once.
  clearTimeout(fitTimer);
  fitTimer = setTimeout(fitCandidates, 350);
}

function hideCandidate(id) {
  state.candidates.get(id)?.remove();
  state.candidates.delete(id);
}

function fitCandidates() {
  const points = [...state.candidates.values()].map((m) => m.getLatLng());
  if (startMarker) points.push(startMarker.getLatLng());
  if (points.length < 2) return;
  map.flyToBounds(L.latLngBounds(points), { padding: [48, 48], maxZoom: 16, duration: reduceMotion ? 0 : 0.8 });
}

/** After a plan: stops get their numbered pins, the rest of the shortlist fades to ghosts. */
function settleCandidates(plan) {
  clearTimeout(fitTimer);
  startMarker?.remove();
  startMarker = null;
  const inPlan = new Set(plan.stops.map((stop) => stop.place.id));
  for (const [id, marker] of state.candidates) {
    if (inPlan.has(id)) {
      hideCandidate(id);
    } else {
      marker.getElement()?.classList.add("is-dropped");
      marker.setTooltipContent(`${state.placeNames.get(id) ?? "Место"} — не вошло в маршрут`);
    }
  }
}

function renderRoute(plan) {
  clearRoute();
  settleCandidates(plan);
  const start = [plan.start.lat, plan.start.lon];
  const stopPoints = plan.stops.map((stop) => [stop.place.lat, stop.place.lon]);
  // Without walking geometry the stops are joined by straight dotted lines, so the map
  // does not pretend to know the streets.
  const hasGeometry = Boolean(plan.geometry?.length);
  const path = hasGeometry ? plan.geometry : [start, ...stopPoints];
  L.polyline(path, { className: "route-casing", interactive: false }).addTo(routeLayer);
  const lineClass = hasGeometry ? "route-line" : "route-line is-straight";
  const line = L.polyline(path, { className: lineClass, interactive: false }).addTo(routeLayer);

  const startPopup = h(
    "div",
    { class: "popup" },
    h("p", { class: "popup-meta" }, `Старт · ${clock(plan.brief.start_time)}`),
    h("h3", {}, plan.start.label ?? "Точка старта"),
  );
  L.marker(start, { icon: pinIcon("start", "", 0), title: "Старт" }).bindPopup(startPopup).addTo(routeLayer);
  plan.stops.forEach((stop, index) => {
    const id = stop.place.id;
    const icon = pinIcon("stop", index + 1, index + 1);
    const marker = L.marker(stopPoints[index], { icon, title: stop.place.name, riseOnHover: true })
      .bindPopup(popupContent(stop, index), { maxWidth: 320 })
      .on("mouseover", () => highlight(id, true))
      .on("mouseout", () => highlight(id, false))
      .on("click", () => selectCard(id))
      .addTo(routeLayer);
    state.markers.set(id, marker);
  });

  // maxZoom: a route of one stop next to the start would otherwise zoom in to single houses
  map.fitBounds(L.latLngBounds([...path, ...stopPoints]), { padding: [56, 56], maxZoom: 17, animate: false });
  if (hasGeometry && !reduceMotion) drawIn(line);
}

function highlight(id, on) {
  state.markers.get(id)?.getElement()?.classList.toggle("is-hot", on);
  state.cards.get(id)?.classList.toggle("is-hot", on);
}

function selectCard(id) {
  for (const [cardId, card] of state.cards) card.classList.toggle("is-active", cardId === id);
  state.cards.get(id)?.scrollIntoView({ block: "nearest", behavior: reduceMotion ? "auto" : "smooth" });
}

function focusStop(id) {
  const marker = state.markers.get(id);
  if (!marker) return;
  selectCard(id);
  map.once("moveend", () => marker.openPopup());
  map.setView(marker.getLatLng(), Math.max(map.getZoom(), 15), { animate: !reduceMotion });
}

/* ---------- itinerary ---------- */

function showEmpty(title, text) {
  state.cards.clear();
  ui.plan.replaceChildren(h("div", { class: "empty" }, h("h2", {}, title), h("p", {}, text)));
}

function stat(value, label) {
  return h("div", { class: "stat" }, h("dt", {}, label), h("dd", {}, value));
}

function endpointItem(time, text, cls = "", order = 0) {
  return h("li", { class: `endpoint ${cls}`, style: `--i: ${order}` }, h("time", {}, time), ` ${text}`);
}

function legItem(stop, index) {
  const meters = stop.walk_meters_from_prev;
  const text = meters ? `${stop.walk_minutes_from_prev} мин пешком · ${distance(meters)}` : "в двух шагах";
  return h("li", { class: "leg", style: `--i: ${index}` }, text);
}

function stopItem(stop, index, violations) {
  const place = stop.place;
  const card = h(
    "li",
    {
      class: "stop",
      style: `--i: ${index}`,
      tabindex: "0",
      onclick: () => focusStop(place.id),
      onkeydown: (e) => {
        if (e.key === "Enter" && e.target === e.currentTarget) focusStop(place.id);
      },
      onmouseenter: () => highlight(place.id, true),
      onmouseleave: () => highlight(place.id, false),
    },
    h("span", { class: "stop-num", "aria-hidden": "true" }, index + 1),
    h(
      "div",
      { class: "stop-body" },
      h(
        "p",
        { class: "stop-meta" },
        h("time", {}, visitTime(stop)),
        h("span", { class: "kind" }, kindLabel(place.kind)),
      ),
      h("h3", { class: "stop-name" }, place.name),
      stop.interests.length ? h("div", { class: "tags" }, stop.interests.map(tag)) : null,
      h("p", { class: "stop-text" }, stop.blurb || stop.reason),
      stop.tip ? h("p", { class: "stop-tip" }, h("b", {}, "Совет. "), stop.tip) : null,
      violations.length ? h("ul", { class: "violations" }, violations.map(violationItem)) : null,
      h(
        "div",
        { class: "stop-foot" },
        sourceLink(place),
        h(
          "button",
          {
            class: "remove",
            type: "button",
            disabled: state.busy,
            onclick: (e) => {
              e.stopPropagation(); // the card itself reacts to clicks by focusing the map
              removeStop(place);
            },
          },
          "Убрать",
        ),
      ),
    ),
  );
  state.cards.set(place.id, card);
  return card;
}

/** Names of the places that did not fit; ids the trace never named are only counted. */
function droppedText(ids) {
  const names = ids.map((id) => state.placeNames.get(id)).filter(Boolean);
  const rest = ids.length - names.length;
  return (rest ? [...names, `ещё ${rest}`] : names).join(", ");
}

function renderPlan(plan, usage) {
  state.cards.clear();
  const { brief, stops } = plan;
  // a violation sits on its stop's card; one about the whole route, or about a place
  // that is not a stop, goes to the header so it is never lost
  const stopIds = new Set(stops.map((stop) => stop.place.id));
  const general = plan.violations.filter((v) => !stopIds.has(v.place_id));
  const violationsOf = (id) => plan.violations.filter((v) => v.place_id === id);
  const last = stops.at(-1);

  ui.plan.replaceChildren(
    h(
      "header",
      { class: "plan-head" },
      h("p", { class: "eyebrow" }, [brief.city, dayLabel(brief.date), timeWindow(brief)].join(" · ")),
      h("h2", { class: "plan-title" }, plan.title || `Прогулка: ${brief.city}`),
      plan.summary ? h("p", { class: "plan-summary" }, plan.summary) : null,
      h(
        "dl",
        { class: "stats" },
        stat(stops.length, plural(stops.length, "остановка", "остановки", "остановок")),
        stat(km(plan.total_walk_m), "км пешком"),
        stat(duration(plan.total_minutes), "вся прогулка"),
      ),
      general.length ? h("ul", { class: "violations" }, general.map(violationItem)) : null,
      usage ? h("p", { class: "usage" }, usageText(usage)) : null,
    ),
    h(
      "ol",
      { class: "timeline" },
      endpointItem(clock(brief.start_time), `Старт · ${plan.start.label ?? "точка старта"}`),
      stops.map((stop, index) => [legItem(stop, index), stopItem(stop, index, violationsOf(stop.place.id))]),
      last ? endpointItem(hhmm(last.depart), "Конец прогулки", "is-end", stops.length) : null,
    ),
    plan.dropped.length
      ? h("p", { class: "dropped" }, h("b", {}, "Не поместились: "), droppedText(plan.dropped))
      : null,
  );
  ui.plan.scrollTop = 0;
}

/* ---------- controls ---------- */

async function sendFeedback(feedback, excludeIds = []) {
  if (!state.sessionId || state.busy) return false;
  const url = `/api/sessions/${encodeURIComponent(state.sessionId)}/feedback`;
  return run(url, { feedback, exclude_place_ids: excludeIds }, "feedback", feedback);
}

function removeStop(place) {
  sendFeedback(`Убери ${place.name}`, [place.id]);
}

const submitOnCmdEnter = (form) => (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) form.requestSubmit();
};

ui.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.busy) return;
  // Only filled fields are sent: the agent infers the rest from the free text.
  const fields = [...new FormData(ui.composer)].map(([key, value]) => [key, value.trim()]);
  const body = Object.fromEntries(fields.filter(([, value]) => value));
  if (body.request) run("/api/plan", body, "plan", body.request);
});
ui.request.addEventListener("keydown", submitOnCmdEnter(ui.composer));

for (const example of document.querySelectorAll("[data-example]")) {
  example.addEventListener("click", () => {
    ui.request.value = example.dataset.example;
    ui.request.focus();
  });
}

ui.feedback.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = ui.feedbackText.value.trim();
  if (!text || state.busy) return;
  // The wish stays in the box until the route is rebuilt, so a failed attempt is one
  // click to retry; a new wish typed meanwhile is kept too.
  if ((await sendFeedback(text)) && ui.feedbackText.value.trim() === text) ui.feedbackText.value = "";
});
ui.feedbackText.addEventListener("keydown", submitOnCmdEnter(ui.feedback));

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error(String(response.status));
    const { model } = await response.json();
    ui.health.textContent = model || "модель не задана";
    ui.health.dataset.state = "ok";
  } catch {
    ui.health.textContent = "сервер недоступен";
    ui.health.dataset.state = "down";
  }
}

showEmpty(
  "Здесь появится маршрут",
  "Опишите прогулку — агент найдёт реальные места в OpenStreetMap, Википедии и Wikivoyage, " +
    "проверит часы работы и погоду и уложит всё во время.",
);
checkHealth();
