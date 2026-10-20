/*
 * Прогулка — the planner: describe a walk, watch the agent build it, then edit it.
 *
 * The server streams the agent's work as NDJSON: one AgentEvent {type, data} per line.
 * Every event type has one handler below that turns it into something visible: a stage
 * strip, a live trace of tool calls and shortlisted places, then the route on the map
 * and as a timeline. The page keeps no plan logic of its own: the server owns the plan
 * and saves every run as a trip, the page only keeps the trip id (in the address too),
 * so a feedback run is just another stream rendered by the same handlers, and
 * /app/trips/{id} reopens a saved trip exactly where its last run stopped.
 *
 * The composer sends only what the traveller set for this walk: the text, the
 * «Уточнить» fields and the «Настроить прогулку» panel. The profile is not copied into
 * the form: the server merges it in, and the parsed brief shows where each interest
 * came from.
 *
 * Text from outside (place names, Wikipedia extracts, LLM output, memory facts) only
 * ever reaches the DOM through text nodes, never innerHTML, so a place name cannot
 * inject markup.
 */

import { ApiError, api, requireUser, streamEvents } from "./api.js";
import { renderNav } from "./nav.js";
import { createRouteMap } from "./planner-map.js";
import { ago, dayLabel, h, plural, toast, usd } from "./ui.js";

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
  ["memory", "Память"],
];
// Stages a fresh plan always goes through; "repair", "memory" and the stages of a
// feedback run appear in the strip only once the server actually starts them.
const PLAN_STAGES = ["brief", "weather", "research", "optimize", "route", "validate", "write"];

// Keys are the agent's tool names (GET /api/tools). A Map, not an object: the name comes
// from the model, and "constructor" must not find Object's.
const TOOL_KINDS = new Map([
  ["get_wikivoyage_listings", ["WV", "src-wikivoyage", "Wikivoyage"]],
  ["search_wikipedia", ["W", "src-wikipedia", "Википедия"]],
  ["search_osm", ["OSM", "src-osm", "OpenStreetMap"]],
  ["get_place_details", ["i", "tool-details", "Подробности о месте"]],
  ["shortlist_place", ["★", "tool-shortlist", "В шорт-лист"]],
  ["remove_place", ["−", "tool-shortlist", "Из шорт-листа"]],
  ["finish_research", ["✓", "tool-finish", "Конец поиска"]],
]);

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
  missing_must_include: "Нет в маршруте",
};

const PACE = { relaxed: "неспешно", active: "активный темп" };

// Where an interest of the brief came from (Interest.source on the server).
const ORIGINS = [
  ["request", "из запроса"],
  ["walk", "из настроек"],
  ["profile", "из профиля"],
];

// The history line the server writes for a run without text: same words here.
const NO_REQUEST_TEXT = "По анкете и настройкам прогулки";
const FIELDS = ["city", "date", "start_time", "end_time", "start_point"];
const KM_MAX = 20; // the distance slider: 0 is "не задано", then 1..20 km

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// What a refusal before the stream means for the traveller, after the server's own words.
const REFUSAL_HINTS = {
  404: "Возможно, его удалили в другой вкладке.",
  409: "Его правят в другом окне: дождитесь конца того запуска.",
  429: "Лимиты берегут общий ключ API демо.",
};

const state = {
  user: null,
  busy: false,
  tripId: null,
  map: null, // RouteMap from planner-map.js, or null when the map could not load
  plan: null, // the plan on screen
  profile: null,
  catalog: null,
  submitted: null, // {request, fields, preferences}: what the folded composer recaps
  tune: { choice: new Map(), pace: null, kids: null, km: null }, // the «Настроить» panel
  run: null, // bookkeeping of the stream being rendered, see beginRun()
  placeNames: new Map(), // place_id -> name, to name the places that did not fit
  cards: new Map(), // place_id -> timeline item
};

const ui = {
  planner: $("#planner"),
  composer: $("#composer"),
  recap: $("#recap"),
  composerBody: $("#composer-body"),
  request: $("#request"),
  requestHint: $("#request-hint"),
  requestError: $("#request-error"),
  tuneBody: $("#tune-body"),
  tuneCount: $("#tune-count"),
  profileNote: $("#profile-note"),
  build: $("#build"),
  cancelEdit: $("#cancel-edit"),
  replanNote: $("#replan-note"),
  health: $("#health"),
  steps: $("#steps"),
  status: $("#status"),
  stop: $("#stop"),
  hint: $("#trace-hint"),
  log: $("#log"),
  map: $("#map"),
  legend: $("#map-legend"),
  mapFallback: $("#map-fallback"),
  plan: $("#plan"),
  feedback: $("#feedback"),
  feedbackText: $("#feedback-text"),
  rebuild: $("#rebuild"),
  reply: $("#reply"),
};

/* ---------- small helpers ---------- */

const clip = (text, max) => (text.length > max ? `${text.slice(0, max - 1)}…` : text);
const hhmm = (iso) => iso.slice(11, 16); // "2026-10-24T11:12:00" -> "11:12", no timezone games
const clock = (time) => time.slice(0, 5); // "11:00:00" -> "11:00"
const visitTime = (stop) => `${hhmm(stop.arrive)}–${hhmm(stop.depart)}`;
const timeWindow = (brief) => `${clock(brief.start_time)}–${clock(brief.end_time)}`;
const km = (meters) => (meters / 1000).toLocaleString("ru-RU", { maximumFractionDigits: 1 });
const distance = (meters) => (meters < 1000 ? `${meters} м` : `${km(meters)} км`);
const seconds = (ms) => `${(ms / 1000).toLocaleString("ru-RU", { maximumFractionDigits: 1 })} с`;
const count = (n, one, few, many) => `${n} ${plural(n, one, few, many)}`;

function duration(minutes) {
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  if (!hours) return `${rest} мин`;
  return rest ? `${hours} ч ${rest} мин` : `${hours} ч`;
}

function usageText(usage) {
  const calls = usage.llm_calls ?? 0;
  const tokens = (usage.prompt_tokens ?? 0) + (usage.completion_tokens ?? 0);
  const parts = [`${count(calls, "вызов", "вызова", "вызовов")} LLM`];
  if (usage.cached_calls) parts.push(`${usage.cached_calls} из кэша`);
  parts.push(`${tokens.toLocaleString("ru-RU")} токенов`);
  if (usage.cost_usd != null) parts.push(usd(usage.cost_usd));
  return parts.join(" · ");
}

const kindLabel = (kind) => KINDS[kind] ?? kind.replaceAll("_", " ");
const interestLabel = (key) => state.catalog?.interests.find((item) => item.key === key)?.label ?? key;
const paceLabel = (key) => state.catalog?.paces.find((item) => item.key === key)?.label ?? key;
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

/** A refusal before the stream (403, 404, 409, 429) or a lost connection, in plain words. */
function refusalText(error) {
  if (!(error instanceof ApiError)) return `Не удалось обработать ответ сервера: ${error.message}`;
  const hint = REFUSAL_HINTS[error.status];
  return hint ? `${error.message}. ${hint}` : error.message;
}

/** Render one stream from request to end; resolves to whether it delivered a plan. */
async function run(url, body, kind, title) {
  setBusy(true, kind);
  beginRun(kind, title);
  let signedOut = false;
  try {
    await streamEvents(url, body, dispatch, { signal: state.run.controller.signal });
  } catch (error) {
    if (error.name === "AbortError") {
      onError({ message: "Остановлено: агент прерван" }); // the stop button
    } else if (error instanceof ApiError && error.status === 401) {
      signedOut = true;
      onError({ message: "Сессия закончилась: войдите снова" });
    } else {
      onError({ message: refusalText(error) });
    }
  }
  endRun();
  setBusy(false);
  if (signedOut) await requireUser(); // sends the visitor to /login and back here
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

function setBusy(busy, kind = null) {
  state.busy = busy;
  document.body.classList.toggle("is-busy", busy);
  // The nav stays usable: leaving the page is allowed (it asks first, see the bottom).
  for (const button of ui.planner.querySelectorAll("button")) button.disabled = busy && button !== ui.stop;
  ui.stop.hidden = !busy;
  ui.build.classList.toggle("is-busy", busy && kind === "plan");
  ui.rebuild.classList.toggle("is-busy", busy && kind === "feedback");
  updateBuildLabel();
}

function beginRun(kind, title) {
  state.run = {
    kind,
    // the stop button aborts the request; the server then cancels the run and, for a
    // feedback run, keeps the saved trip as it was, so the route on screen stays current
    controller: new AbortController(),
    stages: new Map(), // name -> "active" | "done" | "failed"
    stageRows: new Map(), // name -> {row, startedAt}
    toolRows: new Map(), // tool call id -> row
    gotPlan: false,
    failed: false,
    over: false,
    elapsed: null,
  };
  ui.hint.hidden = true;
  if (kind === "plan") {
    ui.log.replaceChildren();
    state.plan = null;
    state.placeNames.clear();
    ui.feedback.hidden = true;
    state.map?.clearRoute();
    state.map?.clearCandidates();
    updateLegend();
    foldComposer();
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
  // no answer is coming, even with a plan: the research time budget can cut off a tool call
  for (const row of current.toolRows.values()) row.classList.remove("is-pending");
  if (current.gotPlan) {
    setStatus(current.elapsed ? `Маршрут готов за ${current.elapsed}` : "Маршрут готов");
  } else {
    for (const [name, status] of current.stages) if (status === "active") current.stages.set(name, "failed");
    if (!current.failed) {
      const text = "Агент закончил работу, но маршрута не прислал. Попробуйте ещё раз или уточните запрос.";
      addRow("error", badge("×"), text);
    }
    setStatus(current.kind === "plan" ? "Маршрут не построен" : "Маршрут не изменён");
    if (current.kind === "plan") {
      showEmpty("Маршрут не построен", "Сообщения агента слева подскажут, что пошло не так. Уточните запрос.");
      unfoldComposer();
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
  if (name === "weather" && detail.start) state.map?.showStart(detail.start);
  if (name === "research") {
    const candidates = detail.candidates ?? 0;
    addRow(
      "note",
      badge("∑", "tool-finish"),
      `Поиск закончен: ${count(detail.steps, "шаг", "шага", "шагов")}, `,
      count(candidates, "кандидат", "кандидата", "кандидатов"),
      detail.stop_reason ? h("code", { class: "stop-reason" }, detail.stop_reason) : null,
    );
  }
}

/** The parsed brief as chips; interests are grouped by where they came from. */
function renderBrief(brief, title = "Агент понял запрос так") {
  const chip = (cls, ...content) => h("span", { class: `chip ${cls}` }, ...content);
  const groups = ORIGINS.map(([origin, label]) => {
    const interests = brief.interests.filter((interest) => (interest.source ?? "request") === origin);
    if (!interests.length) return null;
    return h(
      "div",
      { class: `brief-group is-${origin}` },
      h("span", { class: "brief-origin" }, label),
      h(
        "div",
        { class: "chips" },
        interests.map((interest) =>
          h(
            "span",
            { class: "chip is-interest", title: "вес интереса: 1 — обязательно, меньше — по возможности" },
            interest.label,
            h("b", {}, interest.weight.toFixed(1)),
          ),
        ),
      ),
    );
  });
  addRow(
    "note brief",
    badge("≡", "tool-finish", "Разобранный запрос"),
    h("p", { class: "brief-title" }, title),
    h(
      "div",
      { class: "chips" },
      chip("is-strong", [brief.city, dayLabel(brief.date)].join(", ")),
      chip("is-strong", timeWindow(brief)),
      brief.start_point ? chip("", `старт: ${brief.start_point}`) : null,
      PACE[brief.pace] ? chip("", PACE[brief.pace]) : null,
      brief.with_kids ? chip("", "с детьми") : null,
      brief.max_walk_km ? chip("", `до ${brief.max_walk_km.toLocaleString("ru-RU")} км`) : null,
    ),
    groups,
    brief.must_include.length || brief.avoid.length
      ? h(
          "div",
          { class: "chips" },
          brief.must_include.map((name) => chip("is-must", `обязательно: ${name}`)),
          brief.avoid.map((item) => chip("is-avoid", `без: ${item}`)),
        )
      : null,
  );
}

function onThought({ text }) {
  if (!text?.trim()) return;
  const toggle = (e) => e.currentTarget.classList.toggle("is-open"); // long thoughts are clamped to 4 lines
  addRow("thought", null, h("p", { title: "Нажмите, чтобы раскрыть", onclick: toggle }, text.trim()));
}

function toolBadge(name) {
  const [glyph, cls, title] = TOOL_KINDS.get(name) ?? ["ƒ", "", "Инструмент"];
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

function onToolResult({ id, name, summary, count: found, error }) {
  const row = state.run.toolRows.get(id) ?? onToolCall({ id, name, args: null });
  row.classList.remove("is-pending");
  row.classList.toggle("is-error", Boolean(error));
  const text = typeof error === "string" ? error : summary;
  const counter = found != null ? h("span", { class: "count" }, found) : null;
  row.lastChild.append(h("div", { class: "tool-result" }, counter, text ?? ""));
  follow();
}

function onCandidate({ action, place_id: id, name, kind, lat, lon, score, interests = [], reason }) {
  state.placeNames.set(id, name);
  const added = action === "add";
  if (added && typeof lat === "number") state.map?.addCandidate(id, lat, lon, `${name} · ${kindLabel(kind)}`);
  if (!added) state.map?.removeCandidate(id);
  updateLegend();
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

/** Facts the agent noticed in the traveller's own words; they live on in the profile. */
function onMemory({ facts = [] }) {
  for (const fact of facts) {
    addRow("memory", badge("✎", "tool-memory", "Память агента"), h("span", { class: "memory-label" }, "Запомнил о вас: "), fact);
  }
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

function onPlan({ trip_id: tripId, plan, usage, reply, learned = [] }) {
  if (!plan) return; // endRun reports a run that ended without a route
  showTrip(tripId, plan);
  renderPlan(plan, usage ? usageText(usage) : null);
  state.map?.showPlan(plan);
  updateLegend();
  state.run.gotPlan = true;
  showReply(reply);
  const text = `Маршрут собран: ${count(plan.stops.length, "остановка", "остановки", "остановок")}, ${distance(plan.total_walk_m)} пешком`;
  addRow("note plan-ready", badge("✓", "tool-shortlist"), text);
  // `learned` holds only the facts the server stored, so the toast never promises more
  if (learned.length) {
    const facts = count(learned.length, "новый факт", "новых факта", "новых фактов");
    toast(h("span", {}, `Агент запомнил о вас ${facts}. `, h("a", { href: "/app/profile#memory" }, "Профиль")), "ok");
  }
}

const handlers = {
  stage: onStage,
  thought: onThought,
  tool_call: onToolCall,
  tool_result: onToolResult,
  candidate: onCandidate,
  memory: onMemory,
  warning: onWarning,
  error: onError,
  plan: onPlan,
  done: onDone,
};

/* ---------- map ---------- */

const mapHooks = {
  popup: popupContent,
  startPopup: (plan) =>
    h(
      "div",
      { class: "popup" },
      h("p", { class: "popup-meta" }, `Старт · ${clock(plan.brief.start_time)}`),
      h("h3", {}, plan.start.label ?? "Точка старта"),
    ),
  ghostTitle: (id) => `${state.placeNames.get(id) ?? "Место"} — не вошло в маршрут`,
  onHover: highlight,
  onSelect: selectCard,
};

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

function showMapFallback(error) {
  ui.mapFallback.replaceChildren(
    h("p", { class: "map-fallback-title" }, "Карта не загрузилась"),
    h("p", {}, "Маршрут и расписание работают и без неё."),
    h("p", { class: "map-fallback-detail" }, clip(error?.message ?? "", 160)),
  );
  ui.mapFallback.hidden = false;
  ui.legend.hidden = true;
}

/** What the markers on the map mean, for the kinds that are on it right now. */
function updateLegend() {
  const markers = state.map?.counts();
  if (!markers) return;
  const items = [];
  if (state.plan) items.push(["start", "старт"], ["stop", "остановка"]);
  if (markers.live) items.push(["cand", "найдено агентом"]);
  if (markers.ghosts) items.push(["ghost", "не вошло в маршрут"]);
  ui.legend.replaceChildren(
    ...items.map(([kind, label]) => h("li", {}, h("span", { class: `legend-mark is-${kind}`, "aria-hidden": "true" }), label)),
  );
  ui.legend.hidden = !items.length;
}

function highlight(id, on) {
  state.map?.highlight(id, on);
  state.cards.get(id)?.classList.toggle("is-hot", on);
}

function selectCard(id) {
  for (const [cardId, card] of state.cards) card.classList.toggle("is-active", cardId === id);
  state.cards.get(id)?.scrollIntoView({ block: "nearest", behavior: reduceMotion ? "auto" : "smooth" });
}

function focusStop(id) {
  selectCard(id);
  state.map?.focus(id);
}

/* ---------- itinerary ---------- */

function showEmpty(title, text, ...actions) {
  state.cards.clear();
  ui.plan.replaceChildren(
    h("div", { class: "empty" }, h("h2", {}, title), h("p", {}, text), actions.length ? h("div", { class: "empty-actions" }, actions) : null),
  );
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
  if (!names.length) return count(rest, "место", "места", "мест");
  return (rest ? [...names, `ещё ${rest}`] : names).join(", ");
}

/** The itinerary; `cost` is the run's usage line, or the trip's total for a saved one. */
function renderPlan(plan, cost) {
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
      cost ? h("p", { class: "usage" }, cost) : null,
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

function showReply(reply) {
  ui.reply.replaceChildren(h("b", {}, "Агент: "), reply ?? "");
  ui.reply.hidden = !reply;
  ui.feedback.hidden = false;
}

/** The plan on screen belongs to this trip: remember it and put it in the address. */
function showTrip(tripId, plan) {
  state.plan = plan;
  for (const stop of plan.stops) state.placeNames.set(stop.place.id, stop.place.name);
  document.title = `${plan.title || `Прогулка: ${plan.brief.city}`} — Прогулка`;
  if (!tripId || tripId === state.tripId) return;
  state.tripId = tripId;
  // replaceState, not pushState: "back" leads where the traveller came from, not to an empty form
  history.replaceState(null, "", `/app/trips/${encodeURIComponent(tripId)}`);
  renderNav(state.user, "trips");
  updateBuildLabel();
}

/* ---------- a saved trip ---------- */

/** Reopen a trip: its plan, the history of runs and the request it started from. */
function openTrip({ trip, plan, history: runs, request, preferences }) {
  // The «Уточнить» fields are not saved. A trip planned without text had its city from
  // them or the profile, so the brief's city keeps «Построить новый маршрут» working.
  state.submitted = { request, fields: request ? {} : { city: plan.brief.city }, preferences };
  fillComposer(state.submitted);
  foldComposer();
  showTrip(trip.id, plan);
  const total = runs.reduce((sum, entry) => sum + (entry.cost_usd ?? 0), 0);
  renderPlan(plan, `${count(runs.length, "запуск", "запуска", "запусков")} агента · ${usd(total)}`);
  state.map?.showPlan(plan);
  updateLegend();
  renderHistory(runs, plan.brief);
  const last = runs.at(-1);
  showReply(last?.kind === "feedback" ? last.reply : null);
  setStatus(`Сохранённый маршрут · изменён ${ago(trip.updated_at)}`);
}

function renderHistory(runs, brief) {
  ui.hint.hidden = true;
  addRow("divider", null, "История маршрута");
  for (const entry of runs) {
    const isPlan = entry.kind === "plan";
    addRow(
      "history",
      badge(isPlan ? "◎" : "✎", isPlan ? "tool-finish" : "tool-shortlist", isPlan ? "Маршрут построен" : "Правка"),
      h(
        "p",
        { class: "history-meta" },
        isPlan ? "Маршрут построен" : "Правка",
        h("span", { title: new Date(entry.at).toLocaleString("ru-RU") }, ` · ${ago(entry.at)}`),
        entry.cost_usd ? ` · ${usd(entry.cost_usd)}` : "",
      ),
      h("q", { class: "history-text" }, clip(entry.text, 240)),
      entry.reply ? h("p", { class: "history-reply" }, h("b", {}, "Агент: "), entry.reply) : null,
    );
  }
  renderBrief(brief, "Сейчас маршрут учитывает");
  addRow("note", badge("→", "tool-finish"), "Напишите, что изменить, справа под маршрутом: агент продолжит с того же места.");
}

/** `pathId` is the address segment, still percent-encoded. */
async function loadTrip(pathId) {
  showEmpty("Открываю маршрут…", "Загружаю сохранённую прогулку.");
  try {
    openTrip(await api(`/api/trips/${pathId}`));
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) await requireUser();
    const missing = error instanceof ApiError && error.status === 404;
    showEmpty(
      missing ? "Маршрут не найден" : "Не удалось открыть маршрут",
      missing ? "Возможно, его удалили или он сохранён в другом аккаунте." : error.message,
      h("a", { class: "btn btn-sm", href: "/app/trips" }, "Мои прогулки"),
      h("a", { class: "btn btn-sm btn-primary", href: "/app/new" }, "Новая прогулка"),
    );
  }
}

/* ---------- the «Настроить прогулку» panel ---------- */

const tuneControls = { chips: new Map(), paces: [], kids: null, km: null, kmOutput: null, reset: null };

// A chip cycles neutral -> «больше» -> «меньше» -> neutral.
const CHOICES = [undefined, "more", "less"];
const CHOICE_WORDS = { more: "больше", less: "меньше" };

function tuneSize() {
  const { choice, pace, kids, km: limit } = state.tune;
  return choice.size + (pace ? 1 : 0) + (kids ? 1 : 0) + (limit ? 1 : 0);
}

/** The panel as WalkPreferences for the server; keys keep the catalog's order. */
function preferences() {
  const { choice, pace, kids, km: limit } = state.tune;
  const keys = (wanted) => (state.catalog?.interests ?? []).map(({ key }) => key).filter((key) => choice.get(key) === wanted);
  return { more: keys("more"), less: keys("less"), pace, with_kids: kids, max_walk_km: limit };
}

function syncTune() {
  const { choice, pace, kids, km: limit } = state.tune;
  for (const [key, chip] of tuneControls.chips) {
    const value = choice.get(key);
    // base.css draws a pressed chip as "wanted" and a pressed avoid-toned one as "not this"
    chip.setAttribute("aria-pressed", String(Boolean(value)));
    if (value === "less") chip.dataset.tone = "avoid";
    else delete chip.dataset.tone;
    chip.dataset.choice = value ?? "";
    const word = CHOICE_WORDS[value] ?? "без разницы";
    chip.setAttribute("aria-label", `${interestLabel(key)}: ${word}`);
  }
  for (const radio of tuneControls.paces) radio.checked = radio.value === (pace ?? "");
  if (tuneControls.kids) tuneControls.kids.checked = Boolean(kids);
  if (tuneControls.km) {
    const slider = tuneControls.km;
    slider.value = String(limit ?? 0);
    slider.style.setProperty("--fill", `${((limit ?? 0) / KM_MAX) * 100}%`);
    slider.classList.toggle("is-off", !limit);
    tuneControls.kmOutput.textContent = limit ? `до ${limit} км` : "не задано";
    slider.setAttribute("aria-valuetext", tuneControls.kmOutput.textContent);
  }
  const size = tuneSize();
  ui.tuneCount.textContent = size;
  ui.tuneCount.hidden = !size;
  if (tuneControls.reset) tuneControls.reset.hidden = !size;
}

function interestChips(catalog) {
  return catalog.interests.map(({ key, label }) => {
    const chip = h(
      "button",
      {
        class: "chip tune-chip",
        type: "button",
        onclick: () => {
          const next = CHOICES[(CHOICES.indexOf(state.tune.choice.get(key)) + 1) % CHOICES.length];
          if (next) state.tune.choice.set(key, next);
          else state.tune.choice.delete(key);
          syncTune();
        },
      },
      label,
    );
    tuneControls.chips.set(key, chip);
    return chip;
  });
}

function paceControl(catalog) {
  const options = [{ key: "", label: "Не задано" }, ...catalog.paces].map(({ key, label }) => {
    const input = h("input", { type: "radio", name: "tune-pace", value: key });
    input.addEventListener("change", () => {
      state.tune.pace = key || null;
      syncTune();
    });
    tuneControls.paces.push(input);
    return h("label", {}, input, h("span", {}, label));
  });
  return h("div", { class: "segmented tune-pace", role: "radiogroup", "aria-labelledby": "tune-pace-label" }, options);
}

function kidsControl() {
  const input = h("input", { type: "checkbox", role: "switch" });
  input.addEventListener("change", () => {
    // off means "не задано", not "без детей": the request or the profile may still say so
    state.tune.kids = input.checked || null;
    syncTune();
  });
  tuneControls.kids = input;
  return h(
    "label",
    { class: "toggle" },
    input,
    h("span", { class: "toggle-track" }),
    h("span", {}, "С детьми", h("small", { class: "tune-hint" }, "места, где интересно детям, и переходы покороче")),
  );
}

function distanceControl() {
  const output = h("output", { class: "tune-value", for: "tune-km" });
  const input = h("input", { class: "slider", type: "range", id: "tune-km", min: "0", max: String(KM_MAX), step: "1" });
  input.addEventListener("input", () => {
    state.tune.km = Number(input.value) || null;
    syncTune();
  });
  tuneControls.km = input;
  tuneControls.kmOutput = output;
  return h(
    "div",
    { class: "tune-section" },
    h("div", { class: "tune-head" }, h("label", { class: "tune-label", for: "tune-km" }, "Сколько пройти пешком"), output),
    input,
    h("div", { class: "tune-scale", "aria-hidden": "true" }, h("span", {}, "не задано"), h("span", {}, `${KM_MAX} км`)),
  );
}

function buildTune(catalog) {
  tuneControls.reset = h(
    "button",
    {
      class: "btn btn-ghost btn-sm tune-reset",
      type: "button",
      onclick: () => {
        state.tune = { choice: new Map(), pace: null, kids: null, km: null };
        syncTune();
      },
    },
    "Сбросить настройки",
  );
  ui.tuneBody.replaceChildren(
    h(
      "div",
      { class: "tune-section" },
      h("p", { class: "tune-label", id: "tune-interests-label" }, "Чего больше, чего меньше"),
      h("div", { class: "chips tune-chips", role: "group", "aria-labelledby": "tune-interests-label" }, interestChips(catalog)),
      h("p", { class: "tune-hint" }, "Нажмите раз — «больше», ещё раз — «меньше», третий — без разницы."),
    ),
    h(
      "div",
      { class: "tune-section" },
      h("p", { class: "tune-label", id: "tune-pace-label" }, "Темп"),
      paceControl(catalog),
    ),
    kidsControl(),
    distanceControl(),
    h(
      "p",
      { class: "tune-hint" },
      "Что не задано здесь, агент возьмёт из запроса, а потом из профиля. Настройки сильнее и того и другого.",
    ),
    tuneControls.reset,
  );
  syncTune();
}

/* ---------- composer ---------- */

function updateBuildLabel() {
  if (state.busy) {
    ui.build.textContent = "Агент работает…";
    return;
  }
  // From a saved trip the composer starts another trip; the current one stays saved.
  ui.build.textContent = state.tripId ? "Построить новый маршрут" : "Построить маршрут";
  ui.replanNote.hidden = !state.tripId || ui.composerBody.hidden;
}

function readComposer() {
  const fields = {};
  for (const name of FIELDS) {
    const value = ui.composer.elements[name].value.trim();
    if (value) fields[name] = value;
  }
  return { request: ui.request.value.trim(), fields, preferences: preferences() };
}

function fillComposer({ request, fields, preferences: prefs }) {
  ui.request.value = request;
  for (const name of FIELDS) ui.composer.elements[name].value = fields[name] ?? "";
  const choice = new Map([...prefs.more.map((key) => [key, "more"]), ...prefs.less.map((key) => [key, "less"])]);
  state.tune = { choice, pace: prefs.pace ?? null, kids: prefs.with_kids || null, km: prefs.max_walk_km ?? null };
  syncTune();
}

/** Chips that recap the composer's settings in one glance. */
function settingsChips({ fields, preferences: prefs }) {
  const chip = (cls, text) => h("span", { class: `chip ${cls}` }, text);
  const from = fields.start_time ?? "…";
  const to = fields.end_time ?? "…";
  return [
    fields.city ? chip("is-strong", fields.city) : null,
    fields.date ? chip("", dayLabel(fields.date)) : null,
    fields.start_time || fields.end_time ? chip("", `${from}–${to}`) : null,
    fields.start_point ? chip("", `старт: ${fields.start_point}`) : null,
    prefs.more.map((key) => chip("is-interest", `+ ${interestLabel(key)}`)),
    prefs.less.map((key) => chip("is-avoid", `− ${interestLabel(key)}`)),
    prefs.pace ? chip("", paceLabel(prefs.pace).toLowerCase()) : null,
    prefs.with_kids ? chip("", "с детьми") : null,
    prefs.max_walk_km ? chip("", `до ${prefs.max_walk_km} км`) : null,
  ].flat().filter(Boolean);
}

/** While the agent works the composer folds into a recap, so the trace gets the room. */
function foldComposer() {
  if (!state.submitted) return;
  const chips = settingsChips(state.submitted);
  ui.recap.replaceChildren(
    h(
      "div",
      { class: "recap-head" },
      h("span", { class: "recap-label" }, "Запрос"),
      h("button", { class: "btn btn-ghost btn-sm", type: "button", disabled: state.busy, onclick: unfoldComposer }, "Изменить"),
    ),
    h("q", { class: "recap-text" }, state.submitted.request || NO_REQUEST_TEXT),
    h("div", { class: "chips recap-chips", hidden: !chips.length }, chips),
  );
  ui.recap.hidden = false;
  ui.composerBody.hidden = true;
  updateBuildLabel();
}

function unfoldComposer() {
  ui.recap.hidden = true;
  ui.composerBody.hidden = false;
  ui.cancelEdit.hidden = !state.submitted;
  updateBuildLabel();
  ui.request.focus();
}

function showRequestError(text) {
  ui.requestError.textContent = text;
  ui.request.setAttribute("aria-invalid", String(Boolean(text)));
}

function renderProfileNote(profile) {
  const link = (text) => h("a", { href: "/app/profile" }, text);
  if (!profile) {
    ui.profileNote.replaceChildren("Учтём ваш ", link("профиль"), ".");
    return;
  }
  const interests = Object.keys(profile.interests).length;
  const parts = [
    profile.home_city,
    interests ? count(interests, "интерес", "интереса", "интересов") : null,
    profile.avoid.length ? `не предлагать: ${profile.avoid.length}` : null,
  ].filter(Boolean);
  if (!parts.length) {
    ui.profileNote.replaceChildren("Профиль пуст: ", link("расскажите о себе"), ", и агент будет это учитывать.");
    return;
  }
  ui.profileNote.replaceChildren("Учтём ваш профиль: ", parts.join(" · "), " · ", link("изменить"));
  if (profile.home_city) {
    ui.requestHint.textContent =
      `Как другу: город, время, что любите. Можно ничего не писать — тогда агент соберёт прогулку ` +
      `по городу из профиля (${profile.home_city}) и вашим интересам.`;
  }
}

/* ---------- controls ---------- */

async function sendFeedback(feedback, excludeIds = [], title = feedback) {
  if (!state.tripId || state.busy) return false;
  const url = `/api/trips/${encodeURIComponent(state.tripId)}/feedback`;
  return run(url, { feedback, exclude_place_ids: excludeIds }, "feedback", title);
}

// The button sends only the id, no words: the server removes the stop without asking
// the LLM what was meant, and a place name is never passed off as the traveller's text.
function removeStop(place) {
  sendFeedback("", [place.id], `Убрать: ${place.name}`);
}

const submitOnCmdEnter = (form) => (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) form.requestSubmit();
};

ui.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  if (ui.build.disabled) return; // still starting, or a run is in progress
  const body = readComposer();
  // The server needs a city from somewhere: the text, the form or the profile.
  if (!body.request && !body.fields.city && !state.profile?.home_city) {
    showRequestError("Опишите прогулку или укажите город в «Уточнить».");
    ui.request.focus();
    return;
  }
  showRequestError("");
  state.submitted = body;
  // Only filled fields are sent: the agent infers the rest from the text and the profile.
  run("/api/plan", { request: body.request, ...body.fields, preferences: body.preferences }, "plan", body.request || NO_REQUEST_TEXT);
});
ui.request.addEventListener("keydown", submitOnCmdEnter(ui.composer));
ui.request.addEventListener("input", () => showRequestError(""));
ui.stop.addEventListener("click", () => state.run?.controller.abort());
ui.cancelEdit.addEventListener("click", () => {
  fillComposer(state.submitted);
  foldComposer();
});

for (const example of document.querySelectorAll("[data-example]")) {
  example.addEventListener("click", () => {
    ui.request.value = example.dataset.example;
    showRequestError("");
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

// Leaving mid-run cancels the agent on the server and loses the walk: ask first.
window.addEventListener("beforeunload", (e) => {
  if (state.busy) e.preventDefault();
});

async function checkHealth() {
  try {
    const { model } = await api("/api/health");
    ui.health.textContent = model || "модель не задана";
    ui.health.dataset.state = "ok";
  } catch {
    ui.health.textContent = "сервер недоступен";
    ui.health.dataset.state = "down";
  }
}

/* ---------- start ---------- */

async function startMap() {
  try {
    const routeMap = await createRouteMap(ui.map, mapHooks);
    routeMap.onStyleError(showMapFallback);
    return routeMap;
  } catch (error) {
    showMapFallback(error); // no WebGL, or unpkg unreachable: the rest of the page still works
    return null;
  }
}

async function main() {
  state.user = await requireUser();
  // kept percent-encoded as in the address: it goes back into an API path as is
  const tripId = location.pathname.match(/^\/app\/trips\/([^/]+)\/?$/)?.[1];
  renderNav(state.user, tripId ? "trips" : "new");
  if (tripId) document.title = "Маршрут — Прогулка";
  checkHealth();
  showEmpty(
    "Здесь появится маршрут",
    "Опишите прогулку — агент найдёт реальные места в OpenStreetMap, Википедии и Wikivoyage, " +
      "проверит часы работы и погоду и уложит всё во время.",
  );
  const [catalog, profile, routeMap] = await Promise.all([
    api("/api/catalog").catch((error) => error),
    api("/api/profile").catch(() => null), // only the note and the hint depend on it
    startMap(),
  ]);
  state.map = routeMap;
  state.profile = profile;
  renderProfileNote(profile);
  if (catalog instanceof Error) {
    ui.tuneBody.replaceChildren(h("p", { class: "alert is-error" }, `Не удалось загрузить настройки: ${catalog.message}`));
  } else {
    state.catalog = catalog;
    buildTune(catalog);
  }
  if (tripId) await loadTrip(tripId);
  // Enabled only now: a run started before the map exists would miss its first events.
  ui.build.disabled = false;
}

main().catch((error) => {
  // /api/me itself failed (server down, a 5xx): nothing on the page can work yet
  showEmpty("Не удалось открыть планировщик", error.message, h("a", { class: "btn btn-sm", href: location.pathname }, "Обновить"));
});
