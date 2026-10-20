/*
 * Прогулка — the journey planner: describe a trip, watch the agent plan it day by day,
 * then change any day.
 *
 * A journey is one to seven days in one city, each day with its own route; a single
 * walk is a one-day journey and looks exactly like one. The server streams the
 * agent's work as NDJSON: one AgentEvent {type, data} per line. Events about one day
 * carry data.day (0-based); those about the whole trip (the brief, the plan of the
 * days, memory) do not. Every event type has one handler below that turns it into
 * something visible: a stage strip, a live trace of tool calls and shortlisted places
 * tagged with their day, then the routes on the map and as timelines. The final
 * "journey" event carries the whole JourneyView. The page keeps no plan logic of its
 * own: the server owns the journey and saves every run, the page only keeps the
 * journey id (in the address too), so a feedback run on one day is just another
 * stream rendered by the same handlers, and /app/trips/{id} reopens a saved journey
 * exactly where its last run stopped.
 *
 * One selection drives the page: a day, or «Все дни». It focuses that day's route on
 * the map and dims the others, shows its timeline, filters the trace to its work and
 * names the day the feedback box changes. While the agent works the selection stays
 * free: a traveller can follow one day's research while the others run in parallel.
 *
 * The composer sends only what the traveller set for this trip: the text, the city and
 * the start point, the dates (calendar.js) and the daily window (time-range.js), and
 * the «Настроить» panel. The profile is not copied into the form: the server merges it
 * in, and the parsed brief shows where each interest came from. Voice input (voice.js)
 * only fills the text boxes; nothing is sent until the traveller presses the button.
 *
 * Text from outside (place names, Wikipedia extracts, LLM output, memory facts) only
 * ever reaches the DOM through text nodes, never innerHTML, so a place name cannot
 * inject markup.
 */

import { ApiError, api, requireUser, streamEvents } from "./api.js";
import { RangeCalendar, addDays, daysInRange, rangeLabel, rangeText, shortDay } from "./calendar.js";
import { renderNav } from "./nav.js";
import { createRouteMap, dayColor } from "./planner-map.js";
import { TimeRange, clock as minutesClock, minutesOf, spanText, wireTime } from "./time-range.js";
import { ago, dayLabel, h, plural, toast, usd } from "./ui.js";
import { dictating, voiceInput } from "./voice.js";

const $ = (selector) => document.querySelector(selector);

const SOURCES = { osm: "OpenStreetMap", wikipedia: "Википедия", wikivoyage: "Wikivoyage" };
const MAX_DAYS = 7;

// Stage names in pipeline order with short labels for the strip; the server sends a longer
// human label with each stage, which goes into the trace and the status line. "days" is
// the strip's own summary of every day's stages on a journey of several days.
const STAGES = [
  ["feedback", "Пожелание"],
  ["brief", "Запрос"],
  ["weather", "Погода"],
  ["outline", "План дней"],
  ["days", "Маршруты дней"],
  ["research", "Поиск"],
  ["optimize", "Расписание"],
  ["route", "Пути"],
  ["validate", "Проверка"],
  ["repair", "Исправление"],
  ["write", "Тексты"],
  ["memory", "Память"],
];
// Stages a one-day plan always goes through; "repair", "memory" and the stages of a
// feedback run appear in the strip only once the server actually starts them.
const PLAN_STAGES = ["brief", "weather", "research", "optimize", "route", "validate", "write"];
// A plan of several days shows the trip's own stages; each day's progress is on its tab.
const JOURNEY_STAGES = ["brief", "outline", "days"];

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
const NO_REQUEST_TEXT = "По анкете и настройкам путешествия";
const TEXT_FIELDS = ["city", "start_point"];
const KM_MAX = 20; // the distance slider: 0 is "не задано", then 1..20 km
const DEFAULT_WINDOW = ["10:00", "18:00"];

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// What a refusal before the stream means for the traveller, after the server's own words.
const REFUSAL_HINTS = {
  404: "Возможно, его удалили в другой вкладке.",
  // 409 is a run on this journey in progress, or a stop that run already removed
  409: "Возможно, путешествие меняют в другой вкладке: обновите страницу, когда там закончится запуск.",
  429: "Лимиты берегут общий ключ API демо.",
};

const state = {
  user: null,
  busy: false,
  journeyId: null,
  journey: null, // the JourneyView on screen
  day: null, // the selected day index; null is «Все дни»
  brief: null, // the running plan's parsed brief, until its journey arrives
  outline: null, // the running plan's {title, days}, until its journey arrives
  plannedDays: 1, // how many days the running plan will have, as far as is known yet
  cost: null, // the usage line under the itinerary
  reply: null, // {day, text}: the agent's answer to the last change, shown under that day only
  map: null, // RouteMap from planner-map.js, or null when the map could not load
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
  requestBox: $("#request-box"),
  requestHint: $("#request-hint"),
  requestError: $("#request-error"),
  calendar: $("#calendar"),
  timeWindow: $("#time-window"),
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
  traceFilter: $("#trace-filter"),
  hint: $("#trace-hint"),
  log: $("#log"),
  journeyBar: $("#journey-bar"),
  journeyMeta: $("#journey-meta"),
  journeyTitle: $("#journey-title"),
  dayTabs: $("#day-tabs"),
  map: $("#map"),
  legend: $("#map-legend"),
  mapFallback: $("#map-fallback"),
  plan: $("#plan"),
  feedback: $("#feedback"),
  feedbackPick: $("#feedback-pick"),
  feedbackEdit: $("#feedback-edit"),
  feedbackLabel: $("#feedback-label"),
  feedbackBox: $("#feedback-box"),
  feedbackText: $("#feedback-text"),
  feedbackHint: $("#feedback-hint"),
  rebuild: $("#rebuild"),
  reply: $("#reply"),
};

const calendar = new RangeCalendar({ maxDays: MAX_DAYS, onChange: () => updateBuildLabel() });
// The window is sent only once the traveller moved the slider: until then a time said in
// the request («с 11 до 18») wins, and without one the server uses the same 10:00–18:00.
let windowTouched = false;
const timeRange = new TimeRange({
  value: DEFAULT_WINDOW.map((time) => minutesOf(time)),
  label: "Время каждый день",
  onChange: () => (windowTouched = true),
});

/* ---------- small helpers ---------- */

const clip = (text, max) => (text.length > max ? `${text.slice(0, max - 1)}…` : text);
const hhmm = (iso) => iso.slice(11, 16); // "2026-10-24T11:12:00" -> "11:12", no timezone games
const clock = (time) => minutesClock(minutesOf(time, 1)); // "11:00:00" -> "11:00", "23:59" stays
const visitTime = (stop) => `${hhmm(stop.arrive)}–${hhmm(stop.depart)}`;
const timeWindow = (brief) => `${clock(brief.start_time)}–${clock(brief.end_time)}`;
const km = (meters) => (meters / 1000).toLocaleString("ru-RU", { maximumFractionDigits: 1 });
const distance = (meters) => (meters < 1000 ? `${meters} м` : `${km(meters)} км`);
const seconds = (ms) => `${(ms / 1000).toLocaleString("ru-RU", { maximumFractionDigits: 1 })} с`;
const count = (n, one, few, many) => `${n} ${plural(n, one, few, many)}`;
const sum = (items, value) => items.reduce((total, item) => total + value(item), 0);

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
const stageLabel = (name) => STAGES.find(([stage]) => stage === name)?.[1] ?? name;
const tag = (text) => h("span", { class: "tag" }, text);
const badge = (glyph, cls = "", title = null) => h("span", { class: `badge ${cls}`, title }, glyph);
const swatch = (day) => h("span", { class: "day-swatch", style: `--day: ${dayColor(day)}`, "aria-hidden": "true" });

function sourceLink(place) {
  const label = SOURCES[place.source] ?? place.source;
  if (!place.url || !/^https?:\/\//.test(place.url)) return h("span", { class: "source" }, label);
  return h("a", { class: "source", href: place.url, target: "_blank", rel: "noopener noreferrer" }, `${label} ↗`);
}

function violationItem(v) {
  return h("li", { class: `violation is-${v.severity}` }, h("b", {}, VIOLATIONS[v.code] ?? v.code), " ", v.message);
}

/* ---------- the journey, as far as it is known ---------- */

const dayCount = () => state.journey?.days.length ?? state.outline?.days.length ?? state.plannedDays;
const several = () => dayCount() > 1;
const plannedDay = (index) => state.journey?.days.find((day) => day.index === index);

/** The days for the tabs: the journey's, the running plan's outline, or bare dates. */
function knownDays() {
  if (state.journey) return state.journey.days.map(({ index, date, theme }) => ({ index, date, theme }));
  if (state.outline) return state.outline.days.map((day, i) => ({ index: day.index ?? i, date: day.date, theme: day.theme }));
  const from = state.brief?.date ?? state.submitted?.fields.date_from;
  return Array.from({ length: state.plannedDays }, (_, index) => ({ index, date: from ? addDays(from, index) : null, theme: "" }));
}

function journeyTitle() {
  const { journey } = state;
  if (journey) return journey.title || journey.days[0]?.plan?.title || `Путешествие: ${journey.city}`;
  if (state.outline?.title) return state.outline.title;
  const city = state.brief?.city ?? state.submitted?.fields.city;
  return city ? `Путешествие: ${city}` : "Новое путешествие";
}

function journeyMeta() {
  const { journey, brief } = state;
  const fields = state.submitted?.fields ?? {};
  const city = journey?.city ?? brief?.city ?? fields.city;
  const from = journey?.date_from ?? brief?.date ?? fields.date_from;
  const to = journey?.date_to ?? brief?.date_to ?? fields.date_to ?? from;
  return [city, from ? rangeText(from, to) : null].filter(Boolean).join(" · ");
}

/* ---------- streaming ---------- */

/** A refusal before the stream (403, 404, 409, 429) or a lost connection, in plain words. */
function refusalText(error) {
  if (!(error instanceof ApiError)) return `Не удалось обработать ответ сервера: ${error.message}`;
  const hint = REFUSAL_HINTS[error.status];
  return hint ? `${error.message}. ${hint}` : error.message;
}

/** Render one stream from request to end; resolves to whether it delivered a journey. */
async function run(url, body, kind, title, day = null) {
  setBusy(true, kind);
  beginRun(kind, title, day);
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
  return state.run.gotJourney;
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
  // Only what starts or changes a run is locked: the days, the map and the trace stay
  // usable while the agent works, and leaving the page asks first (see the bottom).
  for (const button of ui.planner.querySelectorAll("[data-run-lock]")) button.disabled = busy;
  ui.stop.hidden = !busy;
  ui.build.classList.toggle("is-busy", busy && kind === "plan");
  ui.rebuild.classList.toggle("is-busy", busy && kind === "feedback");
  updateBuildLabel();
}

function beginRun(kind, title, day) {
  state.run = {
    kind,
    day, // the day a feedback run changes
    // the stop button aborts the request; the server then cancels the run and, for a
    // feedback run, keeps the saved journey as it was, so the routes on screen stay current
    controller: new AbortController(),
    stages: new Map(), // the trip's stages: name -> "active" | "done" | "failed"
    dayStages: new Map(), // day -> the same map for that day's stages
    stageRows: new Map(), // "day:name" -> {row, startedAt}
    toolRows: new Map(), // "day:tool call id" -> row; parallel days may reuse an id
    gotJourney: false,
    failed: false,
    over: false,
    elapsed: null,
  };
  ui.hint.hidden = true;
  if (kind === "plan") {
    ui.log.replaceChildren();
    const fields = state.submitted?.fields ?? {};
    Object.assign(state, {
      journey: null,
      day: null,
      brief: null,
      outline: null,
      cost: null,
      plannedDays: fields.date_from ? daysInRange(fields.date_from, fields.date_to ?? fields.date_from) : 1,
    });
    state.placeNames.clear();
    ui.feedback.hidden = true;
    state.map?.clearRoute();
    state.map?.clearCandidates();
    state.map?.select(null);
    updateLegend();
    foldComposer();
    renderJourneyBar();
    showEmpty(
      "Агент работает",
      several()
        ? "Следите за ним слева: он делит город по дням, ищет места для каждого дня, а расписание и проверка собирают маршруты."
        : "Следите за ним слева: он ищет места, отбирает подходящие, а затем расписание и проверка соберут маршрут.",
    );
  }
  followLog = true;
  const what = kind === "plan" ? "Новое путешествие" : several() ? `Пожелание · день ${day + 1}` : "Пожелание";
  addRow("divider", null, null, what, " ", h("q", {}, clip(title, 160)));
  renderStages();
  paintDays();
  setStatus("Отправляю запрос агенту…", true);
}

function endRun() {
  const current = state.run;
  current.over = true;
  // no answer is coming, even with a journey: the research time budget can cut off a tool call
  for (const row of current.toolRows.values()) row.classList.remove("is-pending");
  if (current.gotJourney) {
    setStatus(current.elapsed ? `Готово за ${current.elapsed}` : "Готово");
  } else {
    for (const stages of [current.stages, ...current.dayStages.values()]) {
      for (const [name, status] of stages) if (status === "active") stages.set(name, "failed");
    }
    if (!current.failed) {
      const text = "Агент закончил работу, но ничего не прислал. Попробуйте ещё раз или уточните запрос.";
      addRow("error", badge("×"), null, text);
    }
    if (current.kind === "plan") {
      setStatus("Путешествие не построено");
      Object.assign(state, { outline: null, brief: null, plannedDays: 1, day: null });
      renderJourneyBar();
      filterTrace();
      showEmpty("Путешествие не построено", "Сообщения агента слева подскажут, что пошло не так. Уточните запрос.");
      unfoldComposer();
    } else {
      setStatus("Маршрут дня не изменён");
    }
  }
  renderStages();
  paintDays();
}

/* ---------- agent trace ---------- */

let followLog = true; // stop auto-scrolling once the user scrolls up to read
ui.log.addEventListener("scroll", () => {
  followLog = ui.log.scrollHeight - ui.log.scrollTop - ui.log.clientHeight < 40;
});

function follow() {
  if (followLog) ui.log.scrollTop = ui.log.scrollHeight;
}

/**
 * One trace row. `day` is the day the event was about (null for the whole trip); on a
 * trip of several days such a row carries the day's colour and number, and hides while
 * another day is selected.
 */
function addRow(cls, icon, day, ...content) {
  const mark = icon ?? h("span", { class: "badge is-blank" });
  const tagged = typeof day === "number" && several();
  const row = h(
    "li",
    {
      class: `row ${cls}${tagged ? " has-day" : ""}`,
      "data-day": typeof day === "number" ? String(day) : null,
      style: tagged ? `--day: ${dayColor(day)}` : null,
    },
    tagged
      ? h("span", { class: "row-mark" }, mark, h("span", { class: "day-dot" }, h("span", { class: "sr-only" }, "День "), String(day + 1)))
      : mark,
    h("div", { class: "row-body" }, ...content),
  );
  row.hidden = !inTrace(row, traceDay());
  ui.log.append(row);
  follow();
  return row;
}

const traceDay = () => (several() ? state.day : null);
const inTrace = (row, only) => only === null || row.dataset.day === undefined || Number(row.dataset.day) === only;

/** Show the selected day's work only; the trip's own rows always stay. */
function filterTrace() {
  const only = traceDay();
  for (const row of ui.log.children) row.hidden = !inTrace(row, only);
  ui.traceFilter.hidden = only === null;
  if (only === null) return;
  ui.traceFilter.replaceChildren(
    swatch(only),
    h("span", {}, `Только день ${only + 1}`),
    h("button", { class: "btn btn-ghost btn-sm", type: "button", onclick: () => selectDay(null) }, "Все дни"),
  );
  follow();
}

function setStatus(text, live = false) {
  ui.status.textContent = text;
  ui.status.classList.toggle("is-live", live);
}

/** The status of a stage over every day: one that runs anywhere is running. */
function mergedStages() {
  const merged = new Map(state.run.stages);
  for (const stages of state.run.dayStages.values()) {
    for (const [name, status] of stages) {
      if (merged.get(name) !== "active" && merged.get(name) !== "failed") merged.set(name, status);
    }
  }
  return merged;
}

function renderStages() {
  const { stages, kind, over } = state.run;
  let shown;
  if (kind === "plan" && several()) {
    const status = (name) => (name === "days" ? daysStatus() : stages.get(name));
    shown = STAGES.filter(([name]) => JOURNEY_STAGES.includes(name) || stages.has(name)).map(([name, label]) => [label, status(name)]);
  } else {
    // one day, or a change to one day: the trip's and the day's stages read as one pipeline
    const merged = mergedStages();
    const expected = kind === "plan" ? PLAN_STAGES : ["feedback"];
    shown = STAGES.filter(([name]) => expected.includes(name) || merged.has(name)).map(([name, label]) => [label, merged.get(name)]);
  }
  ui.steps.replaceChildren(
    ...shown.map(([label, status]) => h("li", { class: `step is-${status ?? (over ? "skipped" : "pending")}` }, label)),
  );
}

/** Where one day of the running stream is: {status, text}, or null when it is not running. */
function dayProgress(index) {
  const current = state.run;
  if (!current || current.over || (current.kind === "feedback" && current.day !== index)) return null;
  const stages = current.dayStages.get(index);
  if (!stages) return { status: "pending", text: state.outline || current.kind === "feedback" ? "в очереди" : "ждёт плана дней" };
  const active = [...stages].filter(([, status]) => status === "active").map(([name]) => name).at(-1);
  if (active) return { status: "active", text: `${stageLabel(active).toLowerCase()}…` };
  if (stages.get("write") === "done") return { status: "done", text: "маршрут готов" };
  return { status: "active", text: "в работе…" };
}

function daysStatus() {
  const all = Array.from({ length: dayCount() }, (_, index) => dayProgress(index)?.status);
  if (state.run.gotJourney || all.every((status) => status === "done")) return "done";
  if (all.some((status) => status === "active" || status === "done")) return "active";
  return state.run.over ? "failed" : null;
}

function onStage({ name, status, label, detail, day }) {
  const current = state.run;
  const onDay = typeof day === "number";
  if (onDay && !current.dayStages.has(day)) current.dayStages.set(day, new Map());
  const stages = onDay ? current.dayStages.get(day) : current.stages;
  const key = `${onDay ? day : "-"}:${name}`;
  if (status === "start") {
    stages.set(name, "active");
    const title = h("span", { class: "stage-name" }, stageLabel(name));
    const row = addRow("stage", null, onDay ? day : null, title, h("span", { class: "stage-label" }, label));
    current.stageRows.set(key, { row, startedAt: performance.now() });
    setStatus(onDay && several() ? `День ${day + 1}: ${label}` : label, true);
  } else {
    stages.set(name, "done");
    const started = current.stageRows.get(key);
    if (started) {
      started.row.classList.add("is-done");
      // the done label is the outcome ("Погода: …", "Вошло мест: 6…", "по прямой: …"),
      // which says more than the intent shown while the stage ran
      if (label) started.row.querySelector(".stage-label").textContent = label;
      started.row.querySelector(".row-body").append(h("span", { class: "stage-time" }, seconds(performance.now() - started.startedAt)));
    }
    showStageDetail(name, detail, onDay ? day : null);
  }
  renderStages();
  paintDays();
}

function showStageDetail(name, detail, day) {
  if (!detail) return;
  if (name === "brief") onBrief(detail);
  if (name === "outline") onOutline(detail);
  // the city and the start point are found: the map flies there
  if (detail.start && day === null) state.map?.showStart(detail.start);
  if (name === "research") {
    const candidates = detail.candidates ?? 0;
    addRow(
      "note",
      badge("∑", "tool-finish"),
      day,
      `Поиск закончен: ${count(detail.steps, "шаг", "шага", "шагов")}, `,
      count(candidates, "кандидат", "кандидата", "кандидатов"),
      detail.stop_reason ? h("code", { class: "stop-reason" }, detail.stop_reason) : null,
    );
  }
}

function onBrief(brief) {
  state.brief = brief;
  if (brief.date && brief.date_to) state.plannedDays = Math.min(MAX_DAYS, daysInRange(brief.date, brief.date_to));
  renderBrief(brief);
  renderJourneyBar();
}

/** The plan of the days arrives before any day is researched: tabs, a board and day starts. */
function onOutline(outline) {
  state.outline = { title: outline.title ?? "", days: outline.days ?? [] };
  state.plannedDays = state.outline.days.length || state.plannedDays;
  renderOutline(state.outline, "План путешествия");
  renderJourneyBar();
  renderBoard();
  state.map?.setDayStarts(state.outline.days.map((day) => day.start ?? null));
  updateLegend();
}

/** The parsed brief as chips; interests are grouped by where they came from. */
function renderBrief(brief, title = "Агент понял запрос так") {
  const chip = (cls, ...content) => h("span", { class: `chip ${cls}` }, ...content);
  const dates = brief.date_to && brief.date_to !== brief.date ? rangeLabel(brief.date, brief.date_to) : dayLabel(brief.date);
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
    null,
    h("p", { class: "brief-title" }, title),
    h(
      "div",
      { class: "chips" },
      chip("is-strong", [brief.city, dates].join(", ")),
      chip("is-strong", several() ? `каждый день ${timeWindow(brief)}` : timeWindow(brief)),
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

/** The plan of the days: what each day is about, where it centres, which interests lead. */
function renderOutline(outline, heading) {
  addRow(
    "note outline",
    badge("☰", "tool-finish", "План по дням"),
    null,
    h("p", { class: "brief-title" }, heading),
    outline.title ? h("p", { class: "outline-title" }, outline.title) : null,
    h(
      "ol",
      { class: "outline-days" },
      outline.days.map((day, i) =>
        h(
          "li",
          { style: `--day: ${dayColor(day.index ?? i)}` },
          h("span", { class: "outline-num", "aria-hidden": "true" }, (day.index ?? i) + 1),
          h(
            "div",
            {},
            h("p", { class: "outline-theme" }, day.theme || `День ${(day.index ?? i) + 1}`),
            h("p", { class: "outline-meta" }, outlineMeta(day)),
          ),
        ),
      ),
    ),
  );
}

function outlineMeta(day) {
  return [
    day.date ? shortDay(day.date) : null,
    day.area ? `район: ${day.area}` : null,
    day.focus?.length ? `главное: ${day.focus.join(", ")}` : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

function onThought({ text, day }) {
  if (!text?.trim()) return;
  const toggle = (e) => e.currentTarget.classList.toggle("is-open"); // long thoughts are clamped to 4 lines
  addRow("thought", null, day, h("p", { title: "Нажмите, чтобы раскрыть", onclick: toggle }, text.trim()));
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

function onToolCall({ id, name, args, day }) {
  const row = addRow(
    "tool is-pending",
    toolBadge(name),
    day,
    h(
      "div",
      { class: "tool-head" },
      h("code", { class: "tool-name" }, name),
      h("span", { class: "tool-args" }, briefArgs(args)),
    ),
  );
  state.run.toolRows.set(`${day}:${id}`, row);
  return row;
}

function onToolResult({ id, name, summary, count: found, error, day }) {
  const row = state.run.toolRows.get(`${day}:${id}`) ?? onToolCall({ id, name, args: null, day });
  row.classList.remove("is-pending");
  row.classList.toggle("is-error", Boolean(error));
  const text = typeof error === "string" ? error : summary;
  const counter = found != null ? h("span", { class: "count" }, found) : null;
  row.querySelector(".row-body").append(h("div", { class: "tool-result" }, counter, text ?? ""));
  follow();
}

function onCandidate({ action, place_id: id, name, kind, lat, lon, score, interests = [], reason, day }) {
  state.placeNames.set(id, name);
  const added = action === "add";
  const onDay = typeof day === "number" ? day : null;
  if (added && typeof lat === "number") state.map?.addCandidate(id, lat, lon, `${name} · ${kindLabel(kind)}`, onDay);
  if (!added) state.map?.removeCandidate(id);
  updateLegend();
  addRow(
    `candidate ${added ? "is-add" : "is-remove"}`,
    badge(added ? "+" : "−", "", added ? "В шорт-лист" : "Из шорт-листа"),
    onDay,
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
    addRow("memory", badge("✎", "tool-memory", "Память агента"), null, h("span", { class: "memory-label" }, "Запомнил о вас: "), fact);
  }
}

function onWarning({ message, day }) {
  addRow("warning", badge("!"), day, message);
}

function onError({ message, day = null }) {
  state.run.failed = true;
  addRow("error", badge("×"), day, message);
}

function onDone({ usage, elapsed_s: elapsed }) {
  const took = typeof elapsed === "number" ? seconds(elapsed * 1000) : null;
  state.run.elapsed = took;
  // a failed run ends with "done" too, to report its time and tokens: it is not «Готово»
  const what = state.run.gotJourney ? "Готово" : "Запуск окончен";
  const text = took ? `${what} за ${took}` : what;
  addRow("note", badge("■", "tool-finish"), null, text, usage ? ` · ${usageText(usage)}` : "");
}

function onJourney({ journey_id: id, journey, usage, reply, learned = [] }) {
  if (!journey) return; // endRun reports a run that ended without a journey
  state.run.gotJourney = true;
  state.cost = usage ? usageText(usage) : null;
  showJourney(id ?? journey.id, journey);
  showReply(reply, state.run.day);
  const plans = journey.days.map((day) => day.plan);
  const stops = sum(plans, (plan) => plan.stops.length);
  const walked = distance(sum(plans, (plan) => plan.total_walk_m));
  const stopsText = count(stops, "остановка", "остановки", "остановок");
  const text =
    state.run.kind === "feedback"
      ? `День ${state.run.day + 1} перестроен`
      : several()
        ? `Путешествие собрано: ${count(journey.days.length, "день", "дня", "дней")}, ${stopsText}, ${walked} пешком`
        : `Маршрут собран: ${stopsText}, ${walked} пешком`;
  addRow("note plan-ready", badge("✓", "tool-shortlist"), state.run.kind === "feedback" ? state.run.day : null, text);
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
  journey: onJourney,
  done: onDone,
};

/* ---------- map ---------- */

const mapHooks = {
  popup: popupContent,
  startPopup: (point) =>
    h(
      "div",
      { class: "popup" },
      h("p", { class: "popup-meta" }, several() ? "Отсюда начинаются дни" : startMeta()),
      h("h3", {}, point.label ?? "Точка старта"),
    ),
  ghostTitle: (id) => `${state.placeNames.get(id) ?? "Место"} — не вошло в маршрут`,
  onHover: highlight,
  onSelect: (id, day) => {
    if (several() && state.day !== day) selectDay(day);
    selectCard(id);
  },
  onDay: (day) => {
    if (several() && state.journey && state.day !== day) selectDay(day);
  },
};

// the start popup is made with the journey's markers, so the journey is known
const startMeta = () => `Старт · ${clock(state.journey.days[0].plan.brief.start_time)}`;

function popupContent(stop, index, day) {
  const place = stop.place;
  const meta = [several() ? `День ${day + 1}` : null, index + 1, visitTime(stop), kindLabel(place.kind)];
  return h(
    "div",
    { class: "popup", style: `--day: ${dayColor(day)}` },
    h("p", { class: "popup-meta" }, meta.filter((part) => part !== null).join(" · ")),
    h("h3", {}, place.name),
    h("p", {}, clip(stop.blurb || stop.reason, 240)),
    sourceLink(place),
  );
}

function showMapFallback(error) {
  ui.mapFallback.replaceChildren(
    h("p", { class: "map-fallback-title" }, "Карта не загрузилась"),
    h("p", {}, "Маршруты и расписание работают и без неё."),
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
  if (state.journey) items.push(["start", several() ? "ваш старт" : "старт"], ["stop", "остановка"]);
  if (markers.dayStarts) items.push(["daystart", "начало дня"]);
  if (markers.live) items.push(["cand", "найдено агентом"]);
  if (markers.ghosts) items.push(["ghost", "не вошло в маршрут"]);
  const all = several() && state.day === null;
  // the stop mark wears the selected day's colour, or a slice of every day's
  const days = dayCount();
  const slices = Array.from({ length: days }, (_, i) => `${dayColor(i)} ${(i / days) * 100}% ${((i + 1) / days) * 100}%`);
  const stopStyle = all ? `background: conic-gradient(${slices.join(", ")})` : `--day: ${dayColor(state.day ?? 0)}`;
  ui.legend.replaceChildren(
    ...items.map(([kind, label]) =>
      h("li", {}, h("span", { class: `legend-mark is-${kind}`, style: kind === "stop" ? stopStyle : null, "aria-hidden": "true" }), label),
    ),
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

/* ---------- days: tabs, selection, the board ---------- */

/** Select a day (null: every day) everywhere at once: tabs, map, itinerary, trace, feedback. */
function selectDay(day) {
  state.day = several() ? day : state.journey ? 0 : null;
  syncTabs();
  state.map?.select(state.day);
  filterTrace();
  renderItinerary();
  renderFeedbackTarget();
  updateLegend();
}

function renderJourneyBar() {
  const multi = several();
  ui.journeyBar.hidden = !multi;
  ui.planner.classList.toggle("has-journey", multi);
  if (!multi) return;
  ui.journeyMeta.textContent = journeyMeta();
  ui.journeyTitle.textContent = journeyTitle();
  const focused = ui.dayTabs.contains(document.activeElement) ? document.activeElement.dataset.tab : null;
  const days = knownDays();
  ui.dayTabs.replaceChildren(
    tab("all", "Все дни", null, h("span", { class: "day-swatch is-all", "aria-hidden": "true" }, days.map(({ index }) => h("i", { style: `--day: ${dayColor(index)}` })))),
    ...days.map((day) => tab(String(day.index), `День ${day.index + 1}`, day, swatch(day.index))),
  );
  syncTabs();
  paintDays();
  if (focused) ui.dayTabs.querySelector(`[data-tab="${focused}"]`)?.focus();
}

function tab(key, label, day, mark) {
  const index = day ? day.index : null;
  return h(
    "button",
    {
      class: `day-tab${day ? "" : " is-all"}`,
      type: "button",
      role: "tab",
      id: `tab-${key}`,
      "aria-controls": "plan",
      "data-tab": key,
      style: day ? `--day: ${dayColor(index)}` : null,
      onclick: () => selectDay(index),
    },
    h("span", { class: "day-tab-top" }, mark, h("b", {}, label), day?.date ? h("span", { class: "day-tab-date" }, shortDay(day.date)) : null),
    h("span", { class: "day-tab-sub", "data-progress": key }),
  );
}

function syncTabs() {
  const key = state.day === null ? "all" : String(state.day);
  for (const button of ui.dayTabs.querySelectorAll("[role=tab]")) {
    const selected = button.dataset.tab === key;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  }
  if (several()) {
    ui.plan.setAttribute("role", "tabpanel");
    ui.plan.setAttribute("aria-labelledby", `tab-${key}`);
  } else {
    ui.plan.removeAttribute("role");
    ui.plan.removeAttribute("aria-labelledby");
  }
}

/** What a day's tab and board card say under its name: progress while it runs, else its theme. */
function daySubline(index) {
  const progress = dayProgress(index);
  if (progress) return progress;
  const planned = plannedDay(index);
  if (planned) {
    const { title, stops } = planned.plan;
    return { status: "ready", text: planned.theme || title || count(stops.length, "остановка", "остановки", "остановок") };
  }
  const theme = state.outline?.days[index]?.theme;
  return { status: state.run?.over ? "failed" : "pending", text: theme || (state.run?.over ? "не построен" : "ждёт плана дней") };
}

/** Refresh every day's progress line in place: tabs and board cards are not rebuilt. */
function paintDays() {
  for (const element of document.querySelectorAll("[data-progress]")) {
    const key = element.dataset.progress;
    const owner = element.closest(".day-tab, .day-card");
    if (key === "all") {
      element.textContent = state.journey ? count(state.journey.days.length, "маршрут", "маршрута", "маршрутов") + " на карте" : "весь город";
      continue;
    }
    const { status, text } = daySubline(Number(key));
    element.textContent = text;
    owner.dataset.status = status;
  }
}

ui.dayTabs.addEventListener("keydown", (e) => {
  const keys = [...ui.dayTabs.querySelectorAll("[role=tab]")].map((button) => button.dataset.tab);
  const at = keys.indexOf(document.activeElement?.dataset.tab);
  if (at < 0) return;
  const moves = { ArrowLeft: at - 1, ArrowRight: at + 1, Home: 0, End: keys.length - 1 };
  if (!(e.key in moves)) return;
  e.preventDefault();
  const key = keys[(moves[e.key] + keys.length) % keys.length];
  selectDay(key === "all" ? null : Number(key));
  ui.dayTabs.querySelector(`[data-tab="${key}"]`)?.focus();
});

/** While a plan of several days runs: the outline as cards whose status lines follow the days. */
function renderBoard() {
  if (state.journey || !state.outline || !several()) return;
  state.cards.clear();
  ui.plan.style.removeProperty("--day");
  ui.plan.replaceChildren(
    h(
      "header",
      { class: "plan-head" },
      h("p", { class: "eyebrow" }, journeyMeta()),
      h("h2", { class: "plan-title" }, journeyTitle()),
      h("p", { class: "plan-summary" }, "Агент разделил город по дням и ищет места для каждого дня параллельно. Маршруты появятся, когда будут готовы все дни."),
    ),
    h(
      "ol",
      { class: "days" },
      state.outline.days.map((day, i) => {
        const index = day.index ?? i;
        return h(
          "li",
          { class: `day-card${state.day === index ? " is-selected" : ""}`, style: `--day: ${dayColor(index)}; --i: ${index}` },
          h("span", { class: "day-num", "aria-hidden": "true" }, index + 1),
          h(
            "div",
            { class: "day-card-body" },
            h("p", { class: "day-card-meta" }, `День ${index + 1}${day.date ? ` · ${dayLabel(day.date)}` : ""}`),
            h("h3", { class: "day-card-title" }, day.theme || `День ${index + 1}`),
            outlineMeta(day) ? h("p", { class: "day-card-stats" }, outlineMeta(day)) : null,
            h("p", { class: "day-card-progress", "data-progress": String(index) }),
          ),
        );
      }),
    ),
  );
  paintDays();
}

/* ---------- itinerary ---------- */

function showEmpty(title, text, ...actions) {
  state.cards.clear();
  ui.plan.style.removeProperty("--day");
  ui.plan.replaceChildren(
    h("div", { class: "empty" }, h("h2", {}, title), h("p", {}, text), actions.length ? h("div", { class: "empty-actions" }, actions) : null),
  );
}

function renderItinerary() {
  if (!state.journey) {
    renderBoard();
    return;
  }
  if (several() && state.day === null) renderAllDays();
  else renderDay(state.day ?? state.journey.days[0].index);
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

function stopItem(stop, index, violations, day) {
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
            "data-run-lock": "",
            disabled: state.busy,
            onclick: (e) => {
              e.stopPropagation(); // the card itself reacts to clicks by focusing the map
              removeStop(place, day);
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

/** One day's itinerary: its header, then the timeline of stops and walks. */
function renderDay(index) {
  state.cards.clear();
  const day = plannedDay(index) ?? state.journey.days[0];
  const multi = several();
  ui.plan.style.setProperty("--day", dayColor(day.index));
  const plan = day.plan;
  const { brief, stops } = plan;
  // a violation sits on its stop's card; one about the whole route, or about a place
  // that is not a stop, goes to the header so it is never lost
  const stopIds = new Set(stops.map((stop) => stop.place.id));
  const general = plan.violations.filter((v) => !stopIds.has(v.place_id));
  const violationsOf = (id) => plan.violations.filter((v) => v.place_id === id);
  const last = stops.at(-1);
  const outline = state.journey.outline?.find((item) => item.index === day.index);
  const eyebrow = multi
    ? [`День ${day.index + 1} из ${dayCount()}`, dayLabel(day.date ?? brief.date), timeWindow(brief)]
    : [brief.city, dayLabel(brief.date), timeWindow(brief)];
  const theme = day.theme && day.theme !== plan.title ? day.theme : null;
  // one day has no plan of days to explain: its focus is simply every interest
  const focus = multi && (theme || outline?.area || outline?.focus?.length);

  ui.plan.replaceChildren(
    h(
      "header",
      { class: "plan-head" },
      h("p", { class: "eyebrow" }, eyebrow.join(" · ")),
      h("h2", { class: "plan-title" }, plan.title || day.theme || `Прогулка: ${brief.city}`),
      focus
        ? h(
            "p",
            { class: "day-focus" },
            theme ? h("span", {}, theme) : null,
            outline?.area ? h("span", {}, `район: ${outline.area}`) : null,
            outline?.focus?.length ? h("span", {}, `главное: ${outline.focus.join(", ")}`) : null,
          )
        : null,
      plan.summary ? h("p", { class: "plan-summary" }, plan.summary) : null,
      h(
        "dl",
        { class: "stats" },
        stat(stops.length, plural(stops.length, "остановка", "остановки", "остановок")),
        stat(km(plan.total_walk_m), "км пешком"),
        stat(spanText(plan.total_minutes), multi ? "весь день" : "вся прогулка"),
      ),
      general.length ? h("ul", { class: "violations" }, general.map(violationItem)) : null,
      state.cost && !multi ? h("p", { class: "usage" }, state.cost) : null,
    ),
    h(
      "ol",
      { class: "timeline" },
      endpointItem(clock(brief.start_time), `Старт · ${plan.start.label ?? "точка старта"}`),
      stops.map((stop, i) => [legItem(stop, i), stopItem(stop, i, violationsOf(stop.place.id), day.index)]),
      last ? endpointItem(hhmm(last.depart), multi ? `Конец дня ${day.index + 1}` : "Конец прогулки", "is-end", stops.length) : null,
    ),
    plan.dropped.length ? h("p", { class: "dropped" }, h("b", {}, "Не поместились: "), droppedText(plan.dropped)) : null,
    multi ? dayNav(day.index) : null,
  );
  ui.plan.scrollTop = 0;
}

/** «← День 1 · День 3 →» under a day: the trip reads in order without the tabs. */
function dayNav(index) {
  const days = state.journey.days;
  const at = days.findIndex((day) => day.index === index);
  const link = (day, cls, text) =>
    day
      ? h(
          "button",
          { class: `day-step ${cls}`, type: "button", style: `--day: ${dayColor(day.index)}`, onclick: () => selectDay(day.index) },
          h("small", {}, text),
          h("span", {}, `День ${day.index + 1}${day.date ? ` · ${shortDay(day.date)}` : ""}`),
        )
      : h("span");
  return h("nav", { class: "day-nav", "aria-label": "Соседние дни" }, link(days[at - 1], "is-prev", "← Предыдущий"), link(days[at + 1], "is-next", "Следующий →"));
}

/** «Все дни»: the whole trip at a glance, one card per day; a card opens its day. */
function renderAllDays() {
  state.cards.clear();
  ui.plan.style.removeProperty("--day");
  const { journey } = state;
  const plans = journey.days.map((day) => day.plan);
  const stops = sum(plans, (plan) => plan.stops.length);
  ui.plan.replaceChildren(
    h(
      "header",
      { class: "plan-head" },
      h("p", { class: "eyebrow" }, journeyMeta()),
      h("h2", { class: "plan-title" }, journeyTitle()),
      h(
        "dl",
        { class: "stats" },
        stat(journey.days.length, plural(journey.days.length, "день", "дня", "дней")),
        stat(stops, plural(stops, "остановка", "остановки", "остановок")),
        stat(km(sum(plans, (plan) => plan.total_walk_m)), "км пешком"),
      ),
      state.cost ? h("p", { class: "usage" }, state.cost) : null,
    ),
    h("ol", { class: "days" }, journey.days.map(dayCard)),
  );
  ui.plan.scrollTop = 0;
}

function dayCard(day) {
  const plan = day.plan;
  const open = h(
    "button",
    { class: "day-card-open", type: "button", onclick: () => selectDay(day.index) },
    `День ${day.index + 1}${day.date ? ` · ${dayLabel(day.date)}` : ""}`,
  );
  const errors = plan.violations.filter((v) => v.severity === "error").length;
  const last = plan.stops.at(-1);
  const names = plan.stops.map((stop) => stop.place.name);
  return h(
    "li",
    { class: "day-card", style: `--day: ${dayColor(day.index)}; --i: ${day.index}` },
    h("span", { class: "day-num", "aria-hidden": "true" }, day.index + 1),
    h(
      "div",
      { class: "day-card-body" },
      h("p", { class: "day-card-meta" }, open, last ? h("time", {}, `${clock(plan.brief.start_time)}–${hhmm(last.depart)}`) : null),
      h("h3", { class: "day-card-title" }, plan.title || day.theme || `День ${day.index + 1}`),
      day.theme && plan.title && day.theme !== plan.title ? h("p", { class: "day-card-theme" }, day.theme) : null,
      h(
        "p",
        { class: "day-card-stats" },
        [count(plan.stops.length, "остановка", "остановки", "остановок"), `${km(plan.total_walk_m)} км пешком`, spanText(plan.total_minutes)].join(" · "),
      ),
      names.length ? h("p", { class: "day-card-stops" }, names.join(" → ")) : null,
      errors ? h("p", { class: "day-card-warn" }, `${count(errors, "проблема", "проблемы", "проблем")} в маршруте`) : null,
    ),
  );
}

function showReply(text, day) {
  state.reply = text ? { day, text } : null;
  renderReply();
}

function renderReply() {
  const { reply } = state;
  const shown = Boolean(reply) && (!several() || state.day === reply.day);
  ui.reply.replaceChildren(h("b", {}, "Агент: "), shown ? reply.text : "");
  ui.reply.hidden = !shown;
}

/** The feedback box changes one day: the selected one, or the traveller picks one first. */
function renderFeedbackTarget() {
  if (!state.journey) {
    ui.feedback.hidden = true;
    return;
  }
  ui.feedback.hidden = false;
  renderReply();
  const picking = several() && state.day === null;
  ui.feedbackPick.hidden = !picking;
  ui.feedbackEdit.hidden = picking;
  if (picking) {
    ui.feedbackPick.replaceChildren(
      h("p", { class: "feedback-title" }, "Что изменить? Выберите день"),
      h(
        "div",
        { class: "chips" },
        state.journey.days.map((day) =>
          h(
            "button",
            {
              class: "chip pick-day",
              type: "button",
              onclick: () => {
                selectDay(day.index);
                ui.feedbackText.focus();
              },
            },
            swatch(day.index),
            `День ${day.index + 1}`,
          ),
        ),
      ),
    );
    return;
  }
  ui.feedbackLabel.textContent = several() ? `Что изменить в дне ${state.day + 1}?` : "Что изменить?";
  ui.feedbackHint.textContent = several()
    ? "Агент продолжит с того же места и не повторит места других дней. ⌘/Ctrl + Enter — отправить."
    : "Агент продолжит с того же места: найденные места он помнит. ⌘/Ctrl + Enter — отправить.";
}

/** The journey on screen: remember it, put it in the address and draw every part of it. */
function showJourney(id, journey) {
  state.journey = journey;
  state.outline = null;
  state.brief = null;
  for (const day of journey.days) for (const stop of day.plan.stops) state.placeNames.set(stop.place.id, stop.place.name);
  document.title = `${journeyTitle()} — Прогулка`;
  if (id && id !== state.journeyId) {
    state.journeyId = id;
    // replaceState, not pushState: "back" leads where the traveller came from, not to an empty form
    history.replaceState(null, "", `/app/trips/${encodeURIComponent(id)}`);
    renderNav(state.user, "trips");
  }
  updateBuildLabel();
  // a change to one day keeps that day in view; a new journey opens on all its days
  if (journey.days.length < 2) state.day = journey.days[0]?.index ?? 0;
  else if (state.day !== null && !plannedDay(state.day)) state.day = null;
  renderJourneyBar();
  state.map?.showJourney(journey, state.day);
  filterTrace();
  renderItinerary();
  renderFeedbackTarget();
  updateLegend();
}

/* ---------- a saved journey ---------- */

/** Reopen a journey: its days, the history of runs and the request it started from. */
function openJourney({ journey, history: runs, request, preferences }) {
  // The composer recaps the request; a journey planned without text had its city from
  // the form or the profile, so the city keeps «Построить новое путешествие» working.
  // Dates are not carried over: a saved trip's days are usually in the past.
  const { brief } = journey.days[0].plan;
  const fields = request ? {} : { city: journey.city };
  Object.assign(fields, { start_time: clock(brief.start_time), end_time: clock(brief.end_time) });
  state.submitted = { request, fields, preferences };
  fillComposer(state.submitted);
  foldComposer();
  const total = sum(runs, (entry) => entry.cost_usd ?? 0);
  state.cost = `${count(runs.length, "запуск", "запуска", "запусков")} агента · ${usd(total)}`;
  state.day = null;
  showJourney(journey.id, journey);
  renderHistory(runs, journey);
  const last = runs.at(-1);
  showReply(last?.kind === "feedback" ? last.reply : null, last?.day);
  setStatus(last ? `Сохранённое путешествие · изменено ${ago(last.at)}` : "Сохранённое путешествие");
}

function renderHistory(runs, journey) {
  const multi = several();
  ui.hint.hidden = true;
  addRow("divider", null, null, multi ? "История путешествия" : "История маршрута");
  for (const entry of runs) {
    const isPlan = entry.kind === "plan";
    const day = typeof entry.day === "number" ? entry.day : null;
    const what = isPlan ? (multi ? "Путешествие построено" : "Маршрут построен") : multi && day !== null ? `Правка дня ${day + 1}` : "Правка";
    addRow(
      "history",
      badge(isPlan ? "◎" : "✎", isPlan ? "tool-finish" : "tool-shortlist", what),
      isPlan ? null : day,
      h(
        "p",
        { class: "history-meta" },
        what,
        h("span", { title: new Date(entry.at).toLocaleString("ru-RU") }, ` · ${ago(entry.at)}`),
        entry.cost_usd ? ` · ${usd(entry.cost_usd)}` : "",
      ),
      h("q", { class: "history-text" }, clip(entry.text, 240)),
      entry.reply ? h("p", { class: "history-reply" }, h("b", {}, "Агент: "), entry.reply) : null,
    );
  }
  if (multi && journey.outline?.length) renderOutline({ title: journey.title, days: journey.outline }, "Дни путешествия");
  if (!multi) renderBrief(journey.days[0].plan.brief, "Сейчас маршрут учитывает");
  const next = multi
    ? "Выберите день и напишите справа, что в нём изменить: агент продолжит с того же места и не повторит места других дней."
    : "Напишите, что изменить, справа под маршрутом: агент продолжит с того же места.";
  addRow("note", badge("→", "tool-finish"), null, next);
}

/** `pathId` is the address segment, still percent-encoded. */
async function loadJourney(pathId) {
  showEmpty("Открываю путешествие…", "Загружаю сохранённые маршруты.");
  try {
    openJourney(await api(`/api/journeys/${pathId}`));
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) await requireUser();
    const missing = error instanceof ApiError && error.status === 404;
    showEmpty(
      missing ? "Путешествие не найдено" : "Не удалось открыть путешествие",
      missing ? "Возможно, его удалили или оно сохранено в другом аккаунте." : error.message,
      h("a", { class: "btn btn-sm", href: "/app/trips" }, "Мои путешествия"),
      h("a", { class: "btn btn-sm btn-primary", href: "/app/new" }, "Новое путешествие"),
    );
  }
}

/* ---------- the «Настроить путешествие» panel ---------- */

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
    tuneControls.kmOutput.textContent = limit ? `до ${limit} км в день` : "не задано";
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
  const range = calendar.value;
  const oneDay = range !== null && range.from === range.to;
  // From a saved journey the composer starts another one; the current one stays saved.
  if (state.journeyId) ui.build.textContent = oneDay ? "Построить новый маршрут" : "Построить новое путешествие";
  else ui.build.textContent = oneDay ? "Построить маршрут" : "Построить путешествие";
  ui.replanNote.hidden = !state.journeyId || ui.composerBody.hidden;
}

function readComposer() {
  const fields = {};
  for (const name of TEXT_FIELDS) {
    const value = ui.composer.elements[name].value.trim();
    if (value) fields[name] = value;
  }
  const range = calendar.value;
  if (range) Object.assign(fields, { date_from: range.from, date_to: range.to });
  if (windowTouched) {
    const { start, end } = timeRange.value;
    Object.assign(fields, { start_time: wireTime(start), end_time: wireTime(end) });
  }
  return { request: ui.request.value.trim(), fields, preferences: preferences() };
}

function fillComposer({ request, fields, preferences: prefs }) {
  ui.request.value = request;
  for (const name of TEXT_FIELDS) ui.composer.elements[name].value = fields[name] ?? "";
  calendar.set(fields.date_from ? { from: fields.date_from, to: fields.date_to ?? fields.date_from } : null);
  const [start, end] = [fields.start_time ?? DEFAULT_WINDOW[0], fields.end_time ?? DEFAULT_WINDOW[1]];
  timeRange.set(minutesOf(start), minutesOf(end));
  windowTouched = Boolean(fields.start_time);
  const choice = new Map([...prefs.more.map((key) => [key, "more"]), ...prefs.less.map((key) => [key, "less"])]);
  state.tune = { choice, pace: prefs.pace ?? null, kids: prefs.with_kids || null, km: prefs.max_walk_km ?? null };
  syncTune();
  updateBuildLabel();
}

/** Chips that recap the composer's settings in one glance. */
function settingsChips({ fields, preferences: prefs }) {
  const chip = (cls, text) => h("span", { class: `chip ${cls}` }, text);
  return [
    fields.city ? chip("is-strong", fields.city) : null,
    fields.date_from ? chip("", rangeLabel(fields.date_from, fields.date_to ?? fields.date_from)) : null,
    fields.start_time ? chip("", `${clock(fields.start_time)}–${clock(fields.end_time)}`) : null,
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
      h("button", { class: "btn btn-ghost btn-sm", type: "button", "data-run-lock": "", disabled: state.busy, onclick: unfoldComposer }, "Изменить"),
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
      `Как другу: что любите, что не надо, с кем едете. Город, даты и время — ниже. Можно ничего ` +
      `не писать — тогда агент соберёт маршрут по городу из профиля (${profile.home_city}) и вашим интересам.`;
  }
}

/* ---------- controls ---------- */

async function sendFeedback(day, feedback, excludeIds = [], title = feedback) {
  if (!state.journeyId || state.busy) return false;
  const url = `/api/journeys/${encodeURIComponent(state.journeyId)}/days/${day}/feedback`;
  return run(url, { feedback, exclude_place_ids: excludeIds }, "feedback", title, day);
}

// The button sends only the id, no words: the server removes the stop without asking
// the LLM what was meant, and a place name is never passed off as the traveller's text.
function removeStop(place, day) {
  sendFeedback(day, "", [place.id], `Убрать: ${place.name}`);
}

// A form sent mid-dictation would go without the words being spoken, and the composer
// would fold over a microphone that is still recording.
const DICTATING = "Сначала закончите диктовку: нажмите на микрофон и дождитесь текста.";
const dictation = new Map(); // text box -> its microphone (voice.js), where the browser can record

const submitOnCmdEnter = (form) => (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) form.requestSubmit();
};

ui.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  if (ui.build.disabled) return; // still starting, or a run is in progress
  if (dictating(dictation.get(ui.request))) {
    showRequestError(DICTATING);
    return;
  }
  const body = readComposer();
  // The server needs a city from somewhere: the text, the form or the profile.
  if (!body.request && !body.fields.city && !state.profile?.home_city) {
    showRequestError("Расскажите о поездке или укажите город.");
    ui.request.focus();
    return;
  }
  showRequestError("");
  state.submitted = body;
  // An empty field or unpicked dates are not sent: the agent takes them from the text,
  // then the profile. The daily window is always sent (see readComposer).
  run("/api/plan", { request: body.request, ...body.fields, preferences: body.preferences }, "plan", body.request || NO_REQUEST_TEXT);
});
ui.request.addEventListener("keydown", submitOnCmdEnter(ui.composer));
ui.request.addEventListener("input", () => showRequestError(""));
ui.composer.elements.city.addEventListener("input", () => showRequestError(""));
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
  if (dictating(dictation.get(ui.feedbackText))) {
    toast(DICTATING);
    return;
  }
  const text = ui.feedbackText.value.trim();
  if (!text || state.busy || (several() && state.day === null)) return;
  // The wish stays in the box until the day is rebuilt, so a failed attempt is one
  // click to retry; a new wish typed meanwhile is kept too.
  if ((await sendFeedback(state.day ?? 0, text)) && ui.feedbackText.value.trim() === text) ui.feedbackText.value = "";
});
ui.feedbackText.addEventListener("keydown", submitOnCmdEnter(ui.feedback));

// Leaving mid-run cancels the agent on the server and loses the run: ask first.
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

function mountInputs() {
  ui.calendar.append(calendar.element);
  ui.timeWindow.append(timeRange.element);
  for (const [box, field, label] of [
    [ui.requestBox, ui.request, "Надиктовать запрос"],
    [ui.feedbackBox, ui.feedbackText, "Надиктовать, что изменить"],
  ]) {
    const voice = voiceInput(field, { label });
    if (voice) {
      box.append(voice);
      dictation.set(field, voice);
    }
    box.classList.toggle("has-voice", Boolean(voice));
  }
}

async function main() {
  mountInputs();
  state.user = await requireUser();
  // kept percent-encoded as in the address: it goes back into an API path as is
  const journeyId = location.pathname.match(/^\/app\/trips\/([^/]+)\/?$/)?.[1];
  renderNav(state.user, journeyId ? "trips" : "new");
  if (journeyId) document.title = "Путешествие — Прогулка";
  checkHealth();
  updateBuildLabel();
  showEmpty(
    "Здесь появится путешествие",
    "Агент найдёт реальные места в OpenStreetMap, Википедии и Wikivoyage, проверит часы работы " +
      "и погоду и разложит город по дням — без вечеров над путеводителем перед поездкой.",
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
  if (journeyId) await loadJourney(journeyId);
  // Enabled only now: a run started before the map exists would miss its first events.
  ui.build.disabled = false;
}

main().catch((error) => {
  // /api/me itself failed (server down, a 5xx): nothing on the page can work yet
  showEmpty("Не удалось открыть планировщик", error.message, h("a", { class: "btn btn-sm", href: location.pathname }, "Обновить"));
});
