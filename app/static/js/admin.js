/*
 * The admin page: how many people use the service, what the LLM costs and how close
 * the public deployment is to its limits. The API serves this to admins only; anyone
 * else gets 403, which the page turns into a plain explanation rather than an error.
 */

import { ApiError, api, requireUser } from "./api.js";
import { renderNav } from "./nav.js";
import { ago, h, plural, usd } from "./ui.js";

const root = document.getElementById("admin");
const refresh = document.getElementById("refresh");
const updated = document.getElementById("admin-updated");

const number = (value) => value.toLocaleString("ru-RU");

function tile(label, value, note, extra = null) {
  return h(
    "div",
    { class: "tile" },
    h("p", { class: "tile-label" }, label),
    h("p", { class: "tile-value" }, value),
    note ? h("p", { class: "tile-note" }, note) : null,
    extra,
  );
}

/** A thin bar for "this much of the limit is used"; the numbers are in the text next to it. */
function meter(used, limit) {
  const share = limit > 0 ? Math.min(used / limit, 1) : 0;
  const level = share >= 0.9 ? "is-high" : share >= 0.6 ? "is-mid" : "";
  return h("div", { class: `meter ${level}`, "aria-hidden": "true" }, h("span", { style: `--fill: ${share * 100}%` }));
}

/** One dot per run slot, filled while an agent occupies it. */
function slots(running, max) {
  return h(
    "div",
    { class: "slots", "aria-hidden": "true" },
    Array.from({ length: max }, (_, i) => h("i", { class: i < running ? "is-on" : null })),
  );
}

// Every limit is off at 0 (app/limits.py), so 0 reads "no limit", not "nothing allowed".
function tiles(stats) {
  const { limits } = stats;
  const perJourney = stats.journeys ? usd(stats.llm_cost_total / stats.journeys) : null;
  const budget = limits.daily_budget_usd;
  const slotsMax = limits.max_concurrent;
  return h(
    "div",
    { class: "tiles" },
    tile("Пользователи", number(stats.users), plural(stats.users, "аккаунт", "аккаунта", "аккаунтов")),
    tile(
      "Путешествия",
      number(stats.journeys),
      `${number(stats.journeys_today)} сегодня · ${number(stats.days)} маршрутов дней, ${number(stats.days_today)} сегодня`,
    ),
    budget
      ? tile("LLM сегодня", usd(stats.llm_cost_today), `из ${usd(budget)} дневного бюджета`, meter(stats.llm_cost_today, budget))
      : tile("LLM сегодня", usd(stats.llm_cost_today), "дневной бюджет не ограничен"),
    tile("LLM за всё время", usd(stats.llm_cost_total), perJourney ? `≈ ${perJourney} за путешествие` : null),
    slotsMax
      ? tile("Агентов сейчас", `${limits.running} из ${slotsMax}`, "одновременных прогонов", slots(limits.running, slotsMax))
      : tile("Агентов сейчас", number(limits.running), "одновременных прогонов, без лимита"),
    limits.per_hour
      ? tile("Лимит запусков", number(limits.per_hour), "в час с одного адреса")
      : tile("Лимит запусков", "нет", "запуски в час не ограничены"),
  );
}

function table(caption, columns, rows) {
  return h(
    "section",
    { class: "admin-section" },
    h("h2", { class: "subtitle" }, caption),
    rows.length
      ? h(
          "div",
          { class: "table-wrap" },
          h(
            "table",
            { class: "table" },
            h("thead", {}, h("tr", {}, columns.map(([label, , numeric]) => h("th", { class: numeric ? "num" : null, scope: "col" }, label)))),
            h(
              "tbody",
              {},
              rows.map((row) =>
                h("tr", {}, columns.map(([, cell, numeric]) => h("td", { class: numeric ? "num" : null }, cell(row)))),
              ),
            ),
          ),
        )
      : h("p", { class: "muted" }, "Пока пусто."),
  );
}

const when = (iso) => h("span", { title: new Date(iso).toLocaleString("ru-RU") }, ago(iso));

const USER_COLUMNS = [
  ["Почта", (user) => user.email],
  ["Имя", (user) => user.name],
  ["Роль", (user) => (user.is_admin ? h("span", { class: "tag is-admin" }, "админ") : "")],
  ["Регистрация", (user) => when(user.created_at)],
  ["Путешествий", (user) => number(user.journeys), true],
  ["Дней", (user) => number(user.days), true],
  ["LLM", (user) => usd(user.cost_usd), true],
];

const TRIP_COLUMNS = [
  ["Путешествие", (trip) => h("span", { class: "cell-title" }, trip.title)],
  ["Город", (trip) => trip.city],
  ["Дней", (trip) => number(trip.days), true],
  ["Владелец", (trip) => trip.owner_email],
  ["Изменено", (trip) => when(trip.updated_at)],
  ["LLM", (trip) => usd(trip.cost_usd), true],
];

function forbidden() {
  root.replaceChildren(
    h(
      "div",
      { class: "card empty admin-forbidden" },
      h("h2", {}, "Сюда пускают только администраторов"),
      h("p", {}, "Здесь статистика сервиса и траты на LLM. Если нужен доступ, напишите команде проекта."),
      h("a", { class: "btn", href: "/app/trips" }, "К моим путешествиям"),
    ),
  );
}

async function load() {
  root.setAttribute("aria-busy", "true");
  refresh.disabled = true;
  try {
    const [stats, users, trips] = await Promise.all([
      api("/api/admin/stats"),
      api("/api/admin/users"),
      api("/api/admin/trips"),
    ]);
    root.replaceChildren(
      tiles(stats),
      table("Пользователи", USER_COLUMNS, users),
      table("Последние путешествия", TRIP_COLUMNS, trips),
    );
    updated.textContent = `Данные на ${new Date().toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" })}`;
    refresh.hidden = false;
  } catch (error) {
    if (error instanceof ApiError && error.status === 403) {
      forbidden();
      return;
    }
    const retry = h("button", { class: "btn btn-sm", type: "button", onclick: load }, "Повторить");
    root.replaceChildren(h("div", { class: "alert is-error" }, `Не удалось загрузить данные: ${error.message} `, retry));
  } finally {
    root.removeAttribute("aria-busy");
    refresh.disabled = false;
  }
}

refresh.addEventListener("click", load);

try {
  renderNav(await requireUser(), "admin");
  await load();
} catch (error) {
  root.replaceChildren(h("div", { class: "alert is-error" }, error.message));
}
