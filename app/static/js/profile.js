/*
 * What the traveller likes, how they walk, and what the agent has learned about them.
 *
 * One page, two modes chosen by the path:
 *   /app/onboarding — the questionnaire right after registration: three short steps,
 *                     each one skippable, so a new user reaches the planner in a minute;
 *   /app/profile    — the same fields as one form, plus the agent's long-term memory.
 * Both edit a single draft shaped exactly like the server's Profile and send it whole
 * with PUT /api/profile, which also marks the traveller as onboarded (so «Пропустить»
 * saves too: the questionnaire is offered once, not on every login).
 */

import { api, requireUser } from "./api.js";
import { renderNav } from "./nav.js";
import { ago, h, toast } from "./ui.js";

const root = document.getElementById("profile");
const ONBOARDING = location.pathname === "/app/onboarding";

// The two interest levels the server keeps; a chip cycles none -> plain -> strong -> none.
const PLAIN = 0.6;
const STRONG = 1.0;

const PACE_HINTS = { relaxed: "больше пауз, короче переходы", normal: "средний шаг", active: "больше мест за день" };
const DIET_SUGGESTIONS = ["вегетарианец", "веган", "без алкоголя", "без глютена", "халяль"];
const DIET_MAX = 10; // the server accepts up to 10 tags of up to 40 characters
const ABOUT_MAX = 500;
// The slider's leftmost notch means "no limit" (null); the rest are kilometres.
const KM_OFF = 1;
const KM_MAX = 20;

const EMPTY_PROFILE = {
  home_city: null,
  interests: {},
  avoid: [],
  pace: "normal",
  max_walk_km: null,
  with_kids: false,
  diet: [],
  about: null,
};

let catalog;
let draft;
let savedSnapshot;
// Controls whose look depends on the draft (chips, radios, slider) re-read it after every
// change: picking an interest must also clear it from "do not suggest", and back.
const syncs = new Set();

/** Canonical JSON of a profile, so "unsaved changes" ignores key order. */
function snapshot(profile) {
  const interests = Object.fromEntries(Object.entries(profile.interests).sort(([a], [b]) => a.localeCompare(b)));
  return JSON.stringify({ ...profile, interests, avoid: [...profile.avoid].sort() });
}

const isDirty = () => snapshot(draft) !== savedSnapshot;

function changed() {
  for (const sync of syncs) sync();
}

/* ---------- building blocks ---------- */

function section(id, title, lead, ...content) {
  return h(
    "section",
    { class: "card card-pad form-section", id, "aria-labelledby": `${id}-title` },
    h("header", { class: "section-head" }, h("h2", { class: "subtitle", id: `${id}-title` }, title), lead ? h("p", { class: "muted" }, lead) : null),
    ...content,
  );
}

function group(label, control, hint = null) {
  return h(
    "div",
    { class: "field-group" },
    h("p", { class: "field-label" }, label),
    hint ? h("p", { class: "field-hint" }, hint) : null,
    control,
  );
}

function dots() {
  return h("span", { class: "chip-dots", "aria-hidden": "true" }, h("i"), h("i"));
}

function interestChips() {
  const chips = catalog.interests.map(({ key, label }) => {
    const level = h("span", { class: "sr-only" });
    const button = h("button", { class: "chip", type: "button" }, label, dots(), level);
    button.addEventListener("click", () => {
      const weight = draft.interests[key];
      if (!weight) {
        draft.interests[key] = PLAIN;
        draft.avoid = draft.avoid.filter((item) => item !== key);
      } else if (weight < STRONG) {
        draft.interests[key] = STRONG;
      } else {
        delete draft.interests[key];
      }
      changed();
    });
    syncs.add(() => {
      const weight = draft.interests[key];
      button.setAttribute("aria-pressed", weight >= STRONG ? "true" : weight ? "mixed" : "false");
      level.textContent = weight >= STRONG ? ": очень" : weight ? ": интересно" : "";
    });
    return button;
  });
  return h(
    "div",
    {},
    h("div", { class: "chips", role: "group", "aria-label": "Интересы" }, chips),
    h(
      "p",
      { class: "chip-legend" },
      "Нажмите раз — ",
      h("span", { class: "legend-mark is-plain" }, dots(), "интересно"),
      ", ещё раз — ",
      h("span", { class: "legend-mark is-strong" }, dots(), "очень"),
      ", третий — снять отметку.",
    ),
  );
}

function avoidChips() {
  const chips = catalog.interests.map(({ key, label }) => {
    const button = h("button", { class: "chip", type: "button", "data-tone": "avoid" }, label);
    button.addEventListener("click", () => {
      if (draft.avoid.includes(key)) {
        draft.avoid = draft.avoid.filter((item) => item !== key);
      } else {
        draft.avoid = [...draft.avoid, key];
        delete draft.interests[key];
      }
      changed();
    });
    syncs.add(() => button.setAttribute("aria-pressed", String(draft.avoid.includes(key))));
    return button;
  });
  return h("div", { class: "chips", role: "group", "aria-label": "Не предлагать" }, chips);
}

function paceControl() {
  const options = catalog.paces.map(({ key, label }) => {
    const input = h("input", { type: "radio", name: "pace", value: key });
    input.addEventListener("change", () => {
      draft.pace = key;
      changed();
    });
    syncs.add(() => {
      input.checked = draft.pace === key;
    });
    return h("label", {}, input, h("span", {}, label, PACE_HINTS[key] ? h("small", {}, PACE_HINTS[key]) : null));
  });
  return h("div", { class: "segmented", role: "radiogroup", "aria-label": "Темп" }, options);
}

function distanceControl() {
  const output = h("output", { class: "slider-value", for: "max-km" });
  const input = h("input", { class: "slider", type: "range", id: "max-km", min: KM_OFF, max: KM_MAX, step: "1" });
  input.addEventListener("input", () => {
    const value = Number(input.value);
    draft.max_walk_km = value === KM_OFF ? null : value;
    changed();
  });
  syncs.add(() => {
    const km = draft.max_walk_km;
    input.value = km ? String(Math.min(Math.max(Math.round(km), KM_OFF + 1), KM_MAX)) : String(KM_OFF);
    input.style.setProperty("--fill", `${((Number(input.value) - KM_OFF) / (KM_MAX - KM_OFF)) * 100}%`);
    input.classList.toggle("is-off", !km);
    output.textContent = km ? `до ${km.toLocaleString("ru-RU")} км` : "не ограничивать";
    input.setAttribute("aria-valuetext", output.textContent);
  });
  return h(
    "div",
    { class: "slider-field" },
    h("div", { class: "slider-head" }, h("label", { class: "field-label", for: "max-km" }, "Сколько готовы пройти за прогулку"), output),
    input,
    h("div", { class: "slider-scale", "aria-hidden": "true" }, h("span", {}, "без ограничения"), h("span", {}, `${KM_MAX} км`)),
  );
}

function kidsToggle() {
  const input = h("input", { type: "checkbox", role: "switch" });
  input.addEventListener("change", () => {
    draft.with_kids = input.checked;
    changed();
  });
  syncs.add(() => {
    input.checked = draft.with_kids;
  });
  return h(
    "label",
    { class: "toggle" },
    input,
    h("span", { class: "toggle-track" }),
    h("span", {}, "Гуляю с детьми", h("small", { class: "toggle-hint" }, "места, где интересно детям, и переходы покороче")),
  );
}

function cityField() {
  const input = h("input", {
    class: "input",
    id: "home-city",
    maxlength: "100",
    autocomplete: "address-level2",
    placeholder: "Например, Санкт-Петербург",
  });
  input.value = draft.home_city ?? "";
  input.addEventListener("input", () => {
    draft.home_city = input.value.trim() || null;
    changed();
  });
  return h(
    "div",
    { class: "field" },
    h("label", { for: "home-city" }, "Город по умолчанию"),
    input,
    h("p", { class: "field-hint" }, "Подставим в новое путешествие, если в запросе город не указан"),
  );
}

function dietField() {
  const tags = h("div", { class: "chips diet-tags" });
  const suggestions = h("div", { class: "chips" });
  const input = h("input", { class: "input", id: "diet-input", maxlength: "40", placeholder: "Например, без лактозы" });
  /** False when the list is full: the typed text stays in the field, with the reason shown. */
  const add = (text) => {
    const value = text.trim();
    if (!value || draft.diet.some((tag) => tag.toLowerCase() === value.toLowerCase())) return true;
    if (draft.diet.length >= DIET_MAX) {
      toast(`Не больше ${DIET_MAX} пунктов — уберите лишний`, "error");
      return false;
    }
    draft.diet = [...draft.diet, value];
    changed();
    return true;
  };
  const addTyped = () => {
    if (add(input.value)) input.value = "";
    input.focus();
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault(); // Enter adds a tag, it must not submit the whole form
      addTyped();
    }
  });
  syncs.add(() => {
    tags.replaceChildren(
      ...draft.diet.map((tag) =>
        h(
          "span",
          { class: "chip is-tag" },
          tag,
          h(
            "button",
            {
              class: "chip-x",
              type: "button",
              "aria-label": `Убрать «${tag}»`,
              onclick: () => {
                draft.diet = draft.diet.filter((item) => item !== tag);
                changed();
                input.focus();
              },
            },
            "×",
          ),
        ),
      ),
    );
    suggestions.replaceChildren(
      ...DIET_SUGGESTIONS.filter((tag) => !draft.diet.includes(tag)).map((tag) =>
        h("button", { class: "chip chip-suggest", type: "button", onclick: () => add(tag) }, `+ ${tag}`),
      ),
    );
  });
  return h(
    "div",
    { class: "field" },
    h("label", { for: "diet-input" }, "Еда и ограничения"),
    tags,
    h("div", { class: "add-row" }, input, h("button", { class: "btn", type: "button", onclick: addTyped }, "Добавить")),
    suggestions,
  );
}

function aboutField() {
  const textarea = h("textarea", {
    class: "textarea",
    id: "about",
    rows: "4",
    maxlength: String(ABOUT_MAX),
    placeholder: "Например: люблю долго гулять вдоль воды, не люблю толпы, интересуюсь советской архитектурой",
  });
  textarea.value = draft.about ?? "";
  const counter = h("span", { class: "counter num" });
  const count = () => {
    counter.textContent = `${textarea.value.length} / ${ABOUT_MAX}`;
  };
  textarea.addEventListener("input", () => {
    draft.about = textarea.value.trim() || null;
    count();
    changed();
  });
  count();
  return h(
    "div",
    { class: "field" },
    h("div", { class: "label-row" }, h("label", { for: "about" }, "О себе"), counter),
    textarea,
    h("p", { class: "field-hint" }, "Агент прочитает это перед поиском мест. Свободным текстом, как другу."),
  );
}

/* ---------- saving ---------- */

async function save(button) {
  const label = button.textContent;
  button.disabled = true;
  button.classList.add("is-busy");
  button.textContent = "Сохраняем…";
  try {
    const result = await api("/api/profile", { method: "PUT", body: draft });
    draft = { ...EMPTY_PROFILE, ...result };
    savedSnapshot = snapshot(draft);
    return true;
  } catch (error) {
    toast(`Не сохранилось: ${error.message}`, "error");
    return false;
  } finally {
    button.classList.remove("is-busy");
    button.textContent = label;
    button.disabled = false;
    changed();
  }
}

/* ---------- /app/profile ---------- */

function memoryItem(fact, onForget) {
  const button = h("button", { class: "btn btn-ghost btn-sm", type: "button", "aria-label": `Забыть: ${fact.text}` }, "Забыть");
  const item = h(
    "li",
    { class: "memory-item" },
    h("p", { class: "memory-text" }, fact.text),
    h("span", { class: "memory-date", title: new Date(fact.created_at).toLocaleString("ru-RU") }, ago(fact.created_at)),
    button,
  );
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await api(`/api/memory/${encodeURIComponent(fact.id)}`, { method: "DELETE" });
      onForget(fact.id);
      toast("Агент это забыл", "ok");
    } catch (error) {
      button.disabled = false;
      toast(error.message, "error");
    }
  });
  return item;
}

function memorySection() {
  const list = h("ul", { class: "memory-list", "aria-live": "polite" }, h("li", { class: "skeleton line wide" }));
  let facts = [];
  const render = () => {
    list.replaceChildren(
      ...(facts.length
        ? facts.map((fact) =>
            memoryItem(fact, (id) => {
              facts = facts.filter((item) => item.id !== id);
              render();
            }),
          )
        : [h("li", { class: "memory-empty" }, "Пока ничего. Факты появятся после первых прогулок и правок.")]),
    );
  };
  api("/api/memory")
    .then((loaded) => {
      facts = loaded;
      render();
    })
    .catch((error) => list.replaceChildren(h("li", { class: "alert is-error" }, `Не удалось загрузить: ${error.message}`)));
  return section(
    "memory",
    "Что агент помнит о вас",
    "Агент сам замечает предпочтения в ваших запросах и правках — «не люблю очереди», «хожу быстро» — и учитывает их в следующих поездках. Любой факт можно удалить.",
    list,
  );
}

function renderProfile(user) {
  const status = h("p", { class: "savebar-status", "aria-live": "polite" });
  const revert = h("button", { class: "btn btn-ghost", type: "button" }, "Отменить");
  const submit = h("button", { class: "btn btn-primary", type: "submit" }, "Сохранить");
  const bar = h("div", { class: "savebar" }, status, h("div", { class: "savebar-actions" }, revert, submit));
  syncs.add(() => {
    const dirty = isDirty();
    status.textContent = dirty ? "Есть несохранённые изменения" : "Все изменения сохранены";
    status.classList.toggle("is-dirty", dirty);
    bar.classList.toggle("is-dirty", dirty);
    revert.hidden = !dirty;
    if (!submit.classList.contains("is-busy")) submit.disabled = !dirty;
  });

  const form = h(
    "form",
    { class: "profile-form", novalidate: true },
    section(
      "interests",
      "Интересы",
      "Из чего собирать маршруты ваших поездок. В планировщике для отдельного путешествия можно попросить больше или меньше чего-то.",
      interestChips(),
    ),
    section("avoid", "Не предлагать", "Это агент не будет ставить в маршрут, даже если место знаменитое.", avoidChips()),
    section(
      "walk",
      "Как вы гуляете",
      null,
      group("Темп", paceControl()),
      distanceControl(),
      kidsToggle(),
      cityField(),
    ),
    section("about", "Еда и о себе", null, dietField(), aboutField()),
    bar,
  );
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (await save(submit)) toast("Профиль сохранён", "ok");
  });
  revert.addEventListener("click", () => {
    draft = JSON.parse(savedSnapshot);
    syncs.clear();
    renderProfile(user); // text fields read the draft only when built, so rebuild them
  });

  const toc = [
    ["interests", "Интересы"],
    ["avoid", "Не предлагать"],
    ["walk", "Как вы гуляете"],
    ["about", "Еда и о себе"],
    ["memory", "Память агента"],
  ];
  root.replaceChildren(
    h(
      "div",
      { class: "page-head" },
      h(
        "div",
        {},
        h("p", { class: "eyebrow" }, "Профиль"),
        h("h1", { class: "title" }, "Что вы любите в поездках"),
        h("p", { class: "muted" }, `${user.name || "Без имени"} · ${user.email}`),
      ),
    ),
    h(
      "div",
      { class: "profile-layout" },
      h(
        "nav",
        { class: "profile-toc", "aria-label": "Разделы профиля" },
        toc.map(([id, label]) => h("a", { href: `#${id}` }, label)),
      ),
      h("div", { class: "profile-sections" }, form, memorySection()),
    ),
  );
  changed();
}

/* ---------- /app/onboarding ---------- */

const STEPS = [
  {
    label: "Интересы",
    title: "Что вам интересно в городе?",
    lead: "Отметьте всё, что нравится, — агент будет искать такие места в первую очередь.",
    fields: () => [interestChips()],
  },
  {
    label: "Как гуляете",
    title: "Как вы гуляете?",
    lead: "Темп и расстояние агент учтёт, когда будет раскладывать день по времени.",
    fields: () => [group("Темп", paceControl()), distanceControl(), kidsToggle(), group("Что не предлагать", avoidChips())],
  },
  {
    label: "О вас",
    title: "Ещё немного о вас",
    lead: "Необязательно, но помогает: город подставится в новое путешествие, а «о себе» агент прочитает перед поиском.",
    fields: () => [cityField(), dietField(), aboutField()],
  },
];

function renderOnboarding(user) {
  let current = 0;
  const stepper = h(
    "ol",
    { class: "stepper" },
    STEPS.map((step, index) => h("li", {}, h("span", { class: "stepper-dot" }, index + 1), h("span", { class: "stepper-label" }, step.label))),
  );
  // Every step is built once and only shown or hidden, so the controls keep their state.
  const panels = STEPS.map((step, index) =>
    h(
      "section",
      { class: "card card-pad onb-step", "aria-labelledby": `step-${index}-title`, hidden: index !== 0 },
      h(
        "header",
        { class: "section-head" },
        h("p", { class: "eyebrow" }, `Шаг ${index + 1} из ${STEPS.length}`),
        h("h2", { class: "subtitle onb-title", id: `step-${index}-title` }, step.title),
        h("p", { class: "muted" }, step.lead),
      ),
      step.fields(),
    ),
  );
  const skip = h("button", { class: "btn btn-ghost", type: "button" }, "Пропустить");
  const back = h("button", { class: "btn", type: "button" }, "Назад");
  const next = h("button", { class: "btn btn-primary btn-lg", type: "submit" });

  const show = (index, { moveFocus = true } = {}) => {
    current = index;
    panels.forEach((panel, i) => {
      panel.hidden = i !== index;
    });
    [...stepper.children].forEach((item, i) => {
      item.className = i < index ? "is-done" : i === index ? "is-current" : "";
      item.toggleAttribute("aria-current", i === index);
    });
    back.hidden = index === 0;
    next.textContent = index === STEPS.length - 1 ? "Готово — к первому путешествию" : "Далее";
    if (moveFocus) {
      // the new step's heading takes focus, so a screen reader announces the step
      panels[index].querySelector("h2").focus({ preventScroll: true });
      const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
      window.scrollTo({ top: 0, behavior: smooth ? "smooth" : "auto" });
    }
  };

  const finish = async (button) => {
    if (await save(button)) location.assign("/app/new");
  };

  const form = h(
    "form",
    { class: "onb-form", novalidate: true },
    panels,
    h("div", { class: "onb-actions" }, skip, h("div", { class: "onb-actions-main" }, back, next)),
  );
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (current < STEPS.length - 1) show(current + 1);
    else finish(next);
  });
  back.addEventListener("click", () => show(current - 1));
  skip.addEventListener("click", () => finish(skip));

  for (const panel of panels) panel.querySelector("h2").tabIndex = -1;
  root.classList.add("onboarding");
  root.replaceChildren(
    h(
      "header",
      { class: "onb-head" },
      h("p", { class: "eyebrow" }, "Анкета · около минуты"),
      h("h1", { class: "title" }, user.name ? `Здравствуйте, ${user.name}!` : "Здравствуйте!"),
      h("p", { class: "lead" }, "Расскажите, что вам интересно, — агент будет подбирать места под вас. Всё можно поменять потом в профиле."),
      stepper,
    ),
    form,
  );
  changed();
  show(0, { moveFocus: false });
}

/* ---------- start ---------- */

try {
  const user = await requireUser();
  renderNav(user, "profile");
  const [loadedCatalog, profile] = await Promise.all([api("/api/catalog"), api("/api/profile")]);
  catalog = loadedCatalog;
  draft = { ...EMPTY_PROFILE, ...profile };
  savedSnapshot = snapshot(draft);
  if (ONBOARDING) {
    document.title = "Анкета — Прогулка";
    renderOnboarding(user);
  } else {
    renderProfile(user);
    // An unsaved profile is easy to lose by clicking a nav link; the browser asks first.
    window.addEventListener("beforeunload", (event) => {
      if (isDirty()) event.preventDefault();
    });
  }
} catch (error) {
  root.replaceChildren(h("div", { class: "alert is-error" }, `Не удалось открыть профиль: ${error.message}`));
} finally {
  root.removeAttribute("aria-busy");
}
