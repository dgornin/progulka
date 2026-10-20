/*
 * Sign in and sign up ("/login", "/register": one page, the path picks the tab).
 *
 * The demo account comes from GET /api/config, so the credentials shown are always
 * the ones the server actually seeded, and «Войти как демо» skips registration and
 * the questionnaire. After signing in the visitor returns to the page that sent them
 * here (?next=...), but only to a path on this site, so a crafted link cannot bounce
 * them to another domain.
 */

import { api, currentUser } from "./api.js";

const $ = (selector) => document.querySelector(selector);

const TABS = {
  login: {
    tab: $("#tab-login"),
    form: $("#login-form"),
    path: "/login",
    title: "С возвращением",
    lead: "Войдите, чтобы вернуться к своим прогулкам.",
    page: "Вход — Прогулка",
  },
  register: {
    tab: $("#tab-register"),
    form: $("#register-form"),
    path: "/register",
    title: "Новый аккаунт",
    lead: "Прогулки сохранятся в кабинете, а агент запомнит, что вы любите.",
    page: "Регистрация — Прогулка",
  },
};

const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const params = new URLSearchParams(location.search);

/** ?next= only if it stays on this site: "/app/trips/1", never "//evil.example". */
function safeNext() {
  const next = params.get("next");
  if (!next?.startsWith("/")) return null;
  // The browser's URL parser decides, not a prefix check: it drops tabs and newlines and
  // reads "\" as "/", so "/\t/evil.example" would pass a check and still leave the site.
  const url = new URL(next, location.origin);
  return url.origin === location.origin ? url.pathname + url.search + url.hash : null;
}

function destination(user) {
  return safeNext() ?? (user.onboarded ? "/app/trips" : "/app/onboarding");
}

function showTab(name, { focus = false } = {}) {
  for (const [key, { tab, form }] of Object.entries(TABS)) {
    const selected = key === name;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    form.hidden = !selected;
  }
  const current = TABS[name];
  $("#auth-title").textContent = current.title;
  $("#auth-lead").textContent = current.lead;
  document.title = current.page;
  if (location.pathname !== current.path) history.replaceState(null, "", current.path + location.search);
  if (focus) current.tab.focus();
}

/* ---------- validation and messages ---------- */

function setFieldError(input, message) {
  input.setAttribute("aria-invalid", message ? "true" : "false");
  document.getElementById(`${input.id}-error`).textContent = message ?? "";
}

function validate(form) {
  const problems = [];
  const { name, email, password } = form.elements;
  const registering = form === TABS.register.form;
  if (registering && !name.value.trim()) problems.push([name, "Напишите, как к вам обращаться"]);
  if (!EMAIL.test(email.value.trim())) problems.push([email, "Нужна почта вида anna@example.com"]);
  if (registering && password.value.length < 8) problems.push([password, "Пароль — не меньше 8 символов"]);
  if (!registering && !password.value) problems.push([password, "Введите пароль"]);
  for (const input of form.querySelectorAll(".input")) setFieldError(input, null);
  for (const [input, message] of problems) setFieldError(input, message);
  problems[0]?.[0].focus();
  return problems.length === 0;
}

function showAlert(container, message) {
  const alert = container.querySelector(".alert");
  alert.textContent = message ?? "";
  alert.hidden = !message;
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  button.classList.toggle("is-busy", busy);
  if (label) button.textContent = label;
}

/* ---------- actions ---------- */

async function signIn(credentials, container, button) {
  const label = button.textContent;
  setBusy(button, true, "Входим…");
  showAlert(container, null);
  try {
    const { user } = await api("/api/auth/login", { method: "POST", body: credentials });
    location.assign(destination(user));
  } catch (error) {
    setBusy(button, false, label);
    showAlert(container, error.message);
  }
}

async function signUp(details, container, button) {
  const label = button.textContent;
  setBusy(button, true, "Создаём аккаунт…");
  showAlert(container, null);
  try {
    await api("/api/auth/register", { method: "POST", body: details });
    location.assign("/app/onboarding");
  } catch (error) {
    setBusy(button, false, label);
    showAlert(container, error.message);
  }
}

TABS.login.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  if (!validate(form)) return;
  const { email, password } = form.elements;
  signIn({ email: email.value.trim(), password: password.value }, form, form.querySelector("[type=submit]"));
});

TABS.register.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  if (!validate(form)) return;
  const { name, email, password } = form.elements;
  const details = { name: name.value.trim(), email: email.value.trim(), password: password.value };
  signUp(details, form, form.querySelector("[type=submit]"));
});

// A field's error goes away as soon as the person starts fixing it.
for (const input of document.querySelectorAll(".auth-form .input")) {
  input.addEventListener("input", () => {
    if (input.getAttribute("aria-invalid") === "true") setFieldError(input, null);
  });
}

for (const toggle of document.querySelectorAll(".password-toggle")) {
  toggle.addEventListener("click", () => {
    const input = document.getElementById(toggle.getAttribute("aria-controls"));
    const shown = input.type === "password";
    input.type = shown ? "text" : "password";
    toggle.setAttribute("aria-pressed", String(shown));
    toggle.textContent = shown ? "Скрыть" : "Показать";
  });
}

/* ---------- tabs ---------- */

const order = Object.keys(TABS);
for (const [name, { tab }] of Object.entries(TABS)) {
  tab.addEventListener("click", () => showTab(name));
  tab.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    const step = event.key === "ArrowRight" ? 1 : -1;
    showTab(order[(order.indexOf(name) + step + order.length) % order.length], { focus: true });
  });
}

/* ---------- demo account ---------- */

async function setUpDemo() {
  const { demo } = await api("/api/config");
  if (!demo) return;
  const card = $("#demo");
  $("#demo-email").textContent = demo.email;
  $("#demo-password").textContent = demo.password;
  card.hidden = false;
  const button = $("#demo-login");
  button.addEventListener("click", () => signIn({ email: demo.email, password: demo.password }, card, button));
  // ?demo=1 comes from the landing's «Попробовать демо»: point straight at the account
  if (params.get("demo") === "1") {
    const { email, password } = TABS.login.form.elements;
    email.value = demo.email;
    password.value = demo.password;
    card.classList.add("is-highlight");
    button.focus({ preventScroll: true });
    card.scrollIntoView({ block: "center", behavior: "smooth" });
  }
}

showTab(location.pathname === "/register" ? "register" : "login");

// Already signed in (an old tab, the back button): no reason to log in again.
currentUser()
  .then((user) => user && location.replace(destination(user)))
  .catch(() => {});
setUpDemo().catch(() => {}); // no demo card if the server cannot say; the forms still work
