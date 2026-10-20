/*
 * The top bar of the signed-in pages, built from the user the page already loaded,
 * so the bar and the page never disagree about who is signed in. The "Админ" link is
 * only a shortcut: the admin API refuses everyone else on its own.
 */

import { logout } from "./api.js";
import { h, svg, toast } from "./ui.js";

const LINKS = [
  ["trips", "/app/trips", "Мои путешествия"],
  ["new", "/app/new", "Новое путешествие"],
  ["profile", "/app/profile", "Профиль"],
  ["admin", "/app/admin", "Админ"],
];

/** The brand mark: a dotted walk from an ink start through a stop to the goal. */
function logo() {
  return svg(
    "svg",
    { class: "logo", viewBox: "0 0 44 44", "aria-hidden": "true" },
    svg("path", {
      d: "M7 35 C 13 18, 26 32, 30 17 S 36 8, 37 8",
      fill: "none",
      stroke: "currentColor",
      "stroke-width": "2.5",
      "stroke-linecap": "round",
      "stroke-dasharray": "0.1 5.5",
    }),
    svg("circle", { class: "logo-start", cx: "7", cy: "35", r: "4.5" }),
    svg("circle", { cx: "19", cy: "25.5", r: "3.5", fill: "currentColor" }),
    svg("circle", { cx: "37", cy: "8", r: "4.5", fill: "currentColor" }),
  );
}

async function signOut(event) {
  const button = event.currentTarget;
  button.disabled = true;
  try {
    await logout();
  } catch (error) {
    button.disabled = false;
    toast(error.message, "error");
  }
}

/** Mount the bar into <header id="topbar">; active is "trips" | "new" | "profile" | "admin". */
export function renderNav(user, active) {
  const header = document.getElementById("topbar");
  const name = user.name || user.email;
  const links = LINKS.filter(([key]) => key !== "admin" || user.is_admin).map(([key, href, label]) =>
    h("a", { href, class: key === "new" ? "nav-new" : null, "aria-current": key === active ? "page" : null }, label),
  );
  header.classList.add("topbar");
  const nav = h("nav", { class: "nav", "aria-label": "Разделы" }, links);
  header.replaceChildren(
    h(
      "div",
      { class: "topbar-inner" },
      h("a", { class: "brand", href: "/", "aria-label": "Прогулка — на главную" }, logo(), h("span", { class: "brand-name" }, "Прогулка")),
      nav,
      h(
        "div",
        { class: "topbar-user" },
        h("span", { class: "avatar", "aria-hidden": "true" }, Array.from(name.trim())[0]?.toUpperCase()),
        h("span", { class: "user-name", title: user.email }, name),
        h("button", { class: "btn btn-ghost btn-sm", type: "button", onclick: signOut }, "Выйти"),
      ),
    ),
  );
  // On a phone the links scroll sideways; start with the current section in view.
  const current = nav.querySelector("[aria-current]");
  if (current && nav.scrollWidth > nav.clientWidth) nav.scrollLeft = current.offsetLeft - nav.offsetLeft - 16;
}
