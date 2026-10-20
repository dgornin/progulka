/*
 * The landing page is static HTML that works without this script. The script only
 * adjusts the calls to action to what is true right now: a visitor who is already
 * signed in gets the way back to their journeys instead of «Спланировать поездку» /
 * «Войти», and the demo buttons disappear when the server has no demo account to offer.
 */

import { api, currentUser } from "./api.js";

const show = (selector, visible) => {
  for (const element of document.querySelectorAll(selector)) element.hidden = !visible;
};

try {
  const [user, config] = await Promise.all([currentUser(), api("/api/config")]);
  if (user) {
    show("[data-guest]", false);
    show("[data-user]", true);
    document.getElementById("user-name").textContent = user.name || user.email;
  } else if (!config.demo) {
    show("[data-demo]", false);
  }
} catch {
  // the server is unreachable: the guest buttons still lead to the sign-in form
}
