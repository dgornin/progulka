/*
 * The planner's map: MapLibre GL JS drawing OpenFreeMap vector tiles.
 *
 * The basemap is OpenFreeMap's "positron", the quietest of its styles: no POI icons to
 * mistake for our stops and grey roads instead of colourful ones. It is recoloured a
 * little warmer (the same colours as the landing picture) and labelled in Russian
 * where OpenStreetMap has a Russian name, so the terracotta route stays the brightest
 * thing on the map and the page reads as one light design. Tiles and style need no key.
 *
 * Places are DOM markers, so they animate with CSS and react to hover like the
 * timeline does; the route is a GeoJSON line layer over a white casing (dotted when the
 * walking router gave no geometry). The route source lives in the map style, so it is
 * (re)installed on every "style.load" from the line kept here, and a plan that
 * arrives before the style has loaded simply appears once it has.
 *
 * Outside text (place names, descriptions) reaches the map only as DOM nodes and text:
 * popups get setDOMContent, never setHTML. The library is imported on demand, so when
 * unpkg or WebGL is unavailable the planner keeps working without the map.
 */

import { h } from "./ui.js";

const STYLE_URL = "https://tiles.openfreemap.org/styles/positron";
const HOME = { center: [34, 57.5], zoom: 3.6 }; // European Russia, before a city is known
const ROUTE = "#c8412a";
const DRAW_MS = 1500;
const FIT_DELAY_MS = 350; // one re-framing for a turn that shortlists ten places at once

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Paint overrides of positron's layers, by layer id.
const PAINT = {
  background: { "background-color": "#f8f7f4" },
  landuse_residential: { "fill-color": "#f3f1ec" },
  building: { "fill-color": "#eeebe4", "fill-outline-color": "#e3dfd6" },
  park: { "fill-color": "#eef3e8" },
  landcover_wood: { "fill-color": "#e3ecdc" },
  water: { "fill-color": "#d3e2eb" },
  waterway: { "line-color": "#c4d7e3" },
  highway_path: { "line-color": "#ffffff" },
  highway_minor: { "line-color": "#ffffff" },
  highway_major_casing: { "line-color": "#e4dfd5" },
  railway: { "line-color": "#e2ded6" },
  railway_transit: { "line-color": "#e2ded6" },
  railway_service: { "line-color": "#e2ded6" },
};
// positron prints "Latin name + native name"; a Russian reader wants one name, in Russian.
const NAME_RU = ["coalesce", ["get", "name:ru"], ["get", "name"]];

const LOCALE = {
  "NavigationControl.ZoomIn": "Приблизить",
  "NavigationControl.ZoomOut": "Отдалить",
  "AttributionControl.ToggleAttribution": "Источники карты",
  "AttributionControl.MapFeedback": "Сообщить об ошибке на карте",
  "Popup.Close": "Закрыть",
  "Map.Title": "Карта маршрута",
};

const width = (low, high) => ["interpolate", ["linear"], ["zoom"], 11, low, 17, high];
const ROUND = { "line-join": "round", "line-cap": "round" };
const ROUTE_LAYERS = [
  {
    id: "route-casing",
    type: "line",
    source: "route",
    layout: ROUND,
    paint: { "line-color": "#ffffff", "line-width": width(6, 14), "line-opacity": 0.95 },
  },
  {
    id: "route-line",
    type: "line",
    source: "route",
    filter: ["!=", ["get", "straight"], true],
    layout: ROUND,
    paint: { "line-color": ROUTE, "line-width": width(3, 6.5) },
  },
  {
    // straight segments between the stops: dots say "we do not know the streets here"
    id: "route-dots",
    type: "line",
    source: "route",
    filter: ["==", ["get", "straight"], true],
    layout: ROUND,
    paint: { "line-color": ROUTE, "line-width": width(4, 7), "line-dasharray": [0, 2] },
  },
];

function lighten(style) {
  for (const layer of style.layers) {
    if (PAINT[layer.id]) layer.paint = { ...layer.paint, ...PAINT[layer.id] };
    const text = JSON.stringify(layer.layout?.["text-field"] ?? "");
    if (layer.type === "symbol" && text.includes('"name')) layer.layout["text-field"] = NAME_RU;
  }
  return style;
}

/**
 * A marker element: a dot or a numbered circle with a hover label (outside text, so a
 * text node). MapLibre positions the element with a transform, so it animates its child.
 */
function pin(kind, number, order, title, clickable) {
  const role = clickable ? { role: "button", tabindex: "0" } : { role: "img" };
  return h(
    "div",
    { class: `pin pin-${kind}`, ...role, "aria-label": title },
    h("span", { class: "pin-dot", style: `--i: ${order}` }, number),
    h("span", { class: "pin-tip", "aria-hidden": "true" }, title),
  );
}

/** Ease-in-out over the route's length, so the line is drawn at a walker's rhythm. */
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

/** The first part of a [lon, lat] path, `share` (0..1) of its length long. */
function partOf(path, cumulative, share) {
  const target = share * cumulative.at(-1);
  let i = 1;
  while (i < path.length - 1 && cumulative[i] < target) i += 1;
  const [a, b] = [path[i - 1], path[i]];
  const span = cumulative[i] - cumulative[i - 1] || 1;
  const k = Math.min(1, Math.max(0, (target - cumulative[i - 1]) / span));
  return [...path.slice(0, i), [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k]];
}

function lengths(path) {
  const cumulative = [0];
  for (let i = 1; i < path.length; i += 1) {
    const [[x1, y1], [x2, y2]] = [path[i - 1], path[i]];
    // a flat-earth step is exact enough to pace an animation across one city
    const dx = (x2 - x1) * Math.cos((y1 * Math.PI) / 180);
    cumulative.push(cumulative[i - 1] + Math.hypot(dx, y2 - y1));
  }
  return cumulative;
}

/**
 * Create the map in `container`. `hooks` supply the content, the map only places it:
 *   popup(stop, index) -> Node, startPopup(plan) -> Node, ghostTitle(placeId) -> string,
 *   onHover(placeId, on), onSelect(placeId).
 * Throws when the library cannot be loaded or WebGL is unavailable.
 */
export async function createRouteMap(container, hooks) {
  const lib = await import("maplibre-gl");
  return new RouteMap(lib, container, hooks);
}

class RouteMap {
  constructor(lib, container, hooks) {
    this.lib = lib;
    this.hooks = hooks;
    this.map = new lib.Map({
      container,
      ...HOME,
      attributionControl: false,
      // a walking map stays north-up: a rotated or tilted map only confuses
      dragRotate: false,
      pitchWithRotate: false,
      touchPitch: false,
      maxPitch: 0,
      locale: LOCALE,
    });
    this.map.touchZoomRotate.disableRotation();
    this.map.addControl(new lib.NavigationControl({ showCompass: false }), "bottom-right");
    // The licence wants the credit on the map. Compact shows it at first and folds it into
    // an "i" on the first drag; a planner map is seldom dragged (it flies to the city and
    // the route by itself), so it folds on the first camera move instead.
    this.map.addControl(new lib.AttributionControl({ compact: true }), "bottom-right");
    this.map.once("moveend", () => {
      container.querySelector(".maplibregl-compact-show")?.classList.remove("maplibregl-compact-show");
    });

    this.line = { path: [], straight: false }; // what the route source shows right now
    this.frame = 0;
    this.fitTimer = 0;
    this.start = null; // marker
    this.stops = new Map(); // place id -> {marker, popup}
    this.candidates = new Map(); // place id -> {marker, tip}
    this.styleReady = false;

    this.map.on("style.load", () => {
      this.styleReady = true;
      this.installRoute();
    });
    this.map.setStyle(STYLE_URL, { transformStyle: (_previous, next) => lighten(next) });
  }

  /**
   * Called when the style fails before it ever loaded: the page shows its own message.
   * Not map.isStyleLoaded(): that is also false while tiles load, so one failed tile on
   * a working map would cover it with the message.
   */
  onStyleError(callback) {
    this.map.on("error", (event) => {
      if (!this.styleReady) callback(event.error);
    });
  }

  installRoute() {
    if (!this.map.getSource("route")) this.map.addSource("route", { type: "geojson", data: this.routeData() });
    for (const layer of ROUTE_LAYERS) if (!this.map.getLayer(layer.id)) this.map.addLayer(layer);
  }

  routeData() {
    const { path, straight } = this.line;
    if (path.length < 2) return { type: "FeatureCollection", features: [] };
    return { type: "Feature", properties: { straight }, geometry: { type: "LineString", coordinates: path } };
  }

  setLine(path, straight) {
    this.line = { path, straight };
    this.map.getSource("route")?.setData(this.routeData());
  }

  /** Grow the line from the start to the last stop; reduced motion shows it at once. */
  drawLine(path, straight) {
    cancelAnimationFrame(this.frame);
    if (reduceMotion || path.length < 2) {
      this.setLine(path, straight);
      return;
    }
    const cumulative = lengths(path);
    const began = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - began) / DRAW_MS);
      this.setLine(t < 1 ? partOf(path, cumulative, ease(t)) : path, straight);
      if (t < 1) this.frame = requestAnimationFrame(step);
    };
    this.frame = requestAnimationFrame(step);
  }

  padding(wanted) {
    const { clientWidth, clientHeight } = this.map.getContainer();
    return Math.max(8, Math.min(wanted, clientWidth / 6, clientHeight / 6));
  }

  fit(points, maxZoom, duration) {
    if (!points.length) return;
    const bounds = new this.lib.LngLatBounds(points[0], points[0]);
    for (const point of points) bounds.extend(point);
    this.map.fitBounds(bounds, { padding: this.padding(56), maxZoom, duration });
  }

  addMarker(element, lngLat) {
    return new this.lib.Marker({ element, anchor: "center" }).setLngLat(lngLat).addTo(this.map);
  }

  setStart(lngLat, title, popupNode) {
    this.start?.remove();
    const element = pin("start", "", 0, title, Boolean(popupNode));
    const marker = this.addMarker(element, lngLat);
    // a marker with a popup opens it on click, Enter and Space by itself
    if (popupNode) marker.setPopup(this.popup(popupNode));
    this.start = marker;
  }

  popup(node) {
    return new this.lib.Popup({ maxWidth: "320px", offset: 20, focusAfterOpen: false, className: "route-popup" })
      .setDOMContent(node);
  }

  /* ---------- live research ---------- */

  /** The city is found: fly there and mark the start (MapLibre skips the flight under reduced motion). */
  showStart({ lat, lon, label }) {
    this.setStart([lon, lat], label ?? "Старт", null);
    this.map.flyTo({ center: [lon, lat], zoom: 14.5, duration: 1200 });
  }

  /** A place the agent shortlisted appears the moment it is chosen. */
  addCandidate(id, lat, lon, title) {
    this.removeCandidate(id);
    const element = pin("cand", "", 0, title, false);
    this.candidates.set(id, { marker: this.addMarker(element, [lon, lat]), tip: element.lastChild });
    clearTimeout(this.fitTimer);
    this.fitTimer = setTimeout(() => this.fitCandidates(), FIT_DELAY_MS);
  }

  removeCandidate(id) {
    this.candidates.get(id)?.marker.remove();
    this.candidates.delete(id);
  }

  fitCandidates() {
    const points = [...this.candidates.values()].map(({ marker }) => marker.getLngLat().toArray());
    if (this.start) points.push(this.start.getLngLat().toArray());
    if (points.length >= 2) this.fit(points, 16, 800);
  }

  clearCandidates() {
    clearTimeout(this.fitTimer);
    for (const { marker } of this.candidates.values()) marker.remove();
    this.candidates.clear();
    this.start?.remove();
    this.start = null;
  }

  /* ---------- the plan ---------- */

  /** Stops get numbered markers; shortlisted places that did not fit stay as ghosts. */
  showPlan(plan) {
    clearTimeout(this.fitTimer);
    this.clearRoute();
    const inPlan = new Set(plan.stops.map((stop) => stop.place.id));
    for (const [id, { marker, tip }] of this.candidates) {
      if (inPlan.has(id)) {
        this.removeCandidate(id);
      } else {
        marker.getElement().classList.add("is-dropped");
        tip.textContent = this.hooks.ghostTitle(id);
        marker.getElement().setAttribute("aria-label", tip.textContent);
      }
    }

    const start = [plan.start.lon, plan.start.lat];
    const points = plan.stops.map((stop) => [stop.place.lon, stop.place.lat]);
    const straight = !plan.geometry?.length;
    // the server sends geometry as [lat, lon]; GeoJSON and MapLibre want [lon, lat]
    const path = straight ? [start, ...points] : plan.geometry.map(([lat, lon]) => [lon, lat]);

    this.setStart(start, plan.start.label ?? "Старт", this.hooks.startPopup(plan));
    plan.stops.forEach((stop, index) => {
      const id = stop.place.id;
      const element = pin("stop", String(index + 1), index + 1, `${index + 1}. ${stop.place.name}`, true);
      const marker = this.addMarker(element, points[index]);
      const popup = this.popup(this.hooks.popup(stop, index));
      marker.setPopup(popup);
      element.addEventListener("mouseenter", () => this.hooks.onHover(id, true));
      element.addEventListener("mouseleave", () => this.hooks.onHover(id, false));
      element.addEventListener("click", () => this.hooks.onSelect(id));
      element.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        // also cancels MapLibre's own keypress toggle, which would close what focus() opens
        event.preventDefault();
        this.focus(id);
        this.hooks.onSelect(id);
      });
      this.stops.set(id, { marker, popup });
    });

    // maxZoom: a route of one stop next to the start would otherwise zoom in to single houses
    this.fit([...path, ...points], 17, 0);
    this.drawLine(path, straight);
  }

  clearRoute() {
    cancelAnimationFrame(this.frame);
    for (const { marker } of this.stops.values()) marker.remove();
    this.stops.clear();
    this.setLine([], false);
  }

  /** How many shortlisted places are on the map: still candidates, or ghosts after a plan. */
  counts() {
    let ghosts = 0;
    for (const { marker } of this.candidates.values()) if (marker.getElement().classList.contains("is-dropped")) ghosts += 1;
    return { live: this.candidates.size - ghosts, ghosts };
  }

  highlight(id, on) {
    this.stops.get(id)?.marker.getElement().classList.toggle("is-hot", on);
  }

  /** From the timeline: centre the stop and open its popup, closing any other. */
  focus(id) {
    const entry = this.stops.get(id);
    if (!entry) return;
    for (const other of this.stops.values()) if (other !== entry && other.popup.isOpen()) other.popup.remove();
    this.start?.getPopup()?.remove();
    const center = entry.marker.getLngLat();
    this.map.easeTo({ center, zoom: Math.max(this.map.getZoom(), 15), duration: 600 });
    if (!entry.popup.isOpen()) entry.marker.togglePopup();
  }
}
