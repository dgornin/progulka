/*
 * The planner's map: MapLibre GL JS drawing OpenFreeMap vector tiles.
 *
 * The basemap is OpenFreeMap's "positron", the quietest of its styles: no POI icons to
 * mistake for our stops and grey roads instead of colourful ones. It is recoloured a
 * little warmer (the same colours as the landing picture) and labelled in Russian
 * where OpenStreetMap has a Russian name, so the routes stay the brightest things on
 * the map and the page reads as one light design. Tiles and style need no key.
 *
 * A journey is drawn whole: every day's route in its own colour from base.css's day
 * palette (terracotta first, so a one-day trip looks as it always did), with markers numbered
 * per day. Selecting a day dims the other days instead of hiding them, so the
 * traveller still sees how the days divide the city. All routes are features of one
 * GeoJSON source whose paint reads the colour and the day from feature properties:
 * selecting a day swaps two expressions, nothing is rebuilt. The source lives in the
 * map style, so it is (re)installed on every "style.load" from the lines kept here,
 * and a journey that arrives before the style has loaded simply appears once it has.
 *
 * Places are DOM markers, so they animate with CSS and react to hover like the
 * timeline does. Outside text (place names, descriptions) reaches the map only as DOM
 * nodes and text: popups get setDOMContent, never setHTML. The library is imported on
 * demand, so when unpkg or WebGL is unavailable the planner keeps working without it.
 */

import { h } from "./ui.js";

const STYLE_URL = "https://tiles.openfreemap.org/styles/positron";
const HOME = { center: [34, 57.5], zoom: 3.6 }; // European Russia, before a city is known
const DRAW_MS = 1500;
// While places stream in, the map re-frames at most this often: parallel days shortlist
// faster than a debounce would ever settle, and a frame per place would only shake.
const FIT_DELAY_MS = 350;
const LEGEND_PX = 32; // the legend over the map's top edge
const IMPORT_TIMEOUT_MS = 12000; // slow venue Wi-Fi still makes it in time
const SAME_PLACE_M = 60; // a day that starts this close to the hotel starts at the hotel

const PALETTE_SIZE = 7; // --day-1 ... --day-7 in base.css, one per day of the longest trip
let palette = null;

/**
 * A day's colour. The palette is base.css's (--day-N), so a day wears the same colour
 * here as on the cabinet's cards; the map's WebGL layers need a literal colour, not a
 * var(), so it is read from the stylesheet once, on first use.
 */
export function dayColor(day) {
  if (!palette) {
    const tokens = getComputedStyle(document.documentElement);
    palette = Array.from({ length: PALETTE_SIZE }, (_, i) => tokens.getPropertyValue(`--day-${i + 1}`).trim());
  }
  return palette[(day ?? 0) % PALETTE_SIZE];
}

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
  "Map.Title": "Карта путешествия",
};

const width = (low, high) => ["interpolate", ["linear"], ["zoom"], 11, low, 17, high];
const ROUND = { "line-join": "round", "line-cap": "round" };
const LINE_LAYERS = ["route-line", "route-dots"]; // the clickable ones

/** The route layers for a selected day (null: all days at full strength). */
function routeLayers(selected) {
  const pick = (strong, dim) => (selected === null ? strong : ["case", ["==", ["get", "day"], selected], strong, dim]);
  // the selected day is drawn over the others where routes cross
  const layout = { ...ROUND, "line-sort-key": pick(1, 0) };
  return [
    {
      id: "route-casing",
      type: "line",
      source: "route",
      layout,
      paint: { "line-color": "#ffffff", "line-width": width(6, 14), "line-opacity": pick(0.95, 0.5) },
    },
    {
      id: "route-line",
      type: "line",
      source: "route",
      filter: ["!=", ["get", "straight"], true],
      layout,
      paint: { "line-color": ["get", "color"], "line-width": width(3, 6.5), "line-opacity": pick(1, 0.3) },
    },
    {
      // straight segments between the stops: dots say "we do not know the streets here"
      id: "route-dots",
      type: "line",
      source: "route",
      filter: ["==", ["get", "straight"], true],
      layout,
      paint: {
        "line-color": ["get", "color"],
        "line-width": width(4, 7),
        "line-dasharray": [0, 2],
        "line-opacity": pick(1, 0.3),
      },
    },
  ];
}

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
 * text node), in its day's colour. MapLibre positions the element with a transform,
 * so it animates its child.
 */
function pin(kind, number, order, title, clickable, day = null) {
  const role = clickable ? { role: "button", tabindex: "0" } : { role: "img" };
  return h(
    "div",
    { class: `pin pin-${kind}`, style: day === null ? null : `--day: ${dayColor(day)}`, ...role, "aria-label": title },
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

// a flat-earth step is exact enough to pace an animation, or compare two points, in one city
function flat([x1, y1], [x2, y2]) {
  return Math.hypot((x2 - x1) * Math.cos((y1 * Math.PI) / 180), y2 - y1);
}

function lengths(path) {
  const cumulative = [0];
  for (let i = 1; i < path.length; i += 1) cumulative.push(cumulative[i - 1] + flat(path[i - 1], path[i]));
  return cumulative;
}

const near = (a, b) => flat(a, b) * 111_320 < SAME_PLACE_M; // degrees of latitude to metres

/**
 * Create the map in `container`. `hooks` supply the content, the map only places it:
 *   popup(stop, index, day) -> Node, startPopup(point) -> Node, ghostTitle(placeId) -> string,
 *   onHover(placeId, on), onSelect(placeId, day), onDay(day) for a click on a day's route.
 * Throws when the library cannot be loaded or WebGL is unavailable. A stalled CDN
 * neither loads nor fails, so it counts as failed after IMPORT_TIMEOUT_MS: the planner
 * waits for the map before it enables its button.
 */
export async function createRouteMap(container, hooks) {
  const stalled = new Promise((_, reject) => {
    setTimeout(() => reject(new Error("библиотека карты не ответила вовремя")), IMPORT_TIMEOUT_MS);
  });
  const lib = await Promise.race([import("maplibre-gl"), stalled]);
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

    this.lines = []; // [{day, path, straight}]: what the route source shows right now
    this.extent = new Map(); // day -> [lon, lat] points of its route, for framing
    this.selected = null; // the focused day; null is every day
    this.frame = 0;
    this.fitTimer = 0;
    this.start = null; // the journey's start marker: where the traveller stays
    this.dayStarts = []; // [{marker, day}]: days that start in their own district
    this.stops = new Map(); // place id -> {marker, popup, day}
    this.candidates = new Map(); // place id -> {marker, tip, day}
    this.styleReady = false;

    this.map.on("style.load", () => {
      this.styleReady = true;
      this.installRoute();
    });
    this.map.setStyle(STYLE_URL, { transformStyle: (_previous, next) => lighten(next) });
    for (const id of LINE_LAYERS) {
      this.map.on("click", id, (event) => {
        const day = event.features?.[0]?.properties?.day;
        if (typeof day === "number") this.hooks.onDay(day);
      });
      this.map.on("mouseenter", id, () => {
        this.map.getCanvas().style.cursor = "pointer";
      });
      this.map.on("mouseleave", id, () => {
        this.map.getCanvas().style.cursor = "";
      });
    }
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
    for (const layer of routeLayers(this.selected)) if (!this.map.getLayer(layer.id)) this.map.addLayer(layer);
  }

  routeData() {
    const features = this.lines
      .filter(({ path }) => path.length >= 2)
      .map(({ day, path, straight }) => ({
        type: "Feature",
        properties: { day, straight, color: dayColor(day) },
        geometry: { type: "LineString", coordinates: path },
      }));
    return { type: "FeatureCollection", features };
  }

  setLines(lines) {
    this.lines = lines;
    this.map.getSource("route")?.setData(this.routeData());
  }

  /** Grow every day's line from its start at once; reduced motion shows them whole. */
  drawLines(lines) {
    cancelAnimationFrame(this.frame);
    if (reduceMotion) {
      this.setLines(lines);
      return;
    }
    const measured = lines.map((line) => ({ line, cumulative: lengths(line.path) }));
    const began = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - began) / DRAW_MS);
      const share = ease(t);
      this.setLines(
        t < 1
          ? measured.map(({ line, cumulative }) =>
              line.path.length < 2 ? line : { ...line, path: partOf(line.path, cumulative, share) },
            )
          : lines,
      );
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
    // The container may have just changed size (the journey bar appeared above it);
    // MapLibre's own resize observer would catch up only after this frame is computed.
    this.map.resize();
    const bounds = new this.lib.LngLatBounds(points[0], points[0]);
    for (const point of points) bounds.extend(point);
    const pad = this.padding(56);
    const padding = { top: pad + Math.min(LEGEND_PX, pad), bottom: pad, left: pad, right: pad };
    this.map.fitBounds(bounds, { padding, maxZoom, duration });
  }

  cancelFit() {
    clearTimeout(this.fitTimer);
    this.fitTimer = 0;
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

  /** Days whose route starts in a district of their own get a start ring in their colour. */
  setDayStarts(points) {
    for (const { marker } of this.dayStarts) marker.remove();
    this.dayStarts = [];
    const home = this.start?.getLngLat().toArray();
    points.forEach((point, day) => {
      if (!point) return;
      const lngLat = [point.lon, point.lat];
      if (home && near(home, lngLat)) return;
      const title = `День ${day + 1}: начало — ${point.label ?? "район дня"}`;
      this.dayStarts.push({ marker: this.addMarker(pin("daystart", "", 0, title, false, day), lngLat), day });
    });
    this.applySelection();
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

  /** A place the agent shortlisted appears the moment it is chosen, in its day's colour. */
  addCandidate(id, lat, lon, title, day = null) {
    this.removeCandidate(id);
    const element = pin("cand", "", 0, title, false, day);
    element.classList.toggle("is-dim", this.dimmed(day));
    this.candidates.set(id, { marker: this.addMarker(element, [lon, lat]), tip: element.lastChild, day });
    this.fitTimer ||= setTimeout(() => {
      this.fitTimer = 0;
      this.fit(this.livePoints(), 16, 800);
    }, FIT_DELAY_MS);
  }

  removeCandidate(id) {
    this.candidates.get(id)?.marker.remove();
    this.candidates.delete(id);
  }

  /** What a run has put on the map so far, for the selected day or for all. */
  livePoints() {
    const shown = (day) => this.selected === null || day === null || day === this.selected;
    const points = [...this.candidates.values()].filter(({ day }) => shown(day)).map(({ marker }) => marker.getLngLat().toArray());
    for (const { marker, day } of this.dayStarts) if (shown(day)) points.push(marker.getLngLat().toArray());
    if (this.start) points.push(this.start.getLngLat().toArray());
    return points.length >= 2 ? points : [];
  }

  clearCandidates() {
    this.cancelFit();
    for (const { marker } of this.candidates.values()) marker.remove();
    this.candidates.clear();
    this.start?.remove();
    this.start = null;
  }

  /* ---------- the journey ---------- */

  /**
   * Every day's stops get numbered markers in the day's colour; shortlisted places that
   * made it into no day stay as ghosts. `journey` is the API's JourneyView; `selected`
   * the day to focus (null: all), framed at once rather than flown to.
   */
  showJourney(journey, selected = null) {
    this.cancelFit();
    this.clearRoute();
    this.selected = selected;
    const { days } = journey;
    const inPlan = new Set(days.flatMap((day) => day.plan.stops.map((stop) => stop.place.id)));
    for (const [id, { marker, tip }] of this.candidates) {
      if (inPlan.has(id)) {
        this.removeCandidate(id);
      } else {
        marker.getElement().classList.add("is-dropped");
        tip.textContent = this.hooks.ghostTitle(id);
        marker.getElement().setAttribute("aria-label", tip.textContent);
      }
    }

    const start = journey.start;
    this.setStart([start.lon, start.lat], start.label ?? "Старт", this.hooks.startPopup(start));
    const several = days.length > 1;
    const lines = [];
    for (const { index: day, plan } of days) {
      const from = [plan.start.lon, plan.start.lat];
      const points = plan.stops.map((stop) => [stop.place.lon, stop.place.lat]);
      const straight = !plan.geometry?.length;
      // the server sends geometry as [lat, lon]; GeoJSON and MapLibre want [lon, lat]
      const path = straight ? [from, ...points] : plan.geometry.map(([lat, lon]) => [lon, lat]);
      lines.push({ day, path, straight });
      this.extent.set(day, [from, ...path, ...points]);
      plan.stops.forEach((stop, index) => this.addStop(stop, index, day, several, points[index]));
    }
    this.setDayStarts(days.map((day) => day.plan.start));
    this.applySelection();
    // maxZoom: a route of one stop next to the start would otherwise zoom in to single houses
    this.fit(this.framed(), 17, 0);
    this.drawLines(lines);
  }

  addStop(stop, index, day, several, lngLat) {
    const id = stop.place.id;
    const title = `${several ? `День ${day + 1} · ` : ""}${index + 1}. ${stop.place.name}`;
    const element = pin("stop", String(index + 1), index + 1, title, true, day);
    const marker = this.addMarker(element, lngLat);
    const popup = this.popup(this.hooks.popup(stop, index, day));
    marker.setPopup(popup);
    element.addEventListener("mouseenter", () => this.hooks.onHover(id, true));
    element.addEventListener("mouseleave", () => this.hooks.onHover(id, false));
    element.addEventListener("click", () => this.hooks.onSelect(id, day));
    element.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      // also cancels MapLibre's own keypress toggle, which would close what focus() opens
      event.preventDefault();
      this.hooks.onSelect(id, day);
      this.focus(id);
    });
    this.stops.set(id, { marker, popup, day });
  }

  clearRoute() {
    cancelAnimationFrame(this.frame);
    for (const { marker } of this.stops.values()) marker.remove();
    this.stops.clear();
    for (const { marker } of this.dayStarts) marker.remove();
    this.dayStarts = [];
    this.extent.clear();
    this.setLines([]);
  }

  /* ---------- selecting a day ---------- */

  /** Focus one day (dim the rest) or, with null, show every day; then frame it. */
  select(day) {
    this.selected = day;
    this.applySelection();
    const points = this.framed();
    this.fit(points.length ? points : this.livePoints(), this.extent.size ? 17 : 16, 700);
  }

  /** The selected day's route, or every route; empty before a journey has arrived. */
  framed() {
    if (this.selected !== null && this.extent.has(this.selected)) return this.extent.get(this.selected);
    return [...this.extent.values()].flat();
  }

  dimmed(day) {
    return this.selected !== null && day !== null && day !== this.selected;
  }

  applySelection() {
    for (const { marker, popup, day } of this.stops.values()) {
      marker.getElement().classList.toggle("is-dim", this.dimmed(day));
      if (this.dimmed(day) && popup.isOpen()) popup.remove();
    }
    for (const { marker, day } of [...this.candidates.values(), ...this.dayStarts]) {
      marker.getElement().classList.toggle("is-dim", this.dimmed(day));
    }
    if (!this.styleReady) return; // installRoute() builds the layers for this selection
    for (const layer of routeLayers(this.selected)) {
      if (!this.map.getLayer(layer.id)) continue;
      this.map.setPaintProperty(layer.id, "line-opacity", layer.paint["line-opacity"]);
      this.map.setLayoutProperty(layer.id, "line-sort-key", layer.layout["line-sort-key"]);
    }
  }

  /** What is on the map right now, for the legend. */
  counts() {
    let ghosts = 0;
    for (const { marker } of this.candidates.values()) if (marker.getElement().classList.contains("is-dropped")) ghosts += 1;
    return { live: this.candidates.size - ghosts, ghosts, dayStarts: this.dayStarts.length };
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
