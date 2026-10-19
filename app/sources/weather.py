"""Hourly weather from Open-Meteo (free, no key) and a one-line summary of it.

The planner needs the forecast per hour to keep outdoor stops out of the rain;
the traveller needs one line. Open-Meteo answers with the location's local time
(timezone=auto), which is the same naive local time the whole plan uses.

Open-Meteo forecasts 16 days (today + 15) and answers HTTP 400 outside that
window. A trip planned for a day without a forecast is normal, so such a day
gives an empty list and no request is made.
"""

from datetime import date, datetime, time, timedelta

from app.models import GeoPoint, HourWeather
from app.sources.http import http

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_DAYS_AHEAD = 15
# An hour counts as rainy above either threshold; planning/validator.py uses the
# same numbers, so the summary and the rain warnings agree.
RAIN_PROBABILITY = 50  # percent
RAIN_MM = 0.3  # per hour

PHRASES = {
    "ru": {"none": "прогноза нет", "dry": "без осадков", "rain": "дождь вероятен", "max": "до"},
    "en": {"none": "no forecast", "dry": "no precipitation", "rain": "rain likely", "max": "up to"},
}


async def hourly(point: GeoPoint, day: date) -> list[HourWeather]:
    today = date.today()
    if not today <= day <= today + timedelta(days=FORECAST_DAYS_AHEAD):
        return []
    payload = await http.get_json(
        FORECAST_URL,
        {
            "latitude": point.lat,
            "longitude": point.lon,
            "hourly": "temperature_2m,precipitation_probability,precipitation",
            "timezone": "auto",
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
        },
    )
    series = payload["hourly"]
    values = zip(
        series["time"],
        series["temperature_2m"],
        series["precipitation_probability"],
        series["precipitation"],
        strict=True,
    )
    return [
        HourWeather(
            time=datetime.fromisoformat(stamp),
            temperature_c=temperature,
            precipitation_probability=probability,
            precipitation_mm=amount,
        )
        for stamp, temperature, probability, amount in values
        # The last forecast day ends early: its later hours are all nulls, and they
        # must read as "no forecast", not as "no rain".
        if (temperature, probability, amount) != (None, None, None)
    ]


def summarize(hours: list[HourWeather], start: time, end: time, lang: str = "ru") -> str:
    """'+8…+11 °C, дождь вероятен 14:00–16:00 (до 70%)' for the hours that overlap [start, end)."""
    words = PHRASES.get(lang, PHRASES["en"])
    start_min, end_min = start.hour * 60 + start.minute, end.hour * 60 + end.minute
    window = [h for h in hours if h.time.hour * 60 + 60 > start_min and h.time.hour * 60 < end_min]
    if not window:
        return words["none"]

    temps = [round(h.temperature_c) for h in window if h.temperature_c is not None]
    parts = []
    if temps:
        low, high = _degrees(min(temps)), _degrees(max(temps))
        parts.append(f"{low} °C" if low == high else f"{low}…{high} °C")

    rainy = [h for h in window if _is_rainy(h)]
    if not rainy:
        parts.append(words["dry"])
    else:
        spans = ", ".join(f"{a:%H:%M}–{b:%H:%M}" for a, b in _spans(rainy))
        rain = f"{words['rain']} {spans}"
        chance = max((h.precipitation_probability or 0) for h in rainy)
        parts.append(f"{rain} ({words['max']} {chance}%)" if chance else rain)
    return ", ".join(parts)


def _degrees(value: int) -> str:
    """'+8', '-3' and a plain '0', as weather forecasts write it."""
    return f"{value:+d}" if value else "0"


def _spans(hours: list[HourWeather]) -> list[tuple[datetime, datetime]]:
    """Merge consecutive hours into (start, end) intervals."""
    spans: list[tuple[datetime, datetime]] = []
    for h in hours:
        end = h.time + timedelta(hours=1)
        if spans and spans[-1][1] == h.time:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((h.time, end))
    return spans


def _is_rainy(hour: HourWeather) -> bool:
    wet_chance = (hour.precipitation_probability or 0) >= RAIN_PROBABILITY
    wet_amount = (hour.precipitation_mm or 0) >= RAIN_MM
    return wet_chance or wet_amount
