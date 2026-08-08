"""
Client for the National Weather Service (NWS) API (api.weather.gov).

No API key required, but NWS asks every client to send a descriptive
User-Agent identifying the application and a contact (they throttle/block
generic ones like "python-requests"). Mirrors massive_client.py's shape:
a thin requests.Session wrapper plus normalization helpers that turn raw
API responses into the document schema weather_documents expects.
"""

import os
import re
from datetime import datetime, timezone
from typing import Any

import requests

_BASE_URL = os.environ.get("WEATHER_API_BASE_URL", "https://api.weather.gov")
_USER_AGENT = os.environ.get("WEATHER_USER_AGENT", "")

_DEFAULT_TIMEOUT = 30

# No free geocoding API is in scope for this assignment, so a small built-in
# lookup resolves the handful of demo locations by name. Anything else must
# be passed as "lat,lon" or {"lat": .., "lon": ..}. Known limitation --
# documented in README_WEATHER.md.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "sacramento, ca": (38.5816, -121.4944),
    "new york, ny": (40.7128, -74.0060),
    "chicago, il": (41.8781, -87.6298),
    "miami, fl": (25.7617, -80.1918),
    "seattle, wa": (47.6062, -122.3321),
    "austin, tx": (30.2672, -97.7431),
    "denver, co": (39.7392, -104.9903),
}

_LATLON_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")


def resolve_location(location: Any) -> tuple[float, float, str]:
    """Resolve a location argument to (lat, lon, label).

    Accepts:
      - {"lat": float, "lon": float, "label": optional str}
      - "lat,lon" string, e.g. "41.8781,-87.6298"
      - a known "City, ST" string from CITY_COORDS (case-insensitive)

    Raises ValueError if the location can't be resolved to coordinates.
    """
    if isinstance(location, dict):
        try:
            lat = float(location["lat"])
            lon = float(location["lon"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Invalid location dict: {location!r}")
        label = location.get("label") or f"{lat},{lon}"
        return lat, lon, label

    if isinstance(location, str):
        key = location.strip().lower()
        if key in CITY_COORDS:
            lat, lon = CITY_COORDS[key]
            return lat, lon, location.strip()

        match = _LATLON_RE.match(location)
        if match:
            lat, lon = float(match.group(1)), float(match.group(2))
            return lat, lon, location.strip()

    raise ValueError(
        f"Could not resolve location {location!r} -- pass a known 'City, ST' "
        f"name ({', '.join(sorted(CITY_COORDS))}), a 'lat,lon' string, or a "
        f"{{'lat': .., 'lon': ..}} dict."
    )


class WeatherClient:
    """Thin wrapper around the NWS API with a session carrying the required
    User-Agent header."""

    def __init__(self, base_url: str | None = None, timeout: int = _DEFAULT_TIMEOUT):
        if not _USER_AGENT:
            raise ValueError(
                "WEATHER_USER_AGENT is not set. NWS requires a descriptive "
                "User-Agent identifying your application and a contact "
                "method, e.g. '(databricks-lakebase-bootcamp, you@example.com)' "
                "-- generic/default User-Agents get throttled or blocked. "
                "Set it in app.yaml or your .env file."
            )
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                "Accept": "application/geo+json",
            }
        )

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        resp = self._session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_point(self, lat: float, lon: float) -> dict:
        """GET /points/{lat},{lon} -- resolves a coordinate to its NWS grid
        office/x/y, needed for the forecast endpoints."""
        return self._get(f"/points/{lat},{lon}")

    def get_active_alerts(self, state: str) -> list[dict]:
        """GET /alerts/active?area={state} -- active alert features for a
        two-letter state code."""
        data = self._get("/alerts/active", params={"area": state})
        return data.get("features", [])

    def get_forecast(self, office: str, grid_x: int, grid_y: int) -> dict:
        """GET /gridpoints/{office}/{x},{y}/forecast -- multi-day narrative
        forecast periods for a grid point."""
        return self._get(f"/gridpoints/{office}/{grid_x},{grid_y}/forecast")

    def get_documents_for_point(
        self, lat: float, lon: float, label: str | None = None, limit: int = 50
    ) -> list[dict]:
        """One-call, normalized fetch for a single location: resolves the
        grid point, then pulls active alerts (for the point's state) and the
        forecast, normalizing both into the weather_documents schema.

        Mirrors MassiveClient.get_news -- callers (the /weather/sync route)
        never touch raw NWS JSON directly.
        """
        point = self.get_point(lat, lon)
        props = point.get("properties", {})
        office = props.get("cwa")
        grid_x = props.get("gridX")
        grid_y = props.get("gridY")
        rel_loc = (props.get("relativeLocation") or {}).get("properties", {})
        state = rel_loc.get("state")

        resolved_label = label or ", ".join(
            filter(None, [rel_loc.get("city"), rel_loc.get("state")])
        ) or f"{lat},{lon}"

        docs: list[dict] = []

        if state:
            alerts = self.get_active_alerts(state)[:limit]
            docs.extend(_normalize_alert(f, resolved_label) for f in alerts)

        if office and grid_x is not None and grid_y is not None:
            forecast = self.get_forecast(office, grid_x, grid_y)
            periods = forecast.get("properties", {}).get("periods", [])[:limit]
            issued_at = forecast.get("properties", {}).get("updated")
            docs.extend(
                _normalize_forecast_period(period, resolved_label, office, grid_x, grid_y, issued_at)
                for period in periods
            )

        return docs


def _normalize_alert(feature: dict, location: str) -> dict:
    """NWS alert GeoJSON Feature -> weather_documents row.

    The alert's own "id" field is NWS's stable URN for that alert (already a
    dedup key) -- reuse it rather than inventing one.
    """
    properties = feature.get("properties", {})
    narrative = "\n\n".join(
        filter(None, [properties.get("description"), properties.get("instruction")])
    )
    return {
        "id": f"alert:{properties.get('id')}",
        "location": location,
        "source_type": "alert",
        "headline": properties.get("headline") or properties.get("event"),
        "narrative_text": narrative,
        "issued_at": properties.get("sent"),
        "effective_at": properties.get("effective"),
        "payload": feature,
    }


def _normalize_forecast_period(
    period: dict, location: str, office: str, grid_x: int, grid_y: int, issued_at: str | None
) -> dict:
    """NWS forecast period -> weather_documents row.

    NWS doesn't give forecast periods a stable id of their own, so one is
    constructed from the grid point + period number. This means each
    location has a fixed rolling set of ~14 forecast document ids that get
    upserted in place on every sync (latest issuance only, no history kept)
    -- documented as a known limitation in README_WEATHER.md.
    """
    period_number = period.get("number")
    return {
        "id": f"forecast:{office}:{grid_x},{grid_y}:{period_number}",
        "location": location,
        "source_type": "forecast",
        "headline": period.get("name"),
        "narrative_text": period.get("detailedForecast") or period.get("shortForecast"),
        "issued_at": issued_at or datetime.now(timezone.utc).isoformat(),
        "effective_at": period.get("startTime"),
        "payload": period,
    }
