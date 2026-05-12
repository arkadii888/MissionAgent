"""Thread-safe, set-once storage of the drone's first-takeoff home location."""
from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True, slots=True)
class HomeLocation:
    """Immutable snapshot of a GPS position used as the RTH destination.

    Attributes:
        latitude_deg: WGS-84 latitude in decimal degrees.
        longitude_deg: WGS-84 longitude in decimal degrees.
        relative_altitude_m: Altitude above the arming/takeoff ground level in metres.
    """

    latitude_deg: float
    longitude_deg: float
    relative_altitude_m: float


def _home_from_env() -> HomeLocation | None:
    """Read a HomeLocation from ``RESCUE_HOME_*`` environment variables.

    Both latitude and longitude must be set; altitude defaults to 0.0 if omitted.

    Returns:
        A HomeLocation if the required variables are present and parseable, otherwise None.
    """
    lat_raw = os.getenv("RESCUE_HOME_LATITUDE_DEG")
    lon_raw = os.getenv("RESCUE_HOME_LONGITUDE_DEG")
    alt_raw = os.getenv("RESCUE_HOME_RELATIVE_ALTITUDE_M", "0.0")
    if lat_raw is None or lon_raw is None:
        return None
    try:
        return HomeLocation(
            latitude_deg=float(lat_raw),
            longitude_deg=float(lon_raw),
            relative_altitude_m=float(alt_raw),
        )
    except ValueError:
        return None


class HomeLocationState:
    """Thread-safe, set-once container for the drone's first-takeoff home position.

    The home location is written exactly once: either via the ``RESCUE_HOME_*``
    environment variable override at startup, or by the first ``takeoff`` intent
    processed by ``_plan_from_prompt``. Subsequent calls to ``set_once`` are silently
    ignored so the home point never drifts.

    Thread safety: ``get`` and ``set_once`` both acquire an internal ``threading.Lock``.
    """

    def __init__(self, *, initial: HomeLocation | None = None) -> None:
        """Initialise the state, optionally with a pre-seeded location.

        Args:
            initial: If provided, used as the home location regardless of env vars.
                Pass None (default) to use the ``RESCUE_HOME_*`` env-var override or
                wait for the first takeoff.
        """
        self._lock = Lock()
        self._home: HomeLocation | None = initial if initial is not None else _home_from_env()

    def get(self) -> HomeLocation | None:
        """Return the stored home location, or None if not yet set.

        Returns:
            The HomeLocation, or None.
        """
        with self._lock:
            return self._home

    def set_once(
        self,
        latitude_deg: float,
        longitude_deg: float,
        relative_altitude_m: float,
    ) -> None:
        """Store the home location if it has not already been set.

        Silently ignores the call if a home location is already stored (including an
        env-var override), so only the *first* takeoff ever sets the home point.

        Args:
            latitude_deg: WGS-84 latitude in decimal degrees.
            longitude_deg: WGS-84 longitude in decimal degrees.
            relative_altitude_m: Altitude AGL at takeoff in metres.
        """
        new_home = HomeLocation(
            latitude_deg=float(latitude_deg),
            longitude_deg=float(longitude_deg),
            relative_altitude_m=float(relative_altitude_m),
        )
        with self._lock:
            if self._home is None:
                self._home = new_home
