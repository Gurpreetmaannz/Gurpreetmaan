"""Time of concentration formulae that need only DEM-derived inputs.

All functions return Tc in minutes. Inputs:

    L_km   main flow path length (km)
    S      main flow path slope (m/m)
    H_m    elevation drop along the flow path (m)
    A_km2  catchment area (km2)
    Hmean_m  mean catchment elevation above the outlet (m)

Each formula was calibrated on a particular region and catchment size.
The range of applicability is stated so that results can be sanity-checked,
not because the code enforces it. Choosing the appropriate method for a
project remains an engineering judgement.

Methods that need land cover, roughness or rainfall (NRCS velocity method,
Kerby-Hathaway, FAA, kinematic wave) are deliberately not included.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


def kirpich(L_km: float, S: float, **_) -> float:
    """Kirpich (1940). Tc[min] = 0.0195 L^0.77 S^-0.385, with L in metres.

    Small agricultural catchments in Tennessee (about 0.5 to 45 ha).
    Commonly extrapolated well beyond that range, which tends to give short
    times for larger or less steep catchments.
    """
    return 0.0195 * (L_km * 1000.0) ** 0.77 * S**-0.385


def california_culverts(L_km: float, H_m: float, **_) -> float:
    """California Culverts Practice (1942), a Kirpich variant.

    Tc[min] = 57 L^1.155 H^-0.385 with L in km and H in metres.
    Developed for small mountainous catchments.
    """
    return 57.0 * L_km**1.155 * H_m**-0.385


def bransby_williams(L_km: float, A_km2: float, S: float, **_) -> float:
    """Bransby-Williams (1922). Tc[min] = 14.6 L A^-0.1 S^-0.2.

    L in km, A in km2, S in m/m. Equivalent to the ARR 1987 form
    58 L / (A^0.1 Se^0.2) with Se in m/km. Rural catchments, mainly used in
    Australia, New Zealand and the UK.
    """
    return 14.6 * L_km * A_km2**-0.1 * S**-0.2


def giandotti(L_km: float, A_km2: float, Hmean_m: float, **_) -> float:
    """Giandotti (1934). Tc[h] = (4 sqrt(A) + 1.5 L) / (0.8 sqrt(Hmean)).

    A in km2, L in km, Hmean is mean catchment elevation above the outlet in
    metres. Italian catchments, mid-sized to large (about 170 to 70,000 km2
    originally), so treat results for small catchments with caution.
    """
    return 60.0 * (4.0 * math.sqrt(A_km2) + 1.5 * L_km) / (0.8 * math.sqrt(Hmean_m))


def pilgrim_mcdermott(A_km2: float, **_) -> float:
    """Pilgrim and McDermott (1982). Tc[h] = 0.76 A^0.38.

    Area-only relation for rural catchments in eastern Australia
    (about 0.04 to 1,000 km2). Needs neither slope nor length.
    """
    return 60.0 * 0.76 * A_km2**0.38


@dataclass(frozen=True)
class TcMethod:
    name: str
    func: Callable[..., float]
    needs: tuple[str, ...]  # keyword arguments the function requires


METHODS: dict[str, TcMethod] = {
    "kirpich": TcMethod("kirpich", kirpich, ("L_km", "S")),
    "california_culverts": TcMethod("california_culverts", california_culverts, ("L_km", "H_m")),
    "bransby_williams": TcMethod("bransby_williams", bransby_williams, ("L_km", "A_km2", "S")),
    "giandotti": TcMethod("giandotti", giandotti, ("L_km", "A_km2", "Hmean_m")),
    "pilgrim_mcdermott": TcMethod("pilgrim_mcdermott", pilgrim_mcdermott, ("A_km2",)),
}

# NRCS convention: lag time is about 0.6 of the time of concentration.
TLAG_FACTOR = 0.6


def compute_all(**inputs: float) -> dict[str, float]:
    """Run every method whose inputs are valid; return Tc in minutes.

    A method is skipped (NaN) when an input is missing, zero or not finite,
    for example a flat catchment with S = 0.
    """
    out: dict[str, float] = {}
    for name, m in METHODS.items():
        args = {k: inputs.get(k, float("nan")) for k in m.needs}
        if all(math.isfinite(v) and v > 0 for v in args.values()):
            out[name] = m.func(**args)
        else:
            out[name] = float("nan")
    return out
