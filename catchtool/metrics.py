"""Catchment descriptors that follow from the DEM alone."""

from __future__ import annotations

import math

import numpy as np
from rasterio.features import shapes
from shapely.geometry import LineString, shape
from shapely.ops import linemerge, unary_union

from . import tc as tcmod
from .hydro import Catchment, Terrain

SLOPE_METHODS = ("overall", "1085", "equal_area")


def horn_slope(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Horn (1981) 3x3 slope as rise/run (m/m). NaN where any neighbour is NaN."""
    z = np.pad(dem, 1, constant_values=np.nan)
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * dy)
    return np.hypot(dzdx, dzdy)


def polygon_of(terrain: Terrain, mask: np.ndarray):
    geoms = [
        shape(g)
        for g, v in shapes(mask.astype("uint8"), mask=mask, transform=terrain.transform)
        if v == 1
    ]
    return unary_union(geoms)


def stream_lines(terrain: Terrain, mask: np.ndarray, min_cells: float):
    """Streams as cell-centre lines: cells in mask with accumulation >= min_cells."""
    nr, nc = mask.shape
    ds = terrain.ds
    sel = np.flatnonzero((mask & (terrain.acc >= min_cells)).ravel())
    segs = []
    for c in sel:
        n = int(ds[c])
        if n < 0 or not mask.ravel()[n]:
            continue
        p0 = terrain.transform @ (c % nc + 0.5, c // nc + 0.5)
        p1 = terrain.transform @ (n % nc + 0.5, n // nc + 0.5)
        segs.append(LineString([p0, p1]))
    if not segs:
        return None
    merged = linemerge(unary_union(segs))
    return merged


def profile_slope(dist_m: np.ndarray, elev_m: np.ndarray, method: str) -> float:
    """Slope (m/m) of a flow path profile; dist_m increases away from the outlet."""
    L = dist_m[-1]
    if L <= 0:
        return float("nan")
    z = elev_m - elev_m[0]  # elevation above the outlet
    if method == "overall":
        return float(z[-1] / L)
    if method == "1085":
        z10, z85 = np.interp([0.10 * L, 0.85 * L], dist_m, z)
        return float((z85 - z10) / (0.75 * L))
    if method == "equal_area":
        # Line through the outlet enclosing the same area as the profile.
        area = np.sum(0.5 * (z[1:] + z[:-1]) * np.diff(dist_m))
        return float(2.0 * area / L**2)
    raise ValueError(f"Unknown slope method '{method}'. Choose from {SLOPE_METHODS}.")


def compute_metrics(
    terrain: Terrain,
    catch: Catchment,
    lfp_rc: np.ndarray,
    polygon,
    streams,
    slope_method: str = "overall",
    tc_method: str = "bransby_williams",
) -> dict:
    mask = catch.mask
    n = int(mask.sum())
    area_km2 = n * terrain.cell_area / 1e6
    elev = terrain.dem[mask]
    elev = elev[~np.isnan(elev)]
    slope = horn_slope(terrain.dem, terrain.dx, terrain.dy)[mask]
    slope = slope[~np.isnan(slope)]

    r_out, c_out = catch.outlet_rc
    z_out = float(terrain.cond[r_out, c_out])
    mean_elev = float(elev.mean())

    # Longest flow path, headwater to outlet.
    d = catch.flow_len[lfp_rc[:, 0], lfp_rc[:, 1]][::-1]  # outlet first
    z = terrain.cond[lfp_rc[:, 0], lfp_rc[:, 1]][::-1]
    L_m = float(d[-1])
    L_km = L_m / 1000.0
    drop = float(z[-1] - z[0])
    S = profile_slope(d, z, slope_method)

    # Flow length from the outlet to the cell nearest the catchment centroid.
    cen = polygon.centroid
    cr, cc = ~terrain.transform @ (cen.x, cen.y)
    rows, cols = np.nonzero(mask)
    k = np.argmin((rows - cr) ** 2 + (cols - cc) ** 2)
    Lc_km = float(catch.flow_len[rows[k], cols[k]]) / 1000.0

    perim_km = polygon.length / 1000.0
    out = {
        "area_km2": area_km2,
        "perimeter_km": perim_km,
        "elev_min_m": float(elev.min()),
        "elev_max_m": float(elev.max()),
        "elev_mean_m": mean_elev,
        "outlet_elev_m": z_out,
        "relief_m": float(elev.max() - z_out),
        "mean_slope_mm": float(slope.mean()),
        "lfp_km": L_km,
        "lfp_drop_m": drop,
        "lfp_slope_overall_mm": profile_slope(d, z, "overall"),
        "lfp_slope_1085_mm": profile_slope(d, z, "1085"),
        "lfp_slope_equal_area_mm": profile_slope(d, z, "equal_area"),
        "lc_km": Lc_km,
        "form_factor": area_km2 / L_km**2 if L_km > 0 else float("nan"),
        "elongation_ratio": (2 * math.sqrt(area_km2 / math.pi)) / L_km if L_km > 0 else float("nan"),
        "compactness_gravelius": 0.2821 * perim_km / math.sqrt(area_km2),
        "circularity_ratio": 4 * math.pi * area_km2 / perim_km**2,
        "touches_dem_edge": catch.touches_edge,
        "slope_method": slope_method,
    }

    if streams is not None:
        out["stream_length_km"] = streams.length / 1000.0
        out["drainage_density_km_per_km2"] = streams.length / 1000.0 / area_km2
    else:
        out["stream_length_km"] = 0.0
        out["drainage_density_km_per_km2"] = 0.0

    tcs = tcmod.compute_all(L_km=L_km, S=S, H_m=drop, A_km2=area_km2, Hmean_m=mean_elev - z_out)
    for name, minutes in tcs.items():
        out[f"tc_{name}_hr"] = minutes / 60.0
    sel = tcs.get(tc_method, float("nan"))
    out["tc_method"] = tc_method
    out["tc_hr"] = sel / 60.0
    out["tlag_hr"] = sel / 60.0 * tcmod.TLAG_FACTOR
    out["_profile"] = (d, z)  # for plotting; stripped before writing tables
    return out
