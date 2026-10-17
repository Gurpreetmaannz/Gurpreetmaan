"""Shared workflow used by both the command line and the browser app."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point

from .hydro import Terrain, delineate, snap_to_flow, trace_longest_path
from .metrics import compute_metrics, polygon_of, stream_lines


@dataclass
class Result:
    rows: list = field(default_factory=list)
    polys: list = field(default_factory=list)
    lfps: list = field(default_factory=list)
    streams: list = field(default_factory=list)
    profiles: list = field(default_factory=list)  # (distance m, elevation m) per outlet
    errors: list = field(default_factory=list)  # (outlet id, message)


def run_outlets(
    terrain: Terrain,
    outlets,
    snap_distance: float = 100.0,
    stream_km2: float = 0.25,
    slope_method: str = "overall",
    tc_method: str = "bransby_williams",
    strict: bool = True,
    log=lambda msg: None,
) -> Result:
    """Delineate and measure each outlet (id, x, y). With strict=False, a failing
    outlet is recorded in Result.errors instead of stopping the run."""
    min_cells = stream_km2 * 1e6 / terrain.cell_area
    res = Result()
    for oid, x, y in outlets:
        try:
            r, c, on_stream = snap_to_flow(terrain, x, y, snap_distance, min_cells)
            sx, sy = terrain.transform @ (c + 0.5, r + 0.5)
            warn = []
            if not on_stream:
                warn.append(
                    f"no cell above the {stream_km2} km2 stream threshold within the snap radius"
                )
            catch = delineate(terrain, r, c)
            if catch.touches_edge:
                warn.append("catchment touches the DEM edge or no-data, so it may be truncated")
            lfp_rc = trace_longest_path(terrain, catch)
            poly = polygon_of(terrain, catch.mask)
            streams = stream_lines(terrain, catch.mask, min_cells)
            m = compute_metrics(terrain, catch, lfp_rc, poly, streams, slope_method, tc_method)
        except (ValueError, RuntimeError) as e:
            if strict:
                raise
            res.errors.append((oid, str(e)))
            continue
        prof = m.pop("_profile")
        m.update(
            id=oid,
            outlet_x=sx,
            outlet_y=sy,
            snap_distance_m=float(np.hypot(sx - x, sy - y)),
            warnings="; ".join(warn),
        )
        for w in warn:
            log(f"warning [{oid}]: {w}")
        res.rows.append(m)
        res.polys.append(poly)
        res.lfps.append(LineString([terrain.transform @ (cc + 0.5, rr + 0.5) for rr, cc in lfp_rc]))
        res.streams.append(streams)
        res.profiles.append(prof)
    return res


def slope_key(method: str) -> str:
    return f"lfp_slope_{method}_mm"


def write_outputs(terrain: Terrain, res: Result, out_dir, plot: bool = False) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = res.rows
    first = ["id", "area_km2", "lfp_km", slope_key(rows[0]["slope_method"]), "tc_method", "tc_hr", "tlag_hr"]
    keys = first + [k for k in rows[0] if k not in first]
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r.get(k) for k in keys} for r in rows)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2, default=float))

    gpkg = out / "catchments.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    crs = terrain.crs
    ids = [r["id"] for r in rows]
    gpd.GeoDataFrame(rows, geometry=res.polys, crs=crs).to_file(gpkg, layer="catchments", driver="GPKG")
    gpd.GeoDataFrame({"id": ids}, geometry=res.lfps, crs=crs).to_file(gpkg, layer="longest_flow_path", driver="GPKG")
    gpd.GeoDataFrame(
        {"id": ids}, geometry=[Point(r["outlet_x"], r["outlet_y"]) for r in rows], crs=crs
    ).to_file(gpkg, layer="outlets", driver="GPKG")
    sel = [(i, s) for i, s in zip(ids, res.streams) if s is not None]
    if sel:
        gpd.GeoDataFrame({"id": [i for i, _ in sel]}, geometry=[s for _, s in sel], crs=crs).to_file(
            gpkg, layer="streams", driver="GPKG"
        )
    if plot:
        from .plot import overview

        items = [
            (r["id"], p, l, s, pr, (r["outlet_x"], r["outlet_y"]))
            for r, p, l, s, pr in zip(rows, res.polys, res.lfps, res.streams, res.profiles)
        ]
        overview(terrain, items, out / "overview.png")
