"""Command line interface: DEM + outlet points in, catchments and metrics out."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point

from . import tc as tcmod
from .hydro import delineate, load_terrain, snap_to_flow, trace_longest_path
from .metrics import SLOPE_METHODS, compute_metrics, polygon_of, stream_lines


def read_outlets(args, crs) -> list[tuple[str, float, float]]:
    outlets: list[tuple[str, float, float]] = []
    for i, s in enumerate(args.outlet or [], start=1):
        x, y = (float(v) for v in s.split(","))
        outlets.append((f"outlet_{i}", x, y))
    if args.outlets:
        p = Path(args.outlets)
        if p.suffix.lower() == ".csv":
            with open(p, newline="") as f:
                for i, row in enumerate(csv.DictReader(f), start=1):
                    outlets.append((row.get("id") or f"outlet_{i}", float(row["x"]), float(row["y"])))
        else:
            gdf = gpd.read_file(p)
            if gdf.crs is not None and crs is not None and gdf.crs != crs:
                gdf = gdf.to_crs(crs)
            idcol = next((c for c in ("id", "name", "ID", "Name") if c in gdf.columns), None)
            for i, (_, r) in enumerate(gdf.iterrows(), start=1):
                g = r.geometry
                outlets.append((str(r[idcol]) if idcol else f"outlet_{i}", g.x, g.y))
    if not outlets:
        sys.exit("No outlets given. Use --outlet X,Y (repeatable) or --outlets FILE.")
    return outlets


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="catchtool",
        description="Delineate catchments from a DEM and compute DEM-only descriptors and Tc.",
    )
    ap.add_argument("dem", help="DEM GeoTIFF in a projected CRS (metres)")
    ap.add_argument("--outlet", action="append", metavar="X,Y", help="outlet in DEM CRS, repeatable")
    ap.add_argument("--outlets", metavar="FILE", help="CSV (id,x,y) or vector point file")
    ap.add_argument("-o", "--out", default="catchtool_out", help="output directory")
    ap.add_argument("--snap-distance", type=float, default=100.0, help="outlet snap radius in metres")
    ap.add_argument(
        "--stream-threshold-km2",
        type=float,
        default=0.25,
        help="contributing area that defines a stream, used for snapping and drainage density",
    )
    ap.add_argument("--slope-method", choices=SLOPE_METHODS, default="overall")
    ap.add_argument("--tc-method", choices=sorted(tcmod.METHODS), default="bransby_williams")
    ap.add_argument("--plot", action="store_true", help="also write an overview PNG")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Conditioning DEM and computing flow directions...", flush=True)
    terrain = load_terrain(args.dem)
    min_cells = args.stream_threshold_km2 * 1e6 / terrain.cell_area
    outlets = read_outlets(args, terrain.crs)

    rows, polys, lfps, strs = [], [], [], []
    plot_items = []
    for oid, x, y in outlets:
        r, c, on_stream = snap_to_flow(terrain, x, y, args.snap_distance, min_cells)
        sx, sy = terrain.transform @ (c + 0.5, r + 0.5)
        warn = []
        if not on_stream:
            warn.append(f"no cell above the {args.stream_threshold_km2} km2 stream threshold within the snap radius")
        catch = delineate(terrain, r, c)
        if catch.touches_edge:
            warn.append("catchment touches the DEM edge or no-data, so it may be truncated")
        lfp_rc = trace_longest_path(terrain, catch)
        poly = polygon_of(terrain, catch.mask)
        streams = stream_lines(terrain, catch.mask, min_cells)
        m = compute_metrics(terrain, catch, lfp_rc, poly, streams, args.slope_method, args.tc_method)
        prof = m.pop("_profile")
        snap_dist = float(np.hypot(sx - x, sy - y))
        m.update(id=oid, outlet_x=sx, outlet_y=sy, snap_distance_m=snap_dist, warnings="; ".join(warn))
        rows.append(m)
        polys.append(poly)
        pts = [terrain.transform @ (cc + 0.5, rr + 0.5) for rr, cc in lfp_rc]
        lfps.append(LineString(pts))
        strs.append(streams)
        plot_items.append((oid, poly, lfps[-1], streams, prof, (sx, sy)))
        for w in warn:
            print(f"  warning [{oid}]: {w}", file=sys.stderr)

    first = ["id", "area_km2", "lfp_km", "lfp_slope_" + {"overall": "overall", "1085": "1085", "equal_area": "equal_area"}[args.slope_method] + "_mm", "tc_method", "tc_hr", "tlag_hr"]
    keys = first + [k for k in rows[0] if k not in first]
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r.get(k) for k in keys} for r in rows)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2, default=float))

    crs = terrain.crs
    gdf = gpd.GeoDataFrame(rows, geometry=polys, crs=crs)
    gdf.to_file(out / "catchments.gpkg", layer="catchments", driver="GPKG")
    gpd.GeoDataFrame({"id": [r["id"] for r in rows]}, geometry=lfps, crs=crs).to_file(
        out / "catchments.gpkg", layer="longest_flow_path", driver="GPKG"
    )
    gpd.GeoDataFrame(
        {"id": [r["id"] for r in rows]},
        geometry=[Point(r["outlet_x"], r["outlet_y"]) for r in rows],
        crs=crs,
    ).to_file(out / "catchments.gpkg", layer="outlets", driver="GPKG")
    sel = [(r["id"], s) for r, s in zip(rows, strs) if s is not None]
    if sel:
        gpd.GeoDataFrame({"id": [i for i, _ in sel]}, geometry=[s for _, s in sel], crs=crs).to_file(
            out / "catchments.gpkg", layer="streams", driver="GPKG"
        )

    if args.plot:
        from .plot import overview

        overview(terrain, plot_items, out / "overview.png")

    print_summary(rows)
    print(f"\nWrote {out}/metrics.csv, metrics.json, catchments.gpkg" + (", overview.png" if args.plot else ""))
    return 0


def print_summary(rows) -> None:
    print("\nCatchment summary (Tc in hours, slope in m/m)")
    hdr = f"{'id':<14}{'A km2':>8}{'LFP km':>8}{'S':>8}{'Tc':>7}{'Tlag':>7}  method"
    print(hdr)
    for r in rows:
        s = r["lfp_slope_" + {"overall": "overall", "1085": "1085", "equal_area": "equal_area"}[r["slope_method"]] + "_mm"]
        print(f"{r['id']:<14}{r['area_km2']:>8.2f}{r['lfp_km']:>8.2f}{s:>8.3f}{r['tc_hr']:>7.2f}{r['tlag_hr']:>7.2f}  {r['tc_method']}")
    print("\nTc by method (hr):")
    names = sorted(tcmod.METHODS)
    print(f"{'id':<14}" + "".join(f"{n[:12]:>14}" for n in names))
    for r in rows:
        print(f"{r['id']:<14}" + "".join(f"{r['tc_' + n + '_hr']:>14.2f}" for n in names))


if __name__ == "__main__":
    raise SystemExit(main())
