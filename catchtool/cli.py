"""Command line interface: DEM + outlet points in, catchments and metrics out."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import geopandas as gpd
from . import tc as tcmod
from .hydro import load_terrain
from .metrics import SLOPE_METHODS
from .pipeline import run_outlets, slope_key, write_outputs


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
    print("Conditioning DEM and computing flow directions...", flush=True)
    terrain = load_terrain(args.dem)
    outlets = read_outlets(args, terrain.crs)
    res = run_outlets(
        terrain,
        outlets,
        args.snap_distance,
        args.stream_threshold_km2,
        args.slope_method,
        args.tc_method,
        strict=True,
        log=lambda m: print("  " + m, file=sys.stderr),
    )
    write_outputs(terrain, res, out, plot=args.plot)
    print_summary(res.rows)
    print(f"\nWrote {out}/metrics.csv, metrics.json, catchments.gpkg" + (", overview.png" if args.plot else ""))
    return 0


def print_summary(rows) -> None:
    print("\nCatchment summary (Tc in hours, slope in m/m)")
    hdr = f"{'id':<14}{'A km2':>8}{'LFP km':>8}{'S':>8}{'Tc':>7}{'Tlag':>7}  method"
    print(hdr)
    for r in rows:
        s = r[slope_key(r["slope_method"])]
        print(f"{r['id']:<14}{r['area_km2']:>8.2f}{r['lfp_km']:>8.2f}{s:>8.3f}{r['tc_hr']:>7.2f}{r['tlag_hr']:>7.2f}  {r['tc_method']}")
    print("\nTc by method (hr):")
    names = sorted(tcmod.METHODS)
    print(f"{'id':<14}" + "".join(f"{n[:12]:>14}" for n in names))
    for r in rows:
        print(f"{r['id']:<14}" + "".join(f"{r['tc_' + n + '_hr']:>14.2f}" for n in names))


if __name__ == "__main__":
    raise SystemExit(main())
