"""Local browser app: `catchtool-gui` opens an interface on this computer.

Nothing leaves the machine. The server listens on 127.0.0.1 only, uses just the
standard library for HTTP, and reuses the same engine as the command line.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import shutil
import tempfile
import threading
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import tc as tcmod
from .hydro import Terrain, load_terrain
from .metrics import SLOPE_METHODS
from .pipeline import run_outlets, write_outputs

STATIC = Path(__file__).parent / "static"
MAX_PREVIEW_PX = 1600


class State:
    """One DEM and its last results; the app is single-user."""

    def __init__(self):
        self.lock = threading.Lock()
        self.work = Path(tempfile.mkdtemp(prefix="catchtool_"))
        self.terrain: Terrain | None = None
        self.dem_name = ""
        self.preview_png = b""
        self.view: dict = {}
        self.result_dir: Path | None = None

    def close(self):
        shutil.rmtree(self.work, ignore_errors=True)


STATE = State()


def _clean(v):
    """Make values JSON safe: NaN and inf become null, numpy scalars become Python."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _make_preview(t: Terrain) -> tuple[bytes, dict]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.colors import LightSource, LinearSegmentedColormap
    from matplotlib.image import imsave

    h, w = t.dem.shape
    f = max(1, math.ceil(max(h, w) / MAX_PREVIEW_PX))
    z = t.dem[::f, ::f]
    ok = ~np.isnan(z)
    lo, hi = np.nanmin(z), np.nanmax(z)
    zf = np.where(ok, z, lo)
    ls = LightSource(azdeg=315, altdeg=40)
    cmap = LinearSegmentedColormap.from_list(
        "elev", ["#2f5d50", "#6b8f5e", "#c9c58a", "#b79a76", "#f2efe9"]
    )
    # Exaggerate relief so gentle terrain still shows form.
    span = max(hi - lo, 1e-6)
    ex = float(np.clip(0.15 * max(z.shape) * t.dx * f / span / 8, 1, 6))
    rgb = ls.shade(zf, cmap=cmap, blend_mode="soft", vert_exag=ex, dx=t.dx * f, dy=t.dy * f,
                   vmin=lo, vmax=hi)
    rgb[..., 3] = np.where(ok, 1.0, 0.0)
    buf = io.BytesIO()
    imsave(buf, rgb, format="png")
    ph, pw = z.shape
    x0, y1 = t.transform.c, t.transform.f
    view = {
        "img_w": pw, "img_h": ph, "factor": f,
        "x0": x0, "y1": y1,
        "x1": x0 + pw * f * t.dx, "y0": y1 - ph * f * t.dy,
        "elev_min": float(lo), "elev_max": float(hi),
        "cell": t.dx, "cols": w, "rows": h,
        "crs": str(t.crs),
    }
    return buf.getvalue(), view


def _install(path: Path, display_name: str) -> dict:
    t = load_terrain(str(path))
    png, view = _make_preview(t)
    STATE.terrain, STATE.dem_name, STATE.preview_png, STATE.view = t, display_name, png, view
    STATE.result_dir = None
    view = dict(view, name=display_name)
    return view


def _demo_dem(path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin

    rows = cols = 201
    cell, cross, along, half, divide = 10.0, 0.25, 0.03, 60, 40
    r = np.arange(rows)[:, None] * np.ones((1, cols))
    c = np.ones((rows, 1)) * np.arange(cols)[None, :]
    off = np.abs(c - (cols - 1) // 2)
    zc = np.where(off <= half, off, 2 * half - off) * cell * cross
    za = np.where(r >= divide, (200 - r) * cell * along,
                  (200 - divide) * cell * along - (divide - r) * cell * 0.1)
    rng = np.random.default_rng(3)
    z = 100 + za + zc + rng.normal(0, 0.15, (rows, cols))
    with rasterio.open(path, "w", driver="GTiff", height=rows, width=cols, count=1, dtype="float32",
                       crs="EPSG:2193", transform=from_origin(1576000.0, 5181000.0, cell, cell),
                       nodata=-9999) as dst:
        dst.write(z.astype("float32"), 1)


def _ring(coords, ndigits=1):
    return [[round(x, ndigits), round(y, ndigits)] for x, y in coords]


def _polygon_rings(geom, tol):
    geom = geom.simplify(tol, preserve_topology=True)
    polys = getattr(geom, "geoms", [geom])
    return [[_ring(p.exterior.coords)] + [_ring(i.coords) for i in p.interiors] for p in polys]


def _lines(geom):
    if geom is None:
        return []
    return [_ring(g.coords) for g in getattr(geom, "geoms", [geom])]


def run_request(body: dict) -> dict:
    t = STATE.terrain
    if t is None:
        raise ValueError("Load a DEM first.")
    outlets = [(str(o["id"]), float(o["x"]), float(o["y"])) for o in body.get("outlets", [])]
    if not outlets:
        raise ValueError("Click the map to place at least one outlet.")
    slope = body.get("slope_method", "overall")
    tcm = body.get("tc_method", "bransby_williams")
    if slope not in SLOPE_METHODS or tcm not in tcmod.METHODS:
        raise ValueError("Unknown slope or Tc method.")
    res = run_outlets(
        t, outlets,
        snap_distance=float(body.get("snap_distance", 100)),
        stream_km2=float(body.get("stream_km2", 0.25)),
        slope_method=slope, tc_method=tcm, strict=False,
    )
    if res.rows:
        out = STATE.work / "results"
        if out.exists():
            shutil.rmtree(out)
        write_outputs(t, res, out, plot=True)
        STATE.result_dir = out
    tol = t.dx * 0.75
    feats = []
    for i, row in enumerate(res.rows):
        d, z = res.profiles[i]
        step = max(1, len(d) // 300)
        feats.append({
            "id": row["id"],
            "polygon": _polygon_rings(res.polys[i], tol),
            "lfp": _ring(res.lfps[i].coords),
            "streams": _lines(res.streams[i]),
            "outlet": [row["outlet_x"], row["outlet_y"]],
            "profile": {"d": [round(float(v), 1) for v in d[::step]],
                        "z": [round(float(v), 2) for v in z[::step]]},
        })
    return _clean({
        "rows": res.rows,
        "features": feats,
        "errors": [{"id": i, "message": m} for i, m in res.errors],
    })


class Handler(BaseHTTPRequestHandler):
    server_version = "catchtool"

    def log_message(self, *a):  # keep the terminal quiet
        pass

    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, allow_nan=False).encode())

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        elif path == "/api/config":
            self._json({
                "slope_methods": list(SLOPE_METHODS),
                "tc_methods": [
                    {"id": k, "doc": (m.func.__doc__ or "").strip().split("\n")[0]}
                    for k, m in tcmod.METHODS.items()
                ],
                "loaded": dict(STATE.view, name=STATE.dem_name) if STATE.terrain else None,
            })
        elif path == "/api/preview.png":
            if not STATE.preview_png:
                self._send(404, b"no dem", "text/plain")
            else:
                self._send(200, STATE.preview_png, "image/png")
        elif path == "/api/download.zip":
            if not STATE.result_dir:
                self._send(404, b"no results yet", "text/plain")
                return
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for f in sorted(STATE.result_dir.iterdir()):
                    z.write(f, f.name)
            self._send(200, buf.getvalue(), "application/zip",
                       {"Content-Disposition": 'attachment; filename="catchtool_results.zip"'})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        url = urlparse(self.path)
        try:
            with STATE.lock:
                if url.path == "/api/upload":
                    name = re.sub(r"[^\w.\- ]", "_", parse_qs(url.query).get("name", ["dem.tif"])[0])
                    dest = STATE.work / "dem.tif"
                    n = int(self.headers.get("Content-Length", 0))
                    with open(dest, "wb") as f:
                        left = n
                        while left > 0:
                            chunk = self.rfile.read(min(1 << 20, left))
                            if not chunk:
                                break
                            f.write(chunk)
                            left -= len(chunk)
                    self._json(_install(dest, name))
                elif url.path == "/api/demo":
                    dest = STATE.work / "demo.tif"
                    _demo_dem(dest)
                    self._json(_install(dest, "demo valley (synthetic)"))
                elif url.path == "/api/run":
                    n = int(self.headers.get("Content-Length", 0))
                    self._json(run_request(json.loads(self.rfile.read(n) or b"{}")))
                else:
                    self._send(404, b"not found", "text/plain")
        except (ValueError, RuntimeError) as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:  # unreadable file, etc.
            msg = str(e)
            if "not recognized" in msg or "RasterioIOError" in type(e).__name__:
                msg = "That file could not be read as a GeoTIFF DEM."
            self._json({"error": msg}, 400)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="catchtool-gui", description="Browser interface for catchtool.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"Could not find a free port starting at {args.port}.")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"catchtool is running at {url}\nPress Ctrl+C in this window to stop it.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        STATE.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
