"""DEM conditioning, D8 routing, outlet snapping and catchment delineation.

Routing is implemented here (numba) rather than via a hydrology package:
priority-flood depression filling with epsilon increments (Barnes et al. 2014),
D8 steepest descent, flow accumulation, and per-outlet catchment and flow length.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
import rasterio
from numba import njit

_DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
_DC = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)


@njit(cache=True)
def _priority_flood(z, valid):
    """Fill depressions and give flats a gradient towards their spill point.

    Cells on the grid edge or next to no-data are treated as drainage outlets.
    Every other cell ends up with a strictly lower neighbour, so D8 routing
    cannot get stuck in a pit or on a flat.
    """
    nr, nc = z.shape
    out = z.copy()
    closed = np.zeros((nr, nc), np.bool_)
    heap = [(np.float64(0.0), np.int64(0))]
    heap.pop()
    first = True
    for r in range(nr):
        for c in range(nc):
            if not valid[r, c]:
                continue
            seed = r == 0 or c == 0 or r == nr - 1 or c == nc - 1
            if not seed:
                for k in range(8):
                    r2, c2 = r + _DR[k], c + _DC[k]
                    if not valid[r2, c2]:
                        seed = True
                        break
            if seed:
                closed[r, c] = True
                if first:
                    heap = [(out[r, c], np.int64(r * nc + c))]
                    first = False
                else:
                    heappush(heap, (out[r, c], np.int64(r * nc + c)))
    while len(heap) > 0:
        zc, ic = heappop(heap)
        r, c = ic // nc, ic % nc
        for k in range(8):
            r2, c2 = r + _DR[k], c + _DC[k]
            if r2 < 0 or r2 >= nr or c2 < 0 or c2 >= nc:
                continue
            if closed[r2, c2] or not valid[r2, c2]:
                continue
            closed[r2, c2] = True
            zn = out[r2, c2]
            if zn <= zc:
                zn = np.nextafter(zc, np.inf)
                out[r2, c2] = zn
            heappush(heap, (zn, np.int64(r2 * nc + c2)))
    return out


@njit(cache=True)
def _d8(z, valid, dx, dy):
    """Steepest-descent downstream cell (flat index, -1 for none) and step length."""
    nr, nc = z.shape
    ds = np.full(nr * nc, -1, np.int64)
    step = np.zeros(nr * nc)
    for r in range(nr):
        for c in range(nc):
            if not valid[r, c]:
                continue
            best = 0.0
            for k in range(8):
                r2, c2 = r + _DR[k], c + _DC[k]
                if r2 < 0 or r2 >= nr or c2 < 0 or c2 >= nc or not valid[r2, c2]:
                    continue
                dist = np.sqrt((_DR[k] * dy) ** 2 + (_DC[k] * dx) ** 2)
                slope = (z[r, c] - z[r2, c2]) / dist
                if slope > best:
                    best = slope
                    ds[r * nc + c] = r2 * nc + c2
                    step[r * nc + c] = dist
    return ds, step


@njit(cache=True)
def _accumulate(order_asc, ds):
    acc = np.ones(ds.shape[0])
    for k in range(order_asc.shape[0] - 1, -1, -1):  # highest first
        c = order_asc[k]
        if ds[c] >= 0:
            acc[ds[c]] += acc[c]
    return acc


@njit(cache=True)
def _catchment_and_length(order_asc, ds, step, out_idx, n):
    """One pass from low to high: a cell is in the catchment if its downstream cell is."""
    d = np.full(n, np.nan)
    d[out_idx] = 0.0
    for k in range(order_asc.shape[0]):
        c = order_asc[k]
        if c == out_idx:
            continue
        nxt = ds[c]
        if nxt >= 0 and not np.isnan(d[nxt]):
            d[c] = d[nxt] + step[c]
    return d


@dataclass
class Terrain:
    """Everything derived from the DEM that is shared by all outlets."""

    dem: np.ndarray  # original elevations, NaN where no data
    cond: np.ndarray  # depressions filled, flats resolved
    ds: np.ndarray  # flat index of downstream cell, -1 if none
    step: np.ndarray  # distance to the downstream cell (m)
    order_asc: np.ndarray  # valid cells, low to high on the conditioned surface
    acc: np.ndarray  # contributing cells (including the cell itself), 2-D
    transform: rasterio.Affine
    crs: object
    dx: float
    dy: float

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy


def load_terrain(dem_path: str) -> Terrain:
    with rasterio.open(dem_path) as src:
        if src.crs is None:
            raise ValueError("DEM has no CRS. Assign a projected CRS (units of metres) first.")
        if src.crs.is_geographic:
            raise ValueError(
                f"DEM CRS is geographic ({src.crs}). Reproject to a projected CRS in metres "
                "(for example with `gdalwarp -t_srs EPSG:<utm>`), because lengths and slopes "
                "are meaningless in degrees."
            )
        unit = (src.crs.linear_units or "").lower()
        if unit not in ("metre", "meter", "metres", "meters", "m"):
            raise ValueError(f"DEM horizontal units are '{unit}'; only metres are supported.")
        t = src.transform
        if t.b != 0 or t.d != 0 or t.e > 0:
            raise ValueError("DEM must be north-up and unrotated.")
        dem = src.read(1, masked=True).astype("float64").filled(np.nan)
        dx, dy = abs(t.a), abs(t.e)
        crs = src.crs

    valid = ~np.isnan(dem)
    if not valid.any():
        raise ValueError("DEM contains no valid cells.")
    cond = _priority_flood(np.where(valid, dem, 0.0), valid)
    ds, step = _d8(cond, valid, dx, dy)
    flat_valid = np.flatnonzero(valid.ravel())
    order_asc = flat_valid[np.argsort(cond.ravel()[flat_valid], kind="stable")]
    acc = _accumulate(order_asc, ds).reshape(dem.shape)
    acc[~valid] = np.nan
    return Terrain(dem, np.where(valid, cond, np.nan), ds, step, order_asc, acc, t, crs, dx, dy)


def snap_to_flow(
    terrain: Terrain, x: float, y: float, radius_m: float, min_cells: float
) -> tuple[int, int, bool]:
    """Move (x, y) to the nearest stream cell within radius_m.

    A stream cell has accumulation >= min_cells. Nearest (not highest) is used so
    the outlet stays at the crossing rather than drifting downstream and picking
    up extra catchment. Ties go to the higher accumulation. If no stream cell is
    in range, the highest accumulation cell in range is used and on_stream is False.
    Returns (row, col, on_stream).
    """
    c0, r0 = ~terrain.transform @ (x, y)
    r0, c0 = int(np.floor(r0)), int(np.floor(c0))
    nr, nc = terrain.acc.shape
    if not (0 <= r0 < nr and 0 <= c0 < nc):
        raise ValueError(f"Outlet ({x}, {y}) lies outside the DEM.")
    rr = int(np.ceil(radius_m / terrain.dy))
    rc = int(np.ceil(radius_m / terrain.dx))
    rows = np.arange(max(0, r0 - rr), min(nr, r0 + rr + 1))
    cols = np.arange(max(0, c0 - rc), min(nc, c0 + rc + 1))
    R, C = np.meshgrid(rows, cols, indexing="ij")
    dist = np.hypot((R - r0) * terrain.dy, (C - c0) * terrain.dx)
    acc = terrain.acc[np.ix_(rows, cols)]
    valid = (dist <= radius_m) & ~np.isnan(acc)
    if not valid.any():
        raise ValueError(f"No valid DEM cell within {radius_m} m of outlet ({x}, {y}).")
    stream = valid & (acc >= min_cells)
    on_stream = bool(stream.any())
    if on_stream:
        cand = np.argwhere(stream)
        key = np.lexsort((-acc[stream], dist[stream]))  # nearest first, then highest acc
    else:
        cand = np.argwhere(valid)
        key = np.lexsort((dist[valid], -acc[valid]))  # highest acc first, then nearest
    i, j = cand[key[0]]
    return int(rows[i]), int(cols[j]), on_stream


@dataclass
class Catchment:
    mask: np.ndarray  # boolean raster
    outlet_rc: tuple[int, int]
    flow_len: np.ndarray  # flow distance to the outlet (m), NaN outside
    touches_edge: bool


def delineate(terrain: Terrain, row: int, col: int) -> Catchment:
    """Catchment draining to cell (row, col), plus flow distance to that cell."""
    nr, nc = terrain.dem.shape
    d = _catchment_and_length(terrain.order_asc, terrain.ds, terrain.step, row * nc + col, nr * nc)
    d = d.reshape(nr, nc)
    mask = ~np.isnan(d)
    edge = mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()
    nodata = np.isnan(terrain.dem)
    if nodata.any():  # also flag catchments that border no-data cells
        pad = np.pad(mask, 1)
        ring = pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
        edge = edge or bool((ring & nodata).any())
    return Catchment(mask=mask, outlet_rc=(row, col), flow_len=d, touches_edge=bool(edge))


def trace_longest_path(terrain: Terrain, catch: Catchment) -> np.ndarray:
    """Cells (row, col) of the longest flow path, from headwater to outlet."""
    nc = catch.mask.shape[1]
    out = catch.outlet_rc[0] * nc + catch.outlet_rc[1]
    path = [int(np.nanargmax(catch.flow_len))]
    while path[-1] != out:
        path.append(int(terrain.ds[path[-1]]))
    path = np.array(path)
    return np.column_stack([path // nc, path % nc])
