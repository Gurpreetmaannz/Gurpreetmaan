"""Synthetic DEMs with known answers."""
import numpy as np
import rasterio
from rasterio.transform import from_origin

X0, Y0, CELL = 500000.0, 5400000.0, 10.0


def write_dem(path, z, cell=CELL, x0=X0, y0=Y0, epsg=32760, nodata=-9999.0):
    h, w = z.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs=f"EPSG:{epsg}", transform=from_origin(x0, y0, cell, cell), nodata=nodata,
    ) as dst:
        dst.write(np.where(np.isnan(z), nodata, z).astype("float32"), 1)


def xy(row, col, cell=CELL, x0=X0, y0=Y0):
    """Cell-centre coordinates."""
    return x0 + (col + 0.5) * cell, y0 - (row + 0.5) * cell


def inclined_plane(rows=50, cols=20, slope=0.05, cell=CELL):
    """Plane falling to the south: every column is its own 1D catchment."""
    r = np.arange(rows)[:, None] * np.ones((1, cols))
    return (rows - 1 - r) * cell * slope + 100.0


def valley(rows=101, cols=101, cell=CELL, cross=0.25, along=0.03, half_width=30, divide_row=20):
    """A V-shaped valley draining south with its own divides on three sides.

    Cells with |col - mid| <= half_width and row >= divide_row drain into the valley
    and out through the bottom-centre. Beyond those divides the ground falls away to
    the grid edge. Valley catchment for an outlet at (row, mid) is therefore a
    rectangle of (row - divide_row + 1) x (2 * half_width + 1) cells, give or take
    the divide cells that split between neighbours.
    """
    r = np.arange(rows)[:, None] * np.ones((1, cols))
    c = np.ones((rows, 1)) * np.arange(cols)[None, :]
    mid = (cols - 1) // 2
    off = np.abs(c - mid)
    zc = np.where(off <= half_width, off, 2 * half_width - off) * cell * cross
    za = np.where(r >= divide_row, (100 - r) * cell * along,
                  (100 - divide_row) * cell * along - (divide_row - r) * cell * 0.1)
    return 100.0 + za + zc
