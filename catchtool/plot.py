"""Overview figure: hillshade, catchments, streams, longest flow paths, outlets."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

from .hydro import Terrain


def overview(terrain: Terrain, items, path) -> None:
    t = terrain.transform
    nr, nc = terrain.dem.shape
    extent = (t.c, t.c + nc * t.a, t.f + nr * t.e, t.f)
    dem = np.ma.masked_invalid(terrain.dem)
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(dem.filled(np.nanmean(terrain.dem)), dx=terrain.dx, dy=terrain.dy)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(shade, cmap="gray", extent=extent, vmin=0, vmax=1)
    ax.imshow(dem, cmap="terrain", extent=extent, alpha=0.45)
    colours = plt.cm.tab10.colors
    for i, (oid, poly, lfp, streams, _, (sx, sy)) in enumerate(items):
        col = colours[i % len(colours)]
        for g in getattr(poly, "geoms", [poly]):
            ax.plot(*g.exterior.xy, color=col, lw=1.8)
        if streams is not None:
            for g in getattr(streams, "geoms", [streams]):
                ax.plot(*g.xy, color="#1f78b4", lw=0.7)
        ax.plot(*lfp.xy, color="k", lw=1.4, ls="--")
        ax.plot(sx, sy, "o", mfc="white", mec="k", ms=7)
        ax.annotate(oid, (sx, sy), xytext=(6, 6), textcoords="offset points", fontsize=9, weight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    ax.set_title("Catchments (colour), streams (blue), longest flow path (dashed)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
