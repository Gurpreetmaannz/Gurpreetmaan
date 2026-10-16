import math

import numpy as np
import pytest
import rasterio

from catchtool import tc
from catchtool.hydro import delineate, load_terrain, snap_to_flow, trace_longest_path
from catchtool.metrics import compute_metrics, horn_slope, polygon_of, profile_slope, stream_lines

from synth import CELL, inclined_plane, valley, write_dem, xy


def run(terrain, x, y, radius=30.0, min_cells=100):
    r, c, on_stream = snap_to_flow(terrain, x, y, radius, min_cells)
    catch = delineate(terrain, r, c)
    lfp = trace_longest_path(terrain, catch)
    poly = polygon_of(terrain, catch.mask)
    m = compute_metrics(terrain, catch, lfp, poly, stream_lines(terrain, catch.mask, min_cells))
    return catch, lfp, poly, m


# ---------- time of concentration: hand-calculated values ----------

def test_tc_formulae_against_hand_calculation():
    L, S, A, H = 3.2, 0.124, 4.8, 3200 * 0.124
    assert tc.kirpich(L_km=L, S=S) == pytest.approx(0.0195 * 3200**0.77 * S**-0.385)
    assert tc.kirpich(L_km=0.49, S=0.05) == pytest.approx(7.28, abs=0.02)  # worked by hand
    assert tc.bransby_williams(L_km=L, A_km2=A, S=S) == pytest.approx(60.7, abs=0.2)
    assert tc.pilgrim_mcdermott(A_km2=1.0) == pytest.approx(45.6)
    # California culverts form is a metric rewrite of Kirpich's: same order of magnitude
    assert tc.california_culverts(L_km=L, H_m=H) == pytest.approx(tc.kirpich(L_km=L, S=S), rel=0.05)
    # Giandotti: (4*sqrt(4)+1.5*2)/(0.8*sqrt(100)) h = 11/8 h
    assert tc.giandotti(L_km=2, A_km2=4, Hmean_m=100) == pytest.approx(60 * 11 / 8)


def test_compute_all_skips_invalid_inputs():
    out = tc.compute_all(L_km=1.0, S=0.0, H_m=0.0, A_km2=1.0, Hmean_m=10.0)
    assert math.isnan(out["kirpich"]) and math.isnan(out["california_culverts"])
    assert not math.isnan(out["pilgrim_mcdermott"])


# ---------- slope of a profile ----------

def test_profile_slope_methods_on_known_shapes():
    L, a = 1000.0, 1e-4
    d = np.linspace(0, L, 2001)
    lin = 0.02 * d
    for m in ("overall", "1085", "equal_area"):
        assert profile_slope(d, lin, m) == pytest.approx(0.02)
    para = a * d**2  # steep headwaters
    assert profile_slope(d, para, "overall") == pytest.approx(a * L)
    assert profile_slope(d, para, "1085") == pytest.approx(0.95 * a * L, rel=1e-3)
    assert profile_slope(d, para, "equal_area") == pytest.approx(2 * a * L / 3, rel=1e-3)


def test_horn_slope_on_plane():
    z = inclined_plane(rows=10, cols=10, slope=0.05)
    s = horn_slope(z, CELL, CELL)
    assert s[1:-1, 1:-1] == pytest.approx(0.05)


# ---------- delineation on terrain with known answers ----------

def test_inclined_plane(tmp_path):
    p = tmp_path / "plane.tif"
    write_dem(p, inclined_plane())
    t = load_terrain(str(p))
    catch, lfp, poly, m = run(t, *xy(49, 10), radius=5, min_cells=1e9)
    assert catch.mask.sum() == 50 and catch.mask[:, 10].all()
    assert m["area_km2"] == pytest.approx(50 * CELL**2 / 1e6)
    assert m["lfp_km"] == pytest.approx(0.49)
    assert m["lfp_drop_m"] == pytest.approx(24.5, abs=1e-3)
    for k in ("overall", "1085", "equal_area"):
        assert m[f"lfp_slope_{k}_mm"] == pytest.approx(0.05, rel=1e-3)
    assert m["tc_kirpich_hr"] * 60 == pytest.approx(7.28, abs=0.02)
    assert m["tlag_hr"] == pytest.approx(0.6 * m["tc_hr"])
    assert poly.area == pytest.approx(50 * CELL**2)


@pytest.fixture(scope="module")
def vterrain(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "valley.tif"
    write_dem(p, valley())
    return load_terrain(str(p))


def test_valley_catchment_geometry(vterrain):
    catch, lfp, poly, m = run(vterrain, *xy(80, 50))
    assert not catch.touches_edge
    assert m["touches_dem_edge"] is False
    # divide to outlet is 61 rows by 61 columns wide; divide cells may split
    assert 55 * 55 < catch.mask.sum() < 63 * 63
    assert m["area_km2"] == pytest.approx(catch.mask.sum() * CELL**2 / 1e6)
    # steepest descent crosses the V first (30 cells) then runs down the valley (60 cells)
    assert m["lfp_km"] == pytest.approx(0.9, abs=0.02)
    # path starts on the divide and ends at the outlet
    assert tuple(lfp[-1]) == catch.outlet_rc
    steps = np.abs(np.diff(lfp, axis=0))
    assert (steps.max(axis=1) <= 1).all()
    # flow length along the path rises monotonically upstream
    fl = catch.flow_len[lfp[:, 0], lfp[:, 1]]
    assert (np.diff(fl) < 0).all()
    # elevations along the conditioned path never rise going downstream
    z = vterrain.cond[lfp[:, 0], lfp[:, 1]]
    assert (np.diff(z) <= 0).all()
    assert m["lfp_drop_m"] > 0 and m["elev_max_m"] >= m["outlet_elev_m"]
    assert m["drainage_density_km_per_km2"] > 0


def test_nested_outlets_are_subsets(vterrain):
    up, *_ , m_up = run(vterrain, *xy(50, 50))
    dn, *_ , m_dn = run(vterrain, *xy(80, 50))
    assert up.mask.sum() < dn.mask.sum()
    assert (up.mask & ~dn.mask).sum() == 0


def test_snapping_stays_at_crossing_and_does_not_drift_downstream(vterrain):
    # an outlet exactly on the valley floor must not slide down to higher-accumulation cells
    r, c, on = snap_to_flow(vterrain, *xy(80, 50), radius_m=50, min_cells=100)
    assert (r, c, on) == (80, 50, True)


def test_snapping_moves_to_valley(vterrain):
    x, y = xy(80, 45)  # five cells off the valley floor
    r, c, on_stream = snap_to_flow(vterrain, x, y, radius_m=80, min_cells=100)
    assert c == 50 and on_stream
    # with a radius too small to reach the valley the flag is false
    _, _, on = snap_to_flow(vterrain, x, y, radius_m=10, min_cells=100)
    assert not on


def test_pit_in_valley_is_filled(tmp_path):
    z = valley()
    ref_p, pit_p = tmp_path / "a.tif", tmp_path / "b.tif"
    write_dem(ref_p, z)
    z2 = z.copy()
    z2[60, 50] -= 5.0  # an artefact sink on the valley floor
    write_dem(pit_p, z2)
    a = run(load_terrain(str(ref_p)), *xy(80, 50))
    b = run(load_terrain(str(pit_p)), *xy(80, 50))
    assert b[0].mask.sum() == a[0].mask.sum()
    assert b[3]["lfp_km"] == pytest.approx(a[3]["lfp_km"], abs=0.02)


def test_nodata_cells_do_not_break_and_flag_truncation(tmp_path):
    z = valley()
    z[40:45, 55:60] = np.nan  # a hole inside the catchment
    p = tmp_path / "hole.tif"
    write_dem(p, z)
    t = load_terrain(str(p))
    catch, *_ = run(t, *xy(80, 50))
    assert catch.touches_edge  # borders no-data, so truncation is flagged
    assert not catch.mask[40:45, 55:60].any()


def test_geographic_crs_is_rejected(tmp_path):
    p = tmp_path / "geo.tif"
    write_dem(p, inclined_plane(), cell=0.0001, x0=170.0, y0=-45.0, epsg=4326)
    with pytest.raises(ValueError, match="geographic"):
        load_terrain(str(p))
