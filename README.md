# catchtool

Delineate catchments from a DEM and compute every descriptor that follows from the DEM alone,
including time of concentration (Tc).

## Easiest: the browser app

1. Install Miniconda once (https://docs.conda.io/en/latest/miniconda.html).
2. **Windows:** double-click `Start catchtool.bat`. **Mac or Linux:** run `./start_catchtool.sh`.
   The first launch installs what it needs (a few minutes). Later launches take seconds.
3. Your browser opens at http://127.0.0.1:8765. Drop in a DEM, click where each culvert is, press
   **Delineate catchments**, then download the results.

It runs only on your computer. Nothing is uploaded anywhere. Close the black terminal window to stop it.
Already installed? `catchtool-gui` does the same thing.

## Command line (batch work)

```bash
pip install -e ".[test]"
catchtool dem.tif --outlets culverts.csv -o out --plot
catchtool dem.tif --outlet 1576800,5180200 --outlet 1577400,5180900 -o out
pytest
```

Inputs
- DEM GeoTIFF in a **projected CRS in metres** (geographic CRS is rejected, not guessed). North-up. Elevations in metres.
- Outlets: `--outlet X,Y` (repeatable, DEM CRS) or `--outlets FILE`, either CSV (`id,x,y`) or any point vector file (reprojected to the DEM CRS).

Options: `--snap-distance` (m, default 100), `--stream-threshold-km2` (default 0.25), `--slope-method overall|1085|equal_area` (default `overall`), `--tc-method` (default `bransby_williams`).

Outputs (in `-o`): `metrics.csv`, `metrics.json`, `catchments.gpkg` (layers: `catchments`, `longest_flow_path`, `outlets`, `streams`), optional `overview.png`.

## What is computed

| Group | Fields |
|---|---|
| Size and shape | area, perimeter, form factor, elongation ratio, Gravelius compactness, circularity ratio |
| Relief | min, max, mean elevation, outlet elevation, relief, mean slope (Horn 3x3) |
| Flow path | longest flow path (LFP) length, elevation drop, slope (overall, 10-85, equal-area), Lc (outlet to centroid along the flow path) |
| Channels | stream length and drainage density above the stream threshold |
| Tc (hours) | Kirpich, California Culverts, Bransby-Williams, Giandotti, Pilgrim-McDermott, plus the selected method and `tlag_hr = 0.6 Tc` |

Not computed, because they need extra data: curve number (land cover and soil), roughness-based Tc
(NRCS velocity, Kerby-Hathaway, kinematic wave), rainfall and runoff. Tlag uses the NRCS rule of thumb
0.6 Tc; if you later derive lag from CN and slope, replace it.

## Method notes

- Depressions are filled by priority-flood with epsilon increments, so flats drain. Routing is D8 steepest descent. Everything is in `catchtool/hydro.py` (numba), about 6 s for a 9-million-cell DEM.
- Outlets snap to the **nearest** cell above the stream threshold within the snap radius, so the outlet stays at the crossing rather than drifting downstream. If none is in range, the highest-accumulation cell is used and a warning is printed.
- Catchments that touch the DEM edge or no-data are flagged (`touches_dem_edge`), since they may be truncated.
- LFP is the longest D8 flow distance from any cell to the outlet, measured between cell centres (diagonals count as sqrt 2 cells). Its elevations come from the conditioned DEM, so the profile never rises downstream.

## Limits to keep in mind

- Results depend on DEM resolution and quality. Culverts and road embankments often block flow in bare-earth DEMs of coarse data, or are missing from the DEM entirely. Burn or cut them before delineating, or the outlet may snap to the wrong channel.
- Each Tc formula was calibrated on specific regions and catchment sizes (see docstrings in `catchtool/tc.py`). The tool reports all of them and does not choose for you. The spread between methods is itself useful information.
- Slope choice changes Tc. Pick one convention per project and state it.
- Tests use synthetic terrain with hand-derived answers; the Tc coefficients should still be checked against your local guideline before design use.
