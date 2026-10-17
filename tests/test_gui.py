import io
import json
import threading
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer

import pytest

from catchtool import gui


@pytest.fixture(scope="module")
def base():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), gui.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def call(base, path, data=None, raw=False):
    req = urllib.request.Request(base + path, data=data, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req) as r:
            body = r.read()
            return r.status, body if raw else json.loads(body)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_and_config(base):
    code, body = call(base, "/", raw=True)
    assert code == 200 and b"catchtool" in body
    code, cfg = call(base, "/api/config")
    assert code == 200 and "bransby_williams" in [m["id"] for m in cfg["tc_methods"]]


def test_run_before_dem_is_a_clear_error(base):
    gui.STATE.terrain = None
    code, out = call(base, "/api/run", json.dumps({"outlets": [{"id": "a", "x": 1, "y": 1}]}).encode())
    assert code == 400 and "Load a DEM" in out["error"]


def test_demo_run_download(base):
    code, v = call(base, "/api/demo", b"")
    assert code == 200 and v["cols"] == 201
    x = (v["x0"] + v["x1"]) / 2
    y = v["y1"] - 120 * v["cell"]  # on the valley floor, 120 cells down
    body = {"outlets": [{"id": "A", "x": x, "y": y}, {"id": "off", "x": 0, "y": 0}],
            "snap_distance": 50, "stream_km2": 0.05}
    code, out = call(base, "/api/run", json.dumps(body).encode())
    assert code == 200
    assert [r["id"] for r in out["rows"]] == ["A"]
    assert out["errors"][0]["id"] == "off" and "outside the DEM" in out["errors"][0]["message"]
    assert out["rows"][0]["area_km2"] == pytest.approx(1e-4 * (120 - 40 + 1) * 121, rel=0.05)
    f = out["features"][0]
    assert f["polygon"] and f["lfp"] and f["profile"]["d"][0] == 0
    json.dumps(out, allow_nan=False)  # no NaN leaks into the response
    code, z = call(base, "/api/download.zip", raw=True)
    names = zipfile.ZipFile(io.BytesIO(z)).namelist()
    assert {"metrics.csv", "catchments.gpkg", "overview.png"} <= set(names)
    code, png = call(base, "/api/preview.png", raw=True)
    assert png[:4] == b"\x89PNG"


def test_non_geotiff_upload_is_rejected_politely(base):
    code, out = call(base, "/api/upload?name=notes.tif", b"this is not a raster")
    assert code == 400 and "GeoTIFF" in out["error"]
