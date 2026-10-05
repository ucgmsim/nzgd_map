"""Route checks for the map, separate report pages and data downloads."""

import html
import os
import re
import tempfile
import time
from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import plotly
import pytest
from flask import Flask, template_rendered
from flask.testing import FlaskClient

from nzgd_map import constants, filters


@contextmanager
def _templates(app: Flask):
    contexts = []

    def capture(sender: Flask, template: object, context: dict, **extra):
        contexts.append(context)

    template_rendered.connect(capture, app)
    try:
        yield contexts
    finally:
        template_rendered.disconnect(capture, app)


@pytest.mark.parametrize(
    "choice, count", [("all", 8), ("available", 4), ("unavailable", 4)]
)
def test_map_availability_choices(
    app: Flask, client: FlaskClient, choice: str, count: int
):
    with _templates(app) as contexts:
        response = client.get("/", query_string={"vs30_availability": choice})
    assert response.status_code == 200
    context = contexts[-1]
    assert context["num_reports"] == count
    assert context["matching_reports"] == 8
    assert context["available_reports"] == 4
    assert context["unavailable_reports"] == 4
    assert context["selected_availability"] == choice
    assert 'id="availability-select"' in response.text
    assert "/assets/plotly-" in response.text
    assert "Date of last update from NZGD: Not provided" in response.text


def test_filter_does_not_clip_estimates(app: Flask, client: FlaskClient):
    with _templates(app) as contexts:
        response = client.get(
            "/", query_string={"query": "vs30 > 90000", "hist_by": "vs30"}
        )
    assert response.status_code == 200
    assert contexts[-1]["num_reports"] == 1
    assert contexts[-1]["histogram_count"] == 1
    assert contexts[-1]["histogram_missing"] == 0


@pytest.mark.parametrize("query", ["nzgd_id < 0", "vs30_stddev.notna()"])
def test_empty_selections_render(app: Flask, client: FlaskClient, query: str):
    with _templates(app) as contexts:
        response = client.get("/", query_string={"query": query})
    assert response.status_code == 200
    assert contexts[-1]["num_reports"] == 0
    assert "No reports match" in response.text


def test_absent_histogram_values_render(app: Flask, client: FlaskClient):
    with _templates(app) as contexts:
        response = client.get(
            "/", query_string={"colour_by": "vs30_stddev", "hist_by": "vs30_stddev"}
        )
    assert response.status_code == 200
    assert contexts[-1]["histogram_missing"] == 8
    assert "No values available" in response.text


@pytest.mark.parametrize(
    "arguments",
    [
        {"vs30_availability": "other"},
        {"colour_by": "unknown"},
        {"hist_by": "unknown"},
        {"cpt_vs_correlation": "unknown"},
        {"query": "vs30 >"},
        {"query": '@__import__("os").getcwd()'},
    ],
)
def test_bad_parameters_return_client_errors(client: FlaskClient, arguments: dict):
    assert client.get("/", query_string=arguments).status_code == 400


@pytest.mark.parametrize("query", ["unknown > 0", '__import__("os").system("id")'])
def test_rejected_query_is_explained_on_map_and_live_validation(
    client: FlaskClient, query: str
):
    with pytest.raises(filters.QueryError) as rejected:
        filters.filter_reports(filters.empty_query_frame(), query)
    message = html.escape(str(rejected.value))
    response = client.get("/", query_string={"query": query})
    assert response.status_code == 400
    assert message in response.text
    assert message in client.get("/validate", query_string={"query": query}).text


def test_query_help_and_validation(client: FlaskClient):
    response = client.get("/query_help")
    assert response.status_code == 200
    assert "record_created_on" in response.text
    assert "ISPT" not in response.text
    assert "investigation_date" not in response.text
    assert client.get("/validate", query_string={"query": "spt_id == 31"}).text == ""
    assert (
        "Unknown field"
        in client.get("/validate", query_string={"query": "unknown > 0"}).text
    )


def test_cpt_report_sections_and_null_estimates(app: Flask, client: FlaskClient):
    with _templates(app) as contexts:
        response = client.get("/cpt/CPT_1001")
    assert response.status_code == 200
    reports = contexts[-1]["reports"]
    assert [item["report_id"] for item in reports] == [13, 17, 19]
    assert [len(item["measurements"]) for item in reports] == [2, 2, 0]
    assert [len(item["estimates"]) for item in reports] == [3, 1, 0]
    assert 'id="cpt-13-plot"' in response.text
    assert 'id="cpt-17-plot"' in response.text
    assert 'id="cpt-19-plot"' not in response.text
    assert "Not available" in response.text
    assert "Record created in NZGD" in response.text
    assert "Investigation date" not in response.text


def test_no_estimate_record_is_accessible(client: FlaskClient):
    response = client.get("/cpt/CPT_1003")
    assert response.status_code == 200
    assert "No Vs30 estimates are stored for this report" in response.text
    assert 'id="cpt-23-plot"' in response.text
    assert client.get("/cpt/CPT_1006").status_code == 200
    assert client.get("/spt/BH_1004").status_code == 200


def test_spt_reports_and_per_estimate_assumptions(app: Flask, client: FlaskClient):
    with _templates(app) as contexts:
        response = client.get("/spt/BH_1002")
    assert response.status_code == 200
    reports = contexts[-1]["reports"]
    assert [item["report_id"] for item in reports] == [31, 47, 49]
    assert [len(item["measurements"]) for item in reports] == [5, 2, 0]
    assert [len(item["soils"]) for item in reports] == [3, 1, 1]
    assert (
        "ISPT_NVAL" in response.text
        and "ISPT_MAIN" in response.text
        and "ISPT_REP" in response.text
    )
    assert re.search(
        r"boore_2004</td>\s*<td>150\.00</td>\s*<td>Auto</td>\s*<td>No</td>\s*<td>Yes</td>",
        response.text,
    )
    assert re.search(
        r"boore_2011</td>\s*<td>200\.00</td>\s*<td>Auto</td>\s*<td>Yes</td>\s*<td>No</td>",
        response.text,
    )


def test_report_downloads_preserve_identity_and_units(client: FlaskClient):
    cpt = client.get("/cpt/CPT_1001/reports/17/data.csv")
    assert cpt.status_code == 200
    data = pd.read_csv(StringIO(cpt.text))
    assert set(data.cpt_id) == {17}
    assert data.qc_MPa.tolist() == [17.1, 17.2]
    assert set(data.nzgd_id) == {1001}
    assert "source_file" in data

    spt = client.get("/spt/BH_1002/reports/31/data.csv")
    assert spt.status_code == 200
    data = pd.read_csv(StringIO(spt.text))
    assert set(data.spt_id) == {31}
    assert set(data.nzgd_id) == {1002}
    assert data.number_of_blows.iloc[:4].tolist() == [12, 0, 0, 20]
    assert {"ISPT_MAIN", "ISPT_NVAL", "ISPT_REP", "n_value_source"} <= set(data.columns)


def test_soil_and_legacy_downloads(client: FlaskClient):
    response = client.get("/spt/BH_1002/reports/47/soil_types.csv")
    assert response.status_code == 200
    data = pd.read_csv(StringIO(response.text))
    assert data.spt_id.tolist() == [47]
    assert data.soil_type.tolist() == ["GRAVEL"]
    assert data.bottom_depth_m.tolist() == [4]

    all_cpt = pd.read_csv(
        StringIO(client.get("/download_cpt_data/CPT_1001_data.csv").text)
    )
    assert set(all_cpt.cpt_id) == {13, 17}
    all_spt = pd.read_csv(
        StringIO(client.get("/download_spt_data/BH_1002_data.csv").text)
    )
    assert set(all_spt.spt_id) == {31, 47}
    assert (
        client.get("/download_spt_soil_types/BH_1002_soil_types.csv").status_code == 200
    )


@pytest.mark.parametrize(
    "path",
    [
        "/cpt/invalid",
        "/cpt/CPT_99999",
        "/cpt/CPT_999999999999999999999",
        "/cpt/BH_1001",
        "/cpt/BH_1002",
        "/spt/BH_1002/reports/81/data.csv",
        "/cpt/CPT_1001/reports/23/data.csv",
        "/download_cpt_data/not-a-record.csv",
    ],
)
def test_unknown_records_and_mismatched_reports(client: FlaskClient, path: str):
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("name", ["CPT_9001", "SCPT_1001"])
def test_merged_and_legacy_record_names_redirect(client: FlaskClient, name: str):
    response = client.get(f"/cpt/{name}")
    assert response.status_code == 301
    assert response.headers["Location"] == "/cpt/CPT_1001"


def test_mounted_urls_and_matching_plotly(client: FlaskClient):
    response = client.get("/", environ_overrides={"SCRIPT_NAME": "/nzgd"})
    assert response.status_code == 200
    assert 'hx-get="/nzgd/validate"' in response.text
    assert f"/nzgd/assets/plotly-{plotly.__version__}.min.js" in response.text
    assert "\\u002fnzgd\\u002fcpt" in response.text or "/nzgd/cpt" in response.text
    javascript = client.get(f"/assets/plotly-{plotly.__version__}.min.js")
    assert javascript.status_code == 200
    assert javascript.mimetype == "text/javascript"
    assert client.get("/assets/plotly-unknown.min.js").status_code == 404


def test_geonet_toggle_preserves_selection(client: FlaskClient):
    target = "/?vs30_availability=unavailable&vs30_correlation=boore_2011"
    response = client.post("/toggle_geonet_visibility", data={"return_to": target})
    assert response.headers["Location"] == target
    with client.session_transaction() as session:
        assert session["show_geonet_stations"] == "off"
    assert client.post("/upload_geonet").headers["Location"] == "/"
    assert (
        client.post("/clear_geonet", data={"return_to": "https://example.com"}).headers[
            "Location"
        ]
        == "/"
    )


def test_geonet_upload_replace_clear_and_fallback(
    app: Flask, client: FlaskClient, tmp_path: Path
):
    uploads = tmp_path / "uploads"
    app.config["GEONET_UPLOAD_FOLDER"] = str(uploads)
    default = tmp_path / "default.ll"
    default.write_text("172.6 -43.5 DEFAULT\n")
    app.config["GEONET_STATIONS_PATH"] = str(default)
    target = "/?vs30_availability=unavailable&query=vs30.isna()"

    def upload(content: bytes, filename: str):
        return client.post(
            "/upload_geonet",
            data={"geonet_file": (BytesIO(content), filename), "return_to": target},
        )

    assert upload(b"ignored", "stations.exe").headers["Location"] == target
    with client.session_transaction() as session:
        assert "user_geonet_file" not in session
    assert (
        upload(
            b"lon,lat,name\n172.5,-43.4,CUSTOM\n500,-43,INVALID\n", "stations.csv"
        ).headers["Location"]
        == target
    )
    with client.session_transaction() as session:
        first = uploads / session["user_geonet_file"]
    assert first.is_file()
    response = client.get("/", query_string={"colour_by": "type_number_code"})
    assert response.status_code == 200
    assert '"CUSTOM"' in response.text
    assert '"INVALID"' not in response.text
    assert '"DEFAULT"' not in response.text

    # A replacement removes the old temporary upload. Invalid station contents
    # fall back to the configured file without stopping the map from loading.
    assert upload(b"invalid\n", "replacement.ll").status_code == 302
    assert not first.exists()
    with client.session_transaction() as session:
        second = uploads / session["user_geonet_file"]
    assert second.is_file()
    assert '"DEFAULT"' in client.get("/").text
    assert (
        client.post("/clear_geonet", data={"return_to": target}).headers["Location"]
        == target
    )
    assert not second.exists()
    with client.session_transaction() as session:
        assert "user_geonet_file" not in session


def test_geonet_uploads_are_kept_out_of_the_shared_temporary_directory(
    app: Flask, client: FlaskClient
):
    client.post(
        "/upload_geonet",
        data={"geonet_file": (BytesIO(b"172.5 -43.4 CUSTOM\n"), "stations.ll")},
    )
    with client.session_transaction() as session:
        name = session["user_geonet_file"]
    assert not (Path(tempfile.gettempdir()) / name).exists()
    assert [path.name for path in Path(app.instance_path).rglob(name)] == [name]
    assert '"CUSTOM"' in client.get("/").text


@pytest.mark.parametrize(
    "name",
    [
        "../outside.ll",
        "<absolute>",
        "inside.ll",
        "0123456789abcdef0123456789abcdef_inside.ll",
        "0123456789ABCDEF0123456789ABCDEF.ll",
        "0123456789abcdef0123456789abcde.ll",
        "0123456789abcdef0123456789abcdef.exe",
        "<link>",
    ],
)
def test_forged_geonet_session_cannot_read_or_delete_other_files(
    app: Flask, client: FlaskClient, tmp_path: Path, name: str
):
    # The session is signed, but a leaked key would let anyone choose its value.
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    app.config["GEONET_UPLOAD_FOLDER"] = str(uploads)
    outside = tmp_path / "outside.ll"
    outside.write_text("172.6 -43.5 OUTSIDE\n")
    if name == "<absolute>":
        name = str(outside)
    elif name == "<link>":
        name = "0123456789abcdef0123456789abcdef.ll"
        (uploads / name).symlink_to(outside)
    elif "/" not in name:
        # Present in the folder, but not a name that upload_geonet creates.
        (uploads / name).write_text("172.6 -43.5 OUTSIDE\n")
    files = set(tmp_path.rglob("*"))
    with client.session_transaction() as session:
        session["user_geonet_file"] = name
    assert '"OUTSIDE"' not in client.get("/").text
    client.post("/clear_geonet", data={"return_to": "/"})
    assert set(tmp_path.rglob("*")) == files


def test_oldest_geonet_uploads_are_removed_beyond_the_storage_limit(
    app: Flask, client: FlaskClient, tmp_path: Path
):
    # Requests without a session never replace an earlier upload, so unbounded
    # uploads could fill the server's disk.
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    app.config.update(GEONET_UPLOAD_FOLDER=str(uploads), GEONET_MAX_UPLOADS=2)
    oldest, older = uploads / f"{'1' * 32}.ll", uploads / f"{'2' * 32}.csv"
    for age, path in [(7200, oldest), (3600, older)]:
        path.write_text("172.6 -43.5 OLD\n")
        os.utime(path, (time.time() - age,) * 2)
    client.post(
        "/upload_geonet",
        data={"geonet_file": (BytesIO(b"172.5 -43.4 NEW\n"), "new.ll")},
    )
    with client.session_transaction() as session:
        newest = uploads / session["user_geonet_file"]
    assert set(uploads.iterdir()) == {older, newest}


def test_malformed_optional_station_file_does_not_break_map(
    app: Flask, client: FlaskClient, tmp_path: Path
):
    broken = tmp_path / "broken.csv"
    broken.write_bytes(b"\xff\xfe\x00")
    app.config["GEONET_STATIONS_PATH"] = str(broken)
    with _templates(app) as contexts:
        response = client.get("/")
    assert response.status_code == 200
    assert not contexts[-1]["geonet_stations_available"]


def test_optional_retrieval_date_and_config_precedence(
    app: Flask, client: FlaskClient, tmp_path: Path
):
    (tmp_path / constants.last_retrieval_date_file_name).write_text("2026-07-09\n")
    assert "Date of last update from NZGD: 2026-07-09" in client.get("/").text
    app.config["LAST_NZGD_RETRIEVAL_DATE"] = "2026-08-01"
    assert "Date of last update from NZGD: 2026-08-01" in client.get("/").text
