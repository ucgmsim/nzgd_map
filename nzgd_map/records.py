"""Detail pages and downloads that preserve individual report identity."""

import re
import sqlite3
from io import BytesIO

import flask
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import query_sqlite_db as queries
from .database import open_database

bp = flask.Blueprint("records", __name__)


@bp.app_template_filter("number")
def format_number(value: object, precision: int = 2) -> str:
    """Format a finite number, explicitly labelling absent values."""
    if value is None or pd.isna(value):
        return "Not available"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Not available"
    return f"{number:.{precision}f}" if np.isfinite(number) else "Not available"


@bp.app_template_filter("yes_no")
def format_yes_no(value: object) -> str:
    """Format nullable database flags without treating missing values as false."""
    if value is None or pd.isna(value):
        return "Not available"
    return {0: "No", 1: "Yes"}.get(value, "Not available")


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def _record(record_name: str, conn: sqlite3.Connection) -> dict:
    match = re.fullmatch(r"(CPT|SCPT|BH)_([0-9]+)", record_name)
    if match is None:
        flask.abort(404)
    nzgd_id = int(match[2])
    # sqlite INTEGER IDs cannot represent arbitrarily large URL parameters.
    if nzgd_id > 2**63 - 1:
        flask.abort(404)
    original = queries.record_metadata(nzgd_id, conn)
    if original is None:
        flask.abort(404)
    if match[1] != original["type_prefix"] and not (
        match[1] == "SCPT" and original["type_prefix"] == "CPT"
    ):
        flask.abort(404)
    record = queries.resolve_record(nzgd_id, conn)
    if record is None:
        flask.abort(404)
    return {key: None if pd.isna(value) else value for key, value in record.items()}


def _cpt_figure(data: pd.DataFrame) -> go.Figure:
    """Plot a single report's CPT measurements in their stored MPa units."""
    figure = make_subplots(rows=1, cols=3, shared_yaxes=True)
    data = data.loc[np.isfinite(pd.to_numeric(data["depth_m"], errors="coerce"))]
    for column, (field, label) in enumerate(
        [
            ("qc_MPa", "Cone resistance, qc (MPa)"),
            ("fs_MPa", "Sleeve friction, fs (MPa)"),
            ("u2_MPa", "Pore pressure, u2 (MPa)"),
        ],
        start=1,
    ):
        figure.add_trace(
            go.Scatter(
                x=pd.to_numeric(data[field], errors="coerce"),
                y=data["depth_m"],
                mode="lines",
                name=label,
                connectgaps=False,
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(title_text=label, row=1, col=column)
    figure.update_yaxes(autorange="reversed")
    figure.update_yaxes(title_text="Depth (m)", row=1, col=1)
    figure.update_layout(showlegend=False, height=550, margin={"t": 30})
    return figure


def _spt_figure(data: pd.DataFrame) -> go.Figure:
    """Plot the agreed N-value selection for a single SPT report."""
    data = data.loc[np.isfinite(pd.to_numeric(data["depth_m"], errors="coerce"))]
    figure = go.Figure(
        go.Scatter(
            x=data["number_of_blows"],
            y=data["depth_m"],
            customdata=data["n_value_source"],
            mode="lines+markers",
            line_shape="vhv",
            connectgaps=False,
            hovertemplate="N: %{x}<br>Depth: %{y} m<br>Source: %{customdata}<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis_title="Number of blows, N",
        yaxis_title="Depth (m)",
        yaxis_autorange="reversed",
        height=550,
        margin={"t": 30},
    )
    return figure


def _render_record(record_name: str, kind: str):
    report_reader, measurement_reader, estimate_reader = {
        "cpt": (
            queries.cpt_reports_for_one_nzgd,
            queries.cpt_measurements_for_one_nzgd,
            queries.cpt_vs30s_for_one_nzgd_id,
        ),
        "spt": (
            queries.spt_reports_for_one_nzgd,
            queries.spt_measurements_for_one_nzgd,
            queries.spt_vs30s_for_one_nzgd_id,
        ),
    }[kind]
    with open_database() as conn:
        record = _record(record_name, conn)
        nzgd_id = int(record["nzgd_id"])
        reports = report_reader(nzgd_id, conn)
        if reports.empty:
            flask.abort(404)
        if record_name != record["record_name"]:
            return flask.redirect(
                flask.url_for(
                    f"records.{kind}_record", record_name=record["record_name"]
                ),
                code=301,
            )
        measurements = measurement_reader(nzgd_id, conn)
        estimates = estimate_reader(nzgd_id, conn)
        soils = (
            queries.spt_soil_types_for_one_nzgd(nzgd_id, conn)
            if kind == "spt"
            else None
        )

    prepared = []
    id_column = f"{kind}_id"
    for report in _records(reports):
        report_id = report[id_column]
        data = measurements.loc[measurements[id_column] == report_id]
        report["report_id"] = report_id
        report["measurements"] = _records(data)
        report["estimates"] = _records(estimates.loc[estimates[id_column] == report_id])
        report["plot"] = ""
        if not data.empty:
            figure = _cpt_figure(data) if kind == "cpt" else _spt_figure(data)
            report["plot"] = figure.to_html(
                full_html=False,
                include_plotlyjs=False,
                div_id=f"{kind}-{report_id}-plot",
                config={"responsive": True},
            )
        if soils is not None:
            report["soils"] = _records(soils.loc[soils["spt_id"] == report_id])
        prepared.append(report)
    return flask.render_template(
        f"views/{kind}_record.html",
        record=record,
        reports=prepared,
        kind=kind,
    )


@bp.route("/cpt/<record_name>")
def cpt_record(record_name: str):
    """Show all CPT reports within an NZGD investigation."""
    return _render_record(record_name, "cpt")


@bp.route("/spt/<record_name>")
def spt_record(record_name: str):
    """Show all SPT reports within an NZGD investigation."""
    return _render_record(record_name, "spt")


def _download(record_name: str, kind: str, report_id: int | None, soil: bool = False):
    with open_database() as conn:
        record = _record(record_name, conn)
        nzgd_id = int(record["nzgd_id"])
        report_reader = (
            queries.cpt_reports_for_one_nzgd
            if kind == "cpt"
            else queries.spt_reports_for_one_nzgd
        )
        reports = report_reader(nzgd_id, conn)
        if reports.empty or (
            report_id is not None and report_id not in reports[f"{kind}_id"].values
        ):
            flask.abort(404)
        if soil:
            data = queries.spt_soil_types_for_one_nzgd(nzgd_id, conn)
        elif kind == "cpt":
            data = queries.cpt_measurements_for_one_nzgd(nzgd_id, conn)
        else:
            data = queries.spt_measurements_for_one_nzgd(nzgd_id, conn)
    if report_id is not None:
        data = data.loc[data[f"{kind}_id"] == report_id]
    suffix = "soil_types" if soil else "data"
    report_part = f"_{kind}_{report_id}" if report_id is not None else ""
    filename = f"{record['record_name']}{report_part}_{suffix}.csv"
    return flask.send_file(
        BytesIO(data.to_csv(index=False).encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


@bp.route("/cpt/<record_name>/reports/<int:report_id>/data.csv")
def download_cpt_report(record_name: str, report_id: int):
    """Download one CPT report, checking that it belongs to the requested record."""
    return _download(record_name, "cpt", report_id)


@bp.route("/spt/<record_name>/reports/<int:report_id>/data.csv")
def download_spt_report(record_name: str, report_id: int):
    """Download one SPT report with the original and selected N-value fields."""
    return _download(record_name, "spt", report_id)


@bp.route("/spt/<record_name>/reports/<int:report_id>/soil_types.csv")
def download_spt_report_soils(record_name: str, report_id: int):
    """Download soil intervals and classifications from one SPT report."""
    return _download(record_name, "spt", report_id, soil=True)


def _legacy_record_name(filename: str, suffix: str) -> str:
    match = re.fullmatch(rf"((?:CPT|SCPT|BH)_[0-9]+)_{suffix}\.csv", filename)
    if match is None:
        flask.abort(404)
    return match[1]


@bp.route("/download_cpt_data/<filename>")
def download_cpt_data(filename: str):
    """Preserve record-wide CSV URLs, including report identity on every row."""
    return _download(_legacy_record_name(filename, "data"), "cpt", None)


@bp.route("/download_spt_data/<filename>")
def download_spt_data(filename: str):
    """Download all SPT measurements without losing their report provenance."""
    return _download(_legacy_record_name(filename, "data"), "spt", None)


@bp.route("/download_spt_soil_types/<filename>")
def download_spt_soil_types(filename: str):
    """Download all soil intervals, preserving separate reports and layers."""
    return _download(
        _legacy_record_name(filename, "soil_types"), "spt", None, soil=True
    )
