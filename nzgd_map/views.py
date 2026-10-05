"""Map, filter help, JavaScript assets, and the optional GeoNet overlay."""

import re
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import flask
import pandas as pd
import plotly

from . import constants, filters, plotting, query_sqlite_db
from .database import open_database

bp = flask.Blueprint("views", __name__)
ALLOWED_EXTENSIONS = {"txt", "ll", "csv"}
# upload_geonet stores station files only under generated names of this form.
UPLOAD_NAME = re.compile(r"[0-9a-f]{32}\.(?:txt|ll|csv)")
AVAILABILITY_OPTIONS = [
    ("all", "All reports with measurements"),
    ("available", "With a Vs30 estimate for the selected correlations"),
    ("unavailable", "Without a Vs30 estimate for the selected correlations"),
]


def allowed_file(filename: str) -> bool:
    """Check whether an uploaded station file has a supported extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _upload_folder() -> Path:
    return Path(flask.current_app.config["GEONET_UPLOAD_FOLDER"])


def _remove_oldest_uploads(folder: Path):
    """Bound disk use: uploads without a session never replace earlier ones."""
    uploads = []
    for path in folder.iterdir():
        if UPLOAD_NAME.fullmatch(path.name):
            try:
                uploads.append((path.stat().st_mtime_ns, path))
            except FileNotFoundError:  # Removed meanwhile by another worker.
                continue
    excess = len(uploads) - flask.current_app.config["GEONET_MAX_UPLOADS"]
    for _, path in sorted(uploads)[: max(excess, 0)]:
        path.unlink(missing_ok=True)


def get_user_geonet_file_path() -> Path | None:
    """Find this session's upload, accepting only names that upload_geonet creates."""
    filename = flask.session.get("user_geonet_file")
    if not isinstance(filename, str) or not UPLOAD_NAME.fullmatch(filename):
        return None
    folder = _upload_folder().resolve()
    path = folder / filename
    # A link would let a valid name refer to a file outside the folder.
    if path.is_symlink() or not path.is_file() or path.resolve().parent != folder:
        return None
    return path


def load_geonet_stations() -> pd.DataFrame:
    """Load uploaded or default stations, allowing an absent optional overlay."""
    configured_path = flask.current_app.config["GEONET_STATIONS_PATH"]
    paths = [
        get_user_geonet_file_path(),
        Path(configured_path) if configured_path else None,
    ]
    for path in paths:
        if path is None or not path.is_file():
            continue
        try:
            stations = pd.read_csv(
                path,
                sep=r"[\s,]+",
                engine="python",
                header=None,
                names=["lon", "lat", "name"],
                comment="/",
            )
            for column in ("lon", "lat"):
                stations[column] = pd.to_numeric(stations[column], errors="coerce")
            stations = stations.loc[
                stations["lon"].between(-180, 180) & stations["lat"].between(-90, 90)
            ]
            if not stations.empty:
                return stations
        except (OSError, ValueError, pd.errors.ParserError) as error:
            flask.current_app.logger.warning(
                "Could not read GeoNet file %s: %s", path, error
            )
    return pd.DataFrame(columns=["lon", "lat", "name"])


@bp.app_context_processor
def plotly_context() -> dict:
    """Point every page at the JavaScript bundled with the installed Plotly."""
    return {"plotly_version": plotly.__version__}


@bp.route("/assets/plotly-<version>.min.js")
def plotly_js(version: str):
    """Serve the matching Plotly JavaScript with a versioned cache URL."""
    if version != plotly.__version__:
        flask.abort(404)
    return flask.send_file(
        Path(plotly.__file__).parent / "package_data" / "plotly.min.js",
        mimetype="text/javascript",
        max_age=31536000,
    )


@bp.route("/")
def index():
    """Show measured reports with independent correlation and availability choices."""
    vs30_correlation = flask.request.args.get(
        "vs30_correlation", constants.default_vs_to_vs30_correlation
    )
    cpt_vs_correlation = flask.request.args.get(
        "cpt_vs_correlation", constants.default_cpt_to_vs_correlation
    )
    spt_vs_correlation = flask.request.args.get(
        "spt_vs_correlation", constants.default_spt_to_vs_correlation
    )
    colour_by = flask.request.args.get("colour_by", "vs30")
    hist_by = flask.request.args.get("hist_by", "vs30_log_residual")
    availability = flask.request.args.get("vs30_availability", "all")
    if colour_by not in plotting.LABELS or hist_by not in plotting.LABELS:
        flask.abort(400, description="Unknown map colour or histogram field.")
    if availability not in dict(AVAILABILITY_OPTIONS):
        flask.abort(400, description="Unknown Vs30 availability option.")
    query = flask.request.args.get("query", "")

    with open_database() as conn:
        vs30_correlations = query_sqlite_db.lookup_values("vstovs30correlation", conn)
        cpt_correlations = query_sqlite_db.lookup_values("cpttovscorrelation", conn)
        spt_correlations = query_sqlite_db.lookup_values("spttovscorrelation", conn)
        try:
            frame = query_sqlite_db.all_vs30s_given_correlations(
                vs30_correlation,
                cpt_vs_correlation,
                spt_vs_correlation,
                "Auto",
                conn,
                include_unestimated=True,
            )
        except ValueError as error:
            flask.abort(400, description=str(error))
    error = None
    try:
        frame = filters.filter_reports(frame, query)
    except filters.QueryError as invalid:
        error = str(invalid)
        frame = frame.iloc[:0].copy()
    matching_reports = len(frame)
    available_reports = int(frame["vs30_available"].sum())
    unavailable_reports = matching_reports - available_reports
    if availability == "available":
        frame = frame.loc[frame["vs30_available"]].copy()
    elif availability == "unavailable":
        frame = frame.loc[~frame["vs30_available"]].copy()
    frame["report_url"] = [
        flask.url_for(
            f"records.{kind.lower()}_record",
            record_name=name,
            _anchor=f"{kind.lower()}-{report_id}",
        )
        for kind, name, report_id in zip(
            frame["report_kind"], frame["record_name"], frame["report_id"], strict=True
        )
    ]
    stations = load_geonet_stations()
    visibility = flask.session.get("show_geonet_stations", "on")
    map_plot, mapped_reports = plotting.map_figure(
        frame, colour_by, stations if visibility == "on" else stations.iloc[:0]
    )
    histogram, histogram_missing = plotting.histogram_figure(frame, hist_by)
    date = flask.current_app.config["LAST_NZGD_RETRIEVAL_DATE"]
    date_path = (
        Path(flask.current_app.instance_path) / constants.last_retrieval_date_file_name
    )
    if not date and date_path.is_file():
        date = date_path.read_text().strip()

    return flask.render_template(
        "views/index.html",
        date_of_last_nzgd_retrieval=date,
        selected_vs30_correlation=vs30_correlation,
        selected_cpt_vs_correlation=cpt_vs_correlation,
        selected_spt_vs_correlation=spt_vs_correlation,
        vs30_correlations=vs30_correlations,
        cpt_vs_correlations=cpt_correlations,
        spt_vs_correlations=spt_correlations,
        availability_options=AVAILABILITY_OPTIONS,
        selected_availability=availability,
        colour_variables=plotting.COLOUR_VARIABLES,
        colour_by=colour_by,
        hist_by=hist_by,
        query=query,
        error=error,
        num_records=frame["nzgd_id"].nunique(),
        num_reports=len(frame),
        matching_reports=matching_reports,
        available_reports=available_reports,
        unavailable_reports=unavailable_reports,
        unmapped_reports=len(frame) - mapped_reports,
        histogram_missing=histogram_missing,
        histogram_count=len(frame) - histogram_missing,
        map=map_plot.to_html(
            full_html=False,
            include_plotlyjs=False,
            default_height="75vh",
            config={"responsive": True},
        ),
        hist_plot=histogram.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"responsive": True},
        ),
        show_geonet_visibility=visibility,
        geonet_stations_available=not stations.empty,
        return_to=flask.request.script_root + flask.request.full_path.rstrip("?"),
    ), 400 if error else 200


@bp.route("/query_help")
def query_help():
    """List the supported filter fields, syntax and location names."""
    with open_database() as conn:
        locations = {
            "region_names": query_sqlite_db.get_region_names(conn),
            "district_names": query_sqlite_db.get_district_names(conn),
            "city_names": query_sqlite_db.get_city_names(conn),
            "suburb_names": query_sqlite_db.get_suburb_names(conn),
        }
    return flask.render_template(
        "views/query_help.html",
        col_names_to_display=", ".join(filters.QUERY_FIELDS),
        **locations,
    )


@bp.route("/validate")
def validate():
    """Validate exactly the filter syntax used for the map."""
    try:
        filters.filter_reports(
            filters.empty_query_frame(), flask.request.args.get("query", "")
        )
    except filters.QueryError as error:
        return flask.render_template("error.html", error=error)
    return ""


def _return_to_map():
    target = flask.request.form.get("return_to", "")
    parts = urlsplit(target)
    if (
        not target.startswith("/")
        or target.startswith("//")
        or parts.scheme
        or parts.netloc
        or "\\" in target
    ):
        target = flask.url_for("views.index")
    return flask.redirect(target)


@bp.route("/upload_geonet", methods=["POST"])
def upload_geonet():
    """Save a station file for this session and preserve the map selection."""
    file = flask.request.files.get("geonet_file")
    if file and file.filename and allowed_file(file.filename):
        previous = get_user_geonet_file_path()
        extension = file.filename.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{extension}"
        folder = _upload_folder()
        folder.mkdir(parents=True, exist_ok=True)
        file.save(folder / filename)
        flask.session["user_geonet_file"] = filename
        if previous:
            previous.unlink(missing_ok=True)
        _remove_oldest_uploads(folder)
    return _return_to_map()


@bp.route("/clear_geonet", methods=["POST"])
def clear_geonet():
    """Remove the session's uploaded station file and use the default overlay."""
    path = get_user_geonet_file_path()
    if path:
        path.unlink(missing_ok=True)
    flask.session.pop("user_geonet_file", None)
    return _return_to_map()


@bp.route("/toggle_geonet_visibility", methods=["POST"])
def toggle_geonet_visibility():
    """Toggle the station overlay while preserving correlation and filter choices."""
    current = flask.session.get("show_geonet_stations", "on")
    flask.session["show_geonet_stations"] = "off" if current == "on" else "on"
    return _return_to_map()
