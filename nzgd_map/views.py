"""
The views module defines the Flask views (web pages) for the application.
Each view is a function that returns an HTML template to render in the browser.
"""

import os
import sqlite3
import tempfile
import uuid
from collections import OrderedDict
from io import StringIO
from pathlib import Path

import flask
import numpy as np
import pandas as pd  # Ensure pandas is imported
import plotly.express as px
import plotly.graph_objects as go  # Add import for plotly.graph_objects
from plotly.subplots import make_subplots
from werkzeug.utils import secure_filename

from . import constants, query_sqlite_db

# Create a Flask Blueprint for the views
bp = flask.Blueprint("views", __name__)

# Directory to store temporary uploaded geonet files
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {"txt", "ll", "csv"}


def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_user_geonet_file_path():
    """Get the path to the user's uploaded GeoNet file if it exists in the session"""
    if "user_geonet_file" in flask.session:
        file_path = Path(UPLOAD_FOLDER) / flask.session["user_geonet_file"]
        if file_path.exists():
            return file_path
    return None


@bp.route("/", methods=["GET"])
def index():
    """Serve the standard index page."""
    # Access the instance folder for application-specific data
    instance_path = Path(flask.current_app.instance_path)

    with open(instance_path / constants.last_retrieval_date_file_name, "r") as file:
        date_of_last_nzgd_retrieval = file.readline()

    # Retrieve selected vs30 correlation. If no selection, default to "boore_2004"
    vs30_correlation = flask.request.args.get(
        "vs30_correlation", default=constants.default_vs_to_vs30_correlation
    )

    # Retrieve selected spt_vs_correlation. If no selection, default to "brandenberg_2010"
    spt_vs_correlation = flask.request.args.get(
        "spt_vs_correlation", default=constants.default_spt_to_vs_correlation
    )

    # Retrieve selected cpt_vs_correlation. If no selection, default to "andrus_2007_pleistocene".
    cpt_vs_correlation = flask.request.args.get(
        "cpt_vs_correlation", default=constants.default_cpt_to_vs_correlation
    )

    # Retrieve selected column to color by on the map. If no selection, default to "vs30".
    colour_by = flask.request.args.get("colour_by", default="vs30")

    # Retrieve selected column to plot as a histogram. If no selection, default to "vs30_log_residual".
    hist_by = flask.request.args.get(
        "hist_by",
        default="vs30_log_residual",  # Default value if no query parameter is provided
    )

    # Retrieve an optional custom query from request arguments
    query = flask.request.args.get("query", default=None)

    with sqlite3.connect(instance_path / constants.database_file_name) as conn:
        vs_to_vs30_correlation_df = pd.read_sql_query(
            "SELECT * FROM vstovs30correlation", conn
        )
        cpt_to_vs_correlation_df = pd.read_sql_query(
            "SELECT * FROM cpttovscorrelation", conn
        )
        spt_to_vs_correlation_df = pd.read_sql_query(
            "SELECT * FROM spttovscorrelation", conn
        )

        database_df = query_sqlite_db.all_vs30s_given_correlations(
            selected_vs30_correlation=vs30_correlation,
            selected_cpt_to_vs_correlation=cpt_vs_correlation,
            selected_spt_to_vs_correlation=spt_vs_correlation,
            selected_hammer_type="Auto",
            conn=conn,
        )

    database_df["vs30"] = query_sqlite_db.clip_highest_and_lowest_percent(
        database_df["vs30"], 0.1, 99.9
    )

    # Retrieve the available correlation options from the database dataframe to
    # populate the dropdowns in the user interface. Ignore None values.
    vs30_correlations = vs_to_vs30_correlation_df["name"].unique()
    cpt_vs_correlations = cpt_to_vs_correlation_df["name"].unique()
    spt_vs_correlations = spt_to_vs_correlation_df["name"].unique()

    # Apply custom query filtering if provided
    if query:
        database_df = database_df.query(query)

    #########################################################################################
    # Centralize the map on New Zealand
    centre_lat = -41.0
    centre_lon = 174.0

    ## Make map marker sizes proportional to the absolute value of the Vs30 log residual.
    ## For records where the Vs30 log residual is unavailable, use the median of absolute value of the Vs30 log residuals.
    ## This calculation is moved inside the 'if not database_df.empty:' block below.
    marker_size_description_text = r"Marker size indicates the magnitude of the Vs30 log residual, given by \(\mathrm{|(\log(SPT_{Vs30}) - \log(Foster2019_{Vs30})|}\). A minimum size is applied for visibility."

    ## Make new columns of string values to display instead of the float values for Vs30 and log residual
    ## so that an explanation can be shown when the vs30 value or the log residual
    database_df["Vs30 (m/s)"] = database_df["vs30"]
    database_df["Vs30_log_resid"] = database_df["vs30_log_residual"]
    if vs30_correlation == "boore_2011":
        reason_text = "Unable to estimate as Boore et al. (2011) Vs to Vs30 correlation requires a depth of at least 5 m"
        min_required_depth = 5
    else:
        reason_text = "Unable to estimate as Boore et al. (2004) Vs to Vs30 correlation requires a depth of at least 10 m"
        min_required_depth = 10
    database_df.loc[database_df["deepest_depth"] < min_required_depth, "Vs30 (m/s)"] = (
        reason_text
    )
    database_df.loc[
        (database_df["deepest_depth"] >= min_required_depth)
        & (np.isnan(database_df["vs30"]) | (database_df["vs30"] == 0)),
        "Vs30 (m/s)",
    ] = "Vs30 calculation failed even though CPT depth is sufficient"
    database_df.loc[
        (database_df["deepest_depth"] >= min_required_depth)
        & ~(np.isnan(database_df["vs30"]) | (database_df["vs30"] == 0)),
        "Vs30 (m/s)",
    ] = database_df["vs30"].apply(lambda x: f"{x:.2f}")
    database_df.loc[(np.isnan(database_df["vs30_log_residual"])), "Vs30_log_resid"] = (
        "Unavailable as Vs30 could not be calculated"
    )
    database_df.loc[~(np.isnan(database_df["vs30_log_residual"])), "Vs30_log_resid"] = (
        database_df["vs30_log_residual"].apply(lambda x: f"{x:.2f}")
    )
    database_df["deepest_depth (m)"] = database_df["deepest_depth"]

    # Initialize a go.Figure for the map
    map_fig = go.Figure()

    # Add database points if available
    if not database_df.empty:
        # Calculate marker sizes for database_df
        abs_residuals = database_df["vs30_log_residual"].abs()
        median_abs_residual = abs_residuals.median()

        # Determine a representative absolute residual value to use for NaN entries.
        # This value will be scaled along with actual residuals.
        default_abs_residual_for_nans = 0.2  # A small, representative absolute residual
        fill_value_for_residual_nans = default_abs_residual_for_nans
        if pd.notna(median_abs_residual) and median_abs_residual > 0:
            # If a positive median absolute residual exists, use it
            fill_value_for_residual_nans = median_abs_residual.round(1)

        # Input for size calculation: absolute residuals, with NaNs filled
        size_determining_residuals = abs_residuals.fillna(fill_value_for_residual_nans)

        # Scale these residual values to marker sizes
        # Final marker size = base_display_size + (size_determining_residuals * residual_scaling_factor)
        base_display_size = 10  # Minimum size for a point (e.g., when residual is zero)
        residual_scaling_factor = (
            8.0  # Factor to scale the contribution of the residual to the size
        )

        database_df["size"] = base_display_size + (
            size_determining_residuals * residual_scaling_factor
        )

        # Ensure a final minimum display size.
        # This reinforces that no point will be smaller than base_display_size.
        database_df["size"] = np.maximum(database_df["size"], base_display_size)

        # Prepare customdata for hover info
        hover_cols_for_customdata = [
            "deepest_depth (m)",
            "Vs30 (m/s)",
            "Vs30_log_resid",
        ]
        custom_data_for_db = []
        if all(col in database_df.columns for col in hover_cols_for_customdata):
            custom_data_for_db = database_df[hover_cols_for_customdata]
        else:
            # Create placeholder customdata if columns are missing to prevent errors
            num_rows = len(database_df)
            custom_data_for_db = pd.DataFrame(
                {
                    "deepest_depth (m)": ["N/A"] * num_rows,
                    "Vs30 (m/s)": ["N/A"] * num_rows,
                    "Vs30_log_resid": ["N/A"] * num_rows,
                }
            )

        map_fig.add_trace(
            go.Scattermapbox(
                lat=database_df["latitude"],
                lon=database_df["longitude"],
                mode="markers",
                marker=go.scattermapbox.Marker(
                    size=database_df[
                        "size"
                    ],  # Use the robustly calculated 'size' column
                    color=database_df[colour_by] if colour_by in database_df else None,
                    colorscale="Viridis",
                    showscale=True if colour_by in database_df else False,
                    colorbar=dict(
                        title=colour_by if colour_by in database_df else "",
                        len=0.92,  # Make colorbar 92% of the plot height
                        y=0.45,  # Position colorbar at 45% from the bottom
                        yanchor="middle",
                    ),
                ),
                text=database_df.get("record_name", ""),  # Use .get for record_name
                customdata=custom_data_for_db,
                hovertemplate=(
                    "<b>%{text}</b><br><br>"
                    + "Deepest Depth (m): %{customdata[0]}<br>"
                    + "Vs30 (m/s): %{customdata[1]}<br>"
                    + "Vs30 Log Resid: %{customdata[2]}"
                    + "<extra></extra>"
                ),
                name="NZGD Records",
            )
        )

    # Load GeoNet station data
    # First check for user-uploaded file, otherwise use default
    user_geonet_file = get_user_geonet_file_path()
    geonet_file_path = (
        user_geonet_file
        if user_geonet_file
        else instance_path / "geoNet_stats+2023-06-28.ll"
    )

    try:
        # Skip the first line if it's a comment, and handle potential comments within the file
        geonet_df = pd.read_csv(
            geonet_file_path,
            delim_whitespace=True,
            header=None,
            names=["longitude", "latitude", "code_name"],
            comment="/",  # Treat lines starting with / as comments
        )
        # Ensure longitude and latitude are numeric, coercing errors to NaN
        geonet_df["longitude"] = pd.to_numeric(geonet_df["longitude"], errors="coerce")
        geonet_df["latitude"] = pd.to_numeric(geonet_df["latitude"], errors="coerce")
        # Drop rows with NaN in longitude or latitude, which might result from parsing issues or comments
        geonet_df.dropna(subset=["longitude", "latitude"], inplace=True)

    except FileNotFoundError:
        geonet_df = pd.DataFrame(columns=["longitude", "latitude", "code_name"])
        flask.current_app.logger.warning(
            f"GeoNet station file not found: {geonet_file_path}"
        )
    except pd.errors.EmptyDataError:  # Handles empty file or file with only comments
        geonet_df = pd.DataFrame(columns=["longitude", "latitude", "code_name"])
        flask.current_app.logger.warning(
            f"GeoNet station file is empty or could not be parsed: {geonet_file_path}"
        )

    # Add GeoNet stations to the map if data is available
    if not geonet_df.empty:
        map_fig.add_trace(
            go.Scattermapbox(
                lat=geonet_df["latitude"],
                lon=geonet_df["longitude"],
                mode="markers",
                marker=go.scattermapbox.Marker(
                    size=12,
                    color="deeppink",
                    opacity=0.8,
                    symbol="circle",
                ),
                text=geonet_df["code_name"],
                hoverinfo="text",
                name="GeoNet Stations",
                showlegend=True,
            )
        )

    # Update map layout
    map_fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=5,
            center=dict(lat=centre_lat, lon=centre_lon),
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=True,
        legend_title_text="Legend",
    )

    # Create an interactive histogram using Plotly
    if not database_df.empty and hist_by in database_df.columns:
        hist_plot = px.histogram(database_df, x=hist_by)
    else:
        hist_plot = go.Figure().update_layout(
            title_text=f"No data for {hist_by}", xaxis_title=hist_by
        )

    hist_description_text = (
        f"Histogram of {hist_by}, showing {len(database_df)} records"
    )

    # If plotting the vs30_log_residual, add a note about the log residual calculation
    if hist_by == "vs30_log_residual":
        residual_description_text = r"Note: Vs30 residuals are given by \(\mathrm{\log(SPT_{Vs30}) - \log(Foster2019_{Vs30})} \)"
    else:
        residual_description_text = ""

    col_names_to_display = [
        "record_name",
        "nzgd_id",
        "cpt_id",
        "vs30",
        "vs30_stddev",
        "type_prefix",
        "original_reference",
        "investigation_date",
        "published_date",
        "latitude",
        "longitude",
        "model_vs30_foster_2019",
        "model_vs30_stddev_foster_2019",
        "model_gwl_westerhoff_2019",
        "cpt_tip_net_area_ratio",
        "measured_gwl",
        "deepest_depth",
        "shallowest_depth",
        "region",
        "district",
        "suburb",
        "city",
        "vs30_log_residual",
        "gwl_residual",
        "spt_efficiency",
    ]
    col_names_to_display_str = ", ".join(col_names_to_display)

    # Render the map and data in an HTML template
    return flask.render_template(
        "views/index.html",
        date_of_last_nzgd_retrieval=date_of_last_nzgd_retrieval,
        map=map_fig.to_html(  # Use map_fig here
            full_html=False,  # Embed only the necessary map HTML
            include_plotlyjs=False,  # Exclude Plotly.js library (assume it's loaded separately)
            default_height="85vh",  # Set the map height
            config={
                "scrollZoom": True,  # Enable scroll zoom in the HTML config too
                "displayModeBar": True,  # Display the mode bar with additional controls
                "modeBarButtonsToAdd": [
                    "zoomIn",
                    "zoomOut",
                ],  # Add explicit zoom buttons
            },
        ),
        selected_vs30_correlation=vs30_correlation,  # Pass the selected vs30_correlation for the template
        selected_spt_vs_correlation=spt_vs_correlation,
        selected_cpt_vs_correlation=cpt_vs_correlation,
        query=query,  # Pass the query back for persistence in UI
        vs30_correlations=vs30_correlations,  # Pass all vs30_correlations for UI dropdown
        spt_vs_correlations=spt_vs_correlations,
        cpt_vs_correlations=cpt_vs_correlations,
        num_records=len(database_df),
        colour_by=colour_by,
        hist_by=hist_by,
        colour_variables=[
            ("vs30", "Inferred Vs30 from data"),
            ("type_number_code", "Type of record"),
            ("vs30_log_residual", "log residual with Foster et al. (2019)"),
            ("deepest_depth", "Record's deepest depth"),
            ("vs30_stddev", "Vs30 standard deviation inferred from data"),
            ("model_vs30_foster_2019", "Vs30 from Foster et al. (2019)"),
            (
                "model_vs30_stddev_foster_2019",
                "Vs30 standard deviation from Foster et al. (2019)",
            ),
            ("shallowest_depth", "Record's shallowest depth"),
            ("measured_gwl", "Measured groundwater level"),
            (
                "model_gwl_westerhoff_2019",
                "Groundwater level from Westerhoff et al. (2019)",
            ),
        ],
        hist_plot=hist_plot.to_html(
            full_html=False,  # Embed only the necessary map HTML
            include_plotlyjs=False,  # Exclude Plotly.js library (assume it's loaded separately)
            default_height="85vh",  # Set the map height
        ),
        marker_size_description_text=marker_size_description_text,
        hist_description_text=hist_description_text,
        residual_description_text=residual_description_text,
        col_names_to_display=col_names_to_display_str,
    )


@bp.route("/spt/<record_name>", methods=["GET"])
def spt_record(record_name: str):
    """
    Render the details page for a given SPT record.

    Parameters
    ----------
    record_name : str
        The name of the record to display.

    Returns
    -------
    The rendered HTML template for the SPT record page.
    """

    # Access the instance folder for application-specific data
    instance_path = Path(flask.current_app.instance_path)

    nzgd_id = int(record_name.split("_")[1])

    with sqlite3.connect(instance_path / constants.database_file_name) as conn:
        spt_measurements_df = query_sqlite_db.spt_measurements_for_one_nzgd(
            nzgd_id, conn
        )
        spt_soil_df = query_sqlite_db.spt_soil_types_for_one_nzgd(nzgd_id, conn)
        vs30s_df = query_sqlite_db.spt_vs30s_for_one_nzgd_id(nzgd_id, conn)

    type_prefix_to_folder = {"CPT": "cpt", "SCPT": "scpt", "BH": "borehole"}

    path_to_files = (
        Path(type_prefix_to_folder[vs30s_df["type_prefix"][0]])
        / vs30s_df["region"][0]
        / vs30s_df["district"][0]
        / vs30s_df["city"][0]
        / vs30s_df["suburb"][0]
        / vs30s_df["record_name"][0]
    )
    url_str = constants.source_files_base_url + str(path_to_files)
    vs30s_df["estimate_number"] = np.arange(1, len(vs30s_df) + 1)

    spt_efficiency = vs30s_df["spt_efficiency"][0]
    if spt_efficiency is None:
        spt_efficiency = "Not available"
    elif isinstance(spt_efficiency, float):
        spt_efficiency = f"{spt_efficiency:.0f}%"

    spt_borehole_diameter = vs30s_df["spt_borehole_diameter"][0]
    if spt_borehole_diameter is None:
        spt_borehole_diameter = "Not available"
    elif isinstance(spt_borehole_diameter, float):
        spt_borehole_diameter = f"{spt_borehole_diameter:.2f}"

    measured_gwl = vs30s_df["measured_gwl"][0]
    if measured_gwl is None:
        measured_gwl = "Not available"
    elif isinstance(measured_gwl, float):
        measured_gwl = f"{measured_gwl:.2f}"

    model_gwl_westerhoff_2019 = vs30s_df["model_gwl_westerhoff_2019"][0]
    if model_gwl_westerhoff_2019 is None:
        model_gwl_westerhoff_2019 = "Not available"
    elif isinstance(model_gwl_westerhoff_2019, float):
        model_gwl_westerhoff_2019 = f"{model_gwl_westerhoff_2019:.2f}"

    model_vs30_foster_2019 = vs30s_df["model_vs30_foster_2019"][0]
    if model_vs30_foster_2019 is None:
        model_vs30_foster_2019 = "Not available"
    elif isinstance(model_vs30_foster_2019, float):
        model_vs30_foster_2019 = f"{model_vs30_foster_2019:.2f}"

    model_vs30_stddev_foster_2019 = vs30s_df["model_vs30_stddev_foster_2019"][0]
    if model_vs30_stddev_foster_2019 is None:
        model_vs30_stddev_foster_2019 = "Not available"
    elif isinstance(model_vs30_stddev_foster_2019, float):
        model_vs30_stddev_foster_2019 = f"{model_vs30_stddev_foster_2019:.2f}"

    spt_vs30_calculation_used_efficiency = vs30s_df[
        "spt_vs30_calculation_used_soil_info"
    ][0]
    if spt_vs30_calculation_used_efficiency == 0:
        spt_vs30_calculation_used_efficiency = "no"
    elif spt_vs30_calculation_used_efficiency == 1:
        spt_vs30_calculation_used_efficiency = "yes"

    spt_vs30_calculation_used_soil_info = vs30s_df[
        "spt_vs30_calculation_used_efficiency"
    ][0]
    if spt_vs30_calculation_used_soil_info == 0:
        spt_vs30_calculation_used_soil_info = "no"
    elif spt_vs30_calculation_used_soil_info == 1:
        spt_vs30_calculation_used_soil_info = "yes"

    spt_measurements_df.rename(
        columns={"n": "Number of blows", "depth": "Depth (m)"}, inplace=True
    )

    # Plot the SPT data. line_shape is set to "vhv" to create a step plot with the correct orientation for vertical depth.
    spt_plot = px.line(
        spt_measurements_df, x="Number of blows", y="Depth (m)", line_shape="vhv"
    )
    # Invert the y-axis
    spt_plot.update_layout(yaxis=dict(autorange="reversed"))

    return flask.render_template(
        "views/spt_record.html",
        record_details=vs30s_df.to_dict(
            orient="records"
        ),  # Pass DataFrame as list of dictionaries
        spt_data=spt_measurements_df.to_dict(orient="records"),
        soil_type=spt_soil_df.to_dict(orient="records"),
        spt_plot=spt_plot.to_html(),
        url_str=url_str,
        spt_efficiency=spt_efficiency,
        spt_borehole_diameter=spt_borehole_diameter,
        measured_gwl=measured_gwl,
        model_vs30_foster_2019=model_vs30_foster_2019,
        model_vs30_stddev_foster_2019=model_vs30_stddev_foster_2019,
        model_gwl_westerhoff_2019=model_gwl_westerhoff_2019,
        max_depth=spt_measurements_df["Depth (m)"].max(),
        min_depth=spt_measurements_df["Depth (m)"].min(),
        spt_vs30_calculation_used_efficiency=spt_vs30_calculation_used_efficiency,
        spt_vs30_calculation_used_soil_info=spt_vs30_calculation_used_soil_info,
    )


@bp.route("/cpt/<record_name>", methods=["GET"])
def cpt_record(record_name: str):
    """
    Render the details page for a given CPT record.

    Parameters
    ----------
    record_name : str
        The name of the record to display.

    Returns
    -------
    The rendered HTML template for the CPT record page.
    """

    # Access the instance folder for application-specific data
    instance_path = Path(flask.current_app.instance_path)

    nzgd_id = int(record_name.split("_")[1])

    with sqlite3.connect(instance_path / constants.database_file_name) as conn:
        cpt_measurements_df = query_sqlite_db.cpt_measurements_for_one_nzgd(
            nzgd_id, conn
        )
        vs30s_df = query_sqlite_db.cpt_vs30s_for_one_nzgd_id(nzgd_id, conn)

    type_prefix_to_folder = {"CPT": "cpt", "SCPT": "scpt", "BH": "borehole"}
    path_to_files = (
        Path(type_prefix_to_folder[vs30s_df["type_prefix"][0]])
        / vs30s_df["region"][0]
        / vs30s_df["district"][0]
        / vs30s_df["city"][0]
        / vs30s_df["suburb"][0]
        / vs30s_df["record_name"][0]
    )
    url_str = constants.source_files_base_url + str(path_to_files)
    vs30s_df["estimate_number"] = np.arange(1, len(vs30s_df) + 1)

    tip_net_area_ratio = vs30s_df["cpt_tip_net_area_ratio"][0]
    if tip_net_area_ratio is None:
        tip_net_area_ratio = "Not available"
    elif isinstance(tip_net_area_ratio, float):
        tip_net_area_ratio = f"{tip_net_area_ratio:.2f}"

    measured_gwl = vs30s_df["measured_gwl"][0]
    if measured_gwl is None:
        measured_gwl = "Not available"
    elif isinstance(measured_gwl, float):
        measured_gwl = f"{measured_gwl:.2f}"

    model_gwl_westerhoff_2019 = vs30s_df["model_gwl_westerhoff_2019"][0]
    if model_gwl_westerhoff_2019 is None:
        model_gwl_westerhoff_2019 = "Not available"
    elif isinstance(model_gwl_westerhoff_2019, float):
        model_gwl_westerhoff_2019 = f"{model_gwl_westerhoff_2019:.2f}"

    model_vs30_foster_2019 = vs30s_df["model_vs30_foster_2019"][0]
    if model_vs30_foster_2019 is None:
        model_vs30_foster_2019 = "Not available"
    elif isinstance(model_vs30_foster_2019, float):
        model_vs30_foster_2019 = f"{model_vs30_foster_2019:.2f}"

    model_vs30_stddev_foster_2019 = vs30s_df["model_vs30_stddev_foster_2019"][0]
    if model_vs30_stddev_foster_2019 is None:
        model_vs30_stddev_foster_2019 = "Not available"
    elif isinstance(model_vs30_stddev_foster_2019, float):
        model_vs30_stddev_foster_2019 = f"{model_vs30_stddev_foster_2019:.2f}"

    type_prefix = vs30s_df["type_prefix"][0]
    if type_prefix is None:
        type_prefix = "Not available"
    elif isinstance(type_prefix, str):
        type_prefix = f"{type_prefix}"

    ## Only show Vs30 values for correlations that could be used given the depth of the record
    max_depth_for_record = vs30s_df["deepest_depth"].unique()[0]

    if max_depth_for_record < 5:
        vs30_correlation_explanation_text = (
            f"Unable to estimate a Vs30 value from {record_name} as it has a maximum depth "
            f"of {max_depth_for_record} m, while depths of at least 10 m and 5 m are required for "
            "the Boore et al. (2004) and Boore et al. (2011) Vs to Vs30 correlations, respectively."
        )
        show_vs30_values = False

    elif 5 <= max_depth_for_record < 10:
        vs30_correlation_explanation_text = (
            f"{record_name} has a maximum depth of {max_depth_for_record:.2f} m so only the Boore et al. (2011) "
            "Vs to Vs30 correlation can be used as it requires a depth of at least 5 m, while the "
            "Boore et al. (2004) correlation requires a depth of at least 10 m."
        )
        show_vs30_values = True
        vs30s_df = vs30s_df[vs30s_df["vs_to_vs30_correlation"] == "boore_2011"]

    else:
        vs30_correlation_explanation_text = (
            f"{record_name} has a maximum depth of {max_depth_for_record:.2f} m so both the Boore et al. (2004) "
            "and Boore et al. (2011) Vs to Vs30 correlations can be used, as they require depths of at least "
            "10 m and 5 m, respectively."
        )
        show_vs30_values = True

    # Plot the CPT data as a subplot with 1 row and 3 columns
    fig = make_subplots(rows=1, cols=3)

    fig.add_trace(
        go.Scatter(x=cpt_measurements_df["qc"], y=cpt_measurements_df["depth"]),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=cpt_measurements_df["fs"], y=cpt_measurements_df["depth"]),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=cpt_measurements_df["u2"], y=cpt_measurements_df["depth"]),
        row=1,
        col=3,
    )

    fig.update_yaxes(title_text="Depth (m)", autorange="reversed", row=1, col=1)
    fig.update_yaxes(title_text="Depth (m)", autorange="reversed", row=1, col=2)
    fig.update_yaxes(title_text="Depth (m)", autorange="reversed", row=1, col=3)

    fig.update_xaxes(title_text=r"Cone resistance, qc (Mpa)", row=1, col=1)
    fig.update_xaxes(title_text="Sleeve friction, fs (Mpa)", row=1, col=2)
    fig.update_xaxes(title_text="Pore pressure, u2 (Mpa)", row=1, col=3)

    fig.update_layout(showlegend=False)

    return flask.render_template(
        "views/cpt_record.html",
        record_details=vs30s_df.to_dict(
            orient="records"
        ),  # Pass DataFrame as list of dictionaries
        cpt_plot=fig.to_html(),
        vs30_correlation_explanation_text=vs30_correlation_explanation_text,
        show_vs30_values=show_vs30_values,
        url_str=url_str,
        tip_net_area_ratio=tip_net_area_ratio,
        measured_gwl=measured_gwl,
        model_gwl_westerhoff_2019=model_gwl_westerhoff_2019,
        record_name=record_name,
        model_vs30_foster_2019=model_vs30_foster_2019,
        model_vs30_stddev_foster_2019=model_vs30_stddev_foster_2019,
        type_prefix=type_prefix,
    )


def remove_file(file_path):
    """Delete the specified file."""
    try:
        os.remove(file_path)
        print(f"Deleting temporary file: {file_path}")
    except OSError as e:
        print(f"Error: {file_path} : {e.strerror}")


@bp.route("/download_cpt_data/<filename>")
def download_cpt_data(filename):
    """Serve a file from the instance path for download and delete it afterwards."""
    instance_path = Path(flask.current_app.instance_path)

    nzgd_id = int(filename.split("_")[1])
    with sqlite3.connect(instance_path / constants.database_file_name) as conn:
        cpt_measurements_df = query_sqlite_db.cpt_measurements_for_one_nzgd(
            nzgd_id, conn
        )

    cpt_measurements_df.rename(
        columns={
            "depth": "depth_(m)",
            "qc": "cone_resistance_qc_(Mpa)",
            "fs": "sleeve_friction_fs_(Mpa)",
            "u2": "pore_pressure_u2_(Mpa)",
        },
        inplace=True,
    )

    # Create a temporary CSV file containing the CPT data
    download_buffer = StringIO()

    cpt_measurements_df[
        [
            "depth_(m)",
            "cone_resistance_qc_(Mpa)",
            "sleeve_friction_fs_(Mpa)",
            "pore_pressure_u2_(Mpa)",
        ]
    ].to_csv(download_buffer, index=False)
    response = flask.make_response(download_buffer.getvalue())
    response.mimetype = "text/csv"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"

    return response


@bp.route("/download_spt_data/<filename>")
def download_spt_data(filename):
    """Serve SPT data as a downloadable CSV file using an in-memory buffer."""
    instance_path = Path(flask.current_app.instance_path)

    nzgd_id = int(filename.split("_")[1])
    with sqlite3.connect(instance_path / constants.database_file_name) as conn:
        spt_measurements_df = query_sqlite_db.spt_measurements_for_one_nzgd(
            nzgd_id, conn
        )

    # Create a buffer for the CSV data
    download_buffer = StringIO()

    # Rename columns and write to buffer
    spt_measurements_df.rename(
        columns={"depth": "depth_m", "n": "number_of_blows"}, inplace=True
    )
    spt_measurements_df[["depth_m", "number_of_blows"]].to_csv(
        download_buffer, index=False
    )

    # Create response directly from the buffer
    response = flask.make_response(download_buffer.getvalue())
    response.mimetype = "text/csv"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"

    return response


@bp.route("/download_spt_soil_types/<filename>")
def download_spt_soil_types(filename):
    """Serve SPT soil types as a downloadable CSV file using an in-memory buffer."""
    instance_path = Path(flask.current_app.instance_path)

    nzgd_id = int(filename.split("_")[1])
    with sqlite3.connect(instance_path / constants.database_file_name) as conn:
        spt_soil_types_df = query_sqlite_db.spt_soil_types_for_one_nzgd(nzgd_id, conn)

    spt_soil_types_df.rename(
        columns={"top_depth": "depth_at_layer_top_m"}, inplace=True
    )

    # Create a buffer for the CSV data
    download_buffer = StringIO()

    # Write the data to the buffer
    spt_soil_types_df[["depth_at_layer_top_m", "soil_type"]].to_csv(
        download_buffer, index=False
    )

    # Create response directly from the buffer
    response = flask.make_response(download_buffer.getvalue())
    response.mimetype = "text/csv"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"

    return response


@bp.route("/query_help", methods=["GET"])
def query_help():
    """
    Display a help page for constructing queries.
    """
    # Access the instance folder for application-specific data
    instance_path = Path(flask.current_app.instance_path)

    # Get column names to display on the query help page
    col_names_to_display = [
        "record_name",
        "nzgd_id",
        "cpt_id",
        "vs30",
        "vs30_stddev",
        "type_prefix",
        "original_reference",
        "investigation_date",
        "published_date",
        "latitude",
        "longitude",
        "model_vs30_foster_2019",
        "model_vs30_stddev_foster_2019",
        "model_gwl_westerhoff_2019",
        "cpt_tip_net_area_ratio",
        "measured_gwl",
        "deepest_depth",
        "shallowest_depth",
        "region",
        "district",
        "suburb",
        "city",
        "vs30_log_residual",
        "gwl_residual",
        "spt_efficiency",
        "spt_borehole_diameter",
    ]
    col_names_to_display_str = ", ".join(col_names_to_display)

    return flask.render_template(
        "views/query_help.html", col_names_to_display=col_names_to_display_str
    )


@bp.route("/validate", methods=["GET"])
def validate():
    """
    Validate a query string against a dummy DataFrame.
    """
    query = flask.request.args.get("query", None)
    if not query:
        return ""

    # Create a dummy dataframe to ensure the column names are present
    dummy_df = pd.DataFrame(
        columns=[
            "cpt_id",
            "nzgd_id",
            "vs30",
            "vs30_stddev",
            "type_prefix",
            "original_reference",
            "investigation_date",
            "published_date",
            "latitude",
            "longitude",
            "model_vs30_foster_2019",
            "model_vs30_stddev_foster_2019",
            "model_gwl_westerhoff_2019",
            "cpt_tip_net_area_ratio",
            "measured_gwl",
            "deepest_depth",
            "shallowest_depth",
            "region",
            "district",
            "suburb",
            "city",
            "record_name",
            "vs30_log_residual",
            "gwl_residual",
            "spt_efficiency",
            "spt_borehole_diameter",
        ]
    )
    try:
        dummy_df.query(query)
    except (
        ValueError,
        SyntaxError,
        UnboundLocalError,
        pd.errors.UndefinedVariableError,
    ) as e:
        return flask.render_template("error.html", error=e)
    return ""


@bp.route("/upload_geonet", methods=["POST"])
def upload_geonet():
    """Handle GeoNet station file upload."""
    if "geonet_file" not in flask.request.files:
        return flask.redirect(flask.request.referrer)

    file = flask.request.files["geonet_file"]

    # If the user does not select a file, the browser submits an empty file
    if file.filename == "":
        return flask.redirect(flask.request.referrer)

    if file and allowed_file(file.filename):
        # Generate a unique filename to prevent conflicts
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        file_path = Path(UPLOAD_FOLDER) / filename

        # Save the file
        file.save(file_path)

        # Store the filename in the session
        flask.session["user_geonet_file"] = filename

    return flask.redirect(flask.request.referrer)


@bp.route("/clear_geonet", methods=["POST"])
def clear_geonet():
    """Remove the user's uploaded GeoNet file."""
    if "user_geonet_file" in flask.session:
        # Delete the file if it exists
        file_path = Path(UPLOAD_FOLDER) / flask.session["user_geonet_file"]
        if file_path.exists():
            os.remove(file_path)

        # Remove from session
        del flask.session["user_geonet_file"]

    return flask.redirect(flask.request.referrer)
