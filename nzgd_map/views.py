"""
The views module defines the Flask views (web pages) for the application.
Each view is a function that returns an HTML template to render in the browser.
"""

import os
import sqlite3
from collections import OrderedDict
from io import StringIO
from pathlib import Path

import flask
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import after_this_request
from plotly.subplots import make_subplots

from . import query_sqlite_db

# Create a Flask Blueprint for the views
bp = flask.Blueprint("views", __name__)


@bp.route("/", methods=["GET"])
def index():
    """Serve the standard index page."""
    # Access the instance folder for application-specific data
    instance_path = Path(flask.current_app.instance_path)

    with open(instance_path / "date_of_last_nzgd_retrieval.txt", "r") as file:
        date_of_last_nzgd_retrieval = file.readline()

    # Retrieve selected vs30 correlation. If no selection, default to "boore_2004"
    vs30_correlation = flask.request.args.get("vs30_correlation", default="boore_2004")

    # Retrieve selected spt_vs_correlation. If no selection, default to "brandenberg_2010"
    spt_vs_correlation = flask.request.args.get(
        "spt_vs_correlation", default="brandenberg_2010"
    )

    # Retrieve selected cpt_vs_correlation. If no selection, default to "andrus_2007_pleistocene".
    cpt_vs_correlation = flask.request.args.get(
        "cpt_vs_correlation", default="andrus_2007_pleistocene"
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

    with sqlite3.connect(instance_path / "extracted_nzgd.db") as conn:
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

    # Calculate the center of the map for visualization
    centre_lat = database_df["latitude"].mean()
    centre_lon = database_df["longitude"].mean()

    ## Make map marker sizes proportional to the absolute value of the Vs30 log residual.
    ## For records where the Vs30 log residual is unavailable, use the median of absolute value of the Vs30 log residuals.
    database_df["size"] = (
        database_df["vs30_log_residual"]
        .abs()
        .fillna(database_df["vs30_log_residual"].abs().median().round(1))
    )
    marker_size_description_text = r"Marker size indicates the magnitude of the Vs30 log residual, given by \(\mathrm{|(\log(SPT_{Vs30}) - \log(Foster2019_{Vs30})|}\)"

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

    # Create an interactive scatter map using Plotly
    map = px.scatter_map(
        database_df,
        lat="latitude",  # Column specifying latitude
        lon="longitude",  # Column specifying longitude
        color=colour_by,  # Column specifying marker color
        hover_name=database_df["record_name"],
        zoom=5,
        size="size",  # Marker size
        center={"lat": centre_lat, "lon": centre_lon},  # Map center
        hover_data=OrderedDict(
            [  # Used to order the items in hover data (but lat and long are always first)
                ("deepest_depth (m)", ":.2f"),
                ("Vs30 (m/s)", True),
                ("Vs30_log_resid", True),
                ("size", False),
                ("vs30", False),
                ("vs30_log_residual", False),
            ]
        ),
    )

    # Create an interactive histogram using Plotly
    hist_plot = px.histogram(database_df, x=hist_by)
    hist_description_text = (
        f"Histogram of {hist_by}, showing {len(database_df)} records"
    )

    # If plotting the vs30_log_residual, add a note about the log residual calculation
    if hist_by == "vs30_log_residual":
        residual_description_text = r"Note: Vs30 residuals are given by \(\mathrm{\log(SPT_{Vs30}) - \log(Foster2019_{Vs30})} \)"
    else:
        residual_description_text = ""

    all_df_column_names = database_df.columns.tolist()
    col_names_to_exclude = [
        "index",
        "type",
        "link_to_pdf",
        "nzgd_url",
        "size",
        "total_depth",
        "original_reference",
        "error_from_data",
    ]
    col_names_to_display = [
        col_name
        for col_name in all_df_column_names
        if col_name not in col_names_to_exclude
    ]
    col_names_to_display_str = ", ".join(col_names_to_display)

    # Render the map and data in an HTML template
    return flask.render_template(
        "views/index.html",
        date_of_last_nzgd_retrieval=date_of_last_nzgd_retrieval,
        map=map.to_html(
            full_html=False,  # Embed only the necessary map HTML
            include_plotlyjs=False,  # Exclude Plotly.js library (assume it's loaded separately)
            default_height="85vh",  # Set the map height
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

    with sqlite3.connect(instance_path / "extracted_nzgd.db") as conn:
        spt_measurements_df = query_sqlite_db.spt_measurements_for_one_nzgd(
            nzgd_id, conn
        )
        spt_soil_df = query_sqlite_db.spt_soil_types_for_one_nzgd(nzgd_id, conn)
        vs30s_df = query_sqlite_db.spt_vs30s_for_one_nzgd_id(nzgd_id, conn)

    type_prefix_to_folder = {"CPT": "cpt", "SCPT": "scpt", "BH": "borehole"}
    url_str_start = "https://quakecoresoft.canterbury.ac.nz/nzgd_source_files/"

    path_to_files = (
        Path(type_prefix_to_folder[vs30s_df["type_prefix"][0]])
        / vs30s_df["region"][0]
        / vs30s_df["district"][0]
        / vs30s_df["city"][0]
        / vs30s_df["suburb"][0]
        / vs30s_df["record_name"][0]
    )
    url_str = url_str_start + str(path_to_files)
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

    with sqlite3.connect(instance_path / "extracted_nzgd.db") as conn:
        cpt_measurements_df = query_sqlite_db.cpt_measurements_for_one_nzgd(
            nzgd_id, conn
        )
        vs30s_df = query_sqlite_db.cpt_vs30s_for_one_nzgd_id(nzgd_id, conn)

    type_prefix_to_folder = {"CPT": "cpt", "SCPT": "scpt", "BH": "borehole"}
    url_str_start = "https://quakecoresoft.canterbury.ac.nz/nzgd_source_files/"
    path_to_files = (
        Path(type_prefix_to_folder[vs30s_df["type_prefix"][0]])
        / vs30s_df["region"][0]
        / vs30s_df["district"][0]
        / vs30s_df["city"][0]
        / vs30s_df["suburb"][0]
        / vs30s_df["record_name"][0]
    )
    url_str = url_str_start + str(path_to_files)
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
    with sqlite3.connect(instance_path / "extracted_nzgd.db") as conn:
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
    with sqlite3.connect(instance_path / "extracted_nzgd.db") as conn:
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
    with sqlite3.connect(instance_path / "extracted_nzgd.db") as conn:
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
            "efficiency",
            "borehole_diameter",
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
