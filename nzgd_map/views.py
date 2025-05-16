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

from . import constants, query_sqlite_db

# Create a Flask Blueprint for the views
bp = flask.Blueprint("views", __name__)


@bp.route("/", methods=["GET", "POST"])  # Allow POST requests
def index():
    """Serve the standard index page."""
    # Access the instance folder for application-specific data
    instance_path = Path(flask.current_app.instance_path)

    with open(instance_path / constants.last_retrieval_date_file_name, "r") as file:
        date_of_last_nzgd_retrieval = file.readline().strip()

    # --- Start: Load uploaded_df from session ---
    uploaded_df = None
    if "uploaded_locations" in flask.session:
        try:
            uploaded_df_json = flask.session["uploaded_locations"]
            # Use pandas to read the JSON string back into a DataFrame
            # Ensure orient matches the one used in to_json()
            uploaded_df = pd.read_json(StringIO(uploaded_df_json), orient="split")

            # Ensure correct dtypes after reading from JSON, especially for lat/lon
            if "latitude" in uploaded_df.columns:
                uploaded_df["latitude"] = pd.to_numeric(
                    uploaded_df["latitude"], errors="coerce"
                )
            if "longitude" in uploaded_df.columns:
                uploaded_df["longitude"] = pd.to_numeric(
                    uploaded_df["longitude"], errors="coerce"
                )

            # Drop rows if conversion to numeric resulted in NaNs for critical columns
            uploaded_df.dropna(subset=["latitude", "longitude"], inplace=True)

            # Filter out rows with lat/lon outside valid geographic ranges after loading from session
            if not uploaded_df.empty:  # Check before accessing columns
                valid_lat = (uploaded_df["latitude"] >= -90) & (
                    uploaded_df["latitude"] <= 90
                )
                valid_lon = (uploaded_df["longitude"] >= -180) & (
                    uploaded_df["longitude"] <= 180
                )
                uploaded_df = uploaded_df[valid_lat & valid_lon]
                if uploaded_df.empty:
                    flask.flash(
                        "Uploaded data from session contained no valid geolocations after range filtering.",
                        "warning",
                    )

            if (
                uploaded_df.empty
            ):  # If all rows became NaN and were dropped or filtered out
                uploaded_df = None  # Ensure uploaded_df is None if it becomes empty
                if (
                    "uploaded_locations" in flask.session
                ):  # Clean up session if df becomes empty
                    del flask.session["uploaded_locations"]
                # flask.flash("Uploaded data from session was invalid or empty after type conversion.", "warning") # Original message, can be removed or kept
            else:
                flask.flash(
                    f"Loaded {len(uploaded_df)} valid locations from session.", "info"
                )
                # Re-create hovertext if not present, as it might not survive JSON serialization well depending on content
                if (
                    "hovertext" not in uploaded_df.columns and not uploaded_df.empty
                ):  # Check not empty again
                    uploaded_df["hovertext"] = "Uploaded: " + uploaded_df.index.astype(
                        str
                    )
        except Exception as e:
            uploaded_df = None  # Error during deserialization
            if "uploaded_locations" in flask.session:  # Check before deleting
                del flask.session["uploaded_locations"]  # Clear corrupted data
            flask.flash(
                f"Error loading uploaded locations from session: {e}. Please re-upload if needed.",
                "error",
            )
    # --- End: Load uploaded_df from session ---

    if flask.request.method == "POST":
        vs30_correlation = flask.request.form.get(
            "vs30_correlation", default=constants.default_vs_to_vs30_correlation
        )
        spt_vs_correlation = flask.request.form.get(
            "spt_vs_correlation", default=constants.default_spt_to_vs_correlation
        )
        cpt_vs_correlation = flask.request.form.get(
            "cpt_vs_correlation", default=constants.default_cpt_to_vs_correlation
        )
        colour_by = flask.request.form.get("colour_by", default="vs30")
        hist_by = flask.request.form.get("hist_by", default="vs30_log_residual")
        query = flask.request.form.get("query")

        # --- Start: Handle CSV file upload ---
        if "csv_file" in flask.request.files:
            file = flask.request.files["csv_file"]
            if file.filename != "":  # A file was actually selected for upload
                if file.filename.endswith(".csv"):
                    try:
                        # Read the CSV file into a pandas DataFrame
                        csv_data = StringIO(file.stream.read().decode("UTF8"))
                        temp_uploaded_df = pd.read_csv(csv_data)

                        # Validate required columns
                        if (
                            "latitude" not in temp_uploaded_df.columns
                            or "longitude" not in temp_uploaded_df.columns
                        ):
                            flask.flash(
                                "CSV must contain 'latitude' and 'longitude' columns.",
                                "error",
                            )
                        else:
                            # Ensure latitude and longitude are numeric, drop rows where they are not
                            temp_uploaded_df["latitude"] = pd.to_numeric(
                                temp_uploaded_df["latitude"], errors="coerce"
                            )
                            temp_uploaded_df["longitude"] = pd.to_numeric(
                                temp_uploaded_df["longitude"], errors="coerce"
                            )
                            temp_uploaded_df.dropna(
                                subset=["latitude", "longitude"], inplace=True
                            )

                            # Filter out rows with lat/lon outside valid geographic ranges
                            if (
                                not temp_uploaded_df.empty
                            ):  # Check before accessing columns
                                valid_lat = (temp_uploaded_df["latitude"] >= -90) & (
                                    temp_uploaded_df["latitude"] <= 90
                                )
                                valid_lon = (temp_uploaded_df["longitude"] >= -180) & (
                                    temp_uploaded_df["longitude"] <= 180
                                )
                                temp_uploaded_df = temp_uploaded_df[
                                    valid_lat & valid_lon
                                ]
                                if temp_uploaded_df.empty and (
                                    "latitude" in temp_uploaded_df.columns
                                ):  # Check if it became empty due to filtering
                                    flask.flash(
                                        "CSV contained no locations within valid geographic ranges after processing.",
                                        "warning",
                                    )

                            if not temp_uploaded_df.empty:
                                uploaded_df = temp_uploaded_df
                                # --- Start: Save uploaded_df to session ---
                                flask.session["uploaded_locations"] = (
                                    uploaded_df.to_json(orient="split")
                                )
                                # --- End: Save uploaded_df to session ---
                                flask.flash(
                                    "CSV file processed and loaded successfully.",
                                    "success",
                                )
                    except Exception as e:
                        flask.flash(f"Error processing new CSV file: {e}", "error")
                else:  # File selected, but not a CSV
                    flask.flash(
                        "Invalid file type. Please upload a CSV file. Previous uploaded data (if any) is retained.",
                        "warning",
                    )
        # --- End: Handle CSV file upload ---
    else:  # GET request
        vs30_correlation = flask.request.args.get(
            "vs30_correlation", default=constants.default_vs_to_vs30_correlation
        )
        spt_vs_correlation = flask.request.args.get(
            "spt_vs_correlation", default=constants.default_spt_to_vs_correlation
        )
        cpt_vs_correlation = flask.request.args.get(
            "cpt_vs_correlation", default=constants.default_cpt_to_vs_correlation
        )
        colour_by = flask.request.args.get("colour_by", default="vs30")
        hist_by = flask.request.args.get("hist_by", default="vs30_log_residual")
        query = flask.request.args.get("query")

    # Ensure query is None if it's an empty string after stripping, to match behavior of default=None
    if query is not None:
        query = query.strip()
        if not query:
            query = None

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

    # Initialize marker_size_description_text, will be updated if database_df is not empty
    marker_size_description_text = ""

    # Determine map center and zoom
    default_nz_lat, default_nz_lon = -41.2865, 174.7762
    centre_lat, centre_lon, current_map_zoom = (
        default_nz_lat,
        default_nz_lon,
        4,
    )  # Base defaults

    if not database_df.empty:
        centre_lat = database_df["latitude"].mean()
        centre_lon = database_df["longitude"].mean()
        current_map_zoom = 5

        # Define marker_size_description_text here as database_df is available
        if "vs30_log_residual" in database_df.columns:
            abs_residuals = database_df["vs30_log_residual"].abs()
            median_abs_residual = abs_residuals.median()

            # Ensure fill_value for NaNs is a small positive number if median is 0/NaN or values are tiny
            fill_value_for_na = (
                median_abs_residual
                if pd.notna(median_abs_residual) and median_abs_residual > 0.01
                else 0.01
            )

            database_df["size"] = abs_residuals.fillna(fill_value_for_na)
            # Ensure all size values are positive for px to scale (px scales these data values)
            database_df["size"] = np.maximum(database_df["size"], 0.01)

            marker_size_description_text = r"Marker size indicates the magnitude of the Vs30 log residual, given by \(\mathrm{|(\log(SPT_{Vs30}) - \log(Foster2019_{Vs30})|)}\)"
        else:
            database_df["size"] = 5  # Default size if vs30_log_residual is not present
            marker_size_description_text = (
                "Marker size is fixed as Vs30 log residual is not available for sizing."
            )

        # Prepare hover data and other column-dependent operations for database_df
        database_df["Vs30 (m/s)"] = database_df.get("vs30")
        database_df["Vs30_log_resid"] = database_df.get("vs30_log_residual")
        if vs30_correlation == "boore_2011":
            reason_text = "Unable to estimate as Boore et al. (2011) Vs to Vs30 correlation requires a depth of at least 5 m"
            min_required_depth = 5
        else:
            reason_text = "Unable to estimate as Boore et al. (2004) Vs to Vs30 correlation requires a depth of at least 10 m"
            min_required_depth = 10

        if "deepest_depth" in database_df.columns:
            database_df.loc[
                database_df["deepest_depth"] < min_required_depth, "Vs30 (m/s)"
            ] = reason_text
            database_df.loc[
                (database_df["deepest_depth"] >= min_required_depth)
                & (np.isnan(database_df.get("vs30")) | (database_df.get("vs30") == 0)),
                "Vs30 (m/s)",
            ] = "Vs30 calculation failed even though CPT depth is sufficient"
            database_df.loc[
                (database_df["deepest_depth"] >= min_required_depth)
                & ~(np.isnan(database_df.get("vs30")) | (database_df.get("vs30") == 0)),
                "Vs30 (m/s)",
            ] = database_df.get("vs30").apply(
                lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
            )

        if "vs30_log_residual" in database_df.columns:
            database_df.loc[
                (np.isnan(database_df["vs30_log_residual"])), "Vs30_log_resid"
            ] = "Unavailable as Vs30 could not be calculated"
            database_df.loc[
                ~(np.isnan(database_df["vs30_log_residual"])), "Vs30_log_resid"
            ] = database_df["vs30_log_residual"].apply(
                lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
            )
        if "deepest_depth" in database_df.columns:
            database_df["deepest_depth (m)"] = database_df["deepest_depth"]

    elif uploaded_df is not None and not uploaded_df.empty:
        centre_lat = uploaded_df["latitude"].mean()
        centre_lon = uploaded_df["longitude"].mean()
        current_map_zoom = 5
        flask.flash(
            "Database query is empty or resulted in no data. Centering map on uploaded locations.",
            "info",
        )
    # else: use NZ default center and zoom if both are empty

    map_obj = None  # Will hold the Plotly Figure object

    # Create the base map
    if not database_df.empty:
        db_hover_data_dict = OrderedDict()
        if "deepest_depth (m)" in database_df.columns:
            db_hover_data_dict["deepest_depth (m)"] = ":.2f"
        if "Vs30 (m/s)" in database_df.columns:
            db_hover_data_dict["Vs30 (m/s)"] = True
        if "Vs30_log_resid" in database_df.columns:
            db_hover_data_dict["Vs30_log_resid"] = True
        # Ensure 'size' is in hover_data if it was calculated for database_df
        if "size" in database_df.columns:
            db_hover_data_dict["size"] = False

        size_col = database_df["size"] if "size" in database_df.columns else 5
        color_col_data = database_df.get(colour_by)
        # Ensure color_col is numeric or None for continuous scale, otherwise treat as discrete or don't color
        color_col = (
            color_col_data
            if color_col_data is not None
            and pd.api.types.is_numeric_dtype(color_col_data)
            else None
        )

        map_obj = px.scatter_map(
            database_df,
            lat="latitude",
            lon="longitude",
            color=color_col,
            size=size_col,
            hover_name=database_df.get("record_name"),
            center={"lat": centre_lat, "lon": centre_lon},
            zoom=current_map_zoom,
            hover_data=db_hover_data_dict,
        )
    elif uploaded_df is not None and not uploaded_df.empty:
        flask.flash(
            f"Database is empty or query yielded no results. Creating map with {len(uploaded_df)} uploaded locations.",
            "info",
        )
        map_obj = px.scatter_map(
            uploaded_df,
            lat="latitude",
            lon="longitude",
            color_discrete_sequence=[
                "red"
            ],  # Ensure uploaded points are distinctly colored
            size_max=10,  # Fixed size for uploaded points for clarity
            center={"lat": centre_lat, "lon": centre_lon},
            zoom=current_map_zoom,
            hover_data={
                "latitude": True,
                "longitude": True,
            },  # Basic hover for uploaded points
        )
        if map_obj.data:
            map_obj.data[0].name = "Uploaded Locations"
    else:
        flask.flash(
            "No locations to display (database query empty and no locations uploaded). Showing an empty map of New Zealand.",
            "info",
        )
        map_obj = go.Figure(
            go.Scattermapbox(
                lat=[centre_lat],
                lon=[centre_lon],
                mode="markers",
                marker={"size": 0, "opacity": 0},
            )
        )
        map_obj.update_layout(
            mapbox_style="open-street-map",
            mapbox_center_lat=centre_lat,
            mapbox_center_lon=centre_lon,
            mapbox_zoom=current_map_zoom,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )

    # If database_df was the primary source for the map, and uploaded_df also exists, add uploaded_df as a new trace.
    if not database_df.empty and (uploaded_df is not None and not uploaded_df.empty):
        flask.flash(
            f"Adding {len(uploaded_df)} uploaded locations as a separate trace to the map.",
            "info",
        )

        # Ensure required columns exist and are numeric for the trace
        if not (
            "latitude" in uploaded_df.columns
            and "longitude" in uploaded_df.columns
            and pd.api.types.is_numeric_dtype(uploaded_df["latitude"])
            and pd.api.types.is_numeric_dtype(uploaded_df["longitude"])
        ):
            flask.flash(
                "Uploaded data is missing valid latitude/longitude for map trace.",
                "error",
            )
        else:
            # Ensure hovertext column exists for customdata/hovertemplate
            if "hovertext" not in uploaded_df.columns:
                # Create a default hovertext if not present
                uploaded_df["hovertext"] = (
                    "Lat: "
                    + uploaded_df["latitude"].round(4).astype(str)
                    + ", Lon: "
                    + uploaded_df["longitude"].round(4).astype(str)
                )

            map_obj.add_trace(
                go.Scattermapbox(
                    lat=uploaded_df["latitude"],
                    lon=uploaded_df["longitude"],
                    mode="markers",
                    marker=go.scattermapbox.Marker(size=10, color="red", opacity=0.7),
                    name="Uploaded Locations",  # For legend
                    text=uploaded_df[
                        "hovertext"
                    ],  # Data for the hovertemplate's %{text}
                    customdata=uploaded_df[
                        ["latitude", "longitude"]
                    ],  # For hovertemplate's %{customdata[0/1]}
                    hovertemplate=(
                        "<b>Uploaded Point</b><br>"
                        + "Lat: %{customdata[0]:.4f}<br>"
                        + "Lon: %{customdata[1]:.4f}<br>"
                        + "Details: %{text}<extra></extra>"  # <extra></extra> removes trace info from hover
                    ),
                )
            )
            # Explicitly update layout AFTER adding the new trace
            map_obj.update_layout(
                mapbox_style="carto-positron",  # Maintain consistency with px default
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.7)",  # Semi-transparent background for legend
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
        map=map_obj.to_html(
            full_html=False,
            include_plotlyjs=False,
            default_height="85vh",
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
