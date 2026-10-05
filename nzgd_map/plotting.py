"""Map and histogram figures with explicit handling of unavailable values."""

from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLOUR_VARIABLES = [
    ("vs30", "Estimated Vs30 (m/s)"),
    ("type_number_code", "Investigation type"),
    ("vs30_log_residual", "Vs30 log residual with Foster (2019)"),
    ("deepest_depth", "Maximum measurement depth (m)"),
    ("vs30_stddev", "Vs30 uncertainty"),
    ("model_vs30_foster_2019", "Foster (2019) Vs30 (m/s)"),
    ("model_vs30_stddev_foster_2019", "Foster (2019) standard deviation (ln units)"),
    ("shallowest_depth", "Minimum measurement depth (m)"),
    ("extracted_gwl", "Extracted groundwater level (m)"),
    ("model_gwl_westerhoff_2018", "Westerhoff (2018) groundwater level (m)"),
    ("gwl_residual", "Groundwater residual with Westerhoff (2018) (m)"),
]
LABELS = dict(COLOUR_VARIABLES)


def _numeric(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )


def _formatted(series: pd.Series, missing: str = "Unavailable") -> pd.Series:
    return _numeric(series).map(
        lambda value: f"{value:.2f}" if pd.notna(value) else missing
    )


def map_figure(
    reports: pd.DataFrame, colour_by: str, stations: pd.DataFrame
) -> tuple[go.Figure, int]:
    """Plot all reports with coordinates, giving missing colours a neutral trace."""
    frame = reports.loc[
        _numeric(reports["latitude"]).between(-90, 90)
        & _numeric(reports["longitude"]).between(-180, 180)
    ].copy()
    frame["_marker_size"] = _numeric(frame["vs30_log_residual"]).abs().fillna(0)
    # Share the scale across colour traces. In particular, the unavailable
    # trace must not independently scale its small markers up to size_max.
    size_reference = (float(frame["_marker_size"].max()) or 1) / 20**2
    if frame.empty:
        size_reference = 1 / 20**2
    # Plotly suppresses zero-sized points even when sizemin is set. Give zero
    # and unavailable residuals a positive size below the visible minimum.
    frame["_marker_size"] = frame["_marker_size"].clip(lower=size_reference)
    frame["_report_label"] = (
        frame["report_kind"] + " report " + frame["report_id"].astype(str)
    )
    frame["_source"] = frame["source_file"].fillna("").map(escape)
    frame["_depth"] = _formatted(frame["deepest_depth"])
    frame["_vs30"] = _formatted(
        frame["vs30"].where(frame["vs30_available"]),
        "Unavailable for selected correlations",
    )
    frame["_residual"] = _formatted(frame["vs30_log_residual"])
    frame["_colour_value"] = (
        frame["type_prefix"]
        if colour_by == "type_number_code"
        else _formatted(frame[colour_by])
    )
    figure = go.Figure()
    custom_data = [
        "report_url",
        "_report_label",
        "_source",
        "_depth",
        "_vs30",
        "_residual",
        "_colour_value",
    ]

    def points(data: pd.DataFrame, **kwargs) -> go.Figure:
        return px.scatter_map(
            data,
            lat="latitude",
            lon="longitude",
            hover_name="record_name",
            size="_marker_size",
            size_max=20,
            custom_data=custom_data,
            **kwargs,
        )

    if colour_by == "type_number_code":
        if not frame.empty:
            figure = points(
                frame, color="type_prefix", labels={"type_prefix": "Investigation type"}
            )
    else:
        values = _numeric(frame[colour_by])
        if colour_by == "vs30":
            values = values.where(frame["vs30_available"])
        missing = frame.loc[values.isna()]
        if not missing.empty:
            for trace in points(missing, color_discrete_sequence=["#888888"]).data:
                trace.name = "Reports with no value"
                trace.showlegend = True
                figure.add_trace(trace)
        available = frame.loc[values.notna()]
        if not available.empty:
            # Limit only the colour scale. Hover values, filters and histograms
            # continue to use the original estimates, including extremes.
            colour_range = None
            if colour_by == "vs30":
                low, high = values.quantile([0.001, 0.999])
                if low < high:
                    colour_range = (low, high)
            coloured = points(
                available,
                color=colour_by,
                labels={colour_by: LABELS[colour_by]},
                range_color=colour_range,
            )
            for trace in coloured.data:
                figure.add_trace(trace)
            figure.update_layout(coloraxis=coloured.layout.coloraxis)

    for trace in figure.data:
        trace.update(
            marker_sizemin=4,
            marker_sizeref=size_reference,
            hovertemplate=(
                "<b>%{hovertext}</b><br>%{customdata[1]}<br>%{customdata[2]}"
                "<br>Maximum depth: %{customdata[3]} m"
                "<br>Vs30: %{customdata[4]}"
                "<br>Vs30 log residual: %{customdata[5]}"
                f"<br>{LABELS[colour_by]}: %{{customdata[6]}}<extra></extra>"
            ),
        )

    if not stations.empty:
        figure.add_trace(
            go.Scattermap(
                lat=stations["lat"],
                lon=stations["lon"],
                text=stations["name"].fillna("").astype(str).map(escape),
                mode="markers",
                marker={"size": 10, "color": "#d95f02"},
                name="GeoNet stations",
                hovertemplate="%{text}<extra>GeoNet station</extra>",
            )
        )

    centre = (
        {
            "lat": float(frame["latitude"].mean()),
            "lon": float(frame["longitude"].mean()),
        }
        if not frame.empty
        else {"lat": -41.0, "lon": 173.0}
    )
    figure.update_layout(
        map={"style": "carto-positron", "center": centre, "zoom": 4},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={"title": "Map key", "itemsizing": "constant", "x": 0.01, "y": 0.99},
        uirevision="nzgd-map",
    )
    if frame.empty:
        figure.add_annotation(
            text="No reports match this selection with available coordinates.",
            x=0.5,
            y=0.95,
            xref="paper",
            yref="paper",
            showarrow=False,
            bgcolor="white",
        )
    return figure, len(frame)


def histogram_figure(reports: pd.DataFrame, field: str) -> tuple[go.Figure, int]:
    """Use original values and count the reports excluded for missing values."""
    values = (
        reports["type_prefix"]
        if field == "type_number_code"
        else _numeric(reports[field])
    )
    available = values.dropna()
    if field == "vs30":
        available = values.where(reports["vs30_available"]).dropna()
    if available.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No values available for this histogram.", showarrow=False
        )
        figure.update_layout(xaxis_title=LABELS[field], yaxis_title="Reports")
    else:
        figure = px.histogram(x=available, labels={"x": LABELS[field], "y": "Reports"})
        figure.update_layout(yaxis_title="Reports")
    return figure, len(reports) - len(available)
