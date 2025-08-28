"""
The views module defines the Flask views (web pages) for the application.
Each view is a function that returns an HTML template to render in the browser.
"""

import os
import flask
import pandas as pd
import plotly.express as px
import os

bp = flask.Blueprint("views", __name__)

@bp.route("/", methods=["GET"])
def index():
    # Path to the CSV file (update this path as needed)
    csv_path = os.path.join(os.path.dirname(__file__), "..", "instance", "vs30_points.csv")
    df = pd.read_csv(csv_path)
    # Expect columns: latitude, longitude, Vs30
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="Vs30",
        zoom=4,
        mapbox_style="open-street-map",
        hover_data={"latitude": True, "longitude": True, "Vs30": True},
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return flask.render_template(
        "views/index.html",
        map=fig.to_html(full_html=False, include_plotlyjs=True, default_height="85vh"),
    )
