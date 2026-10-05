"""Check unavailable values and comparable marker sizes across map layers."""

from pathlib import Path

import pandas as pd

from nzgd_map import plotting
from nzgd_map import query_sqlite_db as queries
from nzgd_map.database import open_database


def test_unavailable_markers_and_shared_size_scale(database_path: Path):
    with open_database(database_path) as conn:
        frame = queries.all_vs30s_given_correlations(
            "boore_2004",
            "andrus_2007_pleistocene",
            "brandenberg_2010",
            "Auto",
            conn,
            include_unestimated=True,
        )
    frame["report_url"] = "/record"
    stations = pd.DataFrame(columns=["lat", "lon", "name"])
    figure, mapped = plotting.map_figure(frame, "vs30", stations)
    assert mapped == 8
    missing, available = figure.data
    assert len(missing.lat) == len(available.lat) == 4
    assert all(0 < size <= missing.marker.sizeref for size in missing.marker.size)
    assert missing.marker.sizeref == available.marker.sizeref
    assert missing.marker.sizemin == available.marker.sizemin == 4

    # Filtering to reports without estimates still produces visible minimum
    # markers and avoids zero/NaN size references.
    figure, mapped = plotting.map_figure(
        frame.loc[~frame.vs30_available], "vs30", stations
    )
    assert mapped == 4
    assert figure.data[0].marker.sizeref > 0
    assert figure.data[0].marker.sizemin == 4
