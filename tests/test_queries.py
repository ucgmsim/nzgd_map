"""Regression checks for report identity, units and incomplete estimates."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from nzgd_map import query_sqlite_db as queries
from nzgd_map.database import open_database


def _map_data(
    conn: sqlite3.Connection, correlation: str = "boore_2004"
) -> pd.DataFrame:
    return queries.all_vs30s_given_correlations(
        correlation,
        "andrus_2007_pleistocene",
        "brandenberg_2010",
        "Auto",
        conn,
        include_unestimated=True,
    )


def test_map_joins_reports_to_their_nzgd_ids(database_path: Path):
    with open_database(database_path) as conn:
        frame = _map_data(conn)
    assert len(frame) == 8
    assert frame["vs30_available"].sum() == 4
    assert not frame.duplicated(["report_kind", "report_id"]).any()
    spt = frame.loc[frame.spt_id == 31].iloc[0]
    assert spt.nzgd_id == 1002
    assert spt.record_name == "BH_1002"
    assert spt.deepest_depth == 14
    assert frame.loc[frame.spt_id == 47, "deepest_depth"].iloc[0] == 7
    assert 19 not in frame.cpt_id.values
    assert 49 not in frame.spt_id.values
    assert 9001 not in frame.nzgd_id.values


def test_changing_correlation_changes_availability_without_losing_reports(
    database_path: Path,
):
    with open_database(database_path) as conn:
        first, second = _map_data(conn), _map_data(conn, "boore_2011")
    assert set(zip(first.report_kind, first.report_id)) == set(
        zip(second.report_kind, second.report_id)
    )
    assert first.loc[first.cpt_id == 17, "vs30_available"].iloc[0] == False
    assert second.loc[second.cpt_id == 17, "vs30_available"].iloc[0] == True
    assert first.loc[first.cpt_id == 24, "vs30"].iloc[0] == 100000
    assert first.loc[first.cpt_id == 24, "vs30_log_residual"].isna().all()
    assert first["vs30_stddev"].isna().all()


def test_reports_without_estimates_retain_metadata_and_measurements(
    database_path: Path,
):
    with open_database(database_path) as conn:
        reports = queries.cpt_reports_for_one_nzgd(1003, conn)
        estimates = queries.cpt_vs30s_for_one_nzgd_id(1003, conn)
        data = queries.cpt_measurements_for_one_nzgd(1003, conn)
    assert reports.cpt_id.tolist() == [23]
    assert estimates.empty
    assert data.qc_MPa.tolist() == [23.1, 23.2]
    assert data.fs_MPa.tolist() == [0.01, 0.02]
    assert data.u2_MPa.iloc[0] == 0.005
    assert data.u2_MPa.isna().iloc[1]


def test_spt_nval_precedence_zero_and_raw_fields(database_path: Path):
    with open_database(database_path) as conn:
        data = queries.spt_measurements_for_one_nzgd(1002, conn)
    first = data.loc[data.spt_id == 31]
    assert first.number_of_blows.iloc[:4].tolist() == [12, 0, 0, 20]
    assert first.n_value_source.iloc[:4].tolist() == [
        "ISPT_NVAL",
        "ISPT_MAIN",
        "ISPT_NVAL",
        "ISPT_NVAL",
    ]
    assert pd.isna(first.number_of_blows.iloc[4])
    assert first.ISPT_REP.iloc[4] == 73
    assert first.ISPT_MAIN.iloc[0] == 5
    assert set(data.nzgd_id) == {1002}
    assert 999 not in data.number_of_blows.values


def test_soil_layers_are_not_merged_across_reports_or_shared_depths(
    database_path: Path,
):
    with open_database(database_path) as conn:
        soils = queries.spt_soil_types_for_one_nzgd(1002, conn).set_index(
            "soil_measurement_id"
        )
    assert len(soils) == 5
    assert soils.loc[100, "soil_type"] == "SAND + SILT"
    assert soils.loc[103, "soil_type"] == "GRAVEL"
    assert soils.loc[100, "layer_thickness_m"] == 2
    assert soils.loc[103, "layer_thickness_m"] == 4
    assert pd.isna(soils.loc[101, "layer_thickness_m"])
    assert soils.loc[102, "layer_thickness_m"] == 3
    assert soils.loc[104, "soil_type"] == ""


def test_spt_assumptions_belong_to_each_estimate(database_path: Path):
    with open_database(database_path) as conn:
        estimates = queries.spt_vs30s_for_one_nzgd_id(1002, conn)
    first = estimates.loc[estimates.spt_id == 31]
    assert first.assumed_borehole_diameter_mm.tolist() == [150, 200]
    assert first.estimate_used_extracted_efficiency.tolist() == [1, 0]
    assert first.estimate_used_extracted_layer_soil_types.tolist() == [0, 1]
    assert np.isfinite(first.vs30_log_residual).all()


def test_unknown_and_merged_records(database_path: Path):
    with open_database(database_path) as conn:
        assert queries.resolve_record(9001, conn)["nzgd_id"] == 1001
        assert queries.record_metadata(99999, conn) is None
        assert queries.spt_measurements_for_one_nzgd(99999, conn).empty
        assert queries.spt_soil_types_for_one_nzgd(99999, conn).empty
        assert queries.spt_vs30s_for_one_nzgd_id(99999, conn).empty
