"""Read report metadata, measurements and stored estimates from the current schema."""

import sqlite3

import numpy as np
import pandas as pd

# Keep existing map/query names where their meaning has not changed. Record
# timestamps deliberately retain their new names: they are not test dates.
RECORD_COLUMNS = """
    n.nzgd_id, t.value AS type_prefix,
    n.original_investigation_name,
    n.original_investigation_name AS original_reference,
    n.record_created_on, n.record_last_modified_on,
    n.latitude, n.longitude,
    n.model_vs30_foster_2019_m_per_s AS model_vs30_foster_2019,
    n.model_vs30_stddev_foster_2019_ln AS model_vs30_stddev_foster_2019,
    n.model_gwl_westerhoff_2018_m AS model_gwl_westerhoff_2018,
    r.value AS region, d.value AS district,
    c.value AS city, s.value AS suburb
"""
RECORD_JOINS = """
    JOIN nzgdrecord n ON n.nzgd_id = p.nzgd_id
    JOIN type t ON t.id = n.type_id
    LEFT JOIN region r ON r.id = n.region_id
    LEFT JOIN district d ON d.id = n.district_id
    LEFT JOIN city c ON c.id = n.city_id
    LEFT JOIN suburb s ON s.id = n.suburb_id
"""
CPT_COLUMNS = """
    p.cpt_id, p.source_file, p.max_depth_m AS deepest_depth,
    p.min_depth_m AS shallowest_depth, p.extracted_gwl_m AS extracted_gwl,
    p.tip_net_area_ratio AS cpt_tip_net_area_ratio,
    p.predrill_depth_m, p.has_cpt_data,
    p.did_explicit_unit_conversion, p.did_inferred_unit_conversion,
    g.value AS ground_water_level_method, tr.value AS termination_reason
"""
CPT_JOINS = """
    LEFT JOIN cptgroundwaterlevelmethod g ON g.id = p.gwl_method_id
    LEFT JOIN terminationreason tr ON tr.id = p.termination_reason_id
"""
SPT_COLUMNS = """
    p.spt_id, p.source_file, p.extracted_gwl_m AS extracted_gwl,
    p.efficiency AS spt_efficiency,
    p.borehole_diameter AS spt_borehole_diameter,
    p.casing_diameter AS spt_casing_diameter
"""
LOOKUP_TABLES = {
    "region",
    "district",
    "city",
    "suburb",
    "cpttovscorrelation",
    "spttovscorrelation",
    "vstovs30correlation",
    "spttovs30hammertype",
}


def lookup_values(table: str, conn: sqlite3.Connection) -> list[str]:
    """Read values from one of the supported lookup tables."""
    if table not in LOOKUP_TABLES:
        raise ValueError(f"Unsupported lookup table: {table}")
    return [row[0] for row in conn.execute(f"SELECT value FROM {table} ORDER BY id")]


def _correlation_id(table: str, value: str, conn: sqlite3.Connection) -> int:
    if table not in LOOKUP_TABLES:
        raise ValueError(f"Unsupported lookup table: {table}")
    row = conn.execute(f"SELECT id FROM {table} WHERE value = ?", (value,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown option for {table}: {value}")
    return row[0]


def _add_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Add display names and residuals without replacing any stored values."""
    frame["record_name"] = (
        frame["type_prefix"].astype(str) + "_" + frame["nzgd_id"].astype(str)
    )
    if "extracted_gwl" in frame:
        frame["gwl_residual"] = pd.to_numeric(
            frame["extracted_gwl"], errors="coerce"
        ) - pd.to_numeric(frame["model_gwl_westerhoff_2018"], errors="coerce")
    if "vs30" in frame:
        values = pd.to_numeric(frame["vs30"], errors="coerce")
        model = pd.to_numeric(frame["model_vs30_foster_2019"], errors="coerce")
        frame["vs30_available"] = np.isfinite(values) & values.gt(0)
        frame["vs30_log_residual"] = np.log(
            values.where(frame["vs30_available"]).astype(float)
        ) - np.log(model.where(np.isfinite(model) & model.gt(0)).astype(float))
    return frame


def record_metadata(nzgd_id: int, conn: sqlite3.Connection) -> dict | None:
    """Read a record independently of its reports and estimates."""
    frame = pd.read_sql_query(
        f"""SELECT {RECORD_COLUMNS}, n.merged_into_nzgd_id
            FROM nzgdrecord p {RECORD_JOINS}
            WHERE n.nzgd_id = ?""",
        conn,
        params=(nzgd_id,),
    )
    if frame.empty:
        return None
    return _add_metadata(frame).iloc[0].to_dict()


def resolve_record(nzgd_id: int, conn: sqlite3.Connection) -> dict | None:
    """Follow deduplication links to the surviving NZGD record."""
    visited = set()
    while nzgd_id not in visited:
        visited.add(nzgd_id)
        record = record_metadata(nzgd_id, conn)
        if record is None or pd.isna(record["merged_into_nzgd_id"]):
            return record
        nzgd_id = int(record["merged_into_nzgd_id"])
    raise ValueError("The database contains a cycle in its record merge links.")


def cpt_reports_for_one_nzgd(nzgd_id: int, conn: sqlite3.Connection) -> pd.DataFrame:
    """Read every CPT report, including reports without stored estimates."""
    frame = pd.read_sql_query(
        f"""SELECT {RECORD_COLUMNS}, {CPT_COLUMNS}
            FROM cptreport p {RECORD_JOINS} {CPT_JOINS}
            WHERE p.nzgd_id = ? ORDER BY p.cpt_id""",
        conn,
        params=(nzgd_id,),
    )
    return _add_metadata(frame)


def spt_reports_for_one_nzgd(nzgd_id: int, conn: sqlite3.Connection) -> pd.DataFrame:
    """Read SPT reports and their individual measurement depth ranges."""
    frame = pd.read_sql_query(
        f"""SELECT {RECORD_COLUMNS}, {SPT_COLUMNS},
                MIN(m.depth_m) AS shallowest_depth,
                MAX(m.depth_m) AS deepest_depth,
                COUNT(m.spt_measurement_id) AS measurement_count
            FROM sptreport p {RECORD_JOINS}
            LEFT JOIN sptmeasurements m ON m.spt_id = p.spt_id
            WHERE p.nzgd_id = ?
            GROUP BY p.spt_id ORDER BY p.spt_id""",
        conn,
        params=(nzgd_id,),
    )
    return _add_metadata(frame)


def all_vs30s_given_correlations(
    selected_vs30_correlation: str,
    selected_cpt_to_vs_correlation: str,
    selected_spt_to_vs_correlation: str,
    selected_hammer_type: str,
    conn: sqlite3.Connection,
    *,
    include_unestimated: bool = False,
) -> pd.DataFrame:
    """Read selected estimates, optionally including measured reports without one."""
    vs30_id = _correlation_id("vstovs30correlation", selected_vs30_correlation, conn)
    cpt_id = _correlation_id("cpttovscorrelation", selected_cpt_to_vs_correlation, conn)
    spt_id = _correlation_id("spttovscorrelation", selected_spt_to_vs_correlation, conn)
    hammer_id = _correlation_id("spttovs30hammertype", selected_hammer_type, conn)
    cpt_frame = pd.read_sql_query(
        f"""SELECT {RECORD_COLUMNS}, {CPT_COLUMNS},
                'CPT' AS report_kind, p.cpt_id AS report_id
            FROM cptreport p {RECORD_JOINS} {CPT_JOINS}
            WHERE n.merged_into_nzgd_id IS NULL AND p.has_cpt_data = 1
            ORDER BY p.cpt_id""",
        conn,
    )
    spt_frame = pd.read_sql_query(
        f"""WITH depths AS (
                SELECT spt_id, MIN(depth_m) AS shallowest_depth,
                    MAX(depth_m) AS deepest_depth
                FROM sptmeasurements GROUP BY spt_id
            )
            SELECT {RECORD_COLUMNS}, {SPT_COLUMNS},
                depths.shallowest_depth, depths.deepest_depth,
                'SPT' AS report_kind, p.spt_id AS report_id
            FROM sptreport p {RECORD_JOINS}
            JOIN depths ON depths.spt_id = p.spt_id
            WHERE n.merged_into_nzgd_id IS NULL
            ORDER BY p.spt_id""",
        conn,
    )
    # Query the selected estimates once. A SQLite LEFT JOIN here otherwise
    # chooses the correlation index for each report on this database, causing
    # repeated scans of hundreds of thousands of estimates.
    cpt_estimates = pd.read_sql_query(
        """SELECT cpt_id, vs30, vs30_stddev FROM cptvs30estimates
            WHERE vs_to_vs30_correlation_id = ? AND cpt_to_vs_correlation_id = ?""",
        conn,
        params=(vs30_id, cpt_id),
    )
    spt_estimates = pd.read_sql_query(
        """SELECT spt_id, vs30, vs30_stddev FROM sptvs30estimates
            WHERE vs_to_vs30_correlation_id = ? AND spt_to_vs_correlation_id = ?
                AND assumed_hammer_type_id = ?""",
        conn,
        params=(vs30_id, spt_id, hammer_id),
    )
    cpt_frame = cpt_frame.merge(
        cpt_estimates, on="cpt_id", how="left", validate="one_to_one"
    )
    spt_frame = spt_frame.merge(
        spt_estimates, on="spt_id", how="left", validate="one_to_one"
    )
    frame = _add_metadata(pd.concat([cpt_frame, spt_frame], ignore_index=True))
    for column in ("cpt_id", "spt_id"):
        frame[column] = frame[column].astype("Int64")
    frame["type_number_code"] = frame["type_prefix"].map({"CPT": 0, "SCPT": 1, "BH": 2})
    if not include_unestimated:
        frame = frame.loc[frame["vs30_available"]].copy()
    return frame


def cpt_measurements_for_one_nzgd(
    selected_nzgd_id: int, conn: sqlite3.Connection
) -> pd.DataFrame:
    """Read CPT measurements with their report identity and explicit units."""
    return pd.read_sql_query(
        """SELECT p.nzgd_id, p.cpt_id, p.source_file,
                m.depth_m, m.qc_MPa, m.fs_MPa, m.u2_MPa
            FROM cptreport p
            JOIN cptmeasurements m ON m.cpt_id = p.cpt_id
            WHERE p.nzgd_id = ?
            ORDER BY p.cpt_id, m.depth_m, m.measurement_id""",
        conn,
        params=(selected_nzgd_id,),
    )


def spt_measurements_for_one_nzgd(
    selected_nzgd_id: int, conn: sqlite3.Connection
) -> pd.DataFrame:
    """Preserve original SPT fields and select finite NVAL, then MAIN, for plots."""
    frame = pd.read_sql_query(
        """SELECT p.nzgd_id, p.spt_id, p.source_file,
                m.depth_m, m.ISPT_MAIN, m.ISPT_NVAL, m.ISPT_REP
            FROM sptreport p
            JOIN sptmeasurements m ON m.spt_id = p.spt_id
            WHERE p.nzgd_id = ?
            ORDER BY p.spt_id, m.depth_m, m.spt_measurement_id""",
        conn,
        params=(selected_nzgd_id,),
    )
    nval = pd.to_numeric(frame["ISPT_NVAL"], errors="coerce")
    main = pd.to_numeric(frame["ISPT_MAIN"], errors="coerce")
    use_nval = np.isfinite(nval)
    use_main = ~use_nval & np.isfinite(main)
    frame["number_of_blows"] = nval.where(use_nval, main.where(use_main))
    frame["n_value_source"] = np.select(
        [use_nval, use_main], ["ISPT_NVAL", "ISPT_MAIN"], default=""
    )
    return frame


def spt_soil_types_for_one_nzgd(
    selected_nzgd_id: int, conn: sqlite3.Connection
) -> pd.DataFrame:
    """Read stored soil intervals, grouping types by layer identity within a report."""
    frame = pd.read_sql_query(
        """SELECT p.nzgd_id, p.spt_id, p.source_file,
                m.soil_measurement_id, m.top_depth_m, m.bottom_depth_m,
                t.value AS soil_type
            FROM sptreport p
            JOIN soilmeasurements m ON m.spt_id = p.spt_id
            LEFT JOIN soilmeasurementsoiltype j
                ON j.soil_measurement_id = m.soil_measurement_id
            LEFT JOIN soiltypes t ON t.id = j.soil_type_id
            WHERE p.nzgd_id = ?
            ORDER BY p.spt_id, m.top_depth_m, m.soil_measurement_id, t.value""",
        conn,
        params=(selected_nzgd_id,),
    )
    # One physical layer may have multiple soil classifications. Layers from
    # different reports, or distinct layers sharing a depth, must stay separate.
    frame = frame.groupby(
        ["spt_id", "soil_measurement_id"], sort=False, as_index=False
    ).agg(
        nzgd_id=("nzgd_id", "first"),
        source_file=("source_file", "first"),
        top_depth_m=("top_depth_m", "first"),
        bottom_depth_m=("bottom_depth_m", "first"),
        soil_type=("soil_type", lambda values: " + ".join(values.dropna().unique())),
    )
    frame["layer_thickness_m"] = pd.to_numeric(
        frame["bottom_depth_m"], errors="coerce"
    ) - pd.to_numeric(frame["top_depth_m"], errors="coerce")
    return frame


def cpt_vs30s_for_one_nzgd_id(
    selected_nzgd_id: int, conn: sqlite3.Connection
) -> pd.DataFrame:
    """Read all stored CPT estimates, retaining their report IDs."""
    frame = pd.read_sql_query(
        f"""SELECT {RECORD_COLUMNS}, {CPT_COLUMNS},
                e.vs30_id, e.vs30, e.vs30_stddev,
                v.value AS cpt_to_vs_correlation,
                v30.value AS vs_to_vs30_correlation
            FROM cptreport p {RECORD_JOINS} {CPT_JOINS}
            JOIN cptvs30estimates e ON e.cpt_id = p.cpt_id
            JOIN cpttovscorrelation v ON v.id = e.cpt_to_vs_correlation_id
            JOIN vstovs30correlation v30 ON v30.id = e.vs_to_vs30_correlation_id
            WHERE p.nzgd_id = ?
            ORDER BY p.cpt_id, v.id, v30.id, e.vs30_id""",
        conn,
        params=(selected_nzgd_id,),
    )
    return _add_metadata(frame)


def spt_vs30s_for_one_nzgd_id(
    selected_nzgd_id: int, conn: sqlite3.Connection
) -> pd.DataFrame:
    """Read SPT estimates and the assumptions stored for each estimate."""
    frame = pd.read_sql_query(
        f"""SELECT {RECORD_COLUMNS}, {SPT_COLUMNS},
                e.vs30_id, e.vs30, e.vs30_stddev,
                e.assumed_borehole_diameter_mm,
                e.estimate_used_extracted_efficiency,
                e.estimate_used_extracted_layer_soil_types,
                v.value AS spt_to_vs_correlation,
                v30.value AS vs_to_vs30_correlation,
                h.value AS hammer_type
            FROM sptreport p {RECORD_JOINS}
            JOIN sptvs30estimates e ON e.spt_id = p.spt_id
            JOIN spttovscorrelation v ON v.id = e.spt_to_vs_correlation_id
            JOIN vstovs30correlation v30 ON v30.id = e.vs_to_vs30_correlation_id
            JOIN spttovs30hammertype h ON h.id = e.assumed_hammer_type_id
            WHERE p.nzgd_id = ?
            ORDER BY p.spt_id, v.id, v30.id, h.id, e.vs30_id""",
        conn,
        params=(selected_nzgd_id,),
    )
    return _add_metadata(frame)


def get_region_names(conn: sqlite3.Connection) -> list[str]:
    """Read region names for query help."""
    return sorted(lookup_values("region", conn))


def get_district_names(conn: sqlite3.Connection) -> list[str]:
    """Read district names for query help."""
    return sorted(lookup_values("district", conn))


def get_city_names(conn: sqlite3.Connection) -> list[str]:
    """Read city names for query help."""
    return sorted(lookup_values("city", conn))


def get_suburb_names(conn: sqlite3.Connection) -> list[str]:
    """Read suburb names for query help."""
    return sorted(lookup_values("suburb", conn))
