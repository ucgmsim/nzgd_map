"""
This module provides a function to extract pre-computed Vs30 values for CPT and SPT from
a SQLite database based on the selected correlations and hammer type.
"""

import pandas as pd
import sqlite3
import time
import numpy as np
from pathlib import Path


def clip_highest_and_lowest_percent(
    data: pd.Series, lower_percent: float, upper_percent: float
) -> pd.Series:
    """
    Clips the highest and lowest percent of values in a Pandas Series based on the specified quantiles.

    Parameters
    ----------
    data : pd.Series
        The data series to be clipped.
    lower_percent : float
        The lower percentile threshold (0-100).
    upper_percent : float
        The upper percentile threshold (0-100).

    Returns
    -------
    pd.Series
        The modified data series with values outside the specified quantiles replaced.
    """
    lower_quantile = data.quantile(lower_percent / 100)
    upper_quantile = data.quantile(upper_percent / 100)

    # Determine the maximum and minimum values within these quantiles
    # Determine the maximum and minimum values within the specified quantiles
    max_within_quantiles = data[
        (data >= lower_quantile) & (data <= upper_quantile)
    ].max()
    min_within_quantiles = data[
        (data >= lower_quantile) & (data <= upper_quantile)
    ].min()

    # Replace the values outside these quantiles
    # Make a copy to avoid SettingWithCopyWarning
    data = data.copy()
    data.loc[data > upper_quantile] = max_within_quantiles
    data.loc[data < lower_quantile] = min_within_quantiles

    return data


def all_vs30s_given_correlations(
    selected_vs30_correlation: str,
    selected_cpt_to_vs_correlation: str,
    selected_spt_to_vs_correlation: str,
    selected_hammer_type: str,
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Extracts CPT and SPT data from the SQLite database based on the selected correlations and hammer type.

    Parameters
    ----------
    selected_vs30_correlation : str
        The selected Vs to Vs30 correlation name.
    selected_cpt_to_vs_correlation : str
        The selected CPT to Vs correlation name.
    selected_spt_to_vs_correlation : str
        The selected SPT to Vs correlation name.
    selected_hammer_type : str
        The selected hammer type name.
    conn : sqlite3.Connection
        The SQLite database connection.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted data.
    """

    vs_to_vs30_correlation_df = pd.read_sql_query(
        "SELECT * FROM vstovs30correlation", conn
    )
    cpt_to_vs_correlation_df = pd.read_sql_query(
        "SELECT * FROM cpttovscorrelation", conn
    )
    spt_to_vs_correlation_df = pd.read_sql_query(
        "SELECT * FROM spttovscorrelation", conn
    )
    hammer_type_df = pd.read_sql_query("SELECT * FROM spttovs30hammertype", conn)

    vs_to_vs30_correlation_id_value = int(
        vs_to_vs30_correlation_df[
            vs_to_vs30_correlation_df["name"] == selected_vs30_correlation
        ]["vs_to_vs30_correlation_id"].values[0]
    )
    cpt_to_vs_correlation_id_value = int(
        cpt_to_vs_correlation_df[
            cpt_to_vs_correlation_df["name"] == selected_cpt_to_vs_correlation
        ]["cpt_to_vs_correlation_id"].values[0]
    )
    spt_to_vs_correlation_id_value = int(
        spt_to_vs_correlation_df[
            spt_to_vs_correlation_df["name"] == selected_spt_to_vs_correlation
        ]["correlation_id"].values[0]
    )
    hammer_type_id_value = int(
        hammer_type_df[hammer_type_df["name"] == selected_hammer_type][
            "hammer_id"
        ].values[0]
    )

    # The SQLite query to extract the pre-computed Vs30 values.
    # It takes too long to extract all pre-computed CPT Vs30s values from the SQLite database,
    # so we only extract the Vs30 values that were calculated with the selected Vs to Vs30 correlation
    # (identified by vs_to_vs30_correlation_id_value) and the selected CPT to Vs correlation
    # (identified by cpt_to_vs_correlation_id_value).
    # We first filter on the Vs to Vs30 correlation, as there are only two, so this halves the number of rows
    # to search through. We also only select certain columns as some like the vs30_id column are not needed, so only
    # waste time if they are selected. We also filter using the integer id values, rather than the names as strings,
    # to save time by avoiding SQLite JOIN operations with the tables that contain the string names of the correlations.
    # In testing, this query takes about 0.4 seconds to run. If this is too slow, write the required
    # data to a parquet file and read that instead, as reading from a parquet file was found to be 10x faster in testing.
    cpt_sql_query = f"""
    WITH filtered_data AS (
        SELECT cpt_id, nzgd_id, cpt_to_vs_correlation_id, vs_to_vs30_correlation_id, vs30, vs30_stddev
        FROM cptvs30estimates
        WHERE vs_to_vs30_correlation_id = {vs_to_vs30_correlation_id_value}  -- First filter
    ), second_filter AS (
        SELECT cpt_id, nzgd_id, cpt_to_vs_correlation_id, vs_to_vs30_correlation_id, vs30, vs30_stddev
        FROM filtered_data
        WHERE cpt_to_vs_correlation_id = {cpt_to_vs_correlation_id_value}   -- Second filter
    )
    SELECT 
        sf.cpt_id, sf.nzgd_id, sf.vs30, sf.vs30_stddev,
        n.type_prefix, n.original_reference, n.investigation_date, n.published_date,
        n.latitude, n.longitude, n.model_vs30_foster_2019, n.model_vs30_stddev_foster_2019,
        n.model_gwl_westerhoff_2019, cr.tip_net_area_ratio, cr.measured_gwl,
        cr.deepest_depth, cr.shallowest_depth,
        r.name AS region_name,
        d.name AS district_name,
        sub.name AS suburb_name,
        cty.name AS city_name
    FROM second_filter AS sf
    JOIN nzgdrecord AS n
        ON sf.nzgd_id = n.nzgd_id
    JOIN region AS r
        ON n.region_id = r.region_id
    JOIN district AS d
        ON n.district_id = d.district_id
    JOIN suburb AS sub
        ON n.suburb_id = sub.suburb_id
    JOIN city AS cty
        ON n.city_id = cty.city_id
    JOIN cptreport AS cr
        ON sf.cpt_id = cr.cpt_id;
    """

    # Extract the CPT data from the SQLite database and store it in a Pandas DataFrame.
    # Also get timing points (t1, t2) to assess performance.
    t1 = time.time()
    cpt_database_df = pd.read_sql_query(cpt_sql_query, conn)
    t2 = time.time()

    # Add columns needed for the web app
    cpt_database_df["record_name"] = (
        cpt_database_df["type_prefix"].astype(str)
        + "_"
        + cpt_database_df["nzgd_id"].astype(str)
    )
    cpt_database_df["vs30_log_residual"] = np.log(cpt_database_df["vs30"]) - np.log(
        cpt_database_df["model_vs30_foster_2019"]
    )
    cpt_database_df["gwl_residual"] = (
        cpt_database_df["measured_gwl"] - cpt_database_df["model_gwl_westerhoff_2019"]
    )

    # The SQLite query to extract the SPT data.
    # There far fewer SPT Vs30 values than CPT Vs30 values, so this should be fast, regardless of the query structure.
    spt_sql_query = f"""
    WITH filtered_data AS (
        SELECT *
        FROM sptvs30estimates
        WHERE vs_to_vs30_correlation_id = {vs_to_vs30_correlation_id_value}  -- First filter
    ), second_filter AS (
        SELECT *
        FROM filtered_data
        WHERE spt_to_vs_correlation_id = {spt_to_vs_correlation_id_value}   -- Second filter
    ), third_filter AS (
        SELECT *
        FROM second_filter
        WHERE hammer_type_id = {hammer_type_id_value}   -- Second filter
    )
    SELECT 
        tf.spt_id, tf.vs30, tf.vs30_stddev,
        n.type_prefix, n.original_reference, n.investigation_date, n.published_date,
        n.latitude, n.longitude, n.model_vs30_foster_2019, n.model_vs30_stddev_foster_2019,
        n.model_gwl_westerhoff_2019, sr.measured_gwl, sr.efficiency, sr.borehole_diameter,
        r.name AS region_name,
        d.name AS district_name,
        sub.name AS suburb_name,
        cty.name AS city_name
    FROM third_filter AS tf
    JOIN nzgdrecord AS n
        ON tf.spt_id = n.nzgd_id
    JOIN sptreport AS sr
        ON tf.spt_id = sr.borehole_id
    JOIN region AS r
        ON n.region_id = r.region_id
    JOIN district AS d
        ON n.district_id = d.district_id
    JOIN suburb AS sub
        ON n.suburb_id = sub.suburb_id
    JOIN city AS cty
        ON n.city_id = cty.city_id;
    """

    # Extract the SPT data from the SQLite database and store it in a Pandas DataFrame.
    # Also get timing points (t3, t4) to assess performance.
    t3 = time.time()
    spt_partial_database_df = pd.read_sql_query(spt_sql_query, conn)
    t4 = time.time()

    spt_measurements_df = pd.read_sql_query("SELECT * FROM sptmeasurements", conn)
    # Use Pandas groupby to quickly calculate the shallowest and deepest depths for each borehole
    depth_stats_df = (
        spt_measurements_df.groupby("borehole_id")["depth"]
        .agg(shallowest_depth="min", deepest_depth="max")
        .reset_index()
    )

    spt_database_df = pd.merge(
        spt_partial_database_df,
        depth_stats_df,
        left_on="spt_id",
        right_on="borehole_id",
        how="left",
    )
    spt_database_df.drop(columns="borehole_id", inplace=True)

    # Rename and add columns needed for the web app
    spt_database_df.rename(columns={"spt_id": "nzgd_id"}, inplace=True)
    spt_database_df["record_name"] = (
        spt_database_df["type_prefix"].astype(str)
        + "_"
        + spt_database_df["nzgd_id"].astype(str)
    )
    spt_database_df["vs30_log_residual"] = np.log(spt_database_df["vs30"]) - np.log(
        spt_database_df["model_vs30_foster_2019"]
    )
    spt_database_df["gwl_residual"] = (
        spt_database_df["measured_gwl"] - spt_database_df["model_gwl_westerhoff_2019"]
    )

    print(f"Time to extract CPT Vs30s and metadata from SQLite: {t2 - t1:.2f} s")
    print(f"Time to extract SPT Vs30s and metadata from SQLite: {t4 - t3:.2f} s")

    # Concatenate the CPT and SPT dataframes so they can both be queried with a single Pandas query.
    # Columns that are only relevant for CPTs will be NaN for rows for SPTs (and vice versa).
    database_df = pd.concat([cpt_database_df, spt_database_df], ignore_index=True)
    database_df["type_number_code"] = database_df["type_prefix"].map(
        {"CPT": 0, "SCPT": 1, "BH": 2}
    )

    # rename the columns to match the web app and add prefixes of cpt spt to columns that only
    # apply to one of the two types of data
    database_df.rename(
        columns={
            "tip_net_area_ratio": "cpt_tip_net_area_ratio",
            "region_name": "region",
            "district_name": "district",
            "suburb_name": "suburb",
            "city_name": "city",
            "efficiency": "spt_efficiency",
            "borehole_diameter": "spt_borehole_diameter",
        },
        inplace=True,
    )

    return database_df


def cpt_measurements_for_one_nzgd(
    selected_nzgd_id: int, conn: sqlite3.Connection
) -> pd.DataFrame:
    """
    Extracts CPT measurements from the SQLite database for a given NZGD ID.
    Note that multiple CPT investigations (with different cpt_ids) can be returned,
    as some NZGD records contain multiple CPT investigations.

    Parameters
    ----------
    selected_nzgd_id : int
        The selected CPT ID.
    conn : sqlite3.Connection
        The SQLite database connection.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the CPT measurements and metadata as columns.
    """

    query = f"""SELECT 
    cptmeasurements.depth,
    cptmeasurements.qc,
    cptmeasurements.fs,
    cptmeasurements.u2,
    cptmeasurements.cpt_id,
    cptreport.nzgd_id
    FROM cptmeasurements
    JOIN cptreport ON cptmeasurements.cpt_id = cptreport.cpt_id
    WHERE cptreport.nzgd_id = {selected_nzgd_id}
    ORDER BY cptmeasurements.depth ASC;"""

    t1 = time.time()
    cpt_measurements_df = pd.read_sql_query(query, conn)
    t2 = time.time()

    print(
        f"Time to extract CPT measurements for cpt_id={selected_nzgd_id} from SQLite: {t2 - t1:.2f} s"
    )

    return cpt_measurements_df


def spt_measurements_for_one_nzgd(
    selected_nzgd_id: int, conn: sqlite3.Connection
) -> pd.DataFrame:
    """
    Extracts SPT measurements from the SQLite database for a given NZGD ID.

    Parameters
    ----------
    selected_nzgd_id : int
        The selected NZGD ID.
    conn : sqlite3.Connection
        The SQLite database connection.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the SPT.
    """

    query = f"""SELECT 
    sptmeasurements.depth,
    sptmeasurements.n,
    sptmeasurements.borehole_id AS nzgd_id
    FROM sptmeasurements
    WHERE sptmeasurements.borehole_id = {selected_nzgd_id}
    ORDER BY sptmeasurements.depth ASC;"""

    t1 = time.time()
    spt_measurements_df = pd.read_sql_query(query, conn)
    t2 = time.time()

    print(
        f"Time to extract CPT measurements for cpt_id={selected_nzgd_id} from SQLite: {t2 - t1:.2f} s"
    )

    return spt_measurements_df


def spt_soil_types_for_one_nzgd(
    selected_nzgd_id: int, conn: sqlite3.Connection
) -> pd.DataFrame:

    query = f"""SELECT *
FROM sptreport
JOIN soilmeasurements ON soilmeasurements.report_id = sptreport.borehole_id
JOIN soilmeasurementsoiltype ON soilmeasurementsoiltype.soil_measurement_id = soilmeasurements.measurement_id
JOIN soiltypes ON soilmeasurementsoiltype.soil_type_id = soiltypes.id
WHERE sptreport.borehole_id = {selected_nzgd_id}
ORDER BY soilmeasurements.top_depth ASC;"""

    t1 = time.time()
    spt_soil_types_df = pd.read_sql_query(query, conn)
    t2 = time.time()

    spt_soil_types_df.rename(columns={"name": "soil_type"}, inplace=True)

    print(
        f"Time to extract SPT soil types for borehole_id={selected_nzgd_id} from SQLite: {t2 - t1:.2f} s"
    )

    return spt_soil_types_df


def cpt_vs30s_for_one_nzgd_id(
    selected_nzgd_id: int, conn: sqlite3.Connection
) -> pd.DataFrame:
    """
    Extracts Vs30 values for a given CPT ID from the SQLite database.
    Note that multiple CPT investigations (with different cpt_ids) can be returned,
    as some NZGD records contain multiple CPT investigations.

    Parameters
    ----------
    selected_nzgd_id : int
        The selected NZGD ID.
    conn : sqlite3.Connection
        The SQLite database connection.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Vs30 values and related metadata.
    """

    query = f"""SELECT 
    cptvs30estimates.cpt_id,
    cptvs30estimates.nzgd_id,
    cptvs30estimates.vs30,
    cptvs30estimates.vs30_stddev, 
    cptreport.cpt_file,
    cptreport.tip_net_area_ratio,
    cptreport.measured_gwl,
    cptreport.deepest_depth,
    cptreport.shallowest_depth,   
    cpttovscorrelation.name AS cpt_to_vs_correlation,
    vstovs30correlation.name AS vs_to_vs30_correlation,
    nzgdrecord.type_prefix,    
    nzgdrecord.original_reference,
    nzgdrecord.investigation_date,
    nzgdrecord.published_date,
    nzgdrecord.latitude,
    nzgdrecord.longitude,
    nzgdrecord.model_vs30_foster_2019,
    nzgdrecord.model_vs30_stddev_foster_2019,
    nzgdrecord.model_gwl_westerhoff_2019,
    region.name AS region,
    district.name AS district,
    city.name AS city,
    suburb.name AS suburb
    FROM cptvs30estimates
    JOIN cpttovscorrelation 
      ON cptvs30estimates.cpt_to_vs_correlation_id = cpttovscorrelation.cpt_to_vs_correlation_id
    JOIN vstovs30correlation 
      ON cptvs30estimates.vs_to_vs30_correlation_id = vstovs30correlation.vs_to_vs30_correlation_id
    JOIN cptreport
      ON cptvs30estimates.cpt_id = cptreport.cpt_id
    JOIN nzgdrecord
      ON cptvs30estimates.nzgd_id = nzgdrecord.nzgd_id
    JOIN region
        ON nzgdrecord.region_id = region.region_id
    JOIN district
        ON nzgdrecord.district_id = district.district_id
    JOIN suburb
        ON nzgdrecord.suburb_id = suburb.suburb_id
    JOIN city
        ON nzgdrecord.city_id = city.city_id
    WHERE cptvs30estimates.nzgd_id = {selected_nzgd_id};"""

    t1 = time.time()
    cpt_vs30_df = pd.read_sql_query(query, conn)

    # Add columns needed for the web app
    cpt_vs30_df["record_name"] = (
        cpt_vs30_df["type_prefix"].astype(str)
        + "_"
        + cpt_vs30_df["nzgd_id"].astype(str)
    )

    # Missing values (nan or None) will raise a TypeError when trying to calculate the residuals
    # so in those cases, set the residuals to nan
    try:
        cpt_vs30_df["vs30_log_residual"] = np.log(cpt_vs30_df["vs30"]) - np.log(
            cpt_vs30_df["model_vs30_foster_2019"]
        )
    except TypeError:
        cpt_vs30_df["vs30_log_residual"] = np.nan

    cpt_vs30_df["gwl_residual"] = (
        cpt_vs30_df["measured_gwl"] - cpt_vs30_df["model_gwl_westerhoff_2019"]
    )

    # rename the columns to match the web app and add prefixes of cpt spt to columns that only
    # apply to one of the two types of data
    cpt_vs30_df.rename(
        columns={
            "tip_net_area_ratio": "cpt_tip_net_area_ratio",
            "efficiency": "spt_efficiency",
            "borehole_diameter": "spt_borehole_diameter",
        },
        inplace=True,
    )

    t2 = time.time()

    print(
        f"Time to extract Vs30s for nzgd_id={selected_nzgd_id} from SQLite: {t2 - t1:.2f} s"
    )

    return cpt_vs30_df


def spt_vs30s_for_one_nzgd_id(
    selected_nzgd_id: int, conn: sqlite3.Connection
) -> pd.DataFrame:
    """
    Extracts Vs30 values for a given SPT ID from the SQLite database.

    Parameters
    ----------
    selected_nzgd_id : int
        The selected NZGD ID.
    conn : sqlite3.Connection
        The SQLite database connection.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Vs30 values and related metadata.
    """
    query = f"""SELECT
    sptvs30estimates.spt_id,
    sptvs30estimates.borehole_diameter AS spt_borehole_diameter_for_vs30_calculation,
    sptvs30estimates.vs30,    
    sptvs30estimates.vs30_stddev,
    sptvs30estimates.vs30_used_efficiency AS spt_vs30_calculation_used_efficiency,
    sptvs30estimates.vs30_used_soil_info AS spt_vs30_calculation_used_soil_info,
    sptreport.borehole_file,
    sptreport.efficiency as spt_efficiency,
    sptreport.borehole_diameter as spt_borehole_diameter,
    sptreport.measured_gwl,
    spttovscorrelation.name AS spt_to_vs_correlation,
    vstovs30correlation.name AS vs_to_vs30_correlation,
    nzgdrecord.type_prefix,
    nzgdrecord.original_reference,
    nzgdrecord.investigation_date,
    nzgdrecord.published_date,
    nzgdrecord.latitude,
    nzgdrecord.longitude,
    nzgdrecord.model_vs30_foster_2019,
    nzgdrecord.model_vs30_stddev_foster_2019,
    nzgdrecord.model_gwl_westerhoff_2019,
    spttovs30hammertype.name AS hammer_type,
    region.name AS region,
    district.name AS district,
    city.name AS city,
    suburb.name AS suburb
    FROM sptvs30estimates
    JOIN spttovscorrelation
      ON sptvs30estimates.spt_to_vs_correlation_id = spttovscorrelation.correlation_id
    JOIN vstovs30correlation
      ON sptvs30estimates.vs_to_vs30_correlation_id = vstovs30correlation.vs_to_vs30_correlation_id
    JOIN sptreport
      ON sptvs30estimates.spt_id = sptreport.borehole_id
    JOIN spttovs30hammertype
      ON sptvs30estimates.hammer_type_id = spttovs30hammertype.hammer_id
    JOIN nzgdrecord
      ON sptvs30estimates.spt_id = nzgdrecord.nzgd_id
    JOIN region
        ON nzgdrecord.region_id = region.region_id
    JOIN district
        ON nzgdrecord.district_id = district.district_id
    JOIN suburb
        ON nzgdrecord.suburb_id = suburb.suburb_id
    JOIN city
        ON nzgdrecord.city_id = city.city_id
    WHERE sptvs30estimates.spt_id = {selected_nzgd_id};"""

    t1 = time.time()
    spt_vs30_df = pd.read_sql_query(query, conn)
    spt_vs30_df.rename(columns={"spt_id": "nzgd_id"}, inplace=True)

    spt_measurements_df = spt_measurements_for_one_nzgd(selected_nzgd_id, conn)

    spt_vs30_df["deepest_depth"] = spt_measurements_df["depth"].max()
    spt_vs30_df["shallowest_depth"] = spt_measurements_df["depth"].min()

    # Add columns needed for the web app
    spt_vs30_df["record_name"] = (
        spt_vs30_df["type_prefix"].astype(str)
        + "_"
        + spt_vs30_df["nzgd_id"].astype(str)
    )
    spt_vs30_df["vs30_log_residual"] = np.log(spt_vs30_df["vs30"]) - np.log(
        spt_vs30_df["model_vs30_foster_2019"]
    )
    spt_vs30_df["gwl_residual"] = (
        spt_vs30_df["measured_gwl"] - spt_vs30_df["model_gwl_westerhoff_2019"]
    )

    t2 = time.time()

    print(
        f"Time to extract Vs30s for nzgd_id={selected_nzgd_id} from SQLite: {t2 - t1:.2f} s"
    )

    return spt_vs30_df


if __name__ == "__main__":

    instance_path = Path("/home/arr65/src/nzgd_map_from_webplate/instance")

    vs30_correlation = "boore_2004"
    cpt_vs_correlation = "andrus_2007_pleistocene"
    spt_vs_correlation = "kwak_2015"

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

        database_df = all_vs30s_given_correlations(
            selected_vs30_correlation=vs30_correlation,
            selected_cpt_to_vs_correlation=cpt_vs_correlation,
            selected_spt_to_vs_correlation=spt_vs_correlation,
            selected_hammer_type="Auto",
            conn=conn,
        )

    database_df.loc[:, "vs30"] = clip_highest_and_lowest_percent(
        database_df["vs30"], 0.1, 99.9
    )

    test_selected_vs30_correlation = "boore_2004"
    test_selected_cpt_to_vs_correlation = "andrus_2007_pleistocene"
    test_selected_spt_to_vs_correlation = "kwak_2015"
    test_selected_hammer_type = "Auto"

    db_conn = sqlite3.connect(
        "/home/arr65/src/nzgd_data_extraction/nzgd_data_extraction/test_andrew_spt_nzgd.db"
    )
    database_df = all_vs30s_given_correlations(
        test_selected_vs30_correlation,
        test_selected_cpt_to_vs_correlation,
        test_selected_spt_to_vs_correlation,
        test_selected_hammer_type,
        db_conn,
    )

    print()

    # #record_name = "CPT_1"
    # # cpt_sqlite_query = f"SELECT depth, qc, fs, u2 FROM cptmeasurements WHERE cpt_id = {record_name.split('_')[1]}"
    # # cpt_measurements_df = pd.read_sql_query(cpt_sqlite_query, db_conn)
    #
    # #     query = """SELECT *
    # # FROM cptmeasurements
    # # JOIN cptreport ON cptmeasurements.cpt_id = cptreport.cpt_id
    # # WHERE cptreport.nzgd_id = 1;"""
    # #
    # #     cpt_measurements_df = pd.read_sql_query(query, db_conn)
    #
    # cpt_measurements_df = cpt_measurements_for_one_nzgd(199657, db_conn)
    #
    # cpt_vs30_df = cpt_vs30s_for_one_nzgd_id(199657, db_conn)
    #
    # spt_measurements_df = spt_measurements_for_one_nzgd(128969, db_conn)
    # spt_vs30_df = spt_vs30s_for_one_nzgd_id(128969, db_conn)
    #
    # spt_soil_df = spt_soil_types_for_one_nzgd(128969, db_conn)
    #
    # type_prefix_to_folder = {"CPT": "cpt", "SCPT": "scpt", "BH": "borehole"}
    # url_str_start = "https://quakecoresoft.canterbury.ac.nz/raw_from_nzgd/"
    # path_to_files = (
    #     Path(type_prefix_to_folder[cpt_vs30_df["type_prefix"][0]])
    #     / cpt_vs30_df["region"][0]
    #     / cpt_vs30_df["district"][0]
    #     / cpt_vs30_df["city"][0]
    #     / cpt_vs30_df["suburb"][0]
    #     / cpt_vs30_df["record_name"][0]
    # )
    # url_str = url_str_start + str(path_to_files)
    #
    # tip_net_area_ratio = cpt_vs30_df["cpt_tip_net_area_ratio"][0]
    # if tip_net_area_ratio is None:
    #     tip_net_area_ratio = "Not available"
    #
    # measured_gwl = cpt_vs30_df["measured_gwl"][0]
    # if measured_gwl is None:
    #     measured_gwl = "Not available"
    #

    print()
