"""Portable current-schema fixtures with synthetic investigations and reports."""

import sqlite3
from pathlib import Path

import pytest
from flask import Flask
from flask.testing import FlaskClient

from nzgd_map import create_app


def _insert(conn: sqlite3.Connection, table: str, rows: list[dict]):
    for row in rows:
        fields = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        conn.execute(
            f"INSERT INTO {table} ({fields}) VALUES ({placeholders})",
            tuple(row.values()),
        )


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    path = tmp_path / "nzgd.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            (Path(__file__).parent / "fixtures/current_schema.sql").read_text()
        )
        conn.execute("PRAGMA foreign_keys = ON")
        lookups = {
            "type": [(7, "CPT"), (11, "BH")],
            "region": [(7, "Canterbury"), (9, "Auckland")],
            "district": [(7, "Christchurch City")],
            "city": [(7, "Christchurch")],
            "suburb": [(7, "Central")],
            "cpttovscorrelation": [
                (17, "andrus_2007_pleistocene"),
                (19, "mcgann_2015"),
            ],
            "spttovscorrelation": [(23, "brandenberg_2010"), (29, "kwak_2015")],
            "vstovs30correlation": [(2, "boore_2004"), (3, "boore_2011")],
            "spttovs30hammertype": [(41, "Auto"), (43, "Safety")],
            "cptgroundwaterlevelmethod": [(7, "Measured")],
            "terminationreason": [(7, "Target depth")],
            "soiltypes": [(1, "SAND"), (2, "SILT"), (3, "CLAY"), (4, "GRAVEL")],
        }
        for table, values in lookups.items():
            conn.executemany(f"INSERT INTO {table} (id, value) VALUES (?, ?)", values)
        for nzgd_id, type_id in [
            (1001, 7),
            (1002, 11),
            (1003, 7),
            (1004, 11),
            (1006, 7),
            (31, 11),
        ]:
            _insert(
                conn,
                "nzgdrecord",
                [
                    {
                        "nzgd_id": nzgd_id,
                        "type_id": type_id,
                        "latitude": -43.5,
                        "longitude": 172.6,
                        "model_vs30_foster_2019_m_per_s": None
                        if nzgd_id == 1006
                        else 250.0,
                        "model_vs30_stddev_foster_2019_ln": None
                        if nzgd_id == 1006
                        else 0.35,
                        "model_gwl_westerhoff_2018_m": None if nzgd_id == 1006 else 1.2,
                        "original_investigation_name": f"Investigation {nzgd_id}",
                        "record_created_on": "2025-03-01",
                        "record_last_modified_on": "2026-07-09",
                        "region_id": 7,
                        "district_id": 7,
                        "city_id": 7,
                        "suburb_id": 7,
                    }
                ],
            )
        conn.execute(
            """INSERT INTO nzgdrecord (
                nzgd_id, type_id, latitude, longitude, region_id, district_id,
                city_id, suburb_id, merged_into_nzgd_id
            ) VALUES (9001, 7, -43.5, 172.6, 7, 7, 7, 7, 1001)"""
        )
        for cpt_id, nzgd_id, depth, has_data in [
            (13, 1001, 20, 1),
            (17, 1001, 8, 1),
            (19, 1001, None, 0),
            (23, 1003, 4, 1),
            (24, 1006, 30, 1),
        ]:
            _insert(
                conn,
                "cptreport",
                [
                    {
                        "cpt_id": cpt_id,
                        "nzgd_id": nzgd_id,
                        "max_depth_m": depth,
                        "min_depth_m": 0.1 if has_data else None,
                        "extracted_gwl_m": 0.0 if cpt_id == 13 else None,
                        "tip_net_area_ratio": 0.8 if cpt_id == 13 else 0.7,
                        "gwl_method_id": 7 if cpt_id == 13 else None,
                        "termination_reason_id": 7 if cpt_id == 13 else None,
                        "has_cpt_data": has_data,
                        "source_file": f"cpt_source_{cpt_id}.ags",
                    }
                ],
            )
            if has_data:
                for step in (1, 2):
                    _insert(
                        conn,
                        "cptmeasurements",
                        [
                            {
                                "cpt_id": cpt_id,
                                "depth_m": step * depth / 2,
                                "qc_MPa": cpt_id + step / 10,
                                "fs_MPa": step / 100,
                                "u2_MPa": None if step == 2 else 0.005,
                            }
                        ],
                    )
        for spt_id, nzgd_id in [
            (31, 1002),
            (47, 1002),
            (49, 1002),
            (71, 1004),
            (81, 31),
        ]:
            _insert(
                conn,
                "sptreport",
                [
                    {
                        "spt_id": spt_id,
                        "nzgd_id": nzgd_id,
                        "efficiency": 80 if spt_id == 31 else None,
                        "extracted_gwl_m": 1.0 if spt_id == 31 else None,
                        "borehole_diameter": 150 if spt_id == 31 else None,
                        "source_file": f"spt_source_{spt_id}.ags"
                        if spt_id == 31
                        else f"spt_source_{spt_id}.pdf",
                    }
                ],
            )
        for spt_id, depth, main, nval, rep in [
            (31, 1, 5, 12, 9),
            (31, 3, 0, None, 88),
            (31, 6, 10, 0, 3),
            (31, 12, 10, 20, 15),
            (31, 14, None, None, 73),
            (47, 2, 40, None, None),
            (47, 7, 15, 30, None),
            (71, 3, 7, None, None),
            (81, 4, 999, None, None),
        ]:
            _insert(
                conn,
                "sptmeasurements",
                [
                    {
                        "spt_id": spt_id,
                        "depth_m": depth,
                        "ISPT_MAIN": main,
                        "ISPT_NVAL": nval,
                        "ISPT_REP": rep,
                    }
                ],
            )
        for cpt_id, vs_method, depth_method, vs30 in [
            (13, 17, 2, 250),
            (13, 17, 3, 260),
            (17, 17, 3, 350),
            (24, 17, 2, 100000),
            (13, 19, 2, 270),
        ]:
            nzgd_id = conn.execute(
                "SELECT nzgd_id FROM cptreport WHERE cpt_id=?", (cpt_id,)
            ).fetchone()[0]
            _insert(
                conn,
                "cptvs30estimates",
                [
                    {
                        "cpt_id": cpt_id,
                        "nzgd_id": nzgd_id,
                        "cpt_to_vs_correlation_id": vs_method,
                        "vs_to_vs30_correlation_id": depth_method,
                        "vs30": vs30,
                    }
                ],
            )
        for spt_id, method, vs30, diameter, efficiency, soils in [
            (31, 2, 200, 150, 1, 0),
            (31, 3, 220, 200, 0, 1),
            (47, 2, 300, 100, 0, 1),
        ]:
            _insert(
                conn,
                "sptvs30estimates",
                [
                    {
                        "spt_id": spt_id,
                        "spt_to_vs_correlation_id": 23,
                        "vs_to_vs30_correlation_id": method,
                        "vs30": vs30,
                        "assumed_borehole_diameter_mm": diameter,
                        "assumed_hammer_type_id": 41,
                        "estimate_used_extracted_efficiency": efficiency,
                        "estimate_used_extracted_layer_soil_types": soils,
                    }
                ],
            )
        for layer_id, spt_id, top, bottom, types in [
            (100, 31, 0, 2, [1, 2]),
            (101, 31, 2, None, [3]),
            (102, 31, 2, 5, [1]),
            (103, 47, 0, 4, [4]),
            (104, 49, 0, 1.5, []),
        ]:
            _insert(
                conn,
                "soilmeasurements",
                [
                    {
                        "soil_measurement_id": layer_id,
                        "spt_id": spt_id,
                        "top_depth_m": top,
                        "bottom_depth_m": bottom,
                    }
                ],
            )
            conn.executemany(
                "INSERT INTO soilmeasurementsoiltype VALUES (?, ?)",
                [(layer_id, soil_type) for soil_type in types],
            )
        assert not conn.execute("PRAGMA foreign_key_check").fetchall()
    return path


@pytest.fixture
def app(database_path: Path, tmp_path: Path) -> Flask:
    return create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "fixture-session-key",
            "DATABASE": database_path,
            "GEONET_STATIONS_PATH": None,
        },
        instance_path=str(tmp_path),
    )


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    return app.test_client()
