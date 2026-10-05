-- Schema only, exported from uc_nzgd_v0p8p2_20260709_deduped.db.
-- Test data is synthetic and is supplied by conftest.py.

CREATE TABLE "city" ("id" INTEGER NOT NULL PRIMARY KEY, "value" TEXT NOT NULL);

CREATE TABLE "cptgroundwaterlevelmethod" ("id" INTEGER NOT NULL PRIMARY KEY, "value" TEXT NOT NULL);

CREATE TABLE "cptmeasurements" ("measurement_id" INTEGER NOT NULL PRIMARY KEY, "cpt_id" INTEGER NOT NULL, "depth_m" REAL, "qc_MPa" REAL, "fs_MPa" REAL, "u2_MPa" REAL, FOREIGN KEY ("cpt_id") REFERENCES "cptreport" ("cpt_id"));

CREATE TABLE "cptreport" ("cpt_id" INTEGER NOT NULL PRIMARY KEY, "nzgd_id" INTEGER NOT NULL, "max_depth_m" REAL, "min_depth_m" REAL, "extracted_gwl_m" REAL, "gwl_method_id" INTEGER, "tip_net_area_ratio" REAL, "predrill_depth_m" REAL, "termination_reason_id" INTEGER, "has_cpt_data" INTEGER NOT NULL, "did_explicit_unit_conversion" INTEGER, "did_inferred_unit_conversion" INTEGER, "source_file" TEXT NOT NULL, FOREIGN KEY ("nzgd_id") REFERENCES "nzgdrecord" ("nzgd_id"), FOREIGN KEY ("gwl_method_id") REFERENCES "cptgroundwaterlevelmethod" ("id"), FOREIGN KEY ("termination_reason_id") REFERENCES "terminationreason" ("id"));

CREATE TABLE "cpttovscorrelation" ("id" INTEGER NOT NULL PRIMARY KEY, "value" TEXT NOT NULL);

CREATE TABLE "cptvs30estimates" ("vs30_id" INTEGER NOT NULL PRIMARY KEY, "cpt_id" INTEGER NOT NULL, "nzgd_id" INTEGER NOT NULL, "cpt_to_vs_correlation_id" INTEGER NOT NULL, "vs_to_vs30_correlation_id" INTEGER NOT NULL, "vs30" REAL, "vs30_stddev" REAL, FOREIGN KEY ("cpt_id") REFERENCES "cptreport" ("cpt_id"), FOREIGN KEY ("nzgd_id") REFERENCES "nzgdrecord" ("nzgd_id"), FOREIGN KEY ("cpt_to_vs_correlation_id") REFERENCES "cpttovscorrelation" ("id"), FOREIGN KEY ("vs_to_vs30_correlation_id") REFERENCES "vstovs30correlation" ("id"));

CREATE TABLE "district" ("id" INTEGER NOT NULL PRIMARY KEY, "value" TEXT NOT NULL);

CREATE TABLE "nzgdrecord" ("nzgd_id" INTEGER NOT NULL PRIMARY KEY, "type_id" INTEGER NOT NULL, "latitude" REAL NOT NULL, "longitude" REAL NOT NULL, "model_vs30_foster_2019_m_per_s" REAL, "model_vs30_stddev_foster_2019_ln" REAL, "model_gwl_westerhoff_2018_m" REAL, "model_gwl_nlm_2025_m" REAL, "model_gwl_nlm_2025_stddev_m" REAL, "original_investigation_name" TEXT, "record_created_on" DATE, "record_last_modified_on" DATE, "region_id" INTEGER NOT NULL, "district_id" INTEGER NOT NULL, "city_id" INTEGER NOT NULL, "suburb_id" INTEGER NOT NULL, merged_into_nzgd_id INTEGER REFERENCES nzgdrecord(nzgd_id), FOREIGN KEY ("type_id") REFERENCES "type" ("id"), FOREIGN KEY ("region_id") REFERENCES "region" ("id"), FOREIGN KEY ("district_id") REFERENCES "district" ("id"), FOREIGN KEY ("city_id") REFERENCES "city" ("id"), FOREIGN KEY ("suburb_id") REFERENCES "suburb" ("id"));

CREATE TABLE "region" ("id" INTEGER NOT NULL PRIMARY KEY, "value" TEXT NOT NULL);

CREATE TABLE "soilmeasurements" ("soil_measurement_id" INTEGER NOT NULL PRIMARY KEY, "spt_id" INTEGER NOT NULL, "top_depth_m" REAL NOT NULL, "bottom_depth_m" REAL, FOREIGN KEY ("spt_id") REFERENCES "sptreport" ("spt_id"));

CREATE TABLE "soilmeasurementsoiltype" ("soil_measurement_id" INTEGER NOT NULL, "soil_type_id" INTEGER NOT NULL, PRIMARY KEY ("soil_measurement_id", "soil_type_id"), FOREIGN KEY ("soil_measurement_id") REFERENCES "soilmeasurements" ("soil_measurement_id"), FOREIGN KEY ("soil_type_id") REFERENCES "soiltypes" ("id"));

CREATE TABLE "soiltypes" ("id" INTEGER NOT NULL PRIMARY KEY, "value" TEXT NOT NULL);

CREATE TABLE "sptmeasurements" ("spt_measurement_id" INTEGER NOT NULL PRIMARY KEY, "spt_id" INTEGER NOT NULL, "depth_m" REAL, "ISPT_MAIN" INTEGER, "ISPT_NVAL" INTEGER, "ISPT_REP" INTEGER, FOREIGN KEY ("spt_id") REFERENCES "sptreport" ("spt_id"));

CREATE TABLE "sptreport" ("spt_id" INTEGER NOT NULL PRIMARY KEY, "nzgd_id" INTEGER NOT NULL, "efficiency" REAL, "extracted_gwl_m" REAL, "borehole_diameter" REAL, "casing_diameter" REAL, "source_file" TEXT NOT NULL, FOREIGN KEY ("nzgd_id") REFERENCES "nzgdrecord" ("nzgd_id"));

CREATE TABLE "spttovs30hammertype" ("id" INTEGER NOT NULL PRIMARY KEY, "value" TEXT NOT NULL);

CREATE TABLE "spttovscorrelation" ("id" INTEGER NOT NULL PRIMARY KEY, "value" TEXT NOT NULL);

CREATE TABLE "sptvs30estimates" ("vs30_id" INTEGER NOT NULL PRIMARY KEY, "spt_id" INTEGER NOT NULL, "spt_to_vs_correlation_id" INTEGER NOT NULL, "vs_to_vs30_correlation_id" INTEGER NOT NULL, "assumed_borehole_diameter_mm" REAL, "assumed_hammer_type_id" INTEGER NOT NULL, "estimate_used_extracted_efficiency" INTEGER, "estimate_used_extracted_layer_soil_types" INTEGER, "vs30" REAL, "vs30_stddev" REAL, FOREIGN KEY ("spt_id") REFERENCES "sptreport" ("spt_id"), FOREIGN KEY ("spt_to_vs_correlation_id") REFERENCES "spttovscorrelation" ("id"), FOREIGN KEY ("vs_to_vs30_correlation_id") REFERENCES "vstovs30correlation" ("id"), FOREIGN KEY ("assumed_hammer_type_id") REFERENCES "spttovs30hammertype" ("id"));

CREATE TABLE "suburb" ("id" INTEGER NOT NULL PRIMARY KEY, "value" TEXT NOT NULL);

CREATE TABLE "terminationreason" ("id" INTEGER NOT NULL PRIMARY KEY, "value" TEXT NOT NULL);

CREATE TABLE "type" ("id" INTEGER NOT NULL PRIMARY KEY, "value" TEXT NOT NULL);

CREATE TABLE "vstovs30correlation" ("id" INTEGER NOT NULL PRIMARY KEY, "value" TEXT NOT NULL);

CREATE INDEX "cptmeasurements_cpt_id" ON "cptmeasurements" ("cpt_id");

CREATE UNIQUE INDEX "cptreport_cpt_id" ON "cptreport" ("cpt_id");

CREATE INDEX "cptreport_gwl_method_id" ON "cptreport" ("gwl_method_id");

CREATE INDEX "cptreport_nzgd_id" ON "cptreport" ("nzgd_id");

CREATE INDEX "cptreport_termination_reason_id" ON "cptreport" ("termination_reason_id");

CREATE INDEX "cptreport_tip_net_area_ratio" ON "cptreport" ("tip_net_area_ratio");

CREATE INDEX "cptvs30estimates_cpt_id" ON "cptvs30estimates" ("cpt_id");

CREATE INDEX "cptvs30estimates_cpt_to_vs_correlation_id" ON "cptvs30estimates" ("cpt_to_vs_correlation_id");

CREATE INDEX "cptvs30estimates_nzgd_id" ON "cptvs30estimates" ("nzgd_id");

CREATE INDEX "cptvs30estimates_vs30" ON "cptvs30estimates" ("vs30");

CREATE INDEX "cptvs30estimates_vs_to_vs30_correlation_id" ON "cptvs30estimates" ("vs_to_vs30_correlation_id");

CREATE INDEX idx_nzgdrecord_merged_into ON nzgdrecord(merged_into_nzgd_id);

CREATE INDEX "nzgdrecord_city_id" ON "nzgdrecord" ("city_id");

CREATE INDEX "nzgdrecord_district_id" ON "nzgdrecord" ("district_id");

CREATE INDEX "nzgdrecord_model_gwl_nlm_2025_m" ON "nzgdrecord" ("model_gwl_nlm_2025_m");

CREATE INDEX "nzgdrecord_model_gwl_nlm_2025_stddev_m" ON "nzgdrecord" ("model_gwl_nlm_2025_stddev_m");

CREATE INDEX "nzgdrecord_model_gwl_westerhoff_2018_m" ON "nzgdrecord" ("model_gwl_westerhoff_2018_m");

CREATE INDEX "nzgdrecord_model_vs30_foster_2019_m_per_s" ON "nzgdrecord" ("model_vs30_foster_2019_m_per_s");

CREATE UNIQUE INDEX "nzgdrecord_nzgd_id" ON "nzgdrecord" ("nzgd_id");

CREATE INDEX "nzgdrecord_region_id" ON "nzgdrecord" ("region_id");

CREATE INDEX "nzgdrecord_suburb_id" ON "nzgdrecord" ("suburb_id");

CREATE INDEX "nzgdrecord_type_id" ON "nzgdrecord" ("type_id");

CREATE INDEX "soilmeasurements_spt_id" ON "soilmeasurements" ("spt_id");

CREATE INDEX "soilmeasurementsoiltype_soil_measurement_id" ON "soilmeasurementsoiltype" ("soil_measurement_id");

CREATE INDEX "soilmeasurementsoiltype_soil_type_id" ON "soilmeasurementsoiltype" ("soil_type_id");

CREATE INDEX "sptmeasurements_spt_id" ON "sptmeasurements" ("spt_id");

CREATE INDEX "sptreport_nzgd_id" ON "sptreport" ("nzgd_id");

CREATE INDEX "sptvs30estimates_assumed_hammer_type_id" ON "sptvs30estimates" ("assumed_hammer_type_id");

CREATE INDEX "sptvs30estimates_spt_id" ON "sptvs30estimates" ("spt_id");

CREATE INDEX "sptvs30estimates_spt_to_vs_correlation_id" ON "sptvs30estimates" ("spt_to_vs_correlation_id");

CREATE INDEX "sptvs30estimates_vs_to_vs30_correlation_id" ON "sptvs30estimates" ("vs_to_vs30_correlation_id");
