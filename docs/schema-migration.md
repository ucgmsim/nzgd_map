# Migration to the July 2026 database

## Scope and status

This note covers database access, the map and filters, record pages, and
downloads for the July 2026 database.

The queries, map, record pages, and downloads now use the current schema. The app
opens SQLite in read-only mode, preserves the configured session secret, and
allows the optional retrieval-date and GeoNet files to be absent. Local startup
instructions are in [the README](../readme.md#local-development).

Configuration can be supplied through `instance/config.py` or these environment
variables:

| Environment variable | Flask config key | Default |
| --- | --- | --- |
| `NZGD_DATABASE_PATH` | `DATABASE` | `instance/extracted_nzgd.db` |
| `SECRET_KEY` | `SECRET_KEY` | Required; no built-in secret |
| `NZGD_LAST_RETRIEVAL_DATE` | `LAST_NZGD_RETRIEVAL_DATE` | Read the optional `instance/date_of_last_nzgd_retrieval.txt`; otherwise show "Not provided" |
| `NZGD_GEONET_STATIONS_PATH` | `GEONET_STATIONS_PATH` | Optional `instance/geoNet_stats+2023-06-28.ll` |

The instance directory is Flask's configured instance path, which can differ
between a source checkout and an installed package. `instance/config.py` takes
precedence over environment defaults.

Inspected on 2026-09-21, from the NZGD extraction project:

- Database: `uc_nzgd_v0p8p2_20260709_deduped.db`
- Schema: `nzgd/db/orm.py`
- Import and deduplication scripts: `nzgd/scripts/db/`
- Vs30 publishing code: `nzgd/scripts/estimate_vs30/batch.py`

## Architecture

This is a server-rendered Flask application using pandas for queries and Plotly
for charts. Responsibilities are split across:

- `database.py`: read-only connections and connection cleanup.
- `query_sqlite_db.py`: SQL and dataframe transformations, preserving report IDs.
- `views.py`: the map, query validation/help, assets, and optional GeoNet overlay.
- `records.py`: CPT/SPT detail pages, per-report plots, and CSV downloads.
- `plotting.py`: map and histogram figures, including unavailable values.
- `filters.py`: the documented pandas-style filter operations.
- `templates/`: shared record metadata, report sections, and estimate tables.

The map starts from measured reports and attaches selected estimates by report
ID. Reports and estimates are queried separately: the SQLite planner chose a
correlation-index scan for each report in a combined LEFT JOIN, making that
version impractically slow. Separate queries and a validated one-to-one pandas
merge read all 65,801 measured reports in approximately 0.8 seconds locally.
Detail-page metadata is independent of whether an estimate exists.

## Schema mappings

| Existing app reference | Current database reference / meaning |
| --- | --- |
| Lookup-table `name` and table-specific ID columns | `value` and `id` |
| `nzgdrecord.type_prefix` | Join `nzgdrecord.type_id` to `type.id`; read `type.value` |
| `original_reference` | `original_investigation_name` |
| `investigation_date`, `published_date` | Removed. `record_created_on` and `record_last_modified_on` are NZGD record timestamps, not investigation/publication dates. |
| `model_vs30_foster_2019` | `model_vs30_foster_2019_m_per_s` |
| `model_vs30_stddev_foster_2019` | `model_vs30_stddev_foster_2019_ln` (natural-log units) |
| `model_gwl_westerhoff_2018` | `model_gwl_westerhoff_2018_m` |
| CPT `deepest_depth`, `shallowest_depth`, `extracted_gwl` | `max_depth_m`, `min_depth_m`, `extracted_gwl_m` |
| `groundwaterlevelmethod`, CPT `ground_water_level_method_id` | `cptgroundwaterlevelmethod`, CPT `gwl_method_id` |
| CPT measurement `depth`, `qc`, `fs`, `u2` | `depth_m`, `qc_MPa`, `fs_MPa`, `u2_MPa` |
| `sptreport.borehole_id` | `sptreport.spt_id`, a report ID distinct from `nzgd_id` |
| `sptreport.borehole_file` | `sptreport.source_file` |
| SPT measurement `borehole_id`, `depth`, `n` | `spt_id`, `depth_m`, and separate `ISPT_MAIN`, `ISPT_NVAL`, `ISPT_REP` fields |
| Soil `report_id`, `measurement_id`, `top_depth` | `spt_id`, `soil_measurement_id`, `top_depth_m`; `bottom_depth_m` is also available |
| SPT estimate `hammer_type_id` | `assumed_hammer_type_id` |
| SPT estimate `borehole_diameter` | `assumed_borehole_diameter_mm` |
| SPT estimate `vs30_used_efficiency` | `estimate_used_extracted_efficiency` |
| SPT estimate `vs30_used_soil_info` | `estimate_used_extracted_layer_soil_types` |

SPT estimates, measurements, and soils must join through `sptreport.spt_id` to
reach `sptreport.nzgd_id`. Joining SPT IDs directly to NZGD IDs returns the wrong
investigation. Soil grouping and plots must preserve report identity.

The current `type` table contains `CPT` and `BH`; SCPT records are represented as
CPT by the metadata import code. Do not infer an SCPT classification absent from
the database.

`nzgdrecord.merged_into_nzgd_id` is added by deduplication, outside the core ORM.
There are 1,286 merged records and 65,452 records without a merge target. Reports
are associated with surviving records; old detail links can use the merge target.

Additional metadata includes `model_gwl_nlm_2025_m`,
`model_gwl_nlm_2025_stddev_m`, predrill depth, CPT unit-conversion flags, and
SPT casing diameter. Exposing additional analysis options is separate from
correctly reading the fields the app already uses.

## Coverage and missing values

| Quantity | Count |
| --- | ---: |
| CPT reports | 54,691 |
| CPT reports flagged as having measurements | 49,489 |
| SPT reports | 22,036 |
| SPT reports with measurement rows | 16,312 |
| CPT reports with at least one Vs30 estimate | 43,231 |
| SPT reports with at least one Vs30 estimate | 6,657 |
| CPT estimates for the default correlation pair | 30,824 |
| SPT estimates for the default correlation pair and Auto hammer | 5,518 |
| NZGD records with multiple CPT reports | 4,495 |
| NZGD records with multiple SPT reports | 6,033 |

There are 514,444 CPT and 24,350 SPT estimate rows. Every stored `vs30_stddev` is
NULL, intentionally: the current publishing code publishes central estimates
without uncertainty. Missing uncertainty must not be presented as zero or
formatted using a numeric-only template expression.

The SPT publishing code prefers finite `ISPT_NVAL`, falling back to `ISPT_MAIN`.
There are 2,576 measurement rows where both fields are present and differ.
`ISPT_REP` is retained in the database but is not used by that selection rule.

The database does not supply the separate NZGD retrieval-date file or default
GeoNet station file expected by the app. Neither file exists in this checkout.
The database filename and record timestamps do not establish the retrieval date.

## Approved behaviour

1. The availability selector defaults to **all reports with measurements**, with
   the existing correlations selected. Its other choices are **with an estimate**
   and **without an estimate**. Availability means a finite, positive stored
   Vs30 for the chosen correlations; uncertainty and a model residual are not
   required. SPT retains the existing Auto hammer selection.
2. Each NZGD detail page contains distinct report sections with their own source
   filename, metadata, plot, stored estimates, and downloads. Metadata-only
   reports remain visible on the detail page. Map links include a report anchor.
3. SPT plots select finite `ISPT_NVAL`, then finite `ISPT_MAIN`, matching the
   publishing code. Zero is a valid value. The rule appears beside the plot;
   original `ISPT_MAIN`, `ISPT_NVAL`, and `ISPT_REP` remain in downloads and the
   report's raw-value table.

The custom query applies before availability filtering. A query such as
`vs30 > 200` therefore excludes missing estimates even when availability is set
to all. Counts distinguish reports from NZGD records. Histograms omit missing
values and report how many were omitted. Grey markers indicate that the chosen
colour variable is unavailable. Reports without a residual remain visible at
the minimum marker size. Overlapping reports share their record's coordinates;
the detail page lists every report at that record.

Map colour limits use quantiles without changing stored values, hover values,
query results, or histogram data. An absent uncertainty is shown as unavailable.
SPT estimate tables display the recorded diameter, hammer type, and efficiency
and soil-information flags separately for each estimate. Soil classifications
are grouped by layer ID within a report; thickness comes from the recorded top
and bottom depths, and remains unknown when the bottom is absent.

## URL, query, and CSV compatibility

Existing `/cpt/CPT_<nzgd_id>` and `/spt/BH_<nzgd_id>` detail URLs remain available.
Merged records redirect to their surviving record. Legacy SCPT names redirect
to CPT when that matches the stored record. Unknown or mismatched records and
reports return 404.

Map URLs add `vs30_availability=all|available|unavailable`. Existing correlation,
colour, histogram, and query parameters remain. GeoNet upload, reset, and toggle
actions retain the current selection. Location names in query help are sorted.

Filters support comparisons, boolean logic, numeric arithmetic, membership,
`isna()`, `notna()`, `abs()`, `between()`, and the documented string methods.
They are interpreted using a limited syntax rather than passing user input to
`DataFrame.query()`, which can execute arbitrary code
([pandas documentation](https://pandas.pydata.org/docs/dev/reference/api/pandas.DataFrame.query.html)).
`str.contains()` now matches literal text; regular expressions, arbitrary Python
calls, and other pandas methods are unsupported. Live validation and submitted
queries use the same interpreter. Saved queries using the removed
`investigation_date` or `published_date` fields need review: the new record
timestamps have different meanings. `original_reference` remains an alias for
`original_investigation_name`.

Record-wide CSV download URLs are retained and include report identity on every
row. New per-report downloads use
`/cpt/<record_name>/reports/<cpt_id>/data.csv` and
`/spt/<record_name>/reports/<spt_id>/data.csv`. Soil downloads use the latter path
with `soil_types.csv`. CSV columns now use explicit units and preserve provenance:

| Download | Columns |
| --- | --- |
| CPT | `nzgd_id`, `cpt_id`, `source_file`, `depth_m`, `qc_MPa`, `fs_MPa`, `u2_MPa` |
| SPT | `nzgd_id`, `spt_id`, `source_file`, `depth_m`, `ISPT_MAIN`, `ISPT_NVAL`, `ISPT_REP`, `number_of_blows`, `n_value_source` |
| Soil | `spt_id`, `soil_measurement_id`, `nzgd_id`, `source_file`, `top_depth_m`, `bottom_depth_m`, `soil_type`, `layer_thickness_m` |

Consumers relying on old CSV column names must update those names. Empty
measurement downloads still include column headers.

## Validation

Portable tests use synthetic data and a schema-only fixture exported from the
supplied database. They cover distinct report/NZGD IDs, multiple reports, missing
estimates and metadata, selected N values, soil intervals, correlation changes,
downloads, filter syntax, invalid URLs, mounted URLs, and GeoNet file handling.
The inherited CI workflow now installs the app's test dependencies and measures
`nzgd_map` coverage, retaining its 95% minimum.

On 2026-09-21, all 95 tests passed with 97.86% statement coverage. Ruff lint and
format checks, dependency checks, and wheel packaging passed. The installed wheel
served map and record pages, data, and JavaScript assets against the supplied
database. Chrome checks covered both fixtures and the full database, with no
JavaScript errors. Full-database browser results:

| Selection | Reports |
| --- | ---: |
| All measured reports | 65,801 |
| Default correlations, with estimate | 36,342 |
| Default CPT/SPT correlations and Boore 2011, with estimate | 47,637 |
| Default CPT/SPT correlations and Boore 2011, without estimate | 18,164 |

The full map response is approximately 13.3 MB before HTTP compression; local
browser startup took about 3.6 seconds. Browser checks exercised the availability
controls, correlation changes, live validation, report plots, and CSV download.
Timing is specific to the development workstation.

The browser now receives the JavaScript bundled with the installed Plotly Python
package through a versioned asset URL. This avoids mismatches with the old
checked-in JavaScript and supports `scatter_map` (Plotly.py 5.24 or later;
[Plotly migration announcement](https://plotly.com/blog/plotly-is-switching-to-maplibre/)).
