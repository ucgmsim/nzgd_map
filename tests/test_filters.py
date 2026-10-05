"""Filter semantics and rejection of executable expressions."""

import ast
import html
import re
from pathlib import Path

import pandas as pd
import pytest
from flask.testing import FlaskClient

from nzgd_map import filters
from nzgd_map import query_sqlite_db as queries
from nzgd_map.database import open_database


@pytest.fixture
def reports(database_path: Path) -> pd.DataFrame:
    with open_database(database_path) as conn:
        return queries.all_vs30s_given_correlations(
            "boore_2004",
            "andrus_2007_pleistocene",
            "brandenberg_2010",
            "Auto",
            conn,
            include_unestimated=True,
        )


@pytest.mark.parametrize(
    "expression, count",
    [
        ("vs30 > 90000", 1),
        ("vs30 >= 200 & vs30 <= 300", 3),
        ('(vs30 > 200) & (region == "Canterbury")', 3),
        ('type_prefix == "BH" and vs30_available', 2),
        ("~vs30_available", 4),
        ("vs30_available == False", 4),
        ('region in ["Canterbury", "Auckland"]', 8),
        ('region.isin(["Canterbury"])', 8),
        ('region.str.contains("CAN", case=False)', 8),
        ('region.str.contains("Can.*")', 0),
        ('source_file.str.endswith(".pdf")', 3),
        ('region.str.startswith("Can")', 8),
        ("vs30.between(200, 300)", 3),
        ("vs30_log_residual.abs() > 0.1", 2),
        ("100 < vs30 < 300", 2),
        ("extracted_gwl.isna()", 6),
        ("extracted_gwl.notna()", 2),
        ("vs30_stddev.abs() > 0", 0),
        ("  vs30 > 90000  ", 1),
        ('region == "Canterbury|Auckland"', 0),
        ('region not in ["Auckland"]', 8),
        ('region != ["Canterbury"]', 0),
        ('region == ["Canterbury"]', 8),
        ("-vs30 < -300", 1),
        ("+vs30 / 100 + 1 >= 4", 2),
        ("(vs30 * 2 - 100) % 100 == 0", 4),
        ("vs30 >= 300 or vs30 < 250", 3),
        ("True", 8),
    ],
)
def test_supported_filters(reports: pd.DataFrame, expression: str, count: int):
    assert len(filters.filter_reports(reports, expression)) == count
    filters.filter_reports(filters.empty_query_frame(), expression)


def _help_page_examples(client: FlaskClient) -> list[str]:
    """Collect complete filters shown on the help page, skipping placeholders."""
    examples = []
    for code in re.findall(r"<code>(.*?)</code>", client.get("/query_help").text):
        expression = html.unescape(code)
        try:
            body = ast.parse(expression, mode="eval").body
        except SyntaxError:
            continue  # Operators and location names, not complete filters.
        names = {node.id for node in ast.walk(body) if isinstance(node, ast.Name)}
        if not isinstance(body, ast.Name) and names <= set(filters.QUERY_FIELDS):
            examples.append(expression)
    return examples


def test_help_page_examples_match_pandas_query(
    client: FlaskClient, reports: pd.DataFrame
):
    examples = _help_page_examples(client)
    assert {'(vs30 < 100) & (region=="Canterbury")', '~(type_prefix=="BH")'} <= set(
        examples
    )
    for expression in examples:
        # Safe here: the expressions are the app's own documented examples.
        expected = reports.query(expression, engine="python")
        actual = filters.filter_reports(reports, expression)
        assert actual.index.tolist() == expected.index.tolist(), expression


@pytest.mark.parametrize(
    "expression",
    [
        "vs30 >",
        'investigation_date > "2024-01-01"',
        "vs30",
        'region.str.contains("x", regex=True)',
        "vs30.__class__",
        '__import__("os").getcwd()',
        "@os.getcwd()",
        "[x for x in region]",
        'region.str.contains("x").__class__()',
        "2 ** 100000000",
        "vs30 > 1e999",
        "nzgd_id in [nzgd_id]",
        "vs30 > [100]",
        "region in 'Canterbury'",
        "vs30 is None",
        'vs30.str.contains("1")',
        'region.str.contains("x", case=True, case=False)',
        "vs30.abs(**None)",
        '__import__("os").system("id")',
        "@pd",
        '@pd.io.common.os.system("id")',
        "vs30[0] > 1",
        'region["x"] == "y"',
        "vs30.values > 1",
        "region.str.__class__",
        "vs30.sum() > 1",
        'region.str.upper() == "X"',
        'region.str.contains("a").any()',
        "vs30.apply(print).any()",
        'getattr(vs30, "abs")() > 1',
        'open("/etc/passwd")',
        "(lambda: True)()",
        "lambda: vs30 > 1",
        "(x := vs30) > 1",
        "vs30 if vs30_available else vs30",
        'f"{vs30}" == "1"',
        "{1: vs30}",
    ],
)
def test_unsupported_filters(reports: pd.DataFrame, expression: str):
    with pytest.raises(filters.QueryError):
        filters.filter_reports(reports, expression)


def test_filter_cannot_write_files(reports: pd.DataFrame, tmp_path: Path):
    target = tmp_path / "must-not-exist.csv"
    for expression in [
        f'vs30.to_csv("{target}")',
        f'@__import__("pathlib").Path("{target}").touch()',
    ]:
        with pytest.raises(filters.QueryError):
            filters.filter_reports(reports, expression)
    assert not target.exists()


def test_blank_filter_and_size_limit(reports: pd.DataFrame):
    assert len(filters.filter_reports(reports, "  ")) == 8
    with pytest.raises(filters.QueryError, match="2,000"):
        filters.filter_reports(reports, "x" * 2001)
    with pytest.raises(filters.QueryError, match="too complex"):
        filters.filter_reports(reports, " or ".join(["vs30 > 0"] * 50))
