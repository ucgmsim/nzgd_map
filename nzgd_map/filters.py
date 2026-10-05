"""Evaluate the documented filter syntax without executing user-supplied Python."""

import ast
import io
import operator
import tokenize
from functools import reduce

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

TEXT_FIELDS = {
    "record_name",
    "report_kind",
    "source_file",
    "type_prefix",
    "original_reference",
    "original_investigation_name",
    "record_created_on",
    "record_last_modified_on",
    "region",
    "district",
    "city",
    "suburb",
    "ground_water_level_method",
    "termination_reason",
}
NUMBER_FIELDS = {
    "nzgd_id",
    "report_id",
    "cpt_id",
    "spt_id",
    "vs30",
    "vs30_stddev",
    "latitude",
    "longitude",
    "model_vs30_foster_2019",
    "model_vs30_stddev_foster_2019",
    "model_gwl_westerhoff_2018",
    "cpt_tip_net_area_ratio",
    "extracted_gwl",
    "deepest_depth",
    "shallowest_depth",
    "vs30_log_residual",
    "gwl_residual",
    "spt_efficiency",
    "spt_borehole_diameter",
    "spt_casing_diameter",
    "type_number_code",
    "predrill_depth_m",
}
QUERY_FIELDS = sorted(TEXT_FIELDS | NUMBER_FIELDS | {"vs30_available"})


class QueryError(ValueError):
    """A filter uses invalid or unsupported syntax."""


def empty_query_frame() -> pd.DataFrame:
    """Provide the same fields and types to live validation as to the map filter."""
    return pd.DataFrame(
        {
            field: pd.Series(
                dtype=(
                    "str"
                    if field in TEXT_FIELDS
                    else "bool"
                    if field == "vs30_available"
                    else "float64"
                )
            )
            for field in QUERY_FIELDS
        }
    )


def _logical_tokens(expression: str) -> str:
    # Match pandas-style precedence for &, | and ~ while leaving quoted strings
    # untouched. Python's own parser then handles grouping and comparisons.
    replacements = {"&": "and", "|": "or", "~": "not"}
    tokens = tokenize.generate_tokens(io.StringIO(expression).readline)
    return tokenize.untokenize(
        [
            (tokenize.NAME, replacements[token.string])
            if token.type == tokenize.OP and token.string in replacements
            else (token.type, token.string)
            for token in tokens
        ]
    )


def _mask(value: object, frame: pd.DataFrame) -> pd.Series:
    if isinstance(value, (bool, np.bool_)):
        return pd.Series(bool(value), index=frame.index, dtype=bool)
    if isinstance(value, pd.Series) and is_bool_dtype(value.dtype):
        return value.fillna(False)
    raise QueryError("A filter must produce a true/false result, such as vs30 > 200.")


def _numeric(value: object) -> bool:
    if isinstance(value, pd.Series):
        return is_numeric_dtype(value.dtype)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _evaluate(node: ast.AST, frame: pd.DataFrame):
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, (str, bool, int, float)):
            if isinstance(node.value, (int, float)) and not np.isfinite(
                float(node.value)
            ):
                raise QueryError("Numeric constants must be finite.")
            return node.value
    elif isinstance(node, ast.Name):
        if node.id not in QUERY_FIELDS:
            raise QueryError(f"Unknown field: {node.id}")
        column = frame[node.id]
        return (
            pd.to_numeric(column, errors="coerce")
            if node.id in NUMBER_FIELDS
            else column
        )
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_evaluate(item, frame) for item in node.elts]
        if any(isinstance(value, (pd.Series, list)) for value in values):
            raise QueryError("Membership lists must contain literal values.")
        return values
    elif isinstance(node, ast.BoolOp):
        operation = operator.and_ if isinstance(node.op, ast.And) else operator.or_
        return reduce(
            operation, (_mask(_evaluate(item, frame), frame) for item in node.values)
        )
    elif isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand, frame)
        if isinstance(node.op, ast.Not):
            return ~_mask(operand, frame)
        if _numeric(operand):
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
    elif isinstance(node, ast.BinOp):
        operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
        }
        left, right = _evaluate(node.left, frame), _evaluate(node.right, frame)
        if type(node.op) in operations and _numeric(left) and _numeric(right):
            return operations[type(node.op)](left, right)
    elif isinstance(node, ast.Compare):
        left = _evaluate(node.left, frame)
        result = pd.Series(True, index=frame.index)
        operations = {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
        }
        for operation, comparator in zip(node.ops, node.comparators, strict=True):
            right = _evaluate(comparator, frame)
            if isinstance(right, list) and isinstance(left, pd.Series):
                if not isinstance(operation, (ast.In, ast.NotIn, ast.Eq, ast.NotEq)):
                    raise QueryError("Lists support only in, not in, == and !=.")
                compared = left.isin(right)
                if isinstance(operation, (ast.NotIn, ast.NotEq)):
                    compared = ~compared
            elif type(operation) in operations:
                compared = operations[type(operation)](left, right)
            else:
                raise QueryError("Use a literal list with in or not in.")
            result &= _mask(compared, frame)
            left = right
        return result
    elif isinstance(node, ast.Call):
        return _call(node, frame)
    raise QueryError(
        "Unsupported filter syntax. See query help for supported operations."
    )


def _call(node: ast.Call, frame: pd.DataFrame):
    if not isinstance(node.func, ast.Attribute):
        raise QueryError("Only the column methods listed in query help are supported.")
    function = node.func
    arguments = [_evaluate(value, frame) for value in node.args]
    keywords = {item.arg: _evaluate(item.value, frame) for item in node.keywords}
    if len(keywords) != len(node.keywords) or None in keywords:
        raise QueryError("Invalid method arguments.")
    if isinstance(function.value, ast.Name):
        column = _evaluate(function.value, frame)
        if function.attr == "isna" and not arguments and not keywords:
            return column.isna()
        if function.attr == "notna" and not arguments and not keywords:
            return column.notna()
        if (
            function.attr == "abs"
            and not arguments
            and not keywords
            and _numeric(column)
        ):
            return column.abs()
        if (
            function.attr == "isin"
            and len(arguments) == 1
            and isinstance(arguments[0], list)
            and not keywords
        ):
            return column.isin(arguments[0])
        if (
            function.attr == "between"
            and len(arguments) == 2
            and not keywords
            and all(_numeric(value) for value in arguments)
        ):
            return column.between(*arguments)
    elif (
        isinstance(function.value, ast.Attribute)
        and function.value.attr == "str"
        and isinstance(function.value.value, ast.Name)
    ):
        name = function.value.value.id
        if name not in TEXT_FIELDS:
            raise QueryError("String methods require a text field.")
        column = frame[name].astype("string")
        if len(arguments) == 1 and isinstance(arguments[0], str):
            if function.attr == "contains" and (
                set(keywords) <= {"case", "na", "regex"}
                and isinstance(keywords.get("case", True), bool)
                and keywords.get("regex", False) is False
                and keywords.get("na", False) is False
            ):
                return column.str.contains(
                    arguments[0], case=keywords.get("case", True), regex=False, na=False
                )
            if not keywords and function.attr == "startswith":
                return column.str.startswith(arguments[0], na=False)
            if not keywords and function.attr == "endswith":
                return column.str.endswith(arguments[0], na=False)
    raise QueryError(
        "Unsupported column method or arguments. String matches use literal text."
    )


def filter_reports(frame: pd.DataFrame, expression: str) -> pd.DataFrame:
    """Apply comparisons, boolean logic and the explicitly supported column methods."""
    if not expression.strip():
        return frame.copy()
    if len(expression) > 2000:
        raise QueryError("Please keep filters under 2,000 characters.")
    try:
        parsed = ast.parse(_logical_tokens(expression.strip()), mode="eval")
        if sum(1 for _ in ast.walk(parsed)) > 200:
            raise QueryError("This filter is too complex; please simplify it.")
        return frame.loc[_mask(_evaluate(parsed.body, frame), frame)].copy()
    except QueryError:
        raise
    except (
        SyntaxError,
        tokenize.TokenError,
        TypeError,
        ValueError,
        KeyError,
        ZeroDivisionError,
        OverflowError,
    ) as error:
        raise QueryError(f"Invalid filter: {error}") from error
