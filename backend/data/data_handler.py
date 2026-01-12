import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

# -----------------------------
# Supported file formats
# -----------------------------
SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
    ".parquet",
}


# -----------------------------
# File Loading
# -----------------------------
def load_dataset(uploaded_file) -> pd.DataFrame:
    """
    Load uploaded dataset into a Pandas DataFrame.
    """
    _, ext = os.path.splitext(uploaded_file.name.lower())

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".csv":
        return pd.read_csv(uploaded_file)

    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)

    if ext == ".json":
        return pd.read_json(uploaded_file)

    if ext == ".parquet":
        return pd.read_parquet(uploaded_file)


# -----------------------------
# Basic inspection utilities
# -----------------------------
def get_columns(df: pd.DataFrame) -> List[str]:
    return list(df.columns)


def get_column_dtypes(df: pd.DataFrame) -> Dict[str, str]:
    return df.dtypes.astype(str).to_dict()


def get_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Perform initial data quality checks.
    """
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_total": int(df.isnull().sum().sum()),
        "missing_values_by_column": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
    }


# -----------------------------
# Basic Cleaning
# -----------------------------
def basic_data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safe initial cleaning:
    - Normalize column names
    - Remove duplicate rows
    """
    cleaned_df = df.copy()
    cleaned_df.columns = (
        cleaned_df.columns.astype(str).str.strip().str.replace(" ", "_")
    )
    cleaned_df = cleaned_df.drop_duplicates()
    return cleaned_df


# -----------------------------
# Conversion helpers & rules
# -----------------------------
def _normalize_dtype_name(pd_dtype: Any) -> str:
    """
    Map pandas dtype to a canonical string used in UI:
    int, float, str, bool, datetime, category, object
    """
    pd_dtype = str(pd_dtype).lower()
    if "int" in pd_dtype and "uint" not in pd_dtype:
        return "int"
    if "float" in pd_dtype:
        return "float"
    if "bool" in pd_dtype:
        return "bool"
    if "datetime" in pd_dtype or "datedatetime" in pd_dtype:
        return "datetime"
    if "category" in pd_dtype:
        return "category"
    if "object" in pd_dtype:
        return "object"
    return "str"


def _is_integer_like(series: pd.Series) -> bool:
    s = series.dropna()
    if s.empty:
        return True
    if pd.api.types.is_integer_dtype(s):
        return True
    # If numeric, check fractional part
    if pd.api.types.is_float_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return np.all(np.equal(np.mod(s.astype(np.float64), 1), 0))
    # If strings that can be converted to numeric, check converted values
    try:
        conv = pd.to_numeric(s, errors="coerce")
        if conv.isna().any():
            return False
        return np.all(np.equal(np.mod(conv.astype(np.float64), 1), 0))
    except Exception:
        return False


def _is_numeric_string(series: pd.Series) -> bool:
    s = series.dropna().astype(str)
    if s.empty:
        return True
    conv = pd.to_numeric(s, errors="coerce")
    return not conv.isna().any()


def _is_datetime_string(series: pd.Series) -> bool:
    s = series.dropna().astype(str)
    if s.empty:
        return True
    conv = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    return not conv.isna().any()


def _is_boolean_string(series: pd.Series) -> bool:
    s = series.dropna().astype(str).str.lower()
    if s.empty:
        return True
    unique = set(s.unique())
    return unique.issubset({"true", "false", "0", "1", "t", "f"})


def get_allowed_conversions(df: pd.DataFrame, column: str) -> List[str]:
    """
    Return allowed target dtype names for a given column based on content and dtype.
    Allowed canonical types: int, float, str, bool, datetime, category
    """
    if column not in df.columns:
        return []

    current = _normalize_dtype_name(df[column].dtype)

    allowed = set()

    if current == "int":
        allowed.update(["int", "float", "str", "object"])
    elif current == "float":
        allowed.update(["float", "int", "str", "object"])
    elif current in ("str", "object"):
        # try numeric
        if _is_numeric_string(df[column]):
            allowed.update(["str", "int", "float", "object"])
        # try datetime
        elif _is_datetime_string(df[column]):
            allowed.update(["str", "datetime", "object"])
        # try boolean
        elif _is_boolean_string(df[column]):
            allowed.update(["str", "bool", "object"])
        else:
            allowed.update(["str", "object"])
    elif current == "bool":
        allowed.update(["bool", "int", "float", "str", "object"])
    elif current == "datetime":
        allowed.update(["datetime", "str", "object"])
    elif current == "category":
        allowed.update(["category", "str", "object"])
    else:
        allowed.add(current)

    # Normalize order and return list
    ordered = ["int", "float", "str", "bool", "datetime", "category", "object"]
    return [t for t in ordered if t in allowed]


def change_column_dtype(df: pd.DataFrame, column: str, target: str) -> Tuple[pd.DataFrame, str]:
    """
    Attempt to convert column to target dtype. Return (new_df, message).
    Raises ValueError if conversion is invalid.
    """
    if column not in df.columns:
        raise ValueError("Column not found")

    s = df[column]
    target = target.lower()

    # identity
    if target == _normalize_dtype_name(s.dtype):
        return df.copy(), f"No change needed: already {target}"

    new_df = df.copy()
    try:
        if target == "int":
            conv = pd.to_numeric(s, errors="coerce")
            # Check for invalid conversions only if original was string/object
            original_dtype = _normalize_dtype_name(s.dtype)
            if original_dtype in ("str", "object") and conv.isna().any():
                raise ValueError("Column contains values that cannot be converted to numeric")
            if np.isinf(conv).any():
                raise ValueError("Column contains infinite values that cannot be converted to int")
            # Check for overflow
            if conv.dropna().astype(float).gt(np.iinfo(np.int64).max).any() or conv.dropna().astype(float).lt(np.iinfo(np.int64).min).any():
                raise ValueError("Column contains values too large or small for int64")
            # choose dtype that supports NA
            if conv.isna().any():
                new_df[column] = conv.astype("Int64")
            else:
                new_df[column] = conv.astype("int64")
        elif target == "float":
            conv = pd.to_numeric(s, errors="coerce")
            # Check for invalid conversions only if original was string/object
            original_dtype = _normalize_dtype_name(s.dtype)
            if original_dtype in ("str", "object") and conv.isna().any():
                raise ValueError("Column contains values that cannot be converted to numeric")
            new_df[column] = conv.astype("float64")
        elif target == "str":
            new_df[column] = s.astype(str)
        elif target == "bool":
            # allow 0/1 or True/False strings
            if pd.api.types.is_bool_dtype(s):
                new_df[column] = s.astype(bool)
            else:
                conv = pd.Series(pd.to_numeric(s, errors="coerce"))
                # If numeric with only 0/1
                if not conv.isna().all() and set(conv.dropna().unique()).issubset({0, 1}):
                    new_df[column] = conv.fillna(0).astype("Int64").astype(bool)
                else:
                    # try boolean strings
                    lowered = s.dropna().astype(str).str.lower()
                    if set(lowered.unique()).issubset({"true", "false", "0", "1", "t", "f"}):
                        mapping = {"true": True, "false": False, "1": True, "0": False, "t": True, "f": False}
                        new_df[column] = s.astype(str).str.lower().map(mapping).astype("boolean")
                    else:
                        raise ValueError("Cannot safely convert column to bool")
        elif target == "datetime":
            conv = pd.to_datetime(s, errors="raise", infer_datetime_format=True)
            new_df[column] = conv
        elif target == "category":
            new_df[column] = s.astype("category")
        elif target == "object":
            new_df[column] = s.astype(object)
        else:
            raise ValueError(f"Unknown target dtype: {target}")

        return new_df, f"Column '{column}' converted to {target}"
    except Exception as e:
        raise ValueError(f"Failed to convert column '{column}' to {target}: {e}")


def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not in dataframe: {missing}")
    new_df = df.copy().drop(columns=columns)
    return new_df


def impute_column(df: pd.DataFrame, column: str, method: str, constant_value: Any = None, percentile: float = 50.0, window: int = 3) -> pd.DataFrame:
    """
    Impute missing values in a single column using a chosen method.
    Supported methods:
      - mean, median, mode
      - constant (requires constant_value)
      - ffill, bfill
      - interpolate (linear) [numeric]
      - percentile (numeric)
      - rolling_mean (numeric) with window param
    """
    if column not in df.columns:
        raise ValueError("Column not found")

    s = df[column]
    new_df = df.copy()

    if method == "mean":
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError("Mean imputation only valid for numeric columns")
        val = s.mean()
        new_df[column] = s.fillna(val)
    elif method == "median":
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError("Median imputation only valid for numeric columns")
        val = s.median()
        new_df[column] = s.fillna(val)
    elif method == "mode":
        try:
            val = s.mode(dropna=True)
            fill = val.iloc[0] if len(val) > 0 else constant_value
            new_df[column] = s.fillna(fill)
        except Exception:
            new_df[column] = s.fillna(constant_value)
    elif method == "constant":
        if constant_value is None:
            raise ValueError("constant_value required for constant imputation")
        new_df[column] = s.fillna(constant_value)
    elif method == "ffill":
        new_df[column] = s.fillna(method="ffill")
    elif method == "bfill":
        new_df[column] = s.fillna(method="bfill")
    elif method == "interpolate":
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError("Interpolate only supported for numeric columns")
        new_df[column] = s.interpolate(method="linear")
    elif method == "percentile":
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError("Percentile fill only supported for numeric columns")
        pval = np.nanpercentile(s.astype(float), percentile)
        new_df[column] = s.fillna(pval)
    elif method == "rolling_mean":
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError("Rolling mean only valid for numeric columns")
        filled = s.copy()
        rolled = s.fillna(s.rolling(window=window, min_periods=1).mean())
        new_df[column] = rolled
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    return new_df


# -----------------------------
# Time Series Data Preparation
# -----------------------------
def prepare_time_series_data(df: pd.DataFrame, date_col: str, value_col: str) -> Tuple[pd.DataFrame, str]:
    """
    Prepare data for time series anomaly detection.
    Ensures date column is datetime and sorts by date.
    Returns (prepared_df, message)
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in dataframe")

    prepared_df = df.copy()

    # Convert date column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(prepared_df[date_col]):
        try:
            prepared_df[date_col] = pd.to_datetime(prepared_df[date_col], errors='raise')
        except Exception as e:
            raise ValueError(f"Cannot convert '{date_col}' to datetime: {e}")

    # Sort by date
    prepared_df = prepared_df.sort_values(date_col).reset_index(drop=True)

    # Check for numeric value column
    if not pd.api.types.is_numeric_dtype(prepared_df[value_col]):
        raise ValueError(f"Value column '{value_col}' must be numeric for anomaly detection")

    message = f"Data prepared: {len(prepared_df)} rows, sorted by {date_col}"
    return prepared_df, message
