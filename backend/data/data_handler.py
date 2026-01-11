import os
import pandas as pd


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
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
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
# Initial Data Checks
# -----------------------------
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
        cleaned_df.columns
        .astype(str)
        .str.strip()
        .str.replace(" ", "_")
    )

    cleaned_df = cleaned_df.drop_duplicates()

    return cleaned_df
