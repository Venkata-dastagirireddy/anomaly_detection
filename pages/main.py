import streamlit as st
st.set_page_config(
        page_title="Anomaly Detection", 
        layout="wide",
        page_icon=":material/area_chart:",
        initial_sidebar_state="collapsed")
import pandas as pd
import math
from backend.data.data_handler import (
    load_dataset,
    get_data_quality_report,
    basic_data_cleaning,
    get_columns,
    get_column_dtypes,
    get_allowed_conversions,
    change_column_dtype,
    drop_columns,
    impute_column,
)


# Utility: store history for undo
def _push_history():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    # store copy of current working df
    st.session_state["history"].append(st.session_state["working_df"].copy())


def _undo_last():
    if "history" in st.session_state and st.session_state["history"]:
        st.session_state["working_df"] = st.session_state["history"].pop()
        st.success("Reverted last operation")
    else:
        st.warning("Nothing to undo")


def _reset_to_original():
    st.session_state["working_df"] = st.session_state["original_df"].copy()
    st.session_state["history"] = []
    # Reset widget states
    st.session_state["col_to_drop"] = []
    st.session_state["drop_na_cols"] = []
    st.toast("Reset to original uploaded data")
    st.rerun()


def _safe_set_working(df: pd.DataFrame, message: str = None):
    _push_history()
    st.session_state["working_df"] = df.copy()
    if message:
        st.success(message)


def app():
    st.title("Anomaly Detection")
    st.caption("Upload, inspect, and prepare your time series data.")

    st.divider()

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=["csv", "xlsx", "xls", "json", "parquet"],
            help="Supported formats: CSV, Excel, JSON, Parquet",
        )
        if uploaded_file is None:
            st.info("Please upload a dataset to proceed.")
            return

    # If first time upload (or new file), load and initialize session state
    is_new_file = "uploaded_filename" not in st.session_state or st.session_state.get("uploaded_filename") != uploaded_file.name
    if is_new_file:
        try:
            with st.spinner("Loading dataset..."):
                df = load_dataset(uploaded_file)
            st.session_state["uploaded_filename"] = uploaded_file.name
            st.session_state["original_df"] = df.copy()
            st.session_state["working_df"] = df.copy()
            st.session_state["history"] = []
            st.toast("Dataset loaded successfully", icon=":material/check_circle:")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return

    # alias for convenience
    df = st.session_state["working_df"]
    original_df = st.session_state["original_df"]


    left_col, right_col = st.columns([0.7, 0.3], gap="medium", border=True)
    with left_col:
        st.subheader(":material/data_table: Original Data Preview")

        # Toggle full / paged display
        show_full = st.checkbox("Show full data", value=False)
        if show_full:
            st.dataframe(original_df, use_container_width=True, hide_index=False)
        else:
            # show paged sample with pagination controls
            preview_rows = st.number_input("Rows to preview", min_value=5, max_value=1000, value=100, step=5)
            st.dataframe(original_df.head(int(preview_rows)), use_container_width=True, hide_index=False)

    with right_col:
        st.subheader(":material/summarize: Original Data Summary")
        st.markdown(
            f"**File:** `{st.session_state.get('uploaded_filename', '')}`"
        )
        report = get_data_quality_report(original_df)

        col1, col2 = st.columns([1,1])
        with col1:
            st.metric("Rows", report["rows"])
            st.metric("Missing Values", report["missing_values_total"])
        with col2:
            st.metric("Columns", report["columns"])
            st.metric("Duplicates", report["duplicate_rows"])

        with st.expander("Column types & counts"):
            dtypes = get_column_dtypes(original_df)
            dt_table = pd.DataFrame.from_dict(dtypes, orient="index", columns=["dtype"])
            dt_table["missing"] = original_df.isnull().sum().astype(int)
            dt_table["unique_values"] = original_df.nunique(dropna=True)
            st.dataframe(dt_table, use_container_width=True, hide_index=False)

        with st.expander("Descriptive statistics"):
            try:
                desc = original_df.describe(include="all").transpose()
                # cast to string for stable display
                st.dataframe(desc.astype(str), use_container_width=True)
            except Exception as e:
                st.write("Unable to compute describe():", e)

    st.divider()

    # Reset button at the top
    if st.button("Reset to Original Dataset"):
        _reset_to_original()

    st.divider()

    st.subheader(":material/view_column: Drop Unnecessary Columns")

    with st.expander("Drop columns"):
        st.write("**Select columns to drop:**")
        
        # Table header
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.write("**Column**")
        with col2:
            st.write("**Type**")
        with col3:
            st.write("**Missing**")
        with col4:
            st.write("**Drop?**")
        
        st.divider()
        
        # Collect selected columns
        cols_to_drop = []
        
        # For each column, create a row
        for col in get_columns(df):
            dtype = str(df[col].dtype)
            missing = int(df[col].isnull().sum())
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.write(f"**{col}**")
            with col2:
                st.write(dtype)
            with col3:
                st.write(missing)
            with col4:
                drop_this = st.checkbox(f"Drop {col}", key=f"drop_{col}", label_visibility="collapsed")
                if drop_this:
                    cols_to_drop.append(col)
        
        
        st.divider()
        
        if st.button("Drop selected columns"):
            if not cols_to_drop:
                st.warning("No columns selected to drop")
            else:
                try:
                    new_df = drop_columns(df, cols_to_drop)
                    _safe_set_working(new_df, f"Dropped columns: {cols_to_drop}")
                    st.rerun()  # Refresh the page to update all sections
                except Exception as e:
                    st.error(f"Failed to drop columns: {e}")

    st.divider()

    with st.expander(":material/view_list: Remove duplicate rows"):
        duplicates = df.duplicated().sum()
        st.write(f"**Number of duplicate rows:** {duplicates}")
        if duplicates > 0:
            keep_option = st.selectbox("Keep", ["first", "last"], key="keep_option")
            if st.button("Remove duplicate rows"):
                try:
                    new_df = df.drop_duplicates(keep=keep_option)
                    _safe_set_working(new_df, f"Removed {duplicates} duplicate rows")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            st.info("No duplicate rows found")

    st.divider()

    st.subheader(":material/data_table: Column Data Types (Change safely)")

    # Use an expander to avoid huge UIs for wide tables
    with st.expander("Change column data types"):
        st.write("**Column Data Types:**")
        
        # Header for the table
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            st.write("**Column Name**")
        with col2:
            st.write("**Current Type**")
        with col3:
            st.write("**New Type**")
        with col4:
            st.write("**Action**")
        
        st.divider()
        
        # For each column, create a row
        for col in get_columns(df):
            current_dtype = str(df[col].dtype)
            allowed = get_allowed_conversions(df, col)
            default_choice = _choose_default_for_col(current_dtype, allowed)
            
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            with col1:
                st.write(f"**{col}**")
            with col2:
                st.write(f"`{current_dtype}`")
            with col3:
                if allowed:
                    new_dtype = st.selectbox(
                        f"New type for {col}",
                        options=allowed,
                        index=allowed.index(default_choice) if default_choice in allowed else 0,
                        key=f"dtype_select_{col}",
                        label_visibility="collapsed"
                    )
                else:
                    st.write("No conversions")
                    new_dtype = None
            with col4:
                if allowed and st.button(f"Update {col}", key=f"update_{col}"):
                    if new_dtype != _normalize_dtype_simple(current_dtype):
                        try:
                            new_df, msg = change_column_dtype(st.session_state["working_df"], col, new_dtype)
                            _safe_set_working(new_df, msg)
                            st.rerun()  # To refresh the display
                        except Exception as e:
                            st.error(f"Failed: {e}")
                    else:
                        st.info("No change")
        

    st.divider()

    st.subheader(":material/unknown_med: Missing Values — Options")

    with st.expander("Impute / Remove missing values"):
        cols_with_na = [c for c in get_columns(df) if int(df[c].isnull().sum()) > 0]
        if not cols_with_na:
            st.info("No missing values detected in the dataset")
        else:
            st.write("**Columns with Missing Values:**")
            
            # Table header
            col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 2, 1])
            with col1:
                st.write("**Column**")
            with col2:
                st.write("**Missing**")
            with col3:
                st.write("**Method**")
            with col4:
                st.write("**Parameters**")
            with col5:
                st.write("**Action**")
            
            st.divider()
            
            # For each column with NA, create a row
            for col in cols_with_na:
                dtype_norm = _normalize_dtype_simple(str(df[col].dtype))
                default_method = _default_impute_method(dtype_norm)
                methods = _methods_for_dtype(dtype_norm)
                missing_count = int(df[col].isnull().sum())
                
                col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 2, 1])
                with col1:
                    st.write(f"**{col}**")
                with col2:
                    st.write(missing_count)
                with col3:
                    method = st.selectbox(
                        f"Method for {col}",
                        options=methods,
                        index=methods.index(default_method) if default_method in methods else 0,
                        key=f"impute_method_{col}",
                        label_visibility="collapsed"
                    )
                with col4:
                    if method == "constant":
                        param = st.text_input(f"Constant value for {col}", key=f"constant_{col}", label_visibility="collapsed")
                    elif method == "percentile":
                        param = st.slider(f"Percentile for {col}", 0.0, 100.0, 50.0, key=f"percentile_{col}", label_visibility="collapsed")
                    elif method == "rolling_mean":
                        param = st.number_input(f"Rolling window for {col}", min_value=1, max_value=1000, value=3, key=f"window_{col}", label_visibility="collapsed")
                    else:
                        param = None
                        st.write("—")
                with col5:
                    if st.button(f"Impute {col}", key=f"impute_{col}"):
                        try:
                            const = param if method == "constant" else None
                            pct = param if method == "percentile" else 50.0
                            win = param if method == "rolling_mean" else 3
                            new_df = impute_column(st.session_state["working_df"], col, method, constant_value=const, percentile=float(pct), window=int(win))
                            _safe_set_working(new_df, f"Imputed {col} with {method}")
                            st.rerun()  # Refresh to update missing values count
                        except Exception as e:
                            st.error(f"Failed to impute {col}: {e}")
            
            
            st.divider()
            
            # Drop options
            drop_selection = st.multiselect("Select columns for row/column dropping", options=cols_with_na, key="drop_na_cols")
            if drop_selection:
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Drop rows with NA in selected columns"):
                        try:
                            _push_history()
                            st.session_state["working_df"] = st.session_state["working_df"].dropna(subset=drop_selection)
                            st.success("Dropped rows containing NA in selected columns")
                            st.rerun()  # Refresh to update all sections
                        except Exception as e:
                            st.error(f"Failed to drop rows: {e}")
                with c2:
                    if st.button("Drop selected columns"):
                        try:
                            _push_history()
                            st.session_state["working_df"] = st.session_state["working_df"].drop(columns=drop_selection)
                            st.success("Dropped selected columns")
                            st.rerun()  # Refresh to update all sections
                        except Exception as e:
                            st.error(f"Failed to drop columns: {e}")

    st.divider()

    # Final preview after operations
    st.subheader(":material/dataset: Current Data Preview (After Operations)")
    
    final_left_col, final_right_col = st.columns([0.7, 0.3], gap="medium", border=True)
    with final_left_col:
        st.dataframe(st.session_state["working_df"].head(50), use_container_width=True)
    
    with final_right_col:
        st.subheader(":material/summarize: Data Summary")
        
        report = get_data_quality_report(df)
        col1, col2 = st.columns([1,1])
        with col1:
            st.metric("Rows", report["rows"])
            st.metric("Missing Values", report["missing_values_total"])
        with col2:
            st.metric("Columns", report["columns"])
            st.metric("Duplicates", report["duplicate_rows"])

        with st.expander("Column types & counts"):
            dtypes = get_column_dtypes(df)
            dt_table = pd.DataFrame.from_dict(dtypes, orient="index", columns=["dtype"])
            dt_table["missing"] = df.isnull().sum().astype(int)
            dt_table["unique_values"] = df.nunique(dropna=True)
            st.dataframe(dt_table, use_container_width=True, hide_index=False)

        with st.expander("Descriptive statistics"):
            try:
                desc = df.describe(include="all").transpose()
                # cast to string for stable display
                st.dataframe(desc.astype(str), use_container_width=True)
            except Exception as e:
                st.write("Unable to compute describe():", e)


def _normalize_dtype_simple(dtype_str: str) -> str:
    s = dtype_str.lower()
    if "int" in s and "uint" not in s:
        return "int"
    if "float" in s:
        return "float"
    if "bool" in s:
        return "bool"
    if "datetime" in s:
        return "datetime"
    if "category" in s:
        return "category"
    if "object" in s:
        return "object"
    return "str"


def _choose_default_for_col(current_dtype_str: str, allowed_list: list):
    curr = _normalize_dtype_simple(current_dtype_str)
    if curr in allowed_list:
        return curr
    # fallback to first allowed
    return allowed_list[0] if allowed_list else None


def _methods_for_dtype(dtype_norm: str):
    """
    Provide list of imputation methods relevant for dtype.
    """
    # exhaustive set of sensible methods
    numeric_methods = ["mean", "median", "mode", "constant", "ffill", "bfill", "interpolate", "percentile", "rolling_mean"]
    str_methods = ["mode", "constant", "ffill", "bfill"]
    bool_methods = ["mode", "constant", "ffill", "bfill"]
    datetime_methods = ["ffill", "bfill", "constant"]
    category_methods = ["mode", "constant", "ffill", "bfill"]

    if dtype_norm in ("int", "float"):
        return numeric_methods
    if dtype_norm in ("str", "object"):
        return str_methods
    if dtype_norm == "bool":
        return bool_methods
    if dtype_norm == "datetime":
        return datetime_methods
    if dtype_norm == "category":
        return category_methods
    return ["mode", "constant"]


def _default_impute_method(dtype_norm: str):
    """
    Default imputation rule (sensible defaults).
    - numeric -> mean
    - str/object -> mode
    - bool -> mode
    - datetime -> ffill
    - category -> mode
    """
    if dtype_norm in ("int", "float"):
        return "mean"
    if dtype_norm in ("str", "object", "category"):
        return "mode"
    if dtype_norm == "bool":
        return "mode"
    if dtype_norm == "datetime":
        return "ffill"
    return "mode"
