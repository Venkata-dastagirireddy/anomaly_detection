import streamlit as st

from backend.data.data_handler import (
    load_dataset,
    get_data_quality_report,
    basic_data_cleaning,
)


def app():
    st.title("Anomaly Detection")
    st.caption("Upload, inspect, and prepare your time series data")

    st.divider()

    # -----------------------------
    # File Upload
    # -----------------------------
    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help="Supported formats: CSV, Excel, JSON, Parquet",
    )

    if uploaded_file is None:
        st.info("Please upload a dataset to proceed.")
        return

    # -----------------------------
    # Load Dataset
    # -----------------------------
    try:
        with st.spinner("Loading dataset..."):
            raw_df = load_dataset(uploaded_file)

            st.toast("Dataset loaded successfully", icon=":material/check:")

    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return

    st.session_state["raw_df"] = raw_df

    # -----------------------------
    # Raw Data Preview
    # -----------------------------
    st.subheader("Raw Data Preview")
    st.dataframe(raw_df, use_container_width=True, hide_index=True)

    # -----------------------------
    # Initial Data Checks
    # -----------------------------
    st.subheader("Initial Data Quality Check")

    report = get_data_quality_report(raw_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", report["rows"])
    c2.metric("Columns", report["columns"])
    c3.metric("Duplicates", report["duplicate_rows"])
    c4.metric("Missing Values", report["missing_values_total"])

    with st.expander("Column Details"):
        st.json(
            {
                "Data Types": report["dtypes"],
                "Missing Values by Column": report["missing_values_by_column"],
            }
        )

    # -----------------------------
    # Initial Cleaning
    # -----------------------------
    st.subheader("Initial Cleaning")

    if st.button("Apply Basic Cleaning"):
        with st.spinner("Applying basic cleaning..."):
            cleaned_df = basic_data_cleaning(raw_df)

        st.toast("Basic cleaning completed", icon=":material/check:")

        st.session_state["cleaned_df"] = cleaned_df

        c1, c2 = st.columns(2)
        c1.metric("Rows (Before)", raw_df.shape[0])
        c2.metric("Rows (After)", cleaned_df.shape[0])

        st.subheader("Cleaned Data Preview")
        st.dataframe(cleaned_df.head(10), use_container_width=True)