import streamlit as st

def app():
    st.title(":material/home: Home")
    st.subheader("Time Series Anomaly Detection Platform")

    # Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #1f77b4; font-size: 3rem;'> Advanced Time Series Anomaly Detection</h1>
        <p style='font-size: 1.2rem; color: #666;'>Professional-grade anomaly detection for time series data with interactive visualizations</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Key Features
    st.header(":material/star: Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### :material/table: Data Preparation
        - **Upload & Preview**: Support for CSV, Excel, JSON, Parquet
        - **Data Cleaning**: Handle missing values, duplicates, type conversions
        - **Column Management**: Drop unwanted columns, impute missing data
        - **Time Series Prep**: Automatic date parsing and sorting
        """)

    with col2:
        st.markdown("""
        ### :material/smart_toy: Anomaly Detection
        - **6 Advanced Algorithms**: Statistical & Machine Learning methods
        - **Interactive Parameters**: Tune thresholds and settings in real-time
        - **Batch Processing**: Efficient detection on large datasets
        - **Result Integration**: Add anomaly flags to your data
        """)

    with col3:
        st.markdown("""
        ### :material/show_chart: Visualization & Analysis
        - **Interactive Plots**: Time series with anomaly highlighting
        - **Multiple Views**: Distribution, rolling stats, heatmaps
        - **ACF/PACF Analysis**: Time series correlation insights
        - **Before/After Comparison**: Clean vs original data
        """)

    st.divider()

    # Supported Algorithms
    st.header(":material/psychology: Detection Algorithms")

    algorithms = {
        "Z-Score": "Statistical method using standard deviations",
        "IQR": "Interquartile range-based detection",
        "Isolation Forest": "Unsupervised ML for high-dimensional data",
        "One-Class SVM": "Support Vector Machine for novelty detection",
        "ARIMA": "Time series modeling with residual analysis",
        "Seasonal Decomposition": "STL decomposition for seasonal patterns"
    }

    cols = st.columns(2)
    for i, (name, desc) in enumerate(algorithms.items()):
        with cols[i % 2]:
            st.markdown(f"**{name}**\n\n{desc}")

    st.divider()

    # How to Use
    st.header(":material/rocket: How to Use")

    st.markdown("""
    ### Step-by-Step Guide

    #### 1. **Upload Your Data**
    - Navigate to the **Anomaly Detection** page
    - Upload your time series dataset (CSV, Excel, JSON, Parquet)
    - Supported formats ensure flexibility for various data sources

    #### 2. **Explore & Clean Data**
    - **Data Preview**: View your data with pagination controls
    - **Data Summary**: Check rows, columns, missing values, duplicates
    - **Column Management**: Drop unnecessary columns
    - **Type Conversions**: Convert data types safely
    - **Missing Value Handling**: Impute or remove missing data
    - **Duplicate Removal**: Clean duplicate rows

    #### 3. **Configure Anomaly Detection**
    - **Select Columns**: Choose your date/time and value columns
    - **Choose Algorithm**: Pick from 6 professional algorithms
    - **Tune Parameters**: Adjust thresholds, orders, multipliers
    - **Date Handling**: Automatic datetime conversion and sorting

    #### 4. **Run Detection & Analyze**
    - **Detect Anomalies**: Click to run the selected algorithm
    - **View Results**: Summary metrics and anomaly counts
    - **Interactive Visualizations**:
        - Main time series plot with anomalies highlighted
        - Distribution pie chart
        - Rolling statistics with confidence bands
        - Anomaly heatmap
        - Before/after comparison
        - ACF/PACF correlation analysis
    - **Export Results**: Add anomaly column to your dataset

    #### 5. **Iterate & Refine**
    - Adjust parameters and re-run detection
    - Compare different algorithms
    - Fine-tune for your specific use case
    """)

    st.divider()

    # Data Requirements
    st.header(":material/checklist: Data Requirements")

    st.markdown("""
    ### Supported File Formats
    - **CSV** (.csv)
    - **Excel** (.xlsx, .xls)
    - **JSON** (.json)
    - **Parquet** (.parquet)

    ### Column Requirements
    - **Date/Time Column**: Must contain temporal data (auto-detected or manual selection)
    - **Value Column**: Numeric column for anomaly detection
    - **Data Quality**: Missing values and duplicates are handled automatically

    ### Recommendations
    - **Time Series Length**: Minimum 10-20 data points for reliable detection
    - **Date Format**: Standard formats automatically parsed
    - **Value Range**: No restrictions, algorithms handle various scales
    - **Frequency**: Regular time intervals preferred but not required
    """)

    st.divider()

    # Call to Action
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background-color: #f0f2f6; border-radius: 10px;'>
        <h2 style='color: #1f77b4;'>Ready to Detect Anomalies?</h2>
        <p style='font-size: 1.1rem;'>Navigate to the Anomaly Detection page to get started!</p>
        <p style='font-size: 0.9rem; color: #666;'>Professional • Reliable • Interactive</p>
    </div>
    """, unsafe_allow_html=True)

