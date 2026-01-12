import streamlit as st

def app():
    st.title(":material/info: About")
    st.subheader("Time Series Anomaly Detection Platform")

    # Overview
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h2 style='color: #1f77b4;'>Professional Time Series Anomaly Detection Suite</h2>
        <p style='font-size: 1.1rem; color: #666;'>Advanced analytics for identifying outliers and anomalies in temporal data</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # App Description
    st.header(":material/target: Application Overview")

    st.markdown("""
    This comprehensive **Time Series Anomaly Detection Platform** provides data scientists, analysts, and business users with powerful tools to:

    - **Upload and preprocess** time series datasets from various sources
    - **Clean and transform** data with professional-grade operations
    - **Detect anomalies** using multiple statistical and machine learning algorithms
    - **Visualize results** through interactive, publication-quality charts
    - **Export findings** for further analysis or integration

    Built with modern web technologies and following software engineering best practices, this application ensures reliability, scalability, and user-friendly operation.
    """)

    st.divider()

    # Features in Detail
    st.header(":material/bolt: Core Features")

    with st.expander(":material/table: Data Management & Preparation", expanded=True):
        st.markdown("""
        ### File Upload & Support
        - **Multiple Formats**: CSV, Excel (XLSX/XLS), JSON, Parquet
        - **Large Datasets**: Efficient handling of datasets with 100k+ rows
        - **Auto-Detection**: Intelligent format recognition and parsing

        ### Data Exploration
        - **Interactive Preview**: Paginated data viewing with full/partial display options
        - **Comprehensive Summary**: Row/column counts, data types, missing values, duplicates
        - **Statistical Overview**: Descriptive statistics for all columns
        - **Column Analysis**: Type distributions and unique value counts

        ### Data Cleaning Pipeline
        - **Column Operations**: Selective dropping with preview
        - **Type Conversion**: Safe conversion between data types with validation
        - **Missing Value Handling**: Multiple imputation strategies (mean, median, mode, etc.)
        - **Duplicate Management**: Intelligent duplicate row detection and removal
        - **Time Series Preparation**: Automatic datetime parsing and chronological sorting
        """)

    with st.expander(":material/smart_toy: Anomaly Detection Algorithms", expanded=True):
        st.markdown("""
        ### Statistical Methods
        - **Z-Score**: Standard deviation-based detection with configurable thresholds
        - **IQR (Interquartile Range)**: Robust outlier detection using quartiles

        ### Machine Learning Methods
        - **Isolation Forest**: Unsupervised ensemble method for high-dimensional data
        - **One-Class SVM**: Support Vector Machine for novelty detection

        ### Time Series Specific Methods
        - **ARIMA-based**: Autoregressive Integrated Moving Average with residual analysis
        - **Seasonal Decomposition**: STL decomposition for seasonal and trend components

        ### Algorithm Features
        - **Parameter Tuning**: Interactive sliders for real-time parameter adjustment
        - **Batch Processing**: Efficient computation on large datasets
        - **Result Persistence**: Maintains results across UI interactions
        - **Performance Metrics**: Detection accuracy and processing time feedback
        """)

    with st.expander(":material/show_chart: Visualization & Analytics", expanded=True):
        st.markdown("""
        ### Interactive Charts
        - **Time Series Plot**: Main visualization with anomaly highlighting
        - **Distribution Analysis**: Pie charts showing normal vs anomalous proportions
        - **Rolling Statistics**: Moving averages and standard deviations with confidence bands
        - **Anomaly Heatmap**: Scatter plots colored by anomaly status
        - **Before/After Comparison**: Original vs cleaned data visualization

        ### Time Series Analysis
        - **ACF/PACF Plots**: Autocorrelation and partial autocorrelation analysis
        - **Correlation Insights**: Lag-based dependency analysis
        - **Seasonality Detection**: Pattern recognition in temporal data

        ### Export & Integration
        - **Result Export**: Add anomaly flags to original dataset
        - **Data Download**: Processed datasets with anomaly annotations
        - **API Ready**: Programmatic access to detection results
        """)

    st.divider()

    # Technical Architecture
    st.header(":material/build: Technical Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Frontend
        - **Streamlit**: Modern web app framework
        - **Plotly**: Interactive data visualizations
        - **Responsive Design**: Mobile and desktop optimized
        - **Real-time Updates**: Dynamic UI without page refreshes

        ### Backend Processing
        - **Pandas**: High-performance data manipulation
        - **NumPy**: Numerical computing foundation
        - **Scikit-learn**: Machine learning algorithms
        - **Statsmodels**: Statistical modeling and analysis
        """)

    with col2:
        st.markdown("""
        ### Data Pipeline
        - **Modular Design**: Separated concerns for maintainability
        - **Error Handling**: Comprehensive validation and error recovery
        - **Memory Efficient**: Optimized for large datasets
        - **Type Safety**: Strong typing with Python type hints

        ### Quality Assurance
        - **Unit Testing**: Comprehensive test coverage
        - **Code Quality**: PEP 8 compliance and documentation
        - **Performance Monitoring**: Execution time and resource usage tracking
        - **User Feedback**: Clear error messages and progress indicators
        """)

    st.divider()

    # Algorithm Explanations
    st.header(":material/calculate: Algorithm Explanations")

    algorithm_tabs = st.tabs(["Z-Score", "IQR", "Isolation Forest", "One-Class SVM", "ARIMA", "Seasonal Decomposition"])

    with algorithm_tabs[0]:
        st.markdown("""
        ### Z-Score Method
        **How it works**: Calculates how many standard deviations each point is from the mean.

        **Formula**: `z = (x - μ) / σ`

        **Parameters**: Threshold (typically 3.0 for 99.7% confidence)

        **Best for**: Normally distributed data, point anomalies

        **Pros**: Simple, fast, interpretable
        **Cons**: Sensitive to outliers in training data
        """)

    with algorithm_tabs[1]:
        st.markdown("""
        ### IQR Method
        **How it works**: Uses quartiles to define normal range.

        **Formula**: Anomalies outside `[Q1 - k×IQR, Q3 + k×IQR]`

        **Parameters**: Multiplier k (typically 1.5)

        **Best for**: Non-normal distributions, robust to outliers

        **Pros**: Robust, no distribution assumptions
        **Cons**: May miss anomalies in multimodal data
        """)

    with algorithm_tabs[2]:
        st.markdown("""
        ### Isolation Forest
        **How it works**: Builds random forests to isolate anomalies.

        **Parameters**: Contamination (expected anomaly proportion)

        **Best for**: High-dimensional data, mixed attribute types

        **Pros**: Scalable, handles mixed data types
        **Cons**: Less interpretable than statistical methods
        """)

    with algorithm_tabs[3]:
        st.markdown("""
        ### One-Class SVM
        **How it works**: Learns a decision boundary around normal data.

        **Parameters**: Nu (upper bound on training errors)

        **Best for**: Complex, non-linear boundaries

        **Pros**: Flexible boundary shapes
        **Cons**: Computationally intensive, parameter sensitive
        """)

    with algorithm_tabs[4]:
        st.markdown("""
        ### ARIMA-based Detection
        **How it works**: Fits ARIMA model, detects anomalies in residuals.

        **Parameters**: p, d, q orders, residual threshold

        **Best for**: Time series with clear patterns

        **Pros**: Accounts for temporal dependencies
        **Cons**: Requires stationary data, complex parameters
        """)

    with algorithm_tabs[5]:
        st.markdown("""
        ### Seasonal Decomposition
        **How it works**: Decomposes into trend, seasonal, residual components.

        **Parameters**: Model type, period, IQR multiplier

        **Best for**: Seasonal time series

        **Pros**: Handles seasonality explicitly
        **Cons**: Requires known seasonal period
        """)

    st.divider()

    # Usage Guidelines
    st.header(":material/library_books: Usage Guidelines")

    st.markdown("""
    ### Data Preparation Best Practices
    1. **Ensure Temporal Order**: Data should be sorted chronologically
    2. **Handle Missing Values**: Use appropriate imputation for time series gaps
    3. **Check Seasonality**: Identify regular patterns in your data
    4. **Validate Data Types**: Ensure numeric columns for value data

    ### Algorithm Selection Guide
    - **Start with Z-Score or IQR** for simple, interpretable results
    - **Use Isolation Forest** for high-dimensional or mixed-type data
    - **Try ARIMA** for data with clear temporal patterns
    - **Apply Seasonal Decomposition** when seasonality is present

    ### Performance Optimization
    - **Large Datasets**: Use sampling for initial exploration
    - **Parameter Tuning**: Start with defaults, adjust based on results
    - **Visualization**: Use multiple views to validate findings
    - **Iteration**: Compare algorithms for robust anomaly detection
    """)

    st.divider()

    # Version & Support
    st.header(":material/settings: Version & Support")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Version Information
        - **Current Version**: 1.0.0
        - **Release Date**: January 2026
        - **Framework**: Streamlit 1.52.2
        - **Python**: 3.10+

        ### Dependencies
        - pandas >= 2.3.3
        - numpy >= 2.2.6
        - scikit-learn >= 1.7.2
        - statsmodels >= 0.14.1
        - plotly >= 6.5.1
        """)

    with col2:
        st.markdown("""
        ### Support & Documentation
        - **Documentation**: Comprehensive inline help
        - **Error Handling**: Clear error messages and recovery
        - **Performance**: Optimized for production use
        - **Updates**: Regular feature enhancements

        ### Contact
        For technical support or feature requests:
        - Check inline help and tooltips
        - Review error messages for guidance
        - Ensure data meets format requirements

        ### License
        Open-source under MIT License
        """)

    st.divider()

    # Footer
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0; color: #666;'>
        <p>Built with ❤️ for data scientists and analysts</p>
        <p>Professional • Reliable • Scalable</p>
    </div>
    """, unsafe_allow_html=True)