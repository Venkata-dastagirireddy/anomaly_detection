import streamlit as st

def app():
    st.title("Home")
    st.subheader("Time Series Anomaly Detection Platform")

    st.markdown(
        """
        Welcome to the **Time Series Anomaly Detection** application.

        ### What you can do:
        - Upload time series datasets
        - Clean & preprocess data
        - Detect anomalies using advanced algorithms
        - Visualize results interactively
        """
    )

