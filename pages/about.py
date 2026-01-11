import streamlit as st

def app():
    st.title("About")

    st.markdown(
        """
        ### Time Series Anomaly Detection App

        **Version:** 0.1.0  
        **Built with:** Streamlit 1.52.2  
        **Purpose:** Detect anomalies in time series data using ML & statistical techniques.

        This application is designed to be:
        - Modular
        - Scalable
        - Production-ready
        """
    )