import streamlit as st
from pages import home, main, about

st.set_page_config(
    page_title="Time Series Anomaly Detection",
    page_icon=":material/area_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.logo("assets/full_logo.png", size="large")

pg = st.navigation(
    [
        st.Page(
            home.app,
            title="Home",
            icon=":material/home_app_logo:",
            url_path="home",         
        ),
        st.Page(
            main.app,
            title="Anomaly Detection",
            icon=":material/bubble_chart:",
            url_path="anomaly-detection",  
        ),
        st.Page(
            about.app,
            title="About",
            icon=":material/info:",
            url_path="about",   
        ),
    ],
    position="top"
)

pg.run()