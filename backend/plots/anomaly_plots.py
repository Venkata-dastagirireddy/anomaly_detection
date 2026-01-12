import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List
from statsmodels.tsa.stattools import acf, pacf
from plotly.subplots import make_subplots
import numpy as np

def plot_time_series_anomalies(df: pd.DataFrame, date_col: str, value_col: str, anomalies: List[bool], title: str = "Time Series Anomaly Detection"):
    """
    Plot the time series with anomalies highlighted.
    """
    fig = go.Figure()

    # Normal data
    normal_df = df[~pd.Series(anomalies)]
    fig.add_trace(go.Scatter(
        x=normal_df[date_col],
        y=normal_df[value_col],
        mode='lines',
        name='Normal',
        line=dict(color='blue')
    ))

    # Anomalies
    anomaly_df = df[pd.Series(anomalies)]
    fig.add_trace(go.Scatter(
        x=anomaly_df[date_col],
        y=anomaly_df[value_col],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=8, symbol='x')
    ))

    fig.update_layout(
        title=title,
        xaxis_title=date_col,
        yaxis_title=value_col,
        template='plotly_white'
    )

    return fig

def plot_anomaly_distribution(anomalies: List[bool], title: str = "Anomaly Distribution"):
    """
    Plot the distribution of anomalies (pie chart).
    """
    normal_count = sum(not a for a in anomalies)
    anomaly_count = sum(anomalies)

    fig = go.Figure(data=[go.Pie(
        labels=['Normal', 'Anomalies'],
        values=[normal_count, anomaly_count],
        marker_colors=['blue', 'red']
    )])

    fig.update_layout(title=title)

    return fig

def plot_rolling_statistics(df: pd.DataFrame, date_col: str, value_col: str, window: int = 7, title: str = "Rolling Statistics"):
    """
    Plot rolling mean and std.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df[date_col], y=df[value_col], mode='lines', name='Original', line=dict(color='gray', width=1)))

    rolling_mean = df[value_col].rolling(window=window).mean()
    rolling_std = df[value_col].rolling(window=window).std()

    fig.add_trace(go.Scatter(x=df[date_col], y=rolling_mean, mode='lines', name=f'Rolling Mean ({window})', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df[date_col], y=rolling_mean + 2*rolling_std, mode='lines', name=f'Rolling +2σ', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df[date_col], y=rolling_mean - 2*rolling_std, mode='lines', name=f'Rolling -2σ', line=dict(color='red', dash='dash')))

    fig.update_layout(title=title, xaxis_title=date_col, yaxis_title=value_col, template='plotly_white')

    return fig

def plot_anomaly_heatmap(df: pd.DataFrame, date_col: str, value_col: str, anomalies: List[bool], title: str = "Anomaly Heatmap"):
    """
    Plot a heatmap of values with anomalies highlighted.
    """
    # Create a pivot or just use scatter with color
    fig = px.scatter(df, x=date_col, y=value_col, color=pd.Series(anomalies).map({True: 'Anomaly', False: 'Normal'}),
                     color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
                     title=title, template='plotly_white')
    return fig

def plot_before_after(df: pd.DataFrame, date_col: str, value_col: str, anomalies: List[bool], title: str = "Before vs After Anomaly Removal"):
    """
    Plot original and cleaned series.
    """
    fig = go.Figure()

    # Original
    fig.add_trace(go.Scatter(x=df[date_col], y=df[value_col], mode='lines', name='Original', line=dict(color='gray')))

    # Cleaned (anomalies set to NaN or interpolated)
    cleaned = df[value_col].copy()
    cleaned[pd.Series(anomalies)] = np.nan
    cleaned = cleaned.interpolate()

    fig.add_trace(go.Scatter(x=df[date_col], y=cleaned, mode='lines', name='Cleaned', line=dict(color='green')))

    fig.update_layout(title=title, xaxis_title=date_col, yaxis_title=value_col, template='plotly_white')

    return fig

def plot_acf_pacf(df: pd.DataFrame, value_col: str, lags: int = 20):
    """
    Plot ACF and PACF using Plotly.
    """
    data = df[value_col].dropna().values
    
    # Compute ACF and PACF
    acf_values = acf(data, nlags=lags, fft=True)
    pacf_values = pacf(data, nlags=lags)
    
    # Create subplots
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Autocorrelation Function (ACF)', 'Partial Autocorrelation Function (PACF)'))
    
    # ACF plot
    fig.add_trace(
        go.Bar(x=list(range(len(acf_values))), y=acf_values, name='ACF', marker_color='blue'),
        row=1, col=1
    )
    
    # Add confidence intervals for ACF (approximate)
    conf_interval = 1.96 / np.sqrt(len(data))
    fig.add_hline(y=conf_interval, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=-conf_interval, line_dash="dash", line_color="red", row=1, col=1)
    
    # PACF plot
    fig.add_trace(
        go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name='PACF', marker_color='green'),
        row=2, col=1
    )
    
    # Add confidence intervals for PACF
    fig.add_hline(y=conf_interval, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-conf_interval, line_dash="dash", line_color="red", row=2, col=1)
    
    fig.update_layout(height=600, title_text="ACF and PACF Analysis", showlegend=False)
    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_yaxes(title_text="Autocorrelation", row=1, col=1)
    fig.update_yaxes(title_text="Partial Autocorrelation", row=2, col=1)
    
    return fig