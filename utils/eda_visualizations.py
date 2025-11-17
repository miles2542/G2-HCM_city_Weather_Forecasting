# utils/eda_visualizations.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple
from statsmodels.tsa.seasonal import STL
from scipy.stats import skew, kurtosis
from sklearn.neighbors import KernelDensity
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate


def plot_target_variable_time_series(
    df: pd.DataFrame, target_col: str, rolling_windows: List[int] = [30, 365]
) -> go.Figure:
    """
    Generates a high-quality, two-panel plot of the target variable, showing
    both rolling means and rolling standard deviations to reveal trends,
    seasonality, and volatility. (Version 2 - Corrected)

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.
        target_col (str): The name of the target variable column to plot.
        rolling_windows (List[int]): A list of window sizes for rolling stats.

    Returns:
        go.Figure: A Plotly Figure object ready for display.
    """
    df_plot = df.copy()

    # --- 1. Prepare Data: Calculate all rolling statistics ---
    colors = {
        rolling_windows[0]: "#0771A4",  # Blue for short-term/seasonal
        rolling_windows[1]: "#A22222",  # Red for long-term/trend
    }

    for window in rolling_windows:
        df_plot[f"mean_{window}d"] = df_plot[target_col].rolling(window=window).mean()
        df_plot[f"std_{window}d"] = df_plot[target_col].rolling(window=window).std()

    # --- 2. Create the Subplot Figure ---
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=(
            "<b>Daily Average Temperature with Rolling Means</b>",
            "<b>Rolling Standard Deviations (Volatility)</b>",
        ),
        row_heights=[0.7, 0.3],  # Give more space to the primary plot
    )

    # --- 3. Populate Top Subplot (Means) ---
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot[target_col],
            mode="lines",
            name="Daily Temp",
            line=dict(width=0.5, color="darkgrey"),  # Thinner grey line
        ),
        row=1,
        col=1,
    )

    for window in rolling_windows:
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[f"mean_{window}d"],
                mode="lines",
                name=f"{window}-Day Mean",
                line=dict(width=2.2, color=colors[window]),
            ),
            row=1,
            col=1,
        )

    # --- 4. Populate Bottom Subplot (Standard Deviations) ---
    for window in rolling_windows:
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[f"std_{window}d"],
                mode="lines",
                name=f"{window}-Day Std",
                line=dict(width=2, color=colors[window]),
            ),
            row=2,
            col=1,
        )

    # --- 5. Apply High-Quality Layout and Annotations ---
    fig.update_layout(
        title_text=f"<b>Analysis of HCMC's Temperature: Trend, Seasonality, and Volatility</b><br><sup>Daily average temperature with {' and '.join(str(window) for window in rolling_windows)}-day rolling statistics</sup>",
        height=750,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        yaxis_title_text="Temperature (°C)",
        yaxis2_title_text="Std Dev (°C)",
        xaxis2_title_text="Date",
        margin=dict(t=100),  # Add top margin to prevent title collision
    )

    # Anchoring annotations to the top subplot's data coordinates for stability.
    fig.add_annotation(
        x="2018-01-01",
        y=32.2,
        xref="x1",
        yref="y1",
        text="<b>Long-Term Trend:</b><br>The 365-day mean<br>reveals a gradual upward drift.",
        showarrow=False,
        align="left",
        bordercolor=colors[365],
        borderpad=4,
        borderwidth=1.5,
        bgcolor="rgba(255, 255, 255, 0.85)",
    )
    fig.add_annotation(
        x="2023-10-01",
        y=24.5,
        xref="x1",
        yref="y1",
        text="<b>Annual Seasonality:</b><br>The 30-day mean<br>captures the yearly cycle.",
        showarrow=False,
        align="right",
        bordercolor=colors[30],
        borderpad=4,
        borderwidth=1.5,
        bgcolor="rgba(255, 255, 255, 0.85)",
    )

    # Final styling touches
    fig.update_annotations(font=dict(size=12))
    fig.update_xaxes(showgrid=False)

    return fig


def plot_monthly_seasonality(df: pd.DataFrame, target_col: str) -> go.Figure:
    """
    Generates an enhanced boxplot showing the distribution of the target
    variable for each month, revealing seasonal patterns and outliers.
    (Code remains the same as before)
    """
    df_plot = df.copy()
    df_plot["month_name"] = df_plot.index.strftime("%b")
    df_plot["month_num"] = df_plot.index.month
    month_order = df_plot.sort_values("month_num")["month_name"].unique()
    monthly_means = df_plot.groupby("month_name")[target_col].mean().reindex(month_order)
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=df_plot["month_name"],
            y=df_plot[target_col],
            name="Monthly Distribution",
            marker_color="#0771A4",
            line_color="#0771A4",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly_means.index,
            y=monthly_means.values,
            mode="lines+markers",
            name="Monthly Mean Temp",
            line=dict(color="#E7B142", width=3),
            marker=dict(size=8, symbol="diamond"),
        )
    )
    fig.update_layout(
        title_text="<b>HCMC's Bimodal Climate Pattern Revealed by Monthly Distributions</b><br><sup>Temperatures peak in April-May before the main rainy season, with a secondary dip mid-year.</sup>",
        xaxis_title="Month",
        yaxis_title="Temperature (°C)",
        xaxis=dict(categoryorder="array", categoryarray=month_order),
        showlegend=False,
        annotations=[
            dict(
                x="Apr",
                y=monthly_means.max() + 0.2,
                text="<b>Primary Peak:</b><br>Pre-monsoon heat",
                showarrow=True,
                arrowhead=1,
                ax=-50,
                ay=-50,
            ),
            dict(
                x="Jul",
                y=df_plot[target_col].quantile(0.25) - 1,
                text="<b>Mid-Year Dip:</b><br>Cooling effect of the<br>Southwest Monsoon's peak",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=70,
            ),
            dict(
                x="Dec",
                y=df_plot[target_col].min() - 0.5,
                text="<b>Outliers:</b><br>Note the presence of low-temp<br>outliers in cooler months",
                showarrow=True,
                arrowhead=1,
                ax=40,
                ay=30,
            ),
        ],
    )
    return fig


def plot_stl_decomposition(df: pd.DataFrame, target_col: str, period: int = 365) -> go.Figure:
    """
    Performs and plots the Seasonal-Trend-Loess (STL) decomposition of a time series,
    providing a clear separation of its core components.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.
        target_col (str): The name of the target variable column.
        period (int): The seasonal period. For daily data, this is typically 365.

    Returns:
        go.Figure: A Plotly Figure object with four subplots.
    """
    # Ensure the series has no missing values for STL
    series = df[target_col].dropna()

    # Use robust=True to handle outliers more effectively
    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    # Create a 4-panel subplot
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("<b>Observed</b>", "<b>Trend</b>", "<b>Seasonal</b>", "<b>Residual</b>"),
    )

    # Add each component to its respective subplot
    fig.add_trace(
        go.Scatter(
            x=result.observed.index,
            y=result.observed,
            mode="lines",
            name="Observed",
            line=dict(color="darkgrey", width=1.6),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=result.trend.index, y=result.trend, mode="lines", name="Trend", line=dict(color="#A22222")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=result.seasonal.index,
            y=result.seasonal,
            mode="lines",
            name="Seasonal",
            line=dict(color="#0771A4", width=1.6),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=result.resid.index,
            y=result.resid,
            mode="markers",
            name="Residual",
            marker=dict(color="#38545f", size=2.5, opacity=0.7),
        ),
        row=4,
        col=1,
    )

    # Apply high-quality layout
    fig.update_layout(
        title_text="<b>STL Decomposition of Daily Temperature</b><br><sup>Separating the series into its Trend, Seasonal, and Residual components</sup>",
        height=800,
        showlegend=False,
        yaxis1_title_text="°C",
        yaxis2_title_text="°C",
        yaxis3_title_text="°C",
        yaxis4_title_text="°C",
        xaxis4_title_text="Date",
        margin=dict(t=100),
    )

    fig.update_xaxes(showgrid=False)

    return fig


def plot_feature_distributions(df: pd.DataFrame, features: List[str], color_map: Dict[str, str]) -> go.Figure:
    """
    Generates a grid of enhanced violin plots paired with box plots to show
    the distribution, quartiles, and outliers for key numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (List[str]): A list of column names to plot.
        color_map (Dict[str, str]): A dictionary mapping feature names to hex color codes.

    Returns:
        go.Figure: A Plotly Figure object.
    """
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=[f"<b>{f.replace('_', ' ').title()}</b>" for f in features]
    )

    for i, feature in enumerate(features):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        color = color_map.get(feature, "#38545f")

        # 1. Violin plot for distribution shape
        fig.add_trace(
            go.Violin(
                y=df[feature],
                name=feature,
                box_visible=False,
                meanline_visible=False,
                points=False,
                fillcolor=color,
                opacity=0.6,
                line_color=color,
            ),
            row=row,
            col=col,
        )

        # 2. Box plot for quartiles, median, and outliers
        fig.add_trace(
            go.Box(
                y=df[feature],
                name=feature,
                marker_color=color,
                boxpoints="outliers",  # Show only outliers
                line_width=2,
                width=0.2,
                fillcolor="rgba(255,255,255,0.8)",
                # Marker for the mean to contrast with the median line
                boxmean=True,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title_text="<b>Distribution & Outlier Analysis for Key Predictors</b><br><sup>Violin plots show probability density; embedded box plots show quartiles and outliers.</sup>",
        height=300 * n_rows,
        showlegend=False,
    )
    fig.update_yaxes(showgrid=True, zeroline=False)

    # --- Add a custom legend-like annotation ---
    fig.add_annotation(
        text="<b>Box plot key:</b><br>━━  Median<br><b>- - - -</b>   Mean",
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1.01,
        y=1.085,  # Position slightly outside the top-right plot area
        bordercolor="#38545f",
        borderwidth=1,
        borderpad=4,
        bgcolor="rgba(255, 255, 255, 0.85)",
    )

    return fig


def plot_seasonal_pairplot(df: pd.DataFrame, features: List[str]) -> go.Figure:
    """
    Generates a high-impact, custom pairplot to analyze bivariate relationships,
    color-coded by season, with seasonally-split KDEs and titles on the diagonal.

    Args:
        df (pd.DataFrame): The input DataFrame with a DatetimeIndex.
        features (List[str]): A list of features to plot (ideally 4-6).

    Returns:
        go.Figure: A Plotly Figure object.
    """
    # 1. Create the 'Season' column for color mapping
    df_plot = df[features].copy()
    df_plot["Season"] = np.where(df_plot.index.month.isin([12, 1, 2, 3, 4]), "Dry Season", "Rainy Season")

    season_colors = {"Dry Season": "#E7B142", "Rainy Season": "#0771A4"}
    n_features = len(features)

    # Use make_subplots without subplot_titles; we will add them as annotations.
    fig = make_subplots(rows=n_features, cols=n_features)

    for i in range(n_features):  # Row index
        for j in range(n_features):  # Col index
            # --- Diagonal: Seasonally-split KDEs with Title Annotation ---
            if i == j:
                for season in ["Dry Season", "Rainy Season"]:
                    data = df_plot[df_plot["Season"] == season][features[i]].dropna()
                    kde = KDEUnivariate(data.values)
                    kde.fit(bw="scott")
                    x_kde, y_kde = kde.support, kde.density

                    fig.add_trace(
                        go.Scattergl(
                            x=x_kde,
                            y=y_kde,
                            mode="lines",
                            fill="tozeroy",
                            name=season,
                            legendgroup=season,
                            showlegend=(i == 0 and j == 0),
                            line=dict(color=season_colors[season], width=1.5),
                        ),
                        row=i + 1,
                        col=j + 1,
                    )

                # Calculate the correct subplot index (1-based)
                subplot_idx = i * n_features + j + 1

                # Handle Plotly's 'x' vs 'x2' naming convention
                xref_val = f"x{subplot_idx} domain" if subplot_idx > 1 else "x domain"
                yref_val = f"y{subplot_idx} domain" if subplot_idx > 1 else "y domain"

                # Add the feature title as an annotation on the diagonal
                fig.add_annotation(
                    text=f"<b>{features[i].replace('_', ' ').title()}</b>",
                    xref=xref_val,
                    yref=yref_val,
                    x=0.5,
                    y=1.1,
                    showarrow=False,
                    font=dict(size=14),
                )

            # --- Lower Triangle: Scatter plots with trendlines ---
            elif i > j:
                for season in ["Dry Season", "Rainy Season"]:
                    season_data = df_plot[df_plot["Season"] == season]
                    fig.add_trace(
                        go.Scattergl(
                            x=season_data[features[j]],
                            y=season_data[features[i]],
                            mode="markers",
                            name=season,
                            legendgroup=season,
                            showlegend=False,
                            marker=dict(color=season_colors[season], size=4, opacity=0.4),
                        ),
                        row=i + 1,
                        col=j + 1,
                    )

                x_fit, y_fit = df_plot[features[j]].dropna(), df_plot[features[i]].dropna()
                common_index = x_fit.index.intersection(y_fit.index)
                x_fit, y_fit = x_fit[common_index], y_fit[common_index]

                if len(x_fit) > 1:
                    model = sm.OLS(y_fit, sm.add_constant(x_fit)).fit()
                    x_range = np.linspace(x_fit.min(), x_fit.max(), 100)
                    y_range = model.predict(sm.add_constant(x_range))
                    fig.add_trace(
                        go.Scattergl(
                            x=x_range,
                            y=y_range,
                            mode="lines",
                            name="OLS Trend",
                            showlegend=False,
                            line=dict(color="#963D4D", width=1.5, dash="dash"),
                        ),
                        row=i + 1,
                        col=j + 1,
                    )

    # --- Final Layout and Styling ---
    fig.update_layout(
        title_text="<b>Seasonal Bivariate Analysis of Key Climate Drivers</b><br><sup>Scatter plots show relationships, colored by season. Diagonals show seasonal distribution shifts.</sup>",
        height=200 * n_features,  # Dynamic height
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=0.98),
        margin=dict(t=100, l=50, r=50, b=50),
        plot_bgcolor="#EFF5F5",
        paper_bgcolor="#EFF5F5",
        hovermode="closest",
    )
    for i in range(1, n_features + 1):
        for j in range(1, n_features + 1):
            fig.update_xaxes(
                row=i, col=j, showgrid=False, zeroline=False, title_text=features[j - 1] if i == n_features else ""
            )
            fig.update_yaxes(row=i, col=j, showgrid=False, zeroline=False, title_text=features[i - 1] if j == 1 else "")
    return fig


def plot_correlation_heatmap_enhanced(df: pd.DataFrame, method: str = "pearson") -> go.Figure:
    """
    Generates a high-quality, masked, lower-triangle correlation heatmap
    with a specific, publication-ready style.

    Args:
        df (pd.DataFrame): DataFrame containing only numerical columns.
        method (str): The method of correlation ('pearson' or 'spearman').

    Returns:
        go.Figure: A Plotly Figure object.
    """
    # 1. Calculate the correlation matrix
    corr_matrix = df.corr(method=method)

    # 2. Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # 3. Generate the heatmap using plotly.graph_objects for finer control
    heatmap_trace = go.Heatmap(
        z=corr_matrix.where(~mask),  # Apply mask here
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        xgap=1,
        ygap=1,  # Add small gaps for clarity
        hoverongaps=False,
    )

    # 4. Create the figure and layout
    fig = go.Figure(data=[heatmap_trace])

    # Add correlation values as text annotations
    annotations = []
    for i, row in enumerate(corr_matrix.values):
        for j, value in enumerate(row):
            if j < i:  # Only annotate the lower triangle
                text_color = "white" if abs(value) > 0.6 else "#38545f"
                annotations.append(
                    go.layout.Annotation(
                        text=f"{value:.2f}",
                        x=corr_matrix.columns[j],
                        y=corr_matrix.columns[i],
                        xref="x1",
                        yref="y1",
                        showarrow=False,
                        font=dict(color=text_color, size=10),
                    )
                )

    method_title = method.title()
    fig.update_layout(
        title_text=f"<b>{method_title} Correlation Matrix of Numerical Features</b><br><sup>Masked upper triangle reveals inter-feature relationships.</sup>",
        height=850,
        width=850,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange="reversed",  # Keep diagonal at top-left
        plot_bgcolor="#EFF5F5",
        annotations=annotations,
        xaxis_tickangle=-45,
        hovermode="closest",
        margin=dict(t=70, l=95, r=65, b=95),
    )

    return fig
