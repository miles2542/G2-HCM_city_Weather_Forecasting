import pandas as pd
import numpy as np
from rich.console import Console
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV, MultiTaskLassoCV
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from rich.console import Console
from rich.table import Table
import joblib


def load_and_clean_hourly_data(path: str) -> pd.DataFrame:
    """
    Loads raw hourly data, cleans it, and performs domain-aware imputation.
    """
    console = Console()

    # 1. Load Data
    df_hourly_raw = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    console.print(f"[bold cyan]Raw hourly data loaded: {df_hourly_raw.shape}[/bold cyan]")

    df_hourly = df_hourly_raw.copy()

    # 2. Drop duplicates
    n_before = len(df_hourly)
    df_hourly.drop_duplicates(inplace=True)
    if (n_before - len(df_hourly)) > 0:
        console.print(f"Dropped {n_before - len(df_hourly)} duplicate rows.")

    # 3. Perform domain-aware imputation
    df_hourly["severerisk"].fillna(0, inplace=True)
    df_hourly["preciptype"].fillna("none", inplace=True)
    df_hourly["precip"].fillna(0.0, inplace=True)
    df_hourly["solarradiation"].fillna(0.0, inplace=True)
    df_hourly["solarenergy"].fillna(0.0, inplace=True)
    df_hourly["uvindex"].fillna(0.0, inplace=True)
    df_hourly["windgust"].fillna(df_hourly["windspeed"], inplace=True)
    df_hourly["visibility"].fillna(method="ffill", inplace=True)
    df_hourly["winddir"].fillna(method="ffill", inplace=True)

    remaining_nans = df_hourly.isnull().sum().sum()
    if remaining_nans == 0:
        console.print(f"[green]✓ Data cleaning complete. No remaining NaNs.[/green]")
    else:
        console.print(f"[bold red]Warning: {remaining_nans} NaNs remain after cleaning.[/bold red]")

    return df_hourly


def create_daily_enriched_features(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates cleaned hourly data to a daily format and engineers a rich feature set.
    """
    console = Console()

    # 1. Aggregate hourly to daily
    NUMERICAL_FEATURES = [
        "temp",
        "feelslike",
        "dew",
        "humidity",
        "precip",
        "windgust",
        "windspeed",
        "sealevelpressure",
        "cloudcover",
        "visibility",
        "solarradiation",
        "uvindex",
    ]
    agg_dict = {feature: ["mean", "std", "min", "max"] for feature in NUMERICAL_FEATURES}
    agg_dict["precip"] = ["sum", "max"]
    agg_dict["severerisk"] = ["max", lambda x: (x > 0).sum()]

    df_daily = df_hourly.resample("D").agg(agg_dict)
    df_daily.columns = ["_".join(col).strip() for col in df_daily.columns.values]
    df_daily.rename(columns={"severerisk_<lambda_0>": "severerisk_hours_count"}, inplace=True)

    # 2. Add advanced intra-day features
    df_daily["diurnal_temp_range"] = df_daily["temp_max"] - df_daily["temp_min"]
    hour_max_temp = df_hourly["temp"].resample("D").apply(pd.Series.idxmax).dropna()
    df_daily["hour_of_max_temp"] = hour_max_temp.dt.hour
    hour_min_temp = df_hourly["temp"].resample("D").apply(pd.Series.idxmin).dropna()
    df_daily["hour_of_min_temp"] = hour_min_temp.dt.hour

    console.print(f"Daily aggregation complete. Shape: {df_daily.shape}")

    # 3. Engineer standard time-series features
    df_fe = df_daily.copy()

    # Temporal & Cyclical
    df_fe["year"] = df_fe.index.year
    df_fe["month"] = df_fe.index.month
    df_fe["day_of_year"] = df_fe.index.dayofyear
    df_fe["sin_doy"] = np.sin(2 * np.pi * df_fe["day_of_year"] / 366.0)
    df_fe["cos_doy"] = np.cos(2 * np.pi * df_fe["day_of_year"] / 366.0)
    df_fe["sin_month"] = np.sin(2 * np.pi * df_fe["month"] / 12.0)
    df_fe["cos_month"] = np.cos(2 * np.pi * df_fe["month"] / 12.0)

    # Lags & Rolling Windows
    FEATURES_FOR_LAGS = [
        "temp_mean",
        "temp_std",
        "humidity_mean",
        "solarradiation_mean",
        "precip_sum",
        "windspeed_mean",
        "cloudcover_mean",
        "sealevelpressure_mean",
    ]
    LAG_PERIODS = [1, 2, 3, 7, 14, 28]
    WINDOW_SIZES = [7, 14, 28]

    for feature in FEATURES_FOR_LAGS:
        for lag in LAG_PERIODS:
            df_fe[f"{feature}_lag_{lag}"] = df_fe[feature].shift(lag)

        series_shifted = df_fe[feature].shift(1)
        for window in WINDOW_SIZES:
            df_fe[f"{feature}_roll_{window}d_mean"] = series_shifted.rolling(window, min_periods=1).mean()
            df_fe[f"{feature}_roll_{window}d_std"] = series_shifted.rolling(window, min_periods=1).std()

    console.print(f"[green]✓ Feature engineering complete. Final shape: {df_fe.shape}[/green]")
    return df_fe


def run_strategy_1_evaluation(df_features: pd.DataFrame, test_size: float) -> pd.DataFrame:
    """
    Runs the full training and evaluation pipeline for Strategy 1 (Aggregate-then-Predict).
    """
    console = Console()
    HORIZONS = [1, 2, 3, 4, 5]

    # 1. Prepare data: create targets and split
    df_model = df_features.copy()
    for h in HORIZONS:
        df_model[f"target_t+{h}"] = df_model["temp_mean"].shift(-h)
    df_model.dropna(inplace=True)

    target_cols = [f"target_t+{h}" for h in HORIZONS]
    feature_cols = [col for col in df_model.columns if col not in target_cols + ["temp_mean"]]

    n_train = int(len(df_model) * (1 - test_size))
    X_train = df_model[feature_cols].iloc[:n_train]
    y_train = df_model[target_cols].iloc[:n_train]
    X_test = df_model[feature_cols].iloc[n_train:]
    y_test = df_model[target_cols].iloc[n_train:]

    # 2. Feature selection with LASSO
    console.print("Running LASSO for linear feature selection...")
    lasso_pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("lasso", MultiTaskLassoCV(cv=TimeSeriesSplit(n_splits=5), random_state=105, n_jobs=-1)),
    ])
    lasso_pipeline.fit(X_train, y_train)
    coef = lasso_pipeline.named_steps["lasso"].coef_
    selected_mask = np.abs(coef).sum(axis=0) > 0
    linear_features = X_train.columns[selected_mask].tolist()

    X_train_linear = X_train[linear_features]
    X_test_linear = X_test[linear_features]
    X_train_tree = X_train  # Use all features for tree models
    X_test_tree = X_test

    console.print(
        f"Feature selection complete: {len(linear_features)} linear features, {len(feature_cols)} tree features."
    )

    # 3. Train models
    all_predictions = {}

    # RidgeCV
    console.print("Training RidgeCV model...")
    ridge_preds = {}
    for h in HORIZONS:
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 100))
        ridge.fit(X_train_linear, y_train[f"target_t+{h}"])
        ridge_preds[h] = ridge.predict(X_test_linear)
    all_predictions["Q8.1_RidgeCV"] = ridge_preds

    # LGBM for Ensemble
    console.print("Training LGBM component for ensemble...")
    lgbm_preds = {}
    for h in HORIZONS:
        lgbm = LGBMRegressor(random_state=105, n_jobs=-1, verbosity=-1)
        lgbm.fit(X_train_tree, y_train[f"target_t+{h}"])
        lgbm_preds[h] = lgbm.predict(X_test_tree)

    # Create Ensemble predictions
    ensemble_preds = {h: (ridge_preds[h] + lgbm_preds[h]) / 2.0 for h in HORIZONS}
    all_predictions["Q8.1_Ensemble"] = ensemble_preds

    # 4. Evaluate all models
    console.print("Evaluating model performance...")
    results_list = []
    for model_name, preds_dict in all_predictions.items():
        r2_scores, rmse_scores, mae_scores = [], [], []
        for h in HORIZONS:
            y_true = y_test[f"target_t+{h}"].values
            y_pred = preds_dict[h]
            r2_scores.append(r2_score(y_true, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae_scores.append(mean_absolute_error(y_true, y_pred))

        results_list.append({
            "Model": model_name,
            "Avg_R2": np.mean(r2_scores),
            "Avg_RMSE": np.mean(rmse_scores),
            "Avg_MAE": np.mean(mae_scores),
        })

    # Add champion for reference
    results_list.append({"Model": "Champion_Daily", "Avg_R2": 0.6195, "Avg_RMSE": 0.9664, "Avg_MAE": 0.7698})

    results_df = pd.DataFrame(results_list).sort_values(by="Avg_R2", ascending=False)
    return results_df


def prepare_strategy_2_data(df_hourly_clean: pd.DataFrame, test_start_timestamp: pd.Timestamp) -> dict:
    """
    Prepares data for Strategy 2: extensive hourly features, 120-hour targets, and train/test splits.
    """
    console = Console()

    # 1. Feature Engineering
    df_hourly_fe = df_hourly_clean.copy()
    temp_for_targets = df_hourly_fe[["temp"]].copy()

    # Temporal & Cyclical Features
    df_hourly_fe["hour"] = df_hourly_fe.index.hour
    df_hourly_fe["day_of_week"] = df_hourly_fe.index.dayofweek
    df_hourly_fe["day_of_year"] = df_hourly_fe.index.dayofyear
    df_hourly_fe["month"] = df_hourly_fe.index.month
    df_hourly_fe["sin_hour"] = np.sin(2 * np.pi * df_hourly_fe["hour"] / 24.0)
    df_hourly_fe["cos_hour"] = np.cos(2 * np.pi * df_hourly_fe["hour"] / 24.0)
    df_hourly_fe["sin_doy"] = np.sin(2 * np.pi * df_hourly_fe["day_of_year"] / 366.0)
    df_hourly_fe["cos_doy"] = np.cos(2 * np.pi * df_hourly_fe["day_of_year"] / 366.0)

    # Lags & Rolling Windows
    FEATURES_HOURLY = ["temp", "humidity", "windspeed", "precip", "cloudcover", "solarradiation"]
    LAG_PERIODS_H = [1, 2, 3, 6, 12, 24, 48, 72]
    ROLLING_WINDOWS_H = [3, 6, 12, 24, 72]

    for feature in FEATURES_HOURLY:
        for lag in LAG_PERIODS_H:
            df_hourly_fe[f"{feature}_lag_{lag}h"] = df_hourly_fe[feature].shift(lag)
        shifted = df_hourly_fe[feature].shift(1)
        for window in ROLLING_WINDOWS_H:
            df_hourly_fe[f"{feature}_roll_{window}h_mean"] = shifted.rolling(window, min_periods=1).mean()

    # Cleanup
    cols_to_drop = FEATURES_HOURLY + [
        "dew",
        "icon",
        "conditions",
        "name",
        "address",
        "resolvedAddress",
        "latitude",
        "longitude",
        "source",
        "preciptype",
    ]
    df_hourly_fe.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    df_hourly_fe = df_hourly_fe.select_dtypes(include=[np.number])
    console.print(f"Hourly feature engineering complete. Shape: {df_hourly_fe.shape}")

    # 2. Target Creation
    HORIZON_H = 120
    df_hourly_model = df_hourly_fe.copy()
    for h in range(1, HORIZON_H + 1):
        df_hourly_model[f"target_h+{h}"] = temp_for_targets["temp"].shift(-h)

    df_hourly_model.dropna(inplace=True)

    # 3. Data Splitting
    train_df = df_hourly_model[df_hourly_model.index < test_start_timestamp]
    test_df = df_hourly_model[df_hourly_model.index >= test_start_timestamp]

    target_cols_h = [f"target_h+{h}" for h in range(1, HORIZON_H + 1)]
    feature_cols_h = [col for col in df_hourly_model.columns if col not in target_cols_h]

    # 4. Organize and Return
    day_targets = {}
    for day in range(1, 6):
        start_h = (day - 1) * 24 + 1
        end_h = day * 24
        day_targets[f"day_{day}"] = [f"target_h+{h}" for h in range(start_h, end_h + 1)]

    console.print(f"Data prepared for Strategy 2. Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    return {
        "X_train": train_df[feature_cols_h],
        "y_train": train_df[target_cols_h],
        "X_test": test_df[feature_cols_h],
        "y_test": test_df[target_cols_h],
        "day_targets": day_targets,
    }


def run_strategy_2_evaluation(X_train, y_train, X_test, y_test, day_targets) -> dict:
    """
    Trains 5 specialist hourly models, predicts, aggregates, and evaluates performance.
    """
    console = Console()
    console.print("\nTraining Strategy 2: 5 specialist hourly models...")

    predictions_s2 = {}
    for day_key, targets in day_targets.items():
        console.print(f"  Training model for {day_key}...", end=" ")
        # Using a lightly-tuned, fast-training LGBM as the base estimator
        model = MultiOutputRegressor(LGBMRegressor(random_state=105, n_jobs=-1, verbosity=-1), n_jobs=-1)
        model.fit(X_train, y_train[targets])

        model_path = f"./assets/models/hourly/hourly_model_{day_key}.pkl"
        joblib.dump(model, model_path)
        console.print(f"✓ (Saved to {model_path})")

        preds = model.predict(X_test)
        predictions_s2[day_key] = pd.DataFrame(preds, index=X_test.index, columns=targets)
        console.print("✓")

    # Aggregate hourly predictions to daily averages
    all_hourly_preds = pd.concat(predictions_s2.values(), axis=1)
    daily_pred_s2 = {}
    daily_true_s2 = {}

    for i in range(1, 6):
        day_key = f"day_{i}"
        targets = day_targets[day_key]
        daily_pred_s2[day_key] = all_hourly_preds[targets].resample("D").mean().mean(axis=1)
        daily_true_s2[day_key] = y_test[targets].resample("D").mean().mean(axis=1)

    df_pred_s2 = pd.DataFrame(daily_pred_s2)
    df_true_s2 = pd.DataFrame(daily_true_s2)
    df_pred_s2, df_true_s2 = df_pred_s2.align(df_true_s2, join="inner", axis=0)

    # Evaluate performance on daily averages
    r2_s2, rmse_s2, mae_s2 = [], [], []
    for i in range(1, 6):
        day_key = f"day_{i}"
        r2_s2.append(r2_score(df_true_s2[day_key], df_pred_s2[day_key]))
        rmse_s2.append(np.sqrt(mean_squared_error(df_true_s2[day_key], df_pred_s2[day_key])))
        mae_s2.append(mean_absolute_error(df_true_s2[day_key], df_pred_s2[day_key]))

    console.print("\n[green]✓ Strategy 2 evaluation complete.[/green]")

    return {
        "Model": "Q8.2_Hourly_MultiModel",
        "Avg_R2": np.mean(r2_s2),
        "Avg_RMSE": np.mean(rmse_s2),
        "Avg_MAE": np.mean(mae_s2),
    }


def run_hourly_deep_dive_analysis(X_test, y_test, day_targets, model_path_template):
    """
    Loads pre-trained hourly models and performs an enhanced, stylized deep-dive
    performance analysis at the hourly resolution.
    """
    console = Console()

    # 1. Load models and generate predictions
    all_hourly_preds = {}
    for day in range(1, 6):
        day_key = f"day_{day}"
        targets = day_targets[day_key]
        model_path = model_path_template.format(day_num=day)
        try:
            model = joblib.load(model_path)
            preds = model.predict(X_test)
            all_hourly_preds[day_key] = pd.DataFrame(preds, index=X_test.index, columns=targets)
        except FileNotFoundError:
            console.print(f"[bold red]Error: Model file not found at {model_path}.[/bold red]")
            return None
    full_hourly_preds_df = pd.concat(all_hourly_preds.values(), axis=1)

    # 2. Calculate per-horizon metrics
    per_horizon_metrics = []
    for h in range(1, 121):
        target_col = f"target_h+{h}"
        y_true_h = y_test[target_col].values
        y_pred_h = full_hourly_preds_df[target_col].values
        rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
        r2 = r2_score(y_true_h, y_pred_h)
        per_horizon_metrics.append({"horizon": h, "rmse": rmse, "r2": r2})
    metrics_df = pd.DataFrame(per_horizon_metrics)

    console.print(
        "\n--- [bold]Overall Average Hourly Performance[/bold] ---\n",
        f"  - Avg. Hourly RMSE: {metrics_df['rmse'].mean():.4f}°C\n",
        f"  - Avg. Hourly R²:   {metrics_df['r2'].mean():.4f}",
        sep="",
    )

    # --- 3. Enhanced Visualization: Hourly Performance Decay Curve ---
    fig_decay = go.Figure()

    day_colors = ["#0771A4", "#24C0D2", "#E7B142", "#963D4D", "#369D8E"]

    # Add a single, continuous background trace to connect the segments
    fig_decay.add_trace(
        go.Scatter(
            x=metrics_df["horizon"],
            y=metrics_df["rmse"],
            mode="lines",
            name="Overall Trend",
            line=dict(color="#D9D9D9", width=4),
            showlegend=False,
            hovertemplate="",
        )
    )

    # Add the colored segments for each day's model
    for day in range(1, 6):
        start_h = (day - 1) * 24 + 1
        end_h = day * 24
        day_metrics = metrics_df[(metrics_df["horizon"] >= start_h) & (metrics_df["horizon"] <= end_h)]
        fig_decay.add_trace(
            go.Scatter(
                x=day_metrics["horizon"],
                y=day_metrics["rmse"],
                mode="lines",
                name=f"Day {day} Forecasts",
                line=dict(color=day_colors[day - 1], width=4),
            )
        )

    # Add the highlight panel for the sharp decay period
    fig_decay.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=0,
        x1=11,
        y0=0,
        y1=1,
        fillcolor="rgba(177, 228, 251, 0.85)",
        opacity=0.2,
        layer="below",  # Draw behind the data lines
        line_width=0,
    )

    fig_decay.add_annotation(
        x=5.5,
        y=0.92,
        yref="paper",
        text="<b>Sharp Decay Period</b>",
        showarrow=False,
        font=dict(color="#38545f", size=12, family="Econ Sans Condensed, Roboto Condensed, Arial Narrow, sans-serif"),
        opacity=0.95,
    )

    fig_decay.update_layout(
        title_text='<b>Hourly Forecast Error Rapidly Increases, Then Stabilizes</b><br><span style="font-size: 16px; color: #38545f;">RMSE of hourly temperature forecasts by hours ahead. The sharpest decay occurs within the first 11 hours.</span>',
        xaxis_title="Forecast Horizon (Hours Ahead)",
        yaxis_title="RMSE (°C)",
        plot_bgcolor="#fafafa",
        paper_bgcolor="#fafafa",
        yaxis_showgrid=True,
        xaxis_showgrid=False,
        yaxis_gridcolor="#D9D9D9",
        xaxis_linecolor="#D9D9D9",
        yaxis_zeroline=False,
        showlegend=True,
        legend=dict(x=0.99, y=0.98, xanchor="right", yanchor="top"),
        height=550,
    )
    fig_decay.show()

    # --- 4. Enhanced Visualization: Performance by Time of Day ---
    metrics_df["hour_of_day"] = (metrics_df["horizon"] - 1) % 24
    rmse_by_hour = metrics_df.groupby("hour_of_day")["rmse"].mean().reset_index()

    # Economist Style Guide sequential colorscale
    economist_blue_seq = [[0.0, "#86d6f7"], [0.5, "#388ac3"], [1.0, "#006092"]]

    fig_by_hour = go.Figure(
        data=go.Bar(
            x=rmse_by_hour["hour_of_day"],
            y=rmse_by_hour["rmse"],
            marker=dict(color=rmse_by_hour["rmse"], colorscale=economist_blue_seq, showscale=False),
            showlegend=False,
        )
    )

    # Find min and max error hours for annotation
    min_rmse_hour = rmse_by_hour.loc[rmse_by_hour["rmse"].idxmin()]
    max_rmse_hour = rmse_by_hour.loc[rmse_by_hour["rmse"].idxmax()]

    fig_by_hour.add_annotation(
        x=max_rmse_hour["hour_of_day"],
        y=max_rmse_hour["rmse"],
        text=f"Peak Error<br><b>{max_rmse_hour['rmse']:.2f}°C</b>",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color="#006092"),
    )
    fig_by_hour.add_annotation(
        x=min_rmse_hour["hour_of_day"],
        y=min_rmse_hour["rmse"],
        text=f"Lowest Error<br><b>{min_rmse_hour['rmse']:.2f}°C</b>",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color="#006092"),
    )

    # Apply Economist Style Guide layout
    fig_by_hour.update_layout(
        title_text='<b>Overnight Forecasts Are Systematically More Accurate</b><br><span style="font-size: 16px; color: #38545f;">Average RMSE by hour of day, across all 5 forecast days. Errors peak during afternoon hours.</span>',
        xaxis_title="Hour of Day (0-23)",
        yaxis_title="Average RMSE (°C)",
        plot_bgcolor="#fafafa",
        paper_bgcolor="#fafafa",
        yaxis_gridcolor="#D9D9D9",
        xaxis_showgrid=False,
        yaxis_showgrid=True,
        xaxis_linecolor="#D9D9D9",
        xaxis=dict(tickmode="linear", dtick=2),
        height=550,
    )
    fig_by_hour.show()

    return metrics_df
