import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rich.console import Console
from rich.table import Table
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class EnsembleDriftDetector:
    """
    Production-grade drift detector with 3-tier priority hierarchy:
    1. HIGH PRIORITY (Immediate): Severe performance degradation
    2. MEDIUM PRIORITY (Proactive): Significant drift patterns
    3. LOW PRIORITY (Scheduled): Time-based maintenance
    """

    def __init__(
        self,
        baseline_rmse=None,
        baseline_r2=None,
        # HIGH PRIORITY: Concept drift thresholds (immediate action)
        critical_rmse_threshold=2.0,  # 100% degradation = critical
        critical_r2_threshold=0.40,  # Massive R² drop
        severe_rmse_threshold=1.50,  # 50% degradation = severe
        severe_r2_threshold=0.30,
        # MEDIUM PRIORITY: Data/Label drift thresholds (proactive)
        rmse_threshold=1.30,
        r2_threshold=0.20,
        data_pvalue=0.001,
        data_feature_count=10,
        label_pvalue=0.001,
        label_shift_magnitude=2.5,
        # LOW PRIORITY: Time-based maintenance
        max_days_without_retrain=90,  # Force retrain after 90 days
    ):
        if baseline_rmse is None or baseline_r2 is None:
            raise ValueError("Must provide baseline_rmse and baseline_r2 from your model!")

        self.baseline_rmse = baseline_rmse
        self.baseline_r2 = baseline_r2
        self.baseline_features = {}
        self.baseline_temp = None

        # HIGH PRIORITY thresholds
        self.critical_rmse_threshold = critical_rmse_threshold
        self.critical_r2_threshold = critical_r2_threshold
        self.severe_rmse_threshold = severe_rmse_threshold
        self.severe_r2_threshold = severe_r2_threshold

        # MEDIUM PRIORITY thresholds
        self.rmse_threshold = rmse_threshold
        self.r2_threshold = r2_threshold
        self.data_pvalue = data_pvalue
        self.data_feature_count = data_feature_count
        self.label_pvalue = label_pvalue
        self.label_shift_magnitude = label_shift_magnitude

        # LOW PRIORITY threshold
        self.max_days_without_retrain = max_days_without_retrain
        self.last_retrain_day = 0

        print(f"✓ Drift Detector initialized (3-TIER PRIORITY SYSTEM)")
        print(f"  Baseline RMSE: {self.baseline_rmse:.4f}°C")
        print(f"  Baseline R²: {self.baseline_r2:.4f}")
        print(f"\n  Priority Thresholds:")
        print(f"  🔴 HIGH (Immediate):  RMSE>{severe_rmse_threshold}× OR R²drop>{severe_r2_threshold}")
        print(f"  🟡 MEDIUM (Proactive): Data+Label drift OR moderate performance drop")
        print(f"  🟢 LOW (Scheduled):    {max_days_without_retrain} days without retrain")

    def set_baseline_distributions(self, X_train, y_train):
        """Store training distributions"""
        for col in X_train.columns:
            self.baseline_features[col] = X_train[col].values
        self.baseline_temp = y_train.values if hasattr(y_train, "values") else y_train
        print(f"✓ Stored {len(self.baseline_features)} feature distributions\n")

    def check_drift(self, X_recent, y_recent, y_pred_recent, current_day):
        """
        Check for all three priority levels of drift

        Args:
            current_day: Current day number in simulation (for LOW priority check)
        """
        from scipy import stats

        print(f"\n{'=' * 70}")
        print(f"DRIFT DETECTION CHECK - DAY {current_day}")
        print(f"{'=' * 70}")

        # Calculate performance metrics
        current_rmse = np.sqrt(np.mean((y_recent - y_pred_recent) ** 2))
        current_r2 = 1 - (np.sum((y_recent - y_pred_recent) ** 2) / np.sum((y_recent - y_recent.mean()) ** 2))

        rmse_ratio = current_rmse / self.baseline_rmse
        r2_drop = self.baseline_r2 - current_r2

        # ============================================================
        # PRIORITY 1: HIGH - CRITICAL PERFORMANCE DEGRADATION
        # ============================================================
        critical_performance = (rmse_ratio > self.critical_rmse_threshold) or (r2_drop > self.critical_r2_threshold)

        severe_performance = (rmse_ratio > self.severe_rmse_threshold) or (r2_drop > self.severe_r2_threshold)

        high_priority = critical_performance or severe_performance

        print(f"\n🔴 PRIORITY 1: HIGH (Immediate Action)")
        print(f"   Performance Status:")
        print(f"   - RMSE: {current_rmse:.4f}°C vs baseline {self.baseline_rmse:.4f}°C")
        print(f"   - Ratio: {rmse_ratio:.4f} ({(rmse_ratio - 1) * 100:+.1f}%)")
        print(f"   - R²: {current_r2:.4f} vs baseline {self.baseline_r2:.4f}")
        print(f"   - R² drop: {r2_drop:+.4f}")

        if critical_performance:
            print(f"   🚨 CRITICAL: RMSE>{self.critical_rmse_threshold}× OR R²drop>{self.critical_r2_threshold}")
            print(f"   → IMMEDIATE RETRAINING REQUIRED")
        elif severe_performance:
            print(f"   ⚠️  SEVERE: RMSE>{self.severe_rmse_threshold}× OR R²drop>{self.severe_r2_threshold}")
            print(f"   → URGENT RETRAINING RECOMMENDED")
        else:
            print(f"   ✓ OK: Performance within acceptable bounds")

        # ============================================================
        # PRIORITY 2: MEDIUM - PROACTIVE DRIFT DETECTION
        # ============================================================

        # Data Drift
        drifted_features = []
        if self.baseline_features:
            for col in X_recent.columns:
                if col in self.baseline_features:
                    _, p_value = stats.ks_2samp(self.baseline_features[col], X_recent[col].values)
                    if p_value < self.data_pvalue:
                        drifted_features.append(col)

        data_drift = len(drifted_features) > self.data_feature_count

        # Label Drift
        _, p_temp = stats.ks_2samp(self.baseline_temp, y_recent)
        temp_shift = abs(y_recent.mean() - self.baseline_temp.mean())
        label_drift = (p_temp < self.label_pvalue) and (temp_shift > self.label_shift_magnitude)

        # Moderate concept drift (not severe, but noticeable)
        moderate_concept = (rmse_ratio > self.rmse_threshold) or (r2_drop > self.r2_threshold)

        # Medium priority if: moderate performance drop OR (data + label drift)
        medium_priority = moderate_concept or (data_drift and label_drift)

        print(f"\n🟡 PRIORITY 2: MEDIUM (Proactive Action)")
        print(f"   Moderate Performance Degradation:")
        print(f"   - Threshold: RMSE>{self.rmse_threshold}× OR R²drop>{self.r2_threshold}")
        print(f"   - Status: {'⚠️  DETECTED' if moderate_concept else '✓ OK'}")

        print(f"   Data Drift (Features):")
        print(f"   - Drifted: {len(drifted_features)}/{len(X_recent.columns)} features")
        print(f"   - Threshold: >{self.data_feature_count} features at p<{self.data_pvalue}")
        print(f"   - Status: {'⚠️  DETECTED' if data_drift else '✓ OK'}")

        print(f"   Label Drift (Temperature):")
        print(f"   - Shift: {y_recent.mean() - self.baseline_temp.mean():+.2f}°C (|{temp_shift:.2f}|°C)")
        print(f"   - Threshold: p<{self.label_pvalue} AND |shift|>{self.label_shift_magnitude}°C")
        print(f"   - Status: {'⚠️  DETECTED' if label_drift else '✓ OK'}")

        if medium_priority and not high_priority:
            print(f"   → PROACTIVE RETRAINING RECOMMENDED")

        # ============================================================
        # PRIORITY 3: LOW - SCHEDULED MAINTENANCE
        # ============================================================
        days_since_retrain = current_day - self.last_retrain_day
        low_priority = days_since_retrain >= self.max_days_without_retrain

        print(f"\n🟢 PRIORITY 3: LOW (Scheduled Maintenance)")
        print(f"   Time-Based Check:")
        print(f"   - Days since last retrain: {days_since_retrain}")
        print(f"   - Threshold: {self.max_days_without_retrain} days")
        print(f"   - Status: {'⏰ DUE' if low_priority else '✓ OK'}")

        if low_priority and not (high_priority or medium_priority):
            print(f"   → SCHEDULED MAINTENANCE RETRAIN")

        # ============================================================
        # FINAL DECISION
        # ============================================================

        # Determine overall action
        if high_priority:
            retrain = True
            priority_level = "HIGH"
            if critical_performance:
                reason = "🔴 CRITICAL performance degradation"
            else:
                reason = "🔴 SEVERE performance degradation"
        elif medium_priority:
            retrain = True
            priority_level = "MEDIUM"
            if moderate_concept:
                reason = "🟡 Moderate performance drop"
            else:
                reason = "🟡 Data + Label drift detected"
        elif low_priority:
            retrain = True
            priority_level = "LOW"
            reason = f"🟢 Scheduled maintenance ({days_since_retrain} days)"
        else:
            retrain = False
            priority_level = None
            reason = None

        print(f"\n{'=' * 70}")
        print(f"DECISION: {'⚠️  RETRAIN RECOMMENDED' if retrain else '✓ MODEL IS STABLE'}")
        if retrain:
            print(f"Priority: {priority_level}")
            print(f"Reason: {reason}")
        print(f"{'=' * 70}\n")

        return retrain, {
            "priority": priority_level,
            "reason": reason,
            "high_priority": high_priority,
            "medium_priority": medium_priority,
            "low_priority": low_priority,
            "critical_performance": critical_performance,
            "severe_performance": severe_performance,
            "moderate_concept": moderate_concept,
            "data_drift": data_drift,
            "label_drift": label_drift,
            "rmse_ratio": rmse_ratio,
            "current_rmse": current_rmse,
            "current_r2": current_r2,
            "temp_shift": temp_shift,
            "days_since_retrain": days_since_retrain,
        }

    def update_last_retrain(self, day):
        """Update the last retrain day after successful retraining"""
        self.last_retrain_day = day


def run_static_prediction(ensemble_models, X_test_linear, X_test_tree, y_test):
    """
    Runs a static (non-adaptive) prediction loop over a test set.

    Args:
        ensemble_models (dict): Pre-trained models, keyed by 'h1', 'h2', etc.
        X_test_linear (pd.DataFrame): Test features for the linear component.
        X_test_tree (pd.DataFrame): Test features for the tree component.
        y_test (pd.DataFrame): True target values.

    Returns:
        dict: A dictionary containing predictions and performance metrics.
    """
    print("Making predictions for all 5 horizons (Static Baseline)...\n")

    static_predictions_all_horizons = {}
    horizon_metrics = {}
    horizons = range(1, 6)

    for h in horizons:
        print(f"  Predicting horizon h{h} (t+{h} days ahead)...")
        horizon_preds = []

        # Iterative prediction to simulate a daily process
        for i in range(len(X_test_linear)):
            X_dict = {"linear": X_test_linear.iloc[i : i + 1], "tree": X_test_tree.iloc[i : i + 1]}
            pred = ensemble_models[f"h{h}"].predict(X_dict)[0]
            horizon_preds.append(pred)

        # Store predictions
        pred_col_name = f"pred_t+{h}"
        static_predictions_all_horizons[pred_col_name] = np.array(horizon_preds)

        # Calculate metrics
        true_col_name = f"target_temp_t+{h}"
        y_true = y_test[true_col_name].values
        y_pred = static_predictions_all_horizons[pred_col_name]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        horizon_metrics[f"h{h}"] = {"rmse": rmse, "mae": mae, "r2": r2}

    # Format predictions into a DataFrame
    predictions_df = pd.DataFrame(static_predictions_all_horizons, index=y_test.index)

    # Print summary table
    print(f"\n{'=' * 60}")
    print("✓ Static Ensemble Performance by Horizon:")
    print(f"{'=' * 60}")
    print(f"{'Horizon':<12} {'RMSE (°C)':<15} {'MAE (°C)':<15} {'R²':<10}")
    print("-" * 60)
    for h in horizons:
        metrics = horizon_metrics[f"h{h}"]
        print(f"t+{h:<10} {metrics['rmse']:<15.4f} {metrics['mae']:<15.4f} {metrics['r2']:<10.4f}")

    # Calculate and print average performance
    avg_rmse = np.mean([m["rmse"] for m in horizon_metrics.values()])
    avg_mae = np.mean([m["mae"] for m in horizon_metrics.values()])
    avg_r2 = np.mean([m["r2"] for m in horizon_metrics.values()])

    print("-" * 60)
    print(f"{'Average':<12} {avg_rmse:<15.4f} {avg_mae:<15.4f} {avg_r2:<10.4f}")
    print(f"{'=' * 60}\n")

    return {
        "predictions": predictions_df,
        "horizon_metrics": horizon_metrics,
        "avg_metrics": {"rmse": avg_rmse, "mae": avg_mae, "r2": avg_r2},
    }


def run_adaptive_simulation(
    initial_models,
    drift_detector,
    X_train_full,
    y_train_full,
    X_test_full,
    y_test_full,
    linear_feature_cols,
    tree_feature_cols,
    retrain_check_interval=30,
    drift_lookback_days=14,
):
    """
    Runs a full adaptive retraining simulation.
    """
    console = Console()

    # --- Initialization ---
    adaptive_predictions_all_horizons = {f"h{h}": [] for h in range(1, 6)}
    retrain_dates = []
    retrain_priorities = []
    drift_history = []

    # Initialize training data slices
    current_X_train = X_train_full.copy()
    current_y_train = y_train_full.copy()

    # Clone initial models to avoid modifying them in place
    adaptive_models = {f"h{h}": clone(initial_models[f"h{h}"]) for h in range(1, 6)}

    console.print("🔄 Starting Adaptive Simulation (3-Tier Priority System)...")
    console.print(f"Drift checks: Every {retrain_check_interval} days")
    console.print(f"Lookback window: {drift_lookback_days} days\n")

    console.print("Performing initial training on baseline data...")
    for h in range(1, 6):
        X_train_dict = {"linear": current_X_train[linear_feature_cols], "tree": current_X_train[tree_feature_cols]}
        adaptive_models[f"h{h}"].fit(X_train_dict, current_y_train[f"target_temp_t+{h}"])
    console.print("✓ Initial training complete.\n")

    # --- Simulation Loop ---
    for i in range(len(X_test_full)):
        current_day = i + 1

        # Prepare feature dictionaries for the current day's prediction
        X_test_linear_day = X_test_full.iloc[i : i + 1][linear_feature_cols]
        X_test_tree_day = X_test_full.iloc[i : i + 1][tree_feature_cols]
        X_dict_day = {"linear": X_test_linear_day, "tree": X_test_tree_day}

        # 1. Make predictions for all horizons
        for h in range(1, 6):
            pred = adaptive_models[f"h{h}"].predict(X_dict_day)[0]
            adaptive_predictions_all_horizons[f"h{h}"].append(pred)

        # 2. Check for drift at specified intervals
        if (current_day % retrain_check_interval == 0) and (current_day >= drift_lookback_days):
            recent_start_idx = max(0, i - drift_lookback_days + 1)

            # Prepare recent data for the detector (using h1 as the reference)
            y_recent_true = y_test_full["target_temp_t+1"].iloc[recent_start_idx : i + 1].values
            y_recent_pred = np.array(adaptive_predictions_all_horizons["h1"][recent_start_idx : i + 1])
            X_recent_tree = X_test_full.iloc[recent_start_idx : i + 1][tree_feature_cols]

            drift_detected, drift_info = drift_detector.check_drift(
                X_recent=X_recent_tree, y_recent=y_recent_true, y_pred_recent=y_recent_pred, current_day=current_day
            )

            drift_history.append({"day": current_day, **drift_info})

            # 3. Retrain if drift is detected
            if drift_detected:
                priority = drift_info["priority"]
                print(f"📍 Day {current_day}: {priority} PRIORITY - {drift_info['reason']}")
                print(f"   Retraining all 5 horizon models...")

                retrain_dates.append(current_day)
                retrain_priorities.append(priority)

                # Expand the training data
                current_X_train = pd.concat([X_train_full, X_test_full.iloc[: i + 1]])
                current_y_train = pd.concat([y_train_full, y_test_full.iloc[: i + 1]])

                # Retrain all horizon models
                for h in range(1, 6):
                    X_retrain_dict = {
                        "linear": current_X_train[linear_feature_cols],
                        "tree": current_X_train[tree_feature_cols],
                    }
                    adaptive_models[f"h{h}"].fit(X_retrain_dict, current_y_train[f"target_temp_t+{h}"])

                drift_detector.update_last_retrain(current_day)
                print(f"✓ Models retrained with {len(current_X_train)} total samples\n")

    # --- Post-Simulation Analysis ---
    # Convert prediction lists to a DataFrame
    predictions_df = pd.DataFrame(
        {f"pred_t+{h}": adaptive_predictions_all_horizons[f"h{h}"] for h in range(1, 6)}, index=y_test_full.index
    )

    # Calculate final performance metrics
    horizon_metrics = {}
    for h in range(1, 6):
        y_true = y_test_full[f"target_temp_t+{h}"].values
        y_pred = predictions_df[f"pred_t+{h}"].values
        horizon_metrics[f"h{h}"] = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

    avg_rmse = np.mean([m["rmse"] for m in horizon_metrics.values()])
    avg_mae = np.mean([m["mae"] for m in horizon_metrics.values()])
    avg_r2 = np.mean([m["r2"] for m in horizon_metrics.values()])

    # --- Print Summary Tables (can be called in notebook) ---
    _display_adaptive_summary(horizon_metrics, avg_rmse, avg_mae, avg_r2, retrain_dates, retrain_priorities)

    return {
        "predictions": predictions_df,
        "horizon_metrics": horizon_metrics,
        "avg_metrics": {"rmse": avg_rmse, "mae": avg_mae, "r2": avg_r2},
        "drift_history": pd.DataFrame(drift_history),
        "retrain_dates": retrain_dates,
        "retrain_priorities": retrain_priorities,
    }


def _display_adaptive_summary(horizon_metrics, avg_rmse, avg_mae, avg_r2, retrain_dates, retrain_priorities):
    """A helper to print the final summary tables for the adaptive run."""
    console = Console()

    # Performance table
    perf_table = Table(title="✓ Adaptive Ensemble Performance by Horizon")
    perf_table.add_column("Horizon", style="cyan")
    perf_table.add_column("RMSE (°C)", justify="right")
    perf_table.add_column("MAE (°C)", justify="right")
    perf_table.add_column("R²", justify="right")

    for h in range(1, 6):
        metrics = horizon_metrics[f"h{h}"]
        perf_table.add_row(f"t+{h}", f"{metrics['rmse']:.4f}", f"{metrics['mae']:.4f}", f"{metrics['r2']:.4f}")

    perf_table.add_section()
    perf_table.add_row("Average", f"{avg_rmse:.4f}", f"{avg_mae:.4f}", f"{avg_r2:.4f}")
    console.print(perf_table)
    print(f"  Retrains: {len(retrain_dates)} times on days {retrain_dates}")

    # Retraining events table
    priority_counts = {p: retrain_priorities.count(p) for p in ["HIGH", "MEDIUM", "LOW"]}

    events_table = Table(title="Retraining Events by Priority Level")
    events_table.add_column("Priority", style="cyan")
    events_table.add_column("Count", justify="center")
    events_table.add_column("Days")

    for priority in ["HIGH", "MEDIUM", "LOW"]:
        days = [d for i, d in enumerate(retrain_dates) if retrain_priorities[i] == priority]
        events_table.add_row(priority, str(priority_counts[priority]), str(days))

    console.print(events_table)


def plot_simulation_results(static_results, adaptive_results, y_test):
    """
    Generates a suite of visualizations comparing static vs. adaptive model performance.
    Preserves all original plots from the simulation.
    """
    # --- Extract Data ---
    static_metrics = static_results["avg_metrics"]
    adaptive_metrics = adaptive_results["avg_metrics"]
    static_h_metrics = static_results["horizon_metrics"]
    adaptive_h_metrics = adaptive_results["horizon_metrics"]

    # We need to handle the predictions carefully.
    # static_results['predictions'] is a DataFrame
    # adaptive_results['predictions'] is a DataFrame
    static_preds = static_results["predictions"]
    adaptive_preds = adaptive_results["predictions"]
    retrain_dates = adaptive_results["retrain_dates"]

    horizons = list(range(1, 6))
    test_dates = y_test.index

    # --- 1. Overall Performance Comparison Bar Chart ---
    fig_comparison = go.Figure()
    metrics = ["RMSE (°C)", "MAE (°C)", "R²"]
    static_vals = [static_metrics["rmse"], static_metrics["mae"], static_metrics["r2"]]
    adaptive_vals = [adaptive_metrics["rmse"], adaptive_metrics["mae"], adaptive_metrics["r2"]]

    fig_comparison.add_trace(
        go.Bar(
            name="Static Ensemble",
            x=metrics,
            y=static_vals,
            marker_color="#FF6B6B",
            text=[f"{v:.4f}" for v in static_vals],
            textposition="auto",
        )
    )
    fig_comparison.add_trace(
        go.Bar(
            name="Adaptive Ensemble",
            x=metrics,
            y=adaptive_vals,
            marker_color="#4ECDC4",
            text=[f"{v:.4f}" for v in adaptive_vals],
            textposition="auto",
        )
    )
    fig_comparison.update_layout(
        title="<b>Static vs Adaptive Ensemble: Overall Performance Comparison</b>",
        barmode="group",
        height=450,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_comparison.show()

    # --- 2. Performance by Horizon Comparison ---
    fig_horizons = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("<b>RMSE by Horizon</b>", "<b>MAE by Horizon</b>", "<b>R² by Horizon</b>"),
        horizontal_spacing=0.12,
    )

    # Helper to extract horizon metrics as lists
    def get_h_vals(metrics_dict, metric_name):
        return [metrics_dict[f"h{h}"][metric_name] for h in horizons]

    # RMSE
    fig_horizons.add_trace(
        go.Scatter(
            x=horizons,
            y=get_h_vals(static_h_metrics, "rmse"),
            mode="lines+markers",
            name="Static",
            line=dict(color="#FF6B6B"),
        ),
        row=1,
        col=1,
    )
    fig_horizons.add_trace(
        go.Scatter(
            x=horizons,
            y=get_h_vals(adaptive_h_metrics, "rmse"),
            mode="lines+markers",
            name="Adaptive",
            line=dict(color="#4ECDC4"),
        ),
        row=1,
        col=1,
    )
    # MAE
    fig_horizons.add_trace(
        go.Scatter(
            x=horizons,
            y=get_h_vals(static_h_metrics, "mae"),
            mode="lines+markers",
            name="Static",
            line=dict(color="#FF6B6B"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig_horizons.add_trace(
        go.Scatter(
            x=horizons,
            y=get_h_vals(adaptive_h_metrics, "mae"),
            mode="lines+markers",
            name="Adaptive",
            line=dict(color="#4ECDC4"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    # R²
    fig_horizons.add_trace(
        go.Scatter(
            x=horizons,
            y=get_h_vals(static_h_metrics, "r2"),
            mode="lines+markers",
            name="Static",
            line=dict(color="#FF6B6B"),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig_horizons.add_trace(
        go.Scatter(
            x=horizons,
            y=get_h_vals(adaptive_h_metrics, "r2"),
            mode="lines+markers",
            name="Adaptive",
            line=dict(color="#4ECDC4"),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig_horizons.update_layout(
        height=450, template="plotly_white", title_text="<b>Performance Across Forecast Horizons</b>", showlegend=True
    )
    fig_horizons.show()

    # --- 3. Improvement Percentages ---
    improvements_rmse = [
        (
            (static_h_metrics[f"h{h}"]["rmse"] - adaptive_h_metrics[f"h{h}"]["rmse"])
            / static_h_metrics[f"h{h}"]["rmse"]
            * 100
        )
        for h in horizons
    ]
    improvements_mae = [
        (
            (static_h_metrics[f"h{h}"]["mae"] - adaptive_h_metrics[f"h{h}"]["mae"])
            / static_h_metrics[f"h{h}"]["mae"]
            * 100
        )
        for h in horizons
    ]
    improvements_r2 = [
        (
            (adaptive_h_metrics[f"h{h}"]["r2"] - static_h_metrics[f"h{h}"]["r2"])
            / abs(static_h_metrics[f"h{h}"]["r2"])
            * 100
        )
        for h in horizons
    ]

    fig_improvements = go.Figure()
    fig_improvements.add_trace(
        go.Bar(
            name="RMSE Imp.",
            x=[f"t+{h}" for h in horizons],
            y=improvements_rmse,
            marker_color="#FF6B6B",
            text=[f"{v:+.1f}%" for v in improvements_rmse],
            textposition="outside",
        )
    )
    fig_improvements.add_trace(
        go.Bar(
            name="MAE Imp.",
            x=[f"t+{h}" for h in horizons],
            y=improvements_mae,
            marker_color="#4ECDC4",
            text=[f"{v:+.1f}%" for v in improvements_mae],
            textposition="outside",
        )
    )
    fig_improvements.add_trace(
        go.Bar(
            name="R² Imp.",
            x=[f"t+{h}" for h in horizons],
            y=improvements_r2,
            marker_color="#95E1D3",
            text=[f"{v:+.1f}%" for v in improvements_r2],
            textposition="outside",
        )
    )

    fig_improvements.update_layout(
        title="<b>Percentage Improvement: Adaptive vs Static</b>",
        yaxis_title="Improvement (%)",
        barmode="group",
        height=450,
        template="plotly_white",
    )
    fig_improvements.show()

    # --- 4. Error Timeline with Retrain Markers ---
    fig_timeline = go.Figure()

    # Calculate absolute errors for horizon 1
    y_true_h1 = y_test["target_temp_t+1"].values
    static_err_h1 = np.abs(y_true_h1 - static_preds["pred_t+1"].values)
    adaptive_err_h1 = np.abs(y_true_h1 - adaptive_preds["pred_t+1"].values)

    fig_timeline.add_trace(
        go.Scatter(
            x=test_dates,
            y=static_err_h1,
            mode="lines",
            name="Static Abs Error",
            line=dict(color="#FF6B6B", width=1),
            opacity=0.5,
            fill="tozeroy",
            fillcolor="rgba(255, 107, 107, 0.15)",
        )
    )
    fig_timeline.add_trace(
        go.Scatter(
            x=test_dates,
            y=adaptive_err_h1,
            mode="lines",
            name="Adaptive Abs Error",
            line=dict(color="#4ECDC4", width=1.5),
            opacity=0.9,
            fill="tozeroy",
            fillcolor="rgba(78, 205, 196, 0.15)",
        )
    )

    # Add vertical lines manually to avoid Timestamp arithmetic bug in add_vline
    for day_num in retrain_dates:
        retrain_date = test_dates[day_num - 1]
        fig_timeline.add_shape(
            type="line",
            x0=retrain_date,
            x1=retrain_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="green", width=2, dash="dash"),
        )
        # Add annotation separately
        fig_timeline.add_annotation(
            x=retrain_date,
            y=1.05,
            yref="paper",
            text=f"Day {day_num}",
            showarrow=False,
            font=dict(size=9, color="green"),
        )

    fig_timeline.update_layout(
        title="<b>Prediction Error Over Time (t+1) with Retrain Events</b>",
        xaxis_title="Date",
        yaxis_title="Absolute Error (°C)",
        height=500,
        template="plotly_white",
    )
    fig_timeline.show()

    # --- 5. Rolling Performance Trends ---
    window_size = 30

    # Calculate rolling RMSE as pandas Series to preserve the index
    static_series = pd.Series(static_err_h1, index=test_dates)
    adaptive_series = pd.Series(adaptive_err_h1, index=test_dates)

    static_roll_rmse = static_series.rolling(window=window_size).apply(lambda x: np.sqrt(np.mean(x**2)))
    adaptive_roll_rmse = adaptive_series.rolling(window=window_size).apply(lambda x: np.sqrt(np.mean(x**2)))

    fig_rolling = go.Figure()
    fig_rolling.add_trace(
        go.Scatter(
            x=static_roll_rmse.index,
            y=static_roll_rmse,
            mode="lines",
            name="Static (30d RMSE)",
            line=dict(color="#FF6B6B", width=3),
            fill="tozeroy",
            fillcolor="rgba(255, 107, 107, 0.15)",
        )
    )
    fig_rolling.add_trace(
        go.Scatter(
            x=adaptive_roll_rmse.index,
            y=adaptive_roll_rmse,
            mode="lines",
            name="Adaptive (30d RMSE)",
            line=dict(color="#4ECDC4", width=3),
            fill="tozeroy",
            fillcolor="rgba(78, 205, 196, 0.15)",
        )
    )

    # Add markers for retrain events
    for day_num in retrain_dates:
        retrain_date = test_dates[day_num - 1]
        fig_rolling.add_shape(
            type="line",
            x0=retrain_date,
            x1=retrain_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="green", width=2, dash="dash"),
        )

    first_valid_date = static_roll_rmse.first_valid_index()
    fig_rolling.update_layout(
        title="<b>30-Day Rolling RMSE Trends</b>",
        xaxis_title="Date",
        yaxis_title="Rolling RMSE (°C)",
        height=450,
        template="plotly_white",
        # 1. Set explicit x-axis range to start from the beginning of the test set
        xaxis=dict(
            range=[first_valid_date, test_dates.max()],
            # 2. Set the tick format to "Month Day" (e.g., "Mar 24")
            tickformat="%b %d",
        ),
    )

    fig_rolling.show()
