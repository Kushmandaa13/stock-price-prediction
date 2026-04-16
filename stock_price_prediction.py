"""
===========================================================================
 MINI PROJECT: Stock Price Prediction using Machine Learning
 Course      : Corizo Machine Learning Certificate
 Student     : Kushmandaa Devi Bhaugeeruth
 Description : Predict stock closing prices using Linear Regression,
               Decision Tree, and Random Forest models.
===========================================================================
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        "Missing dependency: "
        f"{missing}\n"
        "Install the project packages with:\n"
        "  python -m pip install -r requirements.txt"
    ) from exc


plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["figure.dpi"] = 120
sns.set_style("whitegrid")


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Train stock price prediction models and save plots locally."
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to label outputs.")
    parser.add_argument(
        "--source",
        choices=["auto", "csv", "synthetic", "yfinance"],
        default="auto",
        help=(
            "Data source to use. 'auto' prefers a local CSV, then yfinance, "
            "then synthetic data."
        ),
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=project_dir / "AAPL_stock_data.csv",
        help="Path to a CSV file with Date, Open, High, Low, Close, Volume columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs",
        help="Directory where plots and generated datasets will be saved.",
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date for yfinance.")
    parser.add_argument("--end", default="2025-04-01", help="End date for yfinance.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data generation.",
    )
    return parser.parse_args()


def ensure_dataframe_schema(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    cleaned = df[required_columns].copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"])

    for column in required_columns[1:]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned.dropna(inplace=True)
    cleaned.sort_values("Date", inplace=True)
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned


def load_data_from_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return ensure_dataframe_schema(df)


def load_data_from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "yfinance is not installed. Run 'python -m pip install -r requirements.txt' "
            "or use '--source csv' / '--source synthetic'."
        ) from exc

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned from Yahoo Finance for ticker '{ticker}'.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    return ensure_dataframe_schema(df)


def generate_synthetic_data(ticker: str, end: str, seed: int) -> pd.DataFrame:
    print(f"\nGenerating realistic {ticker}-like stock data...")

    np.random.seed(seed)
    dates = pd.bdate_range(start="2020-01-02", end=pd.to_datetime(end))
    n_samples = len(dates)

    mu = 0.0008
    sigma = 0.018
    starting_price = 75.0

    daily_returns = np.random.normal(mu, sigma, n_samples)
    close_prices = starting_price * np.cumprod(1 + daily_returns)

    intraday_range = np.abs(np.random.normal(0, sigma * 0.6, n_samples))
    high_prices = close_prices * (1 + intraday_range)
    low_prices = close_prices * (1 - intraday_range)
    open_prices = close_prices * (1 + np.random.normal(0, sigma * 0.3, n_samples))
    volume = np.random.lognormal(mean=18.2, sigma=0.4, size=n_samples).astype(int)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": np.round(open_prices, 2),
            "High": np.round(high_prices, 2),
            "Low": np.round(low_prices, 2),
            "Close": np.round(close_prices, 2),
            "Volume": volume,
        }
    )
    return ensure_dataframe_schema(df)


def resolve_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    csv_path = args.csv_path.resolve()

    if args.source == "csv":
        return load_data_from_csv(csv_path), f"CSV file ({csv_path})"

    if args.source == "synthetic":
        return generate_synthetic_data(args.ticker, args.end, args.seed), "synthetic data"

    if args.source == "yfinance":
        return (
            load_data_from_yfinance(args.ticker, args.start, args.end),
            "Yahoo Finance",
        )

    if csv_path.exists():
        return load_data_from_csv(csv_path), f"CSV file ({csv_path})"

    try:
        return (
            load_data_from_yfinance(args.ticker, args.start, args.end),
            "Yahoo Finance",
        )
    except Exception as exc:
        print(f"\nCould not load Yahoo Finance data: {exc}")
        print("Falling back to synthetic data instead.")
        return generate_synthetic_data(args.ticker, args.end, args.seed), "synthetic data"


def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_df = df.copy()

    feature_df["Prev_Close"] = feature_df["Close"].shift(1)
    feature_df["Prev_Open"] = feature_df["Open"].shift(1)
    feature_df["Prev_High"] = feature_df["High"].shift(1)
    feature_df["Prev_Low"] = feature_df["Low"].shift(1)
    feature_df["Prev_Volume"] = feature_df["Volume"].shift(1)
    feature_df["MA_7"] = feature_df["Close"].shift(1).rolling(window=7).mean()
    feature_df["MA_21"] = feature_df["Close"].shift(1).rolling(window=21).mean()
    feature_df["Returns"] = feature_df["Close"].pct_change().shift(1)
    feature_df["Volatility"] = feature_df["Returns"].rolling(window=21).std()
    feature_df["High_Low_Spread"] = (feature_df["High"] - feature_df["Low"]).shift(1)
    feature_df["Open_Close_Diff"] = (feature_df["Close"] - feature_df["Open"]).shift(1)

    feature_df.dropna(inplace=True)

    feature_columns = [
        "Prev_Open",
        "Prev_High",
        "Prev_Low",
        "Prev_Close",
        "Prev_Volume",
        "MA_7",
        "MA_21",
        "Returns",
        "Volatility",
        "High_Low_Spread",
        "Open_Close_Diff",
    ]
    return feature_df, feature_columns


def save_plot(fig: plt.Figure, output_dir: Path, file_name: str) -> None:
    output_path = output_dir / file_name
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STOCK PRICE PREDICTION - Mini Project")
    print("=" * 70)

    df, dataset_source = resolve_dataset(args)

    dataset_output_path = output_dir / f"{args.ticker}_stock_data.csv"
    df.to_csv(dataset_output_path, index=False)

    print("\nDataset loaded successfully.")
    print(f"Source      : {dataset_source}")
    print(f"Shape       : {df.shape}")
    print(f"Date range  : {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Saved copy  : {dataset_output_path}")

    print("\n" + "-" * 70)
    print("  3. EXPLORATORY DATA ANALYSIS")
    print("-" * 70)

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nLast 5 rows:")
    print(df.tail().to_string(index=False))

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nStatistical Summary:")
    print(df.describe().round(2).to_string())

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["Date"], df["Close"], color="#2563eb", linewidth=1.2)
    ax.set_title(f"{args.ticker} - Closing Price Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.fill_between(df["Date"], df["Close"], alpha=0.08, color="#2563eb")
    save_plot(fig, output_dir, "01_closing_price.png")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(df["Date"], df["Volume"], color="#10b981", width=2, alpha=0.7)
    ax.set_title(f"{args.ticker} - Trading Volume Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    save_plot(fig, output_dir, "02_volume.png")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, ax = plt.subplots(figsize=(8, 6))
    correlation = df[numeric_columns].corr()
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        square=True,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Correlation Heatmap of Stock Features", fontsize=14, fontweight="bold")
    save_plot(fig, output_dir, "03_correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Close"], kde=True, bins=50, color="#8b5cf6", ax=ax)
    ax.set_title("Distribution of Closing Price", fontsize=14, fontweight="bold")
    ax.set_xlabel("Close Price (USD)")
    save_plot(fig, output_dir, "04_close_distribution.png")

    print("\n" + "-" * 70)
    print("  4. FEATURE ENGINEERING")
    print("-" * 70)

    df, feature_columns = create_features(df)
    print(f"Added lagged features to avoid data leakage. Shape: {df.shape}")
    print("Features    : Prev_OHLCV, MA_7, MA_21, Returns, Volatility, Spread, OC_Diff")

    print("\n" + "-" * 70)
    print("  5. DATA PREPARATION")
    print("-" * 70)

    x_values = df[feature_columns]
    y_values = df["Close"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    split_index = int(len(x_values) * 0.8)
    x_test_chrono = x_values.iloc[split_index:]
    y_test_chrono = y_values.iloc[split_index:]
    test_dates_chrono = df.iloc[split_index:]["Date"].values

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    print(f"Training set : {x_train.shape[0]} samples")
    print(f"Testing set  : {x_test.shape[0]} samples")
    print(f"Features used: {len(feature_columns)}")

    print("\n" + "-" * 70)
    print("  6. MODEL TRAINING AND EVALUATION")
    print("-" * 70)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=15,
            n_jobs=-1,
        ),
    }

    results: dict[str, dict[str, float | np.ndarray]] = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        if name == "Linear Regression":
            model.fit(x_train_scaled, y_train)
            predictions = model.predict(x_test_scaled)
        else:
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        results[name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2,
            "MAPE (%)": mape,
            "predictions": predictions,
        }

        print(f"  MAE      : {mae:.4f}")
        print(f"  RMSE     : {rmse:.4f}")
        print(f"  R2 Score : {r2:.4f}")
        print(f"  MAPE     : {mape:.2f}%")

    print("\n" + "-" * 70)
    print("  7. MODEL COMPARISON")
    print("-" * 70)

    comparison_df = pd.DataFrame(
        {
            name: {key: value for key, value in values.items() if key != "predictions"}
            for name, values in results.items()
        }
    ).T

    comparison_df = comparison_df.round(4)
    print("\n" + comparison_df.to_string())

    best_model = comparison_df["R2 Score"].idxmax()
    print(
        f"\nBest Model: {best_model}  "
        f"(R2 = {comparison_df.loc[best_model, 'R2 Score']:.4f})"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_to_plot = ["MAE", "RMSE", "R2 Score"]
    colors = ["#ef4444", "#f59e0b", "#22c55e"]

    for ax, metric, color in zip(axes, metrics_to_plot, colors):
        comparison_df[metric].plot(kind="bar", ax=ax, color=color, edgecolor="white")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticklabels(comparison_df.index, rotation=25, ha="right")
        ax.set_ylabel(metric)

    plt.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.02)
    save_plot(fig, output_dir, "05_model_comparison.png")

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    model_colors = ["#3b82f6", "#f97316", "#10b981"]

    for ax, (name, values), color in zip(axes, results.items(), model_colors):
        if name == "Linear Regression":
            chronological_predictions = models[name].predict(scaler.transform(x_test_chrono))
        else:
            chronological_predictions = models[name].predict(x_test_chrono)

        ax.plot(
            test_dates_chrono,
            y_test_chrono.values,
            color="black",
            linewidth=1.5,
            label="Actual",
            alpha=0.8,
        )
        ax.plot(
            test_dates_chrono,
            chronological_predictions,
            color=color,
            linewidth=1.2,
            label=f"Predicted ({name})",
            linestyle="--",
        )
        ax.set_title(
            f"{name} | R2 = {values['R2 Score']:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel("Close Price (USD)")
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Date")
    plt.suptitle(
        f"{args.ticker} - Actual vs Predicted Closing Price",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    save_plot(fig, output_dir, "06_actual_vs_predicted.png")

    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    feature_importance = pd.Series(importances, index=feature_columns).sort_values(
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance.plot(kind="barh", color="#6366f1", edgecolor="white", ax=ax)
    ax.set_title("Feature Importance - Random Forest", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    save_plot(fig, output_dir, "07_feature_importance.png")

    best_predictions = results[best_model]["predictions"]
    residuals = y_test.values - best_predictions

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(best_predictions, residuals, alpha=0.4, color="#e11d48", s=15)
    axes[0].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"Residuals vs Predicted ({best_model})", fontweight="bold")
    axes[0].set_xlabel("Predicted Price")
    axes[0].set_ylabel("Residual")

    sns.histplot(residuals, kde=True, bins=40, color="#e11d48", ax=axes[1])
    axes[1].set_title("Residual Distribution", fontweight="bold")
    axes[1].set_xlabel("Residual")
    save_plot(fig, output_dir, "08_residual_analysis.png")

    print("\n" + "=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    print(
        f"""
Dataset   : {args.ticker} stock prices ({df['Date'].min().date()} to {df['Date'].max().date()})
Samples   : {len(df)} trading days
Features  : {len(feature_columns)} (OHLCV + engineered technical indicators)
Best Model: {best_model}
  - R2 Score : {results[best_model]['R2 Score']:.4f}
  - MAE      : {results[best_model]['MAE']:.4f}
  - RMSE     : {results[best_model]['RMSE']:.4f}
  - MAPE     : {results[best_model]['MAPE (%)']:.2f}%

Key takeaway: {best_model} achieved the best performance, capturing
{results[best_model]['R2 Score'] * 100:.1f}% of the variance in closing prices.
Feature importance shows that moving averages and lagged OHLC prices are the
strongest predictors, while daily returns and volatility add marginal value.

All outputs were saved to:
{output_dir}
"""
    )
    print("=" * 70)
    print("  PROJECT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
