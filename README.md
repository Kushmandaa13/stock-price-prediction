# Stock Price Prediction

This mini project predicts stock closing prices using machine learning models. It was built as part of the Corizo Machine Learning Certificate and compares Linear Regression, Decision Tree, and Random Forest on historical Apple (`AAPL`) stock data.

## Project Objective

Stock price forecasting is an important problem for investors, traders, and analysts. The goal of this project is to train machine learning models that can learn patterns from historical stock data and predict the closing price of a stock.

## Dataset

- File used in this project: `AAPL_stock_data.csv`
- Stock: Apple Inc. (`AAPL`)
- Columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- Date range in the current dataset: `2020-01-02` to `2025-03-31`
- Source note: this project uses historical stock data stored locally as CSV, and the script also supports downloading data from Yahoo Finance or using synthetic fallback data

## Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Features Created

To improve prediction quality and reduce leakage, the script engineers lagged and rolling features such as:

- Previous day open, high, low, close, and volume
- 7-day moving average
- 21-day moving average
- Daily returns
- Volatility
- High-low spread
- Open-close difference

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- yfinance

## How to Run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the project:

```bash
python stock_price_prediction.py
```

Optional examples:

```bash
python stock_price_prediction.py --source csv
python stock_price_prediction.py --source yfinance --ticker AAPL
python stock_price_prediction.py --source synthetic
```

## Output

The script saves plots and generated files in the `outputs/` folder, including:

- Closing price trend
- Volume trend
- Correlation heatmap
- Closing price distribution
- Model comparison chart
- Actual vs predicted chart
- Feature importance plot
- Residual analysis

## Current Results

Based on the included dataset run:

- Best model: `Linear Regression`
- R2 Score: `0.9988`
- MAE: `2.5288`
- RMSE: `3.8527`
- MAPE: `1.47%`

These results show that the model explains nearly all of the variance in the closing price on the current dataset.

## Project Structure

```text
.
|-- AAPL_stock_data.csv
|-- outputs/
|-- requirements.txt
|-- stock_price_prediction.py
|-- README.md
```

## Author

- Kushmandaa Devi Bhaugeeruth
