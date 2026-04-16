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

## Findings

### 1. Dataset Summary

The project was run on Apple stock price data with the following raw dataset details:

- Source used: local CSV file
- Raw dataset shape: `1368 x 6`
- Columns available: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- Raw date range: `2020-01-02` to `2025-03-31`
- Missing values: `0` in all columns

After feature engineering and removal of rows with rolling-window null values:

- Final modeling dataset shape: `1346 x 17`
- Final modeled date range: `2020-02-03` to `2025-03-31`
- Number of input features used for prediction: `11`

### 2. Exploratory Data Analysis Findings

- The closing price showed a strong long-term upward trend over the selected period.
- Trading volume fluctuated significantly, with some high-activity days during volatile periods.
- Open, High, Low, and Close prices were strongly positively correlated.
- Volume had a weaker relationship with price variables compared with OHLC values.
- The closing price distribution was broad, reflecting the large price growth across the years covered in the dataset.
- The dataset was clean and ready for modeling after schema validation and type conversion.

### 3. Feature Engineering Findings

The following engineered features were created to improve prediction quality while avoiding data leakage:

- `Prev_Open`
- `Prev_High`
- `Prev_Low`
- `Prev_Close`
- `Prev_Volume`
- `MA_7`
- `MA_21`
- `Returns`
- `Volatility`
- `High_Low_Spread`
- `Open_Close_Diff`

These features helped the models learn short-term price movement, momentum, and volatility patterns from previous trading days.

### 4. Data Preparation Findings

- Training set size: `1076` samples
- Testing set size: `270` samples
- Train-test split: `80:20`
- Standard scaling was applied for Linear Regression
- Tree-based models were trained on the unscaled feature values

### 5. Model Performance Findings

Based on the included dataset run:

| Model | MAE | MSE | RMSE | R2 Score | MAPE (%) |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 2.5288 | 14.8435 | 3.8527 | 0.9988 | 1.4718 |
| Decision Tree | 3.3718 | 30.7801 | 5.5480 | 0.9975 | 1.8847 |
| Random Forest | 2.6478 | 15.2046 | 3.8993 | 0.9988 | 1.5372 |

### 6. Best Model

- Best model: `Linear Regression`
- R2 Score: `0.9988`
- MAE: `2.5288`
- RMSE: `3.8527`
- MAPE: `1.47%`

Linear Regression performed slightly better than Random Forest on this dataset and clearly outperformed the Decision Tree model on the main evaluation metrics.

### 7. Prediction and Residual Findings

- The predicted values closely followed the actual closing prices.
- The residuals were centered near zero, which indicates low systematic prediction error.
- The residual distribution suggested that the best model was stable for this dataset.
- The Actual vs Predicted plot showed that the models captured the overall stock trend very well.

### 8. Feature Importance Findings

From the Random Forest feature importance analysis:

- Lagged OHLC features were among the strongest predictors
- Moving averages (`MA_7` and `MA_21`) contributed strongly to prediction accuracy
- Daily returns and volatility added useful but smaller predictive value

### 9. Final Conclusion

This project successfully demonstrated that machine learning can be used to predict stock closing prices from historical market data. Among the three models tested, Linear Regression produced the best overall performance and explained about `99.9%` of the variance in the target variable on the current dataset.

Overall, the findings show that historical price-based features, especially lagged prices and moving averages, are highly informative for short-term stock price prediction in this project setup.

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
