# Stock Signal Generation: Probabilistic Technical Analysis

**Project Status:** Active | **Last Updated:** [12th December, 2025]

A data-driven system applying quantitative analytic principles to generate probabilistic buy/sell signals across US and UK equity markets. This project was initiated as a personal endeavor, leveraging my professional insights and domain knowledge gained while working at IG Group.

---

## Key Features & Data Science Focus

This project serves as a showcase of a complete data science pipeline, from data acquisition and feature engineering to model creation and interactive visualization.

* **Custom Probabilistic Model:** Developed a proprietary aggregation model that combines signals from multiple technical indicators (e.g., **RSI, MACD, Bollinger Bands, ATR, Trend**) using a weighted-average approach. The output is a consolidated **Buy Probability** and **Sell Probability** (ranging from 0 to 1), providing a measure of signal conviction.
* **Modular Feature Engineering Pipeline:** Logic for calculation and management of technical indicator features is abstracted into a custom, reusable Python package (`trading_tools`), demonstrating software engineering practices and code modularity.
* **Reliable Data Acquisition:** Implemented functions using the **`yfinance` API** to reliably fetch and manage historical daily price data for various stable US and UK Stocks.
* **Interactive Visualization:** Used **Plotly Graph Objects** to create dynamic, interactive charts that overlay the calculated Buy/Sell probabilities onto historical price action. This allows for detailed visual back-testing and signal pattern identification.

---

## 💻 Technical Stack

| Category | Tools & Libraries |
| :--- | :--- | 
| **Language** | `Python` | 
| **Data Analysis** | `Pandas`, `NumPy` | 
| **Visualization** | `Plotly Graph Objects` | 
| **Data Source** | `yfinance` | 
| **Architecture** | Jupyter Notebooks (`.ipynb`), Modular Python Scripts |

---

## Methodology:

The system is designed to provide a highly interpretable, quantified assessment of current market conditions for a given stock.

### Process Flow

1.  **Data Ingestion:** Download historical OHLCV data for selected tickers.
2.  **Feature Calculation:** Calculate a standardised set of technical indicator values (e.g., 14-day RSI, 12/26/9 MACD, 20-period Bollinger Bands) for each day.
3.  **Signal Aggregation (Model Core):** Each indicator generates a 'micro-signal' (e.g., RSI < 30 is a Buy signal). These micro-signals are assigned specific, empirically-derived weights.
4.  **Signal Ranking:** A final summary table ranks the assets by the absolute difference in probabilities (`Latest_Diff_Prob`), identifying the stocks with the highest net conviction (strongest signal) for either a buy or a sell.

---

## Results

A table similar to the below will be showcased in the notebook, ranking by the highest conviction level (`Latest_Diff_Prob`).

| Ticker |  Latest\_Buy\_Prob | Latest\_Sell\_Prob | Latest\_Diff\_Prob (Conviction) |
| :--- |  :--- | :--- | :--- |
| V  | **0.229** | 0.075 | **0.154 (Net Buy)** |
| MA | **0.229** | 0.078 | **0.152 (Net Buy)** |
| HD  | **0.231** | 0.086 | **0.145 (Net Buy)** |
| GOOGL | **0.336** | 0.205 | 0.132 (Net Buy) |
| MRK  | **0.340** | 0.264 | 0.075 (Net Buy) |

---

## 🚀 Getting Started

These instructions will get a copy of the project up and running on your local machine.

### Prerequisites

You will need Python 3.x installed.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/johnnyy-f/Buy-and-Sell-Signals-US-UK
    cd buy-and-sell-signals-US-UK
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the analysis:**
    Open the main Jupyter Notebook and execute the cells in order.
    ```bash
    jupyter notebook "Buy and Sell signals.ipynb"
    ```

---

## Update to Original Project

To improve usability, the project has been updated to allow CSV files containing ticker lists to be uploaded in the `/markets` folder, eliminating the need to hard-code tickers in the Jupyter Notebook.  

For easier navigation and analysis, a Streamlit dashboard has been added. Each uploaded CSV appears as a separate tab, allowing you to switch between markets and view signals interactively.


## Execution Instructions

Follow these steps to refresh your data and launch the dashboard:

---

## 1. Add your Market CSVs

Place your CSV files in the `/markets` folder.  
The script is designed to read tickers from the **first column**, skipping the header rows.

---

## 2. Run the Data Pipeline

Run this script to fetch the latest market data and calculate the technical signal probabilities.  
This must be run whenever you want to update the signals with the latest market closes.

```bash
python pipeline.py
```

The script will log its progress in the terminal as it processes each file.

---

## 3. Launch the Dashboard

Once the pipeline has generated the processed files, launch the interactive web interface:

```bash
streamlit run app.py
```

## Accessing the App

The dashboard will open in your browser at:  
[http://localhost:8501](http://localhost:8501)

**Tabs:** Switch between different markets using the tabs at the top.  
**Dropdowns:** Use the dropdown menu located inside the Plotly charts to switch between specific stocks within that market.

---

## Future Work Aspiration & Machine Learning Integration

To further enhance the predictive power and demonstrate advanced data science skills, the following steps are planned:

* **Machine Learning (ML) Implementation:** Convert the problem into a binary classification task. Define the target variable (Y) as: *Did the stock price increase by 3% or more in the next 5 days?*
* **Model Training:** Train models such as **Random Forest** or **Gradient Boosting (XGBoost)** using the technical indicators as features (X).
* **Performance Backtesting & Risk Metrics:** Develop a dedicated backtesting module to simulate historical trading based on the signals. Report financial performance metrics like **Sharpe Ratio**, **Sortino Ratio**, and **Maximum Drawdown**, demonstrating an understanding of risk-adjusted returns.
* **Cloud Deployment:** Containerize the solution using Docker and deploy it on a cloud platform (e.g., AWS/GCP) to run daily signal generation.
