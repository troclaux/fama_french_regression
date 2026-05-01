# fama_french_regression

The following information is derived from the source provided, which details how to perform a **Fama-French Three-Factor Regression** in Python to evaluate investment performance.

### Implementation Prompt & Objective

**Objective:**
The goal is to evaluate the performance of an investment (in this case, the **Fidelity Growth Company Fund - FDGRX**) by determining how much of its returns are explained by three specific risk factors: **Market Risk**, **Size (SMB)**, and **Value (HML)**,. This process identifies if a fund is adding value through manager skill (**Alpha**) or simply by taking on specific types of risk,.

**Context:**
*   **The Equation:** The regression solves $R_{asset} - R_f = \alpha + \beta(R_{m} - R_f) + sSMB + hHML + \epsilon$, where $R_{asset} - R_f$ is the **excess return** of the asset.
*   **Data Sources:** Financial data is pulled online using `pandas_datareader` from **Yahoo Finance** (for stock prices) and **Kenneth French’s Data Library** (for factor data),,.
*   **Requirements:** You must use Python libraries like `pandas`, `datetime`, and `statsmodels` to process the data and run the Ordinary Least Squares (OLS) regression.

---

### Complete Python Code Implementation

Based on the steps demonstrated in the source, here is the complete code-:

```python
import pandas_datareader as reader
import pandas as pd
import datetime as dt
import statsmodels.api as sm

# 1. Define Time Horizon (5 Years ending June 2020)
end = dt.date(2020, 6, 30)
start = dt.date(end.year - 5, end.month, end.day)

# 2. Define the Asset (Fidelity Growth Company Fund)
funds = ['FDGRX']

# 3. Request Asset Data from Yahoo Finance
stock_data = reader.get_data_yahoo(funds, start, end)['Adj Close']

# 4. Calculate Monthly Cumulative Returns
fund_ret = stock_data.pct_change()
# Using a lambda function to calculate cumulative monthly returns
fund_ret_monthly = fund_ret.resample('M').apply(lambda x: (x + 1).prod() - 1)

# Slice to remove the first row (NaN) to match factor data later
fund_ret_monthly = fund_ret_monthly[1:]

# 5. Request Fama-French Factors from Kenneth French's Library
# 'F-F_Research_Data_Factors' contains Mkt-RF, SMB, HML, and RF
factors = reader.DataReader('F-F_Research_Data_Factors', 'famafrench', start, end)

# Slice factor data to match the asset return timeframe (starting July)
factors = factors[1:]

# 6. Align Indices and Merge Dataframes
# Set asset index equal to factor index for a clean merge
fund_ret_monthly.index = factors.index
merge = pd.merge(fund_ret_monthly, factors, on='Date')

# 7. Scaling and Calculating Excess Returns
# Factor data is in percentages; divide by 100 to match asset return scale
merge[['Mkt-RF', 'SMB', 'HML', 'RF']] = merge[['Mkt-RF', 'SMB', 'HML', 'RF']] / 100

# Calculate the Dependent Variable (Asset Return minus Risk-Free Rate)
merge['FDGRX-RF'] = merge['FDGRX'] - merge['RF']

# 8. Define Variables for Regression
y = merge['FDGRX-RF']                        # Dependent Variable
X = merge[['Mkt-RF', 'SMB', 'HML']]          # Independent Variables

# Add a constant to represent Alpha (the intercept)
X_sm = sm.add_constant(X)

# 9. Run the Ordinary Least Squares (OLS) Regression
model = sm.OLS(y, X_sm)
results = model.fit()

# 10. Display Interpretation Table
print(results.summary())
```

### Key Context for Interpretation
*   **R-squared:** A high value (e.g., 0.94) indicates that the factors explain most of the fund's movements.
*   **Beta (Market Risk):** A coefficient > 1.0 indicates the fund is more volatile than the market.
*   **SMB (Size):** Positive values lean toward small-cap; negative values lean toward large-cap.
*   **HML (Value):** For FDGRX, this is negative (approx. -0.5), confirming it is heavily weighted toward **growth stocks**.
*   **Alpha:** Represents performance above the three factors; however, the source notes you must check the **p-value** to ensure it is statistically significant.
