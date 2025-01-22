import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Load the data
file_path = "dataset/AstoreFinal.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Convert date column to datetime and sort
data['time'] = pd.to_datetime(data['time'], format='%m/%d/%Y')
data = data.sort_values(by='time')

# Prepare data for training
data['day_of_year'] = data['time'].dt.dayofyear
data['year'] = data['time'].dt.year

X = data[['day_of_year', 'year']].values
y = data['max'].values  # Replace 'max' with 'tasmax' if applicable

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_preds)
print(f"Random Forest MSE: {rf_mse}")

# Model 2: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_preds)
print(f"Linear Regression MSE: {lr_mse}")

# Model 3: ARIMA
arima_order = (5, 1, 0)  # Adjust as necessary
arima_model = ARIMA(y, order=arima_order).fit()
arima_preds = arima_model.forecast(steps=len(y_test))
arima_mse = mean_squared_error(y_test[:len(arima_preds)], arima_preds)
print(f"ARIMA MSE: {arima_mse}")

# Future Prediction
future_years = list(range(2018, 2100))  # Adjust as per your requirement
future_dates = [datetime(year, 1, 1) for year in future_years]
future_days = [(date.timetuple().tm_yday, date.year) for date in future_dates]
future_X = np.array(future_days)

rf_future_preds = rf_model.predict(future_X)
lr_future_preds = lr_model.predict(future_X)
arima_future_preds = arima_model.forecast(steps=len(future_X))

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(data['time'], y, label="Actual Data", color="black")
plt.plot(future_dates, rf_future_preds, label="RF Predictions", color="blue")
plt.plot(future_dates, lr_future_preds, label="LR Predictions", color="green")
plt.plot(future_dates, arima_future_preds, label="ARIMA Predictions", color="red")
plt.xlabel("Year")
plt.ylabel("Temperature (tasmax)")
plt.title("Temperature Predictions Until 2099")
plt.legend()
plt.show()
