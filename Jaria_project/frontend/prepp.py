import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Load the precipitation file
file_path = 'Dataset/PeshawarprepFinal.csv'  # Update this to your file path
data = pd.read_csv(file_path)

# Convert 'time' column to datetime
data['time'] = pd.to_datetime(data['time'])

# Filter data for years from 2006 onward
base_data = data[data['time'].dt.year >= 2006]

# Fit an ARIMA model to extrapolate the data
arima_model = ARIMA(base_data['rain'], order=(1, 1, 0))  # Replace 'precip' with the actual precipitation column name
arima_fit = arima_model.fit()

# Generate future dates and times from 2006-01-01 12:00:00 to 2099-12-31 12:00:00
future_dates = pd.date_range(start="2006-01-01 12:00:00", end="2099-12-31 12:00:00", freq='D')

# Predict future precipitation values for these dates
future_precip = arima_fit.forecast(steps=len(future_dates))

# Create a new DataFrame with the extended date-time range and predicted precipitation values
extended_data = pd.DataFrame({
    'time': future_dates,
    'rain': future_precip  # Replace 'precip' if needed
})

# Save the extended data to a CSV file
output_file = 'PeshawarprepFinal_extended_with_time.csv'  # Output file name
extended_data.to_csv(output_file, index=False)

print(f"Extended dataset with time saved to {output_file}")
