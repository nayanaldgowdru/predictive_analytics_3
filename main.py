import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("sales_data.csv")
 
print("HISTORICAL SALES DATA")
print(data)

print("\nFIRST FIVE RECORDS")
print(data.head())

print("\nMISSING VALUES")
print(data.isnull().sum())

print("\n--- DATA CLEANING ---")

data = data.drop_duplicates()

data["Month"] = pd.to_datetime(data["Month"])

data = data.dropna()

data["Month_Number"] = np.arange(len(data))

print("\nCleaned Dataset:")
print(data)

print("\nData cleaning completed successfully!")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = data[["Month_Number"]]
y = data["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- PREDICTIVE MODEL ---")
print("Model training completed!")

print("\nActual Sales:")
print(y_test.values)

print("\nPredicted Sales:")
print(y_pred)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- MODEL EVALUATION ---")
print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

plt.figure(figsize=(10, 5))

plt.plot(
    data["Month"].iloc[len(X_train):],
    y_test,
    marker="o",
    label="Actual Sales"
)

plt.plot(
    data["Month"].iloc[len(X_train):],
    y_pred,
    marker="o",
    linestyle="--",
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Month")
plt.ylabel("Sales Amount")
plt.legend()
plt.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("predictions.png")

plt.show()

print("\nGraph saved successfully!")

print("\n--- FUTURE SALES FORECASTING ---")

future_months = np.arange(len(data), len(data) + 6)

future_sales = model.predict(
    pd.DataFrame({"Month_Number": future_months})
)

future_dates = pd.date_range(
    start=data["Month"].max() + pd.DateOffset(months=1),
    periods=6,
    freq="MS"
)

forecast = pd.DataFrame({
    "Month": future_dates,
    "Predicted_Sales": future_sales.round(2)
})

print("\nNext 6 Months Sales Forecast:")
print(forecast.to_string(index=False))
