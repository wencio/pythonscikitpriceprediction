import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Import the warnings module to handle warning messages
import warnings

# Ignore warning messages
warnings.filterwarnings("ignore")


# Download historical stock price data for a given ticker symbol (e.g., Apple)
ticker_symbol = 'AAPL'
data = yf.download(ticker_symbol, start='2020-01-01', end='2023-01-01')

# Use adjusted closing prices as the target variable
y = data['Adj Close']

# Create features (X) for this example; you can define them as needed
# For instance, in this example, we're using random values as features.
X = pd.DataFrame({'Feature1': range(len(y)), 'Feature2': range(len(y) - 1, -1, -1), 'Feature3': np.random.rand(len(y))})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance

# Calculate Mean Squared Error (MSE) to measure the average squared difference between actual and predicted values.
mse = mean_squared_error(y_test, y_pred)

# Calculate Coefficient of Determination (R-squared) to measure the goodness of fit of the model.
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Coefficient of Determination (R^2): {r2:.2f}")

# Allow the user to input values for making a prediction
valor1 = float(input("Enter value 1: "))  # Enter the first value
valor2 = float(input("Enter value 2: "))  # Enter the second value
valor3 = float(input("Enter value 3: "))  # Enter the third value

# Use the trained model to make predictions with the user-entered values
new_features = np.array([[valor1, valor2, valor3]])
prediction = model.predict(new_features)

# Explain what valor1, valor2, and valor3 could represent
# These values could represent various factors or indicators that affect the stock price.
# For example, valor1 might represent trading volume, valor2 could represent market sentiment,
# and valor3 might represent a financial metric like the price-to-earnings ratio.
# These are just hypothetical examples; you should use meaningful features based on your analysis.
print(f"Predicted stock price: {prediction[0]:.2f}")
