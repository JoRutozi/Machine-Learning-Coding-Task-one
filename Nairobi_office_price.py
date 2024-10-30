import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Nairobi Office Price Ex (1).csv')
x = data['SIZE'].values  # Extract the 'SIZE' column
y = data['PRICE'].values  # Extract the 'PRICE' column

# Function to compute Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent function
def gradient_descent(x, y, m, c, learning_rate):
    n = len(x)
    y_pred = m * x + c
    dm = -(2/n) * np.sum(x * (y - y_pred))
    dc = -(2/n) * np.sum(y - y_pred)
    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c

# Initialize slope (m) and intercept (c) with random values
m = np.random.rand()
c = np.random.rand()

# Training parameters
learning_rate = 0.0001
epochs = 10

# Training the model
for epoch in range(epochs):
    y_pred = m * x + c
    error = mean_squared_error(y, y_pred)
    print(f"Epoch {epoch+1}: Mean Squared Error = {error}")
    m, c = gradient_descent(x, y, m, c, learning_rate)

# Plotting the line of best fit after training
plt.scatter(x, y, color="blue", label="Data points")
plt.plot(x, m * x + c, color="red", label="Line of Best Fit")
plt.xlabel("Office Size (sq. ft)")
plt.ylabel("Office Price")
plt.legend()
plt.show()

# Predicting the price for an office size of 100 sq. ft
size = 100
predicted_price = m * size + c
predicted_price = max(predicted_price, 0)  # Ensures price is non-negative
print(f"Predicted price for 100 sq. ft office: {predicted_price}")
