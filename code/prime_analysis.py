import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Function to generate prime numbers using the Sieve of Eratosthenes
def sieve_of_eratosthenes(n):
    """Generates all prime numbers up to n using the Sieve of Eratosthenes."""
    sieve = [True] * (n + 1)
    sieve[0], sieve[1] = False, False  # 0 and 1 are not primes
    for start in range(2, int(n**0.5) + 1):
        if sieve[start]:
            for multiple in range(start * start, n + 1, start):
                sieve[multiple] = False
    return [num for num, is_prime in enumerate(sieve) if is_prime]

# Generate primes up to a specified limit
upper_limit = 1000000  # Adjust as needed
primes = sieve_of_eratosthenes(upper_limit)

# Create a DataFrame with indexed primes
prime_data = pd.DataFrame({'Index': range(1, len(primes) + 1), 'Prime': primes})

# Calculate additional features
prime_data['Prime_Diff'] = prime_data['Prime'].diff().fillna(0)
prime_data['Prime_Density'] = prime_data['Prime'] / prime_data['Index']

# Save the dataset to a CSV file
prime_data.to_csv('prime_numbers_dataset.csv', index=False)

# Prepare data for multi-linear regression
features = ['Index', 'Prime_Density', 'Prime_Diff']
X = prime_data[features]
y = prime_data['Prime']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the multi-linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_test = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

# Print metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R^2 Score: {r2}")

# Plot comparison of real vs. predicted primes
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.7, label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Ideal Fit")
plt.title("Real vs. Predicted Prime Numbers (Test Set)")
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.grid(True)
plt.savefig('real_vs_predicted.png')
plt.close()

# Plot residuals (error analysis)
residuals = y_test - y_pred_test
plt.figure(figsize=(8, 6))
plt.scatter(y_test, residuals, alpha=0.7, label="Residuals")
plt.axhline(0, color='red', linestyle='--', label="Zero Error Line")
plt.title("Residuals of Predictions (Test Set)")
plt.xlabel("Real Values")
plt.ylabel("Residuals")
plt.legend()
plt.grid(True)
plt.savefig('residuals.png')
plt.close()

print("Plots saved as 'real_vs_predicted.png' and 'residuals.png'")
