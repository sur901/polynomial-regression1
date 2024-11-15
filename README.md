import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create or load the dataset
# Example dataset: Position Level vs Salary
data = {
    "Position Level": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary": [15000, 20000, 30000, 50000, 80000, 110000, 150000, 200000, 300000, 500000]
}
df = pd.DataFrame(data)

# Separate the features and target variable
X = df["Position Level"].values.reshape(-1, 1)  # Features (Position Level)
y = df["Salary"].values  # Target (Salary)

# Step 2: Create polynomial features
degree = 4  # Degree of the polynomial
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# Step 3: Train the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Step 4: Visualize the polynomial regression results
plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X, model.predict(X_poly), color='blue', label=f'Polynomial Regression (Degree {degree})')
plt.title('Position Level vs Salary')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Step 5: Predict salaries for new data
new_data = np.array([6.5, 8.5]).reshape(-1, 1)  # Example: Position levels 6.5 and 8.5
new_data_poly = poly.transform(new_data)
predicted_salaries = model.predict(new_data_poly)

print("Predicted Salaries:")
for pos, sal in zip(new_data.flatten(), predicted_salaries):
    print(f"Position Level {pos}: ${sal:.2f}")

# Step 6: Evaluate the model
y_pred = model.predict(X_poly)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
