from sklearn.linear_model import LinearRegression
import numpy as np

# Assume you have independent variables X and a dependent variable y
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
y = np.array([1, 2, 3, 4])

# Create an instance of the LinearRegression class
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Print the coefficients of the model
print(reg.coef_)
