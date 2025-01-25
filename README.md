# Linear Regression: Predicting House Prices

Linear regression is one of the simplest and most commonly used machine learning models for predictive analysis. In this document, we'll explain the theory behind linear regression and how it can be used to predict house prices.

## What is Linear Regression?

Linear regression is a supervised learning algorithm used to model the relationship between a dependent variable (target) and one or more independent variables (features). It assumes a linear relationship between the variables.

For a single variable (simple linear regression), the relationship is represented as:

For multiple variables (multiple linear regression):

Where:

- : Predicted value (house price in this case)
- : Intercept (value of  when all )
- : Coefficients (weights of features )
- : Independent variables (e.g., number of bedrooms, square footage, etc.)

## Objective

The goal of linear regression is to find the best-fitting line (or hyperplane in multiple dimensions) that minimizes the error between the predicted values () and the actual values (). This error is measured using the **Mean Squared Error (MSE)**:

Where:

- : Number of data points
- : Actual value
- : Predicted value

Minimizing MSE helps in determining the optimal values of .

## Steps to Build a Linear Regression Model

1. **Data Collection**: Gather data with features (independent variables) and the target variable (house price).
2. **Data Preprocessing**:
   - Handle missing values.
   - Normalize or standardize features.
   - Perform feature selection or engineering if necessary.
3. **Model Training**:
   - Split data into training and testing sets.
   - Fit the linear regression model to the training data to learn the coefficients .
4. **Model Evaluation**:
   - Test the model on the test data.
   - Evaluate using metrics like MSE, RMSE (Root Mean Squared Error), or  (coefficient of determination).
5. **Prediction**:
   - Use the trained model to predict house prices for new data.

## Example

Suppose you have a dataset with the following features:

- Number of bedrooms
- Square footage
- Distance to the city center

The relationship might look like this:

### Implementation in Python

Here's a simple example using Python's `scikit-learn` library:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("house_prices.csv")

# Features and target
X = data[["bedrooms", "square_footage", "distance_to_city_center"]]
y = data["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

## Advantages of Linear Regression

- Simple to implement and interpret.
- Computationally efficient.
- Works well when the relationship between variables is approximately linear.

## Limitations

- Assumes linearity between features and target.
- Sensitive to outliers.
- Struggles with multicollinearity (high correlation between independent variables).

## Conclusion

Linear regression is an excellent starting point for predictive modeling, especially when relationships between variables are straightforward. While it has its limitations, understanding and applying linear regression provides a solid foundation for exploring more complex models in machine learning.

