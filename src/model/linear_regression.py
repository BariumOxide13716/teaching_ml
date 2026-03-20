"""
A linear regression model.

Input:
    xs: A list of x values
    ys: A list of y values
Output:
    A tuple of (slope, intercept)
"""

from sklearn.linear_model import LinearRegression
import numpy as np

def linear_regression(xs, ys):
    # Reshape the input data for sklearn
    X = np.array(xs).reshape(-1, 1)
    y = np.array(ys)

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the slope and intercept
    slope = model.coef_[0]
    intercept = model.intercept_

    return slope, intercept