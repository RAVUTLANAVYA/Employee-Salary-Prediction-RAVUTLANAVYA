import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Sample training data (Years, Job Rate) -> Salary
# Format: [ [years, job_rate], ... ]
X = np.array([
    [1, 3.0],
    [2, 3.5],
    [3, 4.0],
    [4, 4.5],
    [5, 5.0],
    [6, 5.5],
    [7, 6.0],
    [8, 6.5],
    [9, 7.0],
    [10, 7.5],
])

# Corresponding salaries (in ₹)
y = np.array([
    25000,
    30000,
    35000,
    40000,
    45000,
    50000,
    55000,
    60000,
    65000,
    70000
])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "linearmodel.pkl")
print("Model saved as linearmodel.pkl")
