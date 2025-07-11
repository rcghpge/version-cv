# save_model.py (Run this once to create your dummy model file)
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# Create a simple dummy model
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([3, 5, 7, 9])
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
model_filename = "my_custom_model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Dummy model saved to {model_filename}")
