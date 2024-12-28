import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import _tree
from sklearn.metrics import mean_squared_error

# Function to print the formula for Random Forest prediction
def print_random_forest_formula(model):
    N = len(model.estimators_)  # Number of trees in the forest
    print(f"Number of trees in the forest (N): {N}")
    
    # Print the formula for the final prediction
    print("\nThe final prediction (y) is the average of all N tree predictions:")
    print("y = (1/N) * Σ Tᵢ(X) for i=1 to N")
    print("Where:")
    print("Tᵢ(X) is the prediction of the i-th decision tree for input X")
    print("The final prediction y is the average of all N tree predictions.\n")
    
    # Optionally, print the predictions of each individual tree
    print("Predictions from each tree in the forest:")
    for i, tree in enumerate(model.estimators_):
        print(f"T_{i+1}(X): {tree.predict(X[0:1])[0]:.4f}")

# Function to choose the best regression model
def choose_best_model(X, y, feature_names):
    # List of regression models to consider
    model = RandomForestRegressor(n_estimators=10, random_state=42)

    # Perform cross-validation with negative mean squared error (neg_mean_squared_error)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mean_score = np.mean(scores)
    print(f"Mean MSE for Random Forest: {-mean_score:.4f}")

    # Fit the model to the full dataset
    model.fit(X, y)

    # Print the formula for the Random Forest model
    print_random_forest_formula(model)

    # Optionally, calculate the final MSE on the full dataset
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"Final Mean Squared Error (MSE) on the dataset: {mse:.4f}")
    
    return model

# Example usage:
if __name__ == "__main__":
    # Example data: 7 samples, 1 feature (coordinates), continuous target variable
    X = np.array([1.8, 2, 2.2, 2.5, 2.55, 2.5, 2.4]).reshape(-1, 1)  # Reshape to (n_samples, n_features)
    Y = np.array([0.19994548, 0.3, 0.5, 1, 1.4, 1.8, 1.95])  # Continuous targets

    feature_names = ["Feature 1"]  # You can add more feature names if you have more features

    # Choose the best model
    best_model = choose_best_model(X, Y, feature_names)
    
    # Calculate feature importance
    print("\nFeature Importance:")
    feature_importances = best_model.feature_importances_
    for name, importance in zip(feature_names, feature_importances):
        print(f"{name}: {importance:.4f}")
