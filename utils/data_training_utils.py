"""
────────────────────────────────────────────────────────────
📄 data_training_utils.py
────────────────────────────────────────────────────────────
This file contains utility functions for training machine learning models.
   It provides functions for:
     1. Training models with hyperparameter tuning using GridSearchCV.
     2. Training a final model on the full dataset for production use.

"""

import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
import pandas as pd
import joblib
import os
from datetime import datetime

def train_test_model_with_hyperparameter_tuning(model, param_grid, data, test_size=0.2, target="price", drop_features=None, random_state=42, scale_data=False):
    """ Train and test a machine learning model with hyperparameter tuning using GridSearchCV. 
    
    Parameters: 
        model: An untrained machine learning model (e.g., from sklearn). 
        param_grid: A dictionary specifying the hyperparameters to tune. 
        data: A pandas DataFrame containing the features and target variable. 
        test_size: Proportion of the dataset to include in the test split. 
        target: The name of the target variable column in the DataFrame. 
        drop_features: List of feature names to exclude from training (default=None). 
        random_state: Random seed for reproducibility. 
        scale_data: If True, scales numeric features using StandardScaler (default=False).
        
    Returns: 
        best_model: The trained model with the best hyperparameters. 
        best_params: The best hyperparameters found during tuning. 
        mae_train: Mean Absolute Error on the train set. 
        r2_train: R-squared score on the train set. 
        mae_test: Mean Absolute Error on the test set. 
        r2_test: R-squared score on the test set. 
    """
    
    # Split data into features and target
    X = data.drop(columns=[target])
    y = data[target]

    # Drop optional features if specified
    if drop_features is not None:
        X = X.drop(columns=drop_features, errors="ignore")

    # Scale numeric features if requested
    if scale_data:
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Define cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define scoring metrics
    scoring = {
        'MAE': make_scorer(mean_absolute_error),
        'R2': make_scorer(r2_score)
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit='MAE',   # Ajusta el mejor modelo según el menor MAE
        verbose=2,
        n_jobs=-1
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    print(f"Best model found: {best_model}")
    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)

    mae_train = grid_search.cv_results_['mean_test_MAE'][grid_search.best_index_]
    r2_train = grid_search.cv_results_['mean_test_R2'][grid_search.best_index_]
    print("Best training MAE:", mae_train)
    print("Best training R2:", r2_train)

    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)
    print("·········································")
    print("MAE test:", mae_test)
    print("R2 test:", r2_test)

    return best_model, best_params, mae_train, r2_train, mae_test, r2_test


def train_final_model_for_production(best_model, data, target="price", drop_features=None, model_dir=".", scale_data=False):
    """ 
    Train the final production model using all available data and save it with joblib. 
    
    Parameters: 
        best_model: Best model obtained from hyperparameter tuning. 
        best_params: Dictionary with the best hyperparameters obtained from GridSearchCV. 
        data: A pandas DataFrame containing the features and target variable. 
        target: The name of the target variable column in the DataFrame. 
        drop_features: List of feature names to exclude from training (default=None). 
        model_dir: Directory where the trained model will be saved. Default is current folder. 
        scale_data: If True, scales numeric features using StandardScaler (default=False).
    
    Returns: 
        final_model: The trained model fitted on the full dataset. 
        model_path: The full path of the saved model file. """

    # Split features and target
    X = data.drop(columns=[target])
    y = data[target]

    # Drop optional features if specified
    if drop_features is not None:
        X = X.drop(columns=drop_features, errors="ignore")

    # Scale numeric features if requested
    if scale_data:
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Train model with all data
    best_model.fit(X, y)

    # Predictions on training data (to report scores)
    y_pred = best_model.predict(X)

    # Calculate scorers
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    model_name = type(best_model).__name__ 

    print("·········································")
    print(f"Final {model_name} trained with ALL data ✅")
    print(f"MAE (on full data): {mae:.4f}")
    print(f"R2  (on full data): {r2:.4f}")

    # Create timestamp for unique file name
    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    # Ensure output directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Build full file path
    model_path = os.path.join(model_dir, f"{model_name}_production_{timestamp}.joblib")

    # Save model
    joblib.dump(best_model, model_path)
    print(f"Model saved to: {model_path}")

    return best_model, model_path

