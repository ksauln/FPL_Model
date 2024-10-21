'''
This code defines a series of functions for training, updating, and evaluating a machine learning model that predicts player performance in Fantasy Premier League (FPL). The model is based on a Random Forest Regressor, and grid search is used to tune hyperparameters. Here's a breakdown of the key components:

### Key Functions and Their Roles:

1. **Data Preparation** (`prepare_data`):
   - This function prepares the player history dataset for modeling. It:
     - Imputes missing values in numeric columns with the mean.
     - Converts categorical columns to the `category` data type.
     - Ensures that the `game` column is treated as an ordered categorical variable.

2. **Season Weights** (`add_season_weights`):
   - Adds a season weight to the dataset, scaling the weight of older seasons relative to the current season. This allows the model to give more importance to recent data.

3. **Model Training with Grid Search** (`train_model_with_grid_search`):
   - Trains a Random Forest Regressor using grid search to find the optimal hyperparameters.
   - The custom scoring function (based on Mean Absolute Error, MAE) handles missing data and can incorporate sample weights for each observation.
   - The grid search tunes parameters like the number of trees, depth, and splitting criteria.

4. **Model Update** (`update_model`):
   - After an initial model is trained, this function updates the model with new data.
   - It combines old training data with new test data and refits the model.
   - It also tracks MAE (before and after the update) to gauge improvement.

5. **Model Evaluation** (`evaluate_model`):
   - Evaluates the model on test data by calculating the MAE and R² score, providing a measure of how well the model performs.

6. **Feature Importance Plotting** (`plot_feature_importance`):
   - Plots the feature importances from the trained Random Forest model, showing which features were most influential in the predictions.

7. **Model Saving and Loading** (`save_model` and `load_model`):
   - These functions save the trained model to disk using `joblib` and load it back for future use. If no model exists, the function returns `None`, signaling that a new model should be trained.

8. **Main Training Function** (`FPLModelTrainer_RF`):
   - This is the main function that orchestrates the training process across multiple game weeks:
     1. Prepares and preprocesses player data.
     2. Loads or trains a new model if none exists.
     3. Iteratively trains and updates the model using past data (train set) and predicts player performance for the current game week (test set).
     4. Evaluates the model after each game week and appends predictions.
     5. Updates the player history with new predictions and evaluates overall performance across game weeks using MAE and R².

   - **Important steps**:
     - For each game week, the data is split into training and test sets.
     - The model is trained or updated and predictions are made.
     - The process continues for all game weeks in `current_season_games`.
     - After completing all game weeks, it plots feature importance and saves the model.

### Summary:
This code sets up a Random Forest-based model for predicting FPL player performance. It preprocesses data, applies grid search to optimize hyperparameters, updates the model iteratively with new data, and evaluates performance using metrics like MAE and R². The code also ensures that the model can be loaded from a saved file, making it reusable across sessions, and supports weighted training based on seasons to prioritize recent data.
'''

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
import matplotlib.pyplot as plt
import joblib
import time
from datetime import datetime
from config import (features, target)

def prepare_data(player_history, features, target):
    numeric_features = player_history[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = player_history[features].select_dtypes(include=[object]).columns.tolist()

    numeric_imputer = SimpleImputer(strategy='mean')
    player_history[numeric_features] = numeric_imputer.fit_transform(player_history[numeric_features])

    for col in categorical_features:
        player_history[col] = player_history[col].astype('category')
        
    if 'game' in player_history.columns:
        player_history['game'] = pd.Categorical(player_history['game'], ordered=True)
    
    return player_history

def add_season_weights(player_history):
    player_history['season'] = player_history['season'].str.split('-').str[0]
    player_history['season'] = pd.to_numeric(player_history['season'], errors='coerce')
    player_history = player_history.dropna(subset=['season'])
    
    current_season = player_history['season'].max()
    min_season = player_history['season'].min()
    
    player_history['season_weight'] = (player_history['season'] - min_season + 1) / (current_season - min_season + 1)
    
    return player_history

def train_model_with_grid_search(X_train, y_train, sample_weights=None):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    def custom_scorer(y_true, y_pred, sample_weight=None):
        y_true = pd.to_numeric(y_true, errors='coerce')
        y_pred = pd.to_numeric(y_pred, errors='coerce')
        if sample_weight is not None:
            sample_weight = pd.to_numeric(sample_weight, errors='coerce')
        
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if sample_weight is not None:
            sample_weight = sample_weight[mask]
        
        return -mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    
    scorer = make_scorer(custom_scorer, greater_is_better=False)
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                             scoring=scorer, cv=3, n_jobs=-1, verbose=0)
    
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    
    print("Best parameters found: ", grid_search.best_params_)
    print("Best score: ", -grid_search.best_score_)
    
    return grid_search.best_estimator_

def update_model(model, X_train, y_train, X_test, y_test, sample_weights=None):
    print("Updating model...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"MAE before update: {mae}")
    
    X_train_updated = pd.concat([X_train, X_test])
    y_train_updated = pd.concat([y_train, y_test])
    
    if sample_weights is not None:
        sample_weights_updated = np.concatenate([sample_weights, np.ones(len(X_test))])
    else:
        sample_weights_updated = None
    
    model.fit(X_train_updated, y_train_updated, sample_weight=sample_weights_updated)
    
    predictions_after = model.predict(X_test)
    mae_after = mean_absolute_error(y_test, predictions_after)
    print(f"MAE after update: {mae_after}")
    
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mae, r2

def plot_feature_importance(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename):
    try:
        return joblib.load(filename)
    except FileNotFoundError:
        print(f"Model not found at {filename}. Training a new model.")
        return None

def FPLModelTrainer_RF(player_history, features, target, current_season_games, model_name):
    print("Start time:", datetime.now().strftime("%m-%d-%Y %H:%M:%S"))
    
    player_history = prepare_data(player_history, features, target)
    player_history = add_season_weights(player_history)
    
    latest_model = load_model(model_name)
    
    player_history = player_history.sort_values('game')
    
    maes = []
    r2s = []
    all_predictions = pd.DataFrame()
    
    for game in current_season_games:
        print(f"\nProcessing game week: {game}")
        start_time = time.time()
        
        train_data = player_history[player_history['game'] < game]
        test_data = player_history[player_history['game'] == game]
        
        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]
        
        sample_weights = train_data['season_weight'].values
        
        if latest_model is None:
            print("Performing initial model training with grid search...")
            model = train_model_with_grid_search(X_train, y_train, sample_weights)
        else:
            model = update_model(latest_model, X_train, y_train, X_test, y_test, sample_weights)
        
        latest_model = model
        predictions = model.predict(X_test)
        
        mae, r2 = evaluate_model(model, X_test, y_test)
        maes.append(mae)
        r2s.append(r2)
        
        test_data.loc[:, 'predicted_points'] = predictions
        all_predictions = pd.concat([all_predictions, test_data[['PlayerUniqueID', 'game', target, 'predicted_points']]])
        
        print(f"MAE: {mae}, R2: {r2}")
        
        player_history.loc[player_history['game'] == game, 'predicted_points'] = predictions
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f"Training time: {minutes} minutes and {seconds:.2f} seconds for game week: {game}")
    
    overall_mae = np.mean(maes)
    overall_r2 = np.mean(r2s)
    
    print(f"Overall MAE: {overall_mae}")
    print(f"Overall R2: {overall_r2}")
    
    if latest_model:
        plot_feature_importance(latest_model, features)
        save_model(latest_model, model_name)
    
    return all_predictions, overall_mae, overall_r2, latest_model