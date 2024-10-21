'''
This script is an `FPLModelTrainer_XG` class for training and updating an XGBoost-based model that predicts player performance in Fantasy Premier League (FPL). The code focuses on preparing data, adding season weights, training with grid search, updating models, evaluating performance, and saving/loading the model. Here's a breakdown of the key components:

### Key Components and Their Functions:

1. **Initialization (`__init__`)**:
   - Initializes the class with a default model file name (`FPL_XGBoost_model.joblib`) for saving/loading the XGBoost model.

2. **Data Preparation (`prepare_data`)**:
   - Prepares player history data for model training. It:
     - Identifies numeric and categorical features.
     - Imputes missing numeric values using the mean.
     - Converts categorical columns (like `position` and `game`) to the `category` data type.
     - Orders categories where needed (e.g., positions ordered as `GK`, `DEF`, `MID`, `FWD`).

3. **Season Weights (`add_season_weights`)**:
   - Adds a weight for each player's history based on the season, giving more weight to recent data. The weights are calculated relative to the minimum and maximum seasons in the dataset.

4. **Model Training with Grid Search (`train_model_with_grid_search`)**:
   - Trains an XGBoost model with grid search for hyperparameter tuning. The grid includes parameters like `n_estimators`, `learning_rate`, `max_depth`, regularization (`reg_alpha`, `reg_lambda`), and subsample ratios.
   - Detects if a GPU is available and uses `gpu_hist` for faster training if present; otherwise, it defaults to the `hist` tree method.
   - A custom scoring function (based on Mean Absolute Error) handles missing values and applies sample weights if provided.

5. **Model Update (`update_model`)**:
   - Updates the model incrementally with new game data. It:
     - Concatenates the new game week's data with the existing training data.
     - Refits the model and evaluates the performance before and after the update by calculating MAE (Mean Absolute Error).

6. **Model Evaluation (`evaluate_model`)**:
   - Evaluates the model's predictions on test data, calculating both MAE and R² scores, which provide a sense of prediction accuracy and how well the model fits.

7. **Feature Importance Plotting (`plot_feature_importance`)**:
   - Plots the feature importance from the XGBoost model to visualize which features had the most impact on predictions.

8. **Model Saving and Loading (`save_model`, `load_model`)**:
   - Saves the trained model to disk using `joblib` for persistence across sessions.
   - Loads a previously saved model if it exists; otherwise, it returns `None`, indicating that a new model needs to be trained.

9. **Training and Updating (`train_and_update`)**:
   - The main function that coordinates the training process across multiple game weeks:
     1. Prepares player data.
     2. Adds season weights to give more importance to recent performance.
     3. Loads an existing model if available, or trains a new model using grid search.
     4. For each game week in `current_season_games`, the data is split into training (past games) and test (current game) sets.
     5. The model is updated or trained anew, then used to predict player performance for the current game week.
     6. Predictions are stored, and the model is evaluated for each game week (MAE and R²).
     7. After all game weeks, it plots feature importance and saves the model for future use.

### Key Features of the Code:
- **GPU Support**: Automatically detects if a GPU is available to speed up training.
- **Grid Search**: Optimizes the model's hyperparameters, ensuring a more accurate and reliable model.
- **Model Update**: Supports incremental learning, where the model is updated as new game week data becomes available, rather than retraining from scratch each time.
- **Season Weighting**: Weights historical data by season, giving higher importance to recent data when training the model.

### Summary:
This class sets up a pipeline for training and updating an XGBoost model to predict FPL player performance. It preprocesses the data, applies hyperparameter tuning using grid search, and iteratively updates the model as new data from each game week becomes available. The system also evaluates the model’s performance, saves it, and plots feature importances, making it suitable for real-time or season-long forecasting in FPL.
'''

import numpy as np
import pandas as pd
import joblib
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from xgboost import XGBRegressor
from config import (features, target)

class FPLModelTrainer_XG:
    def __init__(self, model_name='FPL_XGBoost_model.joblib'):
        self.model_name = model_name
        
    def prepare_data(self, player_history, features, target):
        # Identify numeric and categorical features
        numeric_features = player_history[features].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = player_history[features].select_dtypes(include=[object]).columns.tolist()

        # Impute numeric features
        numeric_imputer = SimpleImputer(strategy='mean')
        player_history[numeric_features] = numeric_imputer.fit_transform(player_history[numeric_features])

        # Convert object columns to 'category' dtype
        for col in categorical_features:
            player_history[col] = player_history[col].astype('category')
            
        # Handle ordered categorical variables
        if 'game' in player_history.columns:
            player_history['game'] = pd.Categorical(player_history['game'], ordered=True)
        if 'position' in player_history.columns:
            player_history['position'] = pd.Categorical(player_history['position'], 
                                                      categories=['GK', 'DEF', 'MID', 'FWD'], 
                                                      ordered=True)
        
        return player_history

    def add_season_weights(self, player_history):
        # Extract and convert season to numeric
        player_history['season'] = player_history['season'].str.split('-').str[0]
        player_history['season'] = pd.to_numeric(player_history['season'], errors='coerce')
        player_history = player_history.dropna(subset=['season'])
        
        # Calculate weights based on season
        current_season = player_history['season'].max()
        min_season = player_history['season'].min()
        player_history['season_weight'] = (player_history['season'] - min_season + 1) / (current_season - min_season + 1)
        
        return player_history

    def train_model_with_grid_search(self, X_train, y_train, sample_weights=None):
        # Detect GPU availability
        try:
            import xgboost as xgb
            gpu_available = len(xgb.device_enumeration()) > 0
            tree_method = 'gpu_hist' if gpu_available else 'hist'
        except:
            tree_method = 'hist'
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'reg_alpha': [0.5, 1, 2],
            'reg_lambda': [0.5, 1, 2],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'enable_categorical': [True],
            'tree_method': [tree_method]
        }
        
        xgb = XGBRegressor(random_state=42, n_jobs=-1)
        
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
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, 
                                 scoring=scorer, cv=3, n_jobs=-1, verbose=0)
        
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        
        print("Best parameters found: ", grid_search.best_params_)
        print("Best score: ", -grid_search.best_score_)
        
        return grid_search.best_estimator_

    def update_model(self, model, X_train, y_train, X_test, y_test, sample_weights=None):
        print("Updating model...")
        
        # Evaluate before update
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f"MAE before update: {mae}")
        
        # Update with new data
        X_train_updated = pd.concat([X_train, X_test])
        y_train_updated = pd.concat([y_train, y_test])
        
        if sample_weights is not None:
            sample_weights_updated = np.concatenate([sample_weights, np.ones(len(X_test))])
        else:
            sample_weights_updated = None
        
        model.fit(X_train_updated, y_train_updated, sample_weight=sample_weights_updated)
        
        # Evaluate after update
        predictions_after = model.predict(X_test)
        mae_after = mean_absolute_error(y_test, predictions_after)
        print(f"MAE after update: {mae_after}")
        
        return model

    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mae, r2

    def plot_feature_importance(self, model, features):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

    def save_model(self, model):
        joblib.dump(model, self.model_name)
        print(f"Model saved as {self.model_name}")

    def load_model(self):
        try:
            return joblib.load(self.model_name)
        except FileNotFoundError:
            print(f"Model not found at {self.model_name}. Training a new model.")
            return None

    def train_and_update(self, player_history, features, target, current_season_games):
        print("Start time:", datetime.now().strftime("%m-%d-%Y %H:%M:%S"))
        
        player_history = self.prepare_data(player_history, features, target)
        player_history = self.add_season_weights(player_history)
        
        latest_model = self.load_model()
        player_history = player_history.sort_values('game')
        
        maes = []
        r2s = []
        all_predictions = pd.DataFrame()
        
        for game in current_season_games:
            print(f"\nProcessing game week: {game}")
            start_time = time.time()
            
            # Split data
            train_data = player_history[player_history['game'] < game]
            test_data = player_history[player_history['game'] == game]
            
            X_train = train_data[features]
            y_train = train_data[target]
            X_test = test_data[features]
            y_test = test_data[target]
            
            sample_weights = train_data['season_weight'].values
            
            # Train or update model
            if latest_model is None:
                print("Performing initial model training with grid search...")
                model = self.train_model_with_grid_search(X_train, y_train, sample_weights)
            else:
                model = self.update_model(latest_model, X_train, y_train, X_test, y_test, sample_weights)
            
            latest_model = model
            
            # Make predictions and evaluate
            predictions = model.predict(X_test)
            mae, r2 = self.evaluate_model(model, X_test, y_test)
            maes.append(mae)
            r2s.append(r2)
            
            # Store predictions
            test_data.loc[:, 'predicted_points'] = predictions
            all_predictions = pd.concat([all_predictions, 
                                      test_data[['PlayerUniqueID', 'game', target, 'predicted_points']]])
            
            print(f"MAE: {mae}, R2: {r2}")
            
            # Update player history
            player_history.loc[player_history['game'] == game, 'predicted_points'] = predictions
            
            # Print timing information
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
            self.plot_feature_importance(latest_model, features)
            self.save_model(latest_model)
        
        return all_predictions, overall_mae, overall_r2, latest_model