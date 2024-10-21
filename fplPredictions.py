'''
This code defines a class `FPLPredictionProcessor` that processes Fantasy Premier League (FPL) data and generates predictions using a pre-trained model, specifically designed for predicting game outcomes and player performance for an upcoming game week.

### Key Methods and Their Roles:

1. **Initialization**:
   - The class is initialized with the model's name (`model_name`), which by default points to a specific XGBoost model (`latest_fpl_xgboost_model_with_updates_and_grid_search.joblib`).
   - The model will be loaded for making predictions in later stages.

2. **Loading and Preprocessing Data** (`load_and_preprocess_data`):
   - This method loads the raw player data (via `RAW_URL`), filters and renames the necessary columns, and computes new metrics such as:
     - `game`: A unique identifier for the game week.
     - `ppm`: Points per minute played.
     - `minutes`: The average number of minutes a player has played per game week so far.

3. **Loading Fixtures Data** (`load_and_process_fixtures`):
   - This method loads the fixture data for the current season, specifically focusing on the next game week (`CURRENT_GW`).

4. **Merging Player and Fixture Data** (`merge_player_and_fixture_data`):
   - Merges the player data (`next_gw`) with the fixture data based on whether the player's team is playing at home or away. 
   - Calculates additional metrics such as `transfers_balance` (difference between transfers in and out), player cost, and whether the player’s team is playing at home (`was_home`).

5. **Calculating Rolling Averages** (`calculate_rolling_averages`):
   - Computes rolling averages (for the last 5 game weeks) for team goals scored and goals conceded using a sliding window method, which gives insight into team form.

6. **Calculating Team Difficulty** (`calculate_team_difficulty`):
   - Calculates the rolling difficulty rating for each team based on the average home and away difficulty ratings, providing a measure of how tough their opponents have been.

7. **Preparing Data for Prediction** (`prepare_for_prediction`):
   - Merges the processed fixture and player data with team and player unique IDs, ensuring consistency across the datasets.
   - Performs additional merging and rolling average calculations, followed by dropping unnecessary columns to prepare the final dataset for predictions.

8. **Converting Data Types for the Model** (`convert_dtypes_for_model`):
   - Converts specific columns (e.g., `position`, `PlayerUniqueID`, `TeamUniqueID`) to categorical types for compatibility with the model.

9. **Calculating Advanced Metrics** (`calculate_advanced_metrics`):
   - Adds advanced player metrics, including:
     - `expected_goals`: Calculated based on expected goal involvements.
     - `goal_scoring_probability`: The player's likelihood of scoring a goal, calculated based on their expected goals per minute, clipped between 0 and 100.

10. **Processing Data and Making Predictions** (`process_and_predict`):
   - This is the main method that:
     1. Loads the pre-trained model.
     2. Processes the player and fixture data by calling the aforementioned methods.
     3. Prepares the data for prediction and computes advanced metrics.
     4. Uses the model to predict player performance (`predicted_points`) for the upcoming game week.
   - The method handles errors gracefully, returning `None` if an exception occurs.

### Summary:
This class automates the process of gathering and cleaning FPL data, integrating fixtures, calculating team and player form (through rolling averages), and generating predictions for player performance in upcoming games. It ensures that the data is prepped and ready for prediction with a pre-trained model, adding critical performance metrics like expected goals and scoring probability to enhance predictions.

'''

import pandas as pd
import numpy as np
import joblib
from config import (RAW_URL, FIXTURES_URL, CURRENT_SEASON, 
                   GAME_ID, REQUIRED_COLUMNS, RAW_COLUMNS, COLUMN_RENAME_MAPPING, CURRENT_GW)

class FPLPredictionProcessor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        
    def load_and_preprocess_data(self):
        # Load raw player data
        raw_2024_25 = pd.read_csv(RAW_URL)
        
        # Select and rename columns
        next_gw = raw_2024_25[RAW_COLUMNS].rename(columns=COLUMN_RENAME_MAPPING)
        
        # Add game week info and calculate new columns
        next_gw['game'] = GAME_ID
        next_gw['ppm'] = next_gw['cumulative_points'] / next_gw['cumulative_minutes'].replace(0, 1)
        next_gw['minutes'] = next_gw['cumulative_minutes'] / (CURRENT_GW - 1)
        return next_gw

    def load_and_process_fixtures(self):
        fixtures_2024_25 = pd.read_csv(FIXTURES_URL, usecols=['event', 'team_a', 'team_h', 'team_h_difficulty', 'team_a_difficulty'])
        return fixtures_2024_25[fixtures_2024_25['event'] == CURRENT_GW]

    def merge_player_and_fixture_data(self, next_gw, fixtures):
        merged_home = pd.merge(next_gw, fixtures, left_on='team', right_on='team_h', how='inner')
        merged_away = pd.merge(next_gw, fixtures, left_on='team', right_on='team_a', how='inner')
        final_merged_df = pd.concat([merged_home, merged_away])
        
        final_merged_df['transfers_balance'] = final_merged_df['transfers_in'] - final_merged_df['transfers_out']
        final_merged_df['cost'] = final_merged_df['value'] / 10
        final_merged_df['was_home'] = np.where(final_merged_df['team'] == final_merged_df['team_h'], 1, 0)
        return final_merged_df

    def calculate_rolling_averages(self, df, window=5):
        df = df.sort_values(['team', 'game'])
        
        df['rolling_avg_goals_scored'] = df.groupby('team')['goals_scored'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df['rolling_avg_goals_conceded'] = df.groupby('team')['goals_conceded'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        
        return df

    def calculate_team_difficulty(self, df):
        team_difficulty = df.groupby('team')[['team_h_difficulty', 'team_a_difficulty']].mean().mean(axis=1)
        df['rolling_team_difficulty'] = df['team'].map(team_difficulty)
        return df

    def prepare_for_prediction(self, df, teams, players_unique):
        teams_new = teams[teams['season'] == CURRENT_SEASON]
        df = df.merge(teams_new, left_on='team', right_on='id', how='left')
        df = df.merge(teams_new, left_on='team_h', right_on='id', how='left', suffixes=('', '_oppo'))
        
        columns_to_drop = ['id', 'id_oppo', 'team_name', 'team_name_oppo', 'season', 'season_oppo']
        df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
        
        df = df.merge(players_unique, on=['first_name', 'second_name'], how='left')
        df = self.calculate_rolling_averages(df)
        df = self.calculate_team_difficulty(df)
        
        return df

    def convert_dtypes_for_model(self, df):
        if df is None:
            return None
        category_columns = ['position', 'PlayerUniqueID', 'TeamUniqueID', 'TeamUniqueID_oppo', 'game']
        for col in category_columns:
            df[col] = df[col].astype('category')
        return df

    def calculate_advanced_metrics(self, df):
        df['expected_goals'] = (df['expected_goal_involvements'] * 0.6)
        df['expected_assists'] = df['expected_assists']
        df['goal_scoring_probability'] = (df['expected_goals'] / df['minutes'].replace(0, 1)) * 100
        df['goal_scoring_probability'] = df['goal_scoring_probability'].clip(0, 100)
        return df

    def process_and_predict(self, teams, players_unique):
        try:
            # Load the model
            self.model = joblib.load(self.model_name)
            
            # Process data
            next_gw = self.load_and_preprocess_data()
            fixtures = self.load_and_process_fixtures()
            final_merged_df = self.merge_player_and_fixture_data(next_gw, fixtures)
            
            upcoming_game_week_data = self.prepare_for_prediction(final_merged_df, teams, players_unique)
            upcoming_game_week_data = self.convert_dtypes_for_model(upcoming_game_week_data)
            upcoming_game_week_data = self.calculate_advanced_metrics(upcoming_game_week_data)
            
            # Make predictions
            predictions_upcoming = self.model.predict(upcoming_game_week_data[REQUIRED_COLUMNS])
            upcoming_game_week_data['predicted_points'] = predictions_upcoming
            
            return upcoming_game_week_data
            
        except Exception as e:
            print(f"Error in processing and prediction: {str(e)}")
            return None