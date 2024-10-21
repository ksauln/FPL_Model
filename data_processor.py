'''
This code defines a class `FPLDataProcessor` to clean, transform, and process Fantasy Premier League (FPL) data. The class integrates various datasets, creates unique IDs, computes rolling averages, and ensures the data is formatted for further analysis.

1. **Initialization**:
   - The class is initialized with a list of required columns needed in the processed data.

2. **Creating Unique IDs** (`create_unique_ids`):
   - This method generates unique identifiers (`PlayerUniqueID` and `TeamUniqueID`) for players and teams to ensure consistency when merging datasets.

3. **Merging IDs and Data** (`merge_unique_ids`):
   - Combines player and team unique IDs with player history, players, and fixtures data.
   - Merges team data for both home and away matches into the fixtures and player history datasets.

4. **Cleaning Data**:
   - **Players** (`clean_players`): Handles missing data, removes unnecessary columns, and converts certain fields (e.g., `cost` and `form`) to appropriate data types.
   - **Player History** (`clean_player_history`): Removes rows with missing player IDs, fills missing values in performance metrics, and calculates cumulative statistics like total points, minutes played, and points per minute (`ppm`). It also generates rolling averages for the past 5 game weeks.

5. **Calculating Rolling Averages**:
   - **Overall Averages** (`calculate_overall_rolling_averages`): Calculates rolling averages for goals scored and conceded by teams over a specified window (default 5 weeks).
   - **Opponent Difficulty** (`calculate_opponent_rolling_difficulty`): Computes rolling averages for team difficulty, based on fixture difficulty data.

6. **Merging Rolling Averages** (`merge_rolling_avgs`):
   - Integrates rolling averages (goals scored, conceded, and team difficulty) into the player history dataset.

7. **Formatting Columns** (`format_columns`):
   - Ensures that certain columns are treated as strings (e.g., IDs) and rounds numeric columns to five decimal places for precision.

8. **Processing Data** (`process_data`):
   - This method orchestrates the entire processing flow, from generating unique IDs to cleaning data, calculating rolling averages, and ensuring all required columns are present. The final output includes cleaned and processed datasets for players, player history, teams, and fixtures.
'''


import numpy as np
import pandas as pd

class FPLDataProcessor:
    def __init__(self):
        self.required_columns = [
            'position', 'xP', 'assists', 'clean_sheets', 'creativity', 'expected_assists',
            'expected_goal_involvements', 'expected_goals', 'expected_goals_conceded', 'goals_conceded', 
            'goals_scored', 'influence', 'minutes', 'own_goals', 'penalties_missed', 'penalties_saved', 
            'red_cards', 'saves', 'starts', 'threat', 'transfers_balance', 'cost', 'was_home',
            'yellow_cards', 'PlayerUniqueID', 'TeamUniqueID', 'TeamUniqueID_oppo', 'cumulative_points',
            'cumulative_minutes', 'ppm', 'rolling_avg_points', 'rolling_avg_goals_scored', 
            'rolling_avg_goals_conceded', 'rolling_team_difficulty', 'game'
        ]

    def create_unique_ids(self, player_ids, teams):
        player_ids['name'] = player_ids['first_name'] + ' ' + player_ids['second_name']
        players_unique = player_ids.drop(['id', 'season'], axis=1).drop_duplicates()
        players_unique['PlayerUniqueID'] = players_unique.index

        teams_unique = teams.drop(['id', 'season'], axis=1).drop_duplicates().reset_index(drop=True)
        teams_unique['TeamUniqueID'] = teams_unique.index + 1

        return players_unique, teams_unique

    def merge_unique_ids(self, players, player_history, teams, players_unique, teams_unique, fixtures):
        # Merge player unique IDs
        players = players.merge(players_unique, on=['first_name', 'second_name'], how='left')
        player_history = player_history.merge(players_unique, on='name', how='left')

        # Merge team unique IDs
        teams = teams.merge(teams_unique, on='name', how='left')
        teams.rename(columns={'name': 'team_name'}, inplace=True)
        teams_opponent = teams.rename(columns={
            'team_name': 'team_name_oppo', 
            'TeamUniqueID': 'TeamUniqueID_oppo'
        })

        # Merge team IDs to players and player_history
        players = players.merge(teams, left_on=['team', 'season'], 
                              right_on=['id', 'season'], how='left')
        player_history = player_history.merge(teams, left_on=['team', 'season'], 
                                            right_on=['team_name', 'season'], how='left')
        player_history = player_history.merge(teams_opponent, left_on=['opponent_team', 'season'], 
                                            right_on=['id', 'season'], how='left')
        player_history.drop(['id_x', 'id_y'], axis=1, inplace=True)
        
        # Add unique team ids to fixtures
        fixtures = fixtures.merge(teams, left_on=['team_h','season'], 
                                right_on=['id','season'], how='left')
        fixtures = fixtures.merge(teams_opponent, left_on=['team_a','season'], 
                                right_on=['id','season'], how='left')
        fixtures.drop(['id_x','id_y'], axis=1, inplace=True)
        fixtures['game'] = fixtures['season'].str.replace('-','') + \
                          fixtures['event'].astype(str).str.zfill(2)

        return players, player_history, teams, fixtures

    def clean_players(self, df):
        replace_cols = [
            'chance_of_playing_next_round', 'chance_of_playing_this_round',
            'corners_and_indirect_freekicks_order', 'direct_freekicks_order',
            'penalties_order', 'clean_sheets_per_90', 'expected_assists',
            'expected_assists_per_90', 'expected_goal_involvements',
            'expected_goal_involvements_per_90', 'expected_goals',
            'expected_goals_conceded', 'expected_goals_conceded_per_90',
            'expected_goals_per_90', 'form_rank', 'form_rank_type',
            'goals_conceded_per_90', 'now_cost_rank', 'now_cost_rank_type',
            'points_per_game_rank', 'points_per_game_rank_type', 'saves_per_90',
            'selected_rank', 'selected_rank_type', 'starts', 'starts_per_90'
        ]

        drop_cols = [
            'corners_and_indirect_freekicks_text', 'direct_freekicks_text',
            'news', 'news_added', 'penalties_text', 'photo', 'special',
            'squad_number'
        ]

        df['form'] = df['form'].astype(float)
        df['total_points'] = df['total_points'].astype(float)
        df['minutes'] = df['minutes'].astype(float)
        df['cost'] = df['now_cost'] / 10
        df['name'] = df['first_name'] + ' ' + df['second_name']

        df[replace_cols] = df[replace_cols].fillna(0).replace('None', 0)
        df[replace_cols] = df[replace_cols].apply(pd.to_numeric, errors='coerce')
        df.drop(columns=drop_cols, inplace=True)

        return df

    def clean_player_history(self, df):
        df.dropna(subset=["PlayerUniqueID"], inplace=True)
        
        replace_cols = ['expected_assists', 'expected_goal_involvements', 
                       'expected_goals', 'expected_goals_conceded', 'starts']
        df[replace_cols] = df[replace_cols].fillna(0)

        for col in ['influence', 'creativity', 'threat', 'ict_index']:
            df[col] = df[col].astype(float)

        df['cost'] = df['value'] / 10
        df['game'] = df['season'].str.replace('-', '') + df['GW'].astype(str).str.zfill(2)
        df['was_home'] = df['was_home'].astype(int)

        # Calculate cumulative statistics
        df = df.sort_values(by=['PlayerUniqueID', 'season', 'GW'])
        df['cumulative_points'] = df.groupby(['PlayerUniqueID', 'season'])['total_points'].cumsum()
        df['cumulative_minutes'] = df.groupby(['PlayerUniqueID', 'season'])['minutes'].cumsum()
        df['ppm'] = (df['cumulative_points'] / df['cumulative_minutes']).fillna(0) \
                     .replace([np.inf, -np.inf], 0).round(5)
        df['points_per_cost'] = (df['cumulative_points'] / df['cost']).round(5)
        df['rolling_avg_points'] = df.groupby('PlayerUniqueID')['total_points'] \
                                   .rolling(window=5, min_periods=1).mean() \
                                   .reset_index(level=0, drop=True).round(5)

        df['position'] = df['position'].map({'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4})
        df.dropna(subset=['position'], inplace=True)

        return df

    def calculate_overall_rolling_averages(self, df, window=5):
        team_data = pd.concat([
            df[['game', 'TeamUniqueID', 'team_h_score', 'team_a_score']] \
              .rename(columns={'TeamUniqueID': 'team_id', 
                             'team_h_score': 'goals_scored', 
                             'team_a_score': 'goals_conceded'}),
            df[['game', 'TeamUniqueID_oppo', 'team_a_score', 'team_h_score']] \
              .rename(columns={'TeamUniqueID_oppo': 'team_id', 
                             'team_a_score': 'goals_scored', 
                             'team_h_score': 'goals_conceded'})
        ])
        
        team_data = team_data.sort_values(['team_id', 'game'])
        
        team_data['rolling_avg_goals_scored'] = team_data.groupby('team_id')['goals_scored'] \
            .rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
        team_data['rolling_avg_goals_conceded'] = team_data.groupby('team_id')['goals_conceded'] \
            .rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
        
        return team_data

    def calculate_opponent_rolling_difficulty(self, df, window=5):
        team_data_diff = pd.concat([
            df[['game', 'TeamUniqueID', 'team_h_difficulty']] \
              .rename(columns={'TeamUniqueID': 'team_id', 
                             'team_h_difficulty': 'team_difficulty'}),
            df[['game', 'TeamUniqueID_oppo', 'team_a_difficulty']] \
              .rename(columns={'TeamUniqueID_oppo': 'team_id', 
                             'team_a_difficulty': 'team_difficulty'})
        ])
        
        team_data_diff = team_data_diff.sort_values(['team_id', 'game'])
        team_data_diff['rolling_team_difficulty'] = team_data_diff.groupby('team_id')['team_difficulty'] \
            .rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
        
        return team_data_diff

    def merge_rolling_avgs(self, df, overall_averages, fixture_difficulty):
        df = pd.merge(
            df, 
            overall_averages[['game', 'team_id', 'rolling_avg_goals_scored', 'rolling_avg_goals_conceded']],
            left_on=['game', 'TeamUniqueID'],
            right_on=['game', 'team_id'],
            how='left'
        )
        df.drop(['team_id'], axis=1, inplace=True)

        df = pd.merge(
            df,
            fixture_difficulty[['game', 'team_id', 'team_difficulty', 'rolling_team_difficulty']],
            left_on=['game', 'TeamUniqueID'],
            right_on=['game', 'team_id'],
            how='left'
        )
        df.drop(['team_id'], axis=1, inplace=True)
        
        return df

    def format_columns(self, df):
        strings = ['position', 'PlayerUniqueID', 'TeamUniqueID', 
                  'TeamUniqueID_oppo', 'game']
        nums = ['xP', 'assists', 'clean_sheets', 'creativity', 'expected_assists',
               'expected_goal_involvements', 'expected_goals', 'expected_goals_conceded',
               'goals_conceded', 'goals_scored', 'influence', 'minutes', 'own_goals',
               'penalties_missed', 'penalties_saved', 'red_cards', 'saves', 'selected',
               'starts', 'threat', 'transfers_balance', 'cost', 'was_home',
               'yellow_cards', 'cumulative_points', 'cumulative_minutes', 'ppm',
               'rolling_avg_points', 'rolling_avg_goals_scored',
               'rolling_avg_goals_conceded', 'rolling_team_difficulty']

        df[strings] = df[strings].astype(str)
        df[nums] = df[nums].round(5)
        
        return df

    def process_data(self, player_ids, teams, players, player_history, fixtures):
        players_unique, teams_unique = self.create_unique_ids(player_ids, teams)
        players, player_history, teams, fixtures = self.merge_unique_ids(
            players, player_history, teams, players_unique, teams_unique, fixtures
        )
        
        players = self.clean_players(players)
        player_history = self.clean_player_history(player_history)
        
        overall_averages = self.calculate_overall_rolling_averages(fixtures)
        fixture_difficulty = self.calculate_opponent_rolling_difficulty(fixtures)
        
        player_history = self.merge_rolling_avgs(player_history, overall_averages, fixture_difficulty)
        player_history = self.format_columns(player_history)
        
        player_history['TeamUniqueID'] = player_history['TeamUniqueID'].astype(str)
        players['TeamUniqueID'] = players['TeamUniqueID'].astype(str)
        
        # Ensure all required columns exist
        for col in self.required_columns:
            if col not in player_history.columns:
                player_history[col] = np.nan
                
        return (players, player_history, teams, fixtures, players_unique, 
                self.required_columns, overall_averages, fixture_difficulty)