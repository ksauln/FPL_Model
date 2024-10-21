'''
Contains certain configurations for the entire program including:

1. **Model Training Setup**:
   - **Features**: Includes 36 variables such as player position, xP (expected points), assists, clean sheets, expected goals, creativity, minutes played, and rolling averages (e.g., rolling goals scored and conceded).
   - **Target**: The `total_points` a player will score in the next game week.

2. **Predicting Points for the Next Game Week**:
   - **Data Sources**: Pulls raw player data and fixture data for the 2024-25 season from external URLs.
   - **Model Columns**: Uses necessary features such as xP, goals scored, assists, and team data to train the model and predict the next game week's points.
   - **Raw Data Columns**: Selected from the raw dataset, including attributes like bonus points, minutes played, creativity, and cost.

3. **Team Selection Optimization**:
   - **Column Renaming**: Maps raw data columns to more meaningful names for processing (e.g., `element_type` to `position`, `now_cost` to `value`).
   - **Team Setup**: Defines the allowable number of players for each position (GK, DEF, MID, FWD).
   - **Display Configuration**: Specifies which columns to show in the final output, including player name, team, predicted points, expected goals (Xg), expected assists (Xa), and goal-scoring probability.

'''


'''
************************************************************************************************************************
                                            Model training information
************************************************************************************************************************
'''
#features for the models
features = [
    'position', 'xP', 'assists', 'clean_sheets', 'creativity', 'expected_assists',
    'expected_goal_involvements', 'expected_goals', 'expected_goals_conceded', 'goals_conceded', 
    'goals_scored', 'influence', 'minutes', 'own_goals', 'penalties_missed', 'penalties_saved', 
    'red_cards', 'saves', 'starts', 'threat', 'transfers_balance', 'cost', 'was_home',
    'yellow_cards', 'PlayerUniqueID', 'TeamUniqueID', 'TeamUniqueID_oppo', 'cumulative_points',
    'cumulative_minutes', 'ppm', 'rolling_avg_points', 'rolling_avg_goals_scored', 
    'rolling_avg_goals_conceded', 'rolling_team_difficulty', 'game'
]

#target for the models
target = 'total_points'


'''
************************************************************************************************************************
                                            For Predicting Points for the next game week
************************************************************************************************************************
'''

# FPL Data URLs and Configuration
RAW_URL = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/players_raw.csv'
FIXTURES_URL = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/fixtures.csv'
CURRENT_SEASON = '2024-25'
CURRENT_GW = 8
GAME_ID = f'2024{CURRENT_GW:02d}'

# Model Required Columns
REQUIRED_COLUMNS = [
    'position', 'xP', 'assists', 'clean_sheets', 'creativity', 'expected_assists',
    'expected_goal_involvements', 'expected_goals', 'expected_goals_conceded', 'goals_conceded', 
    'goals_scored', 'influence', 'minutes', 'own_goals', 'penalties_missed', 'penalties_saved', 
    'red_cards', 'saves', 'starts', 'threat', 'transfers_balance', 'cost', 'was_home',
    'yellow_cards', 'PlayerUniqueID', 'TeamUniqueID', 'TeamUniqueID_oppo', 'cumulative_points',
    'cumulative_minutes', 'ppm', 'rolling_avg_points', 'rolling_avg_goals_scored', 
    'rolling_avg_goals_conceded', 'rolling_team_difficulty', 'game'
]

# Column selections for raw data
RAW_COLUMNS = [
    'element_type', 'ep_next', 'assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
    'expected_assists', 'expected_goal_involvements', 'expected_goals', 'expected_goals_conceded',
    'goals_conceded', 'goals_scored', 'influence', 'minutes', 'own_goals', 'penalties_missed',
    'penalties_saved', 'red_cards', 'saves', 'selected_by_percent', 'starts', 'threat',
    'transfers_in', 'transfers_out', 'now_cost', 'yellow_cards', 'total_points',
    'points_per_game', 'team', 'team_code', 'first_name', 'second_name'
]

'''
************************************************************************************************************************
                                                    For Team Selection
************************************************************************************************************************
'''
# Column renaming mappings
COLUMN_RENAME_MAPPING = {
    'element_type': 'position', 
    'ep_next': 'xP', 
    'now_cost': 'value',
    'minutes': 'cumulative_minutes', 
    'total_points': 'cumulative_points',
    'points_per_game': 'rolling_avg_points'
}

# Team Optimization Constants
POSITION_MAPPING = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
POSITION_COUNTS = {
    'GK': (1, 2),
    'DEF': (3, 5),
    'MID': (3, 5),
    'FWD': (1, 3)
}

# Display columns configuration
DISPLAY_COLUMNS = [
    'team_name', 'name', 'position_txt', 'cost', 'predicted_points', 
    'team_name_oppo', 'was_home', 'expected_goals', 'expected_assists', 
    'goal_scoring_probability'
]

# Column renaming for output
COLUMN_DISPLAY_MAPPING = {
    'team_name': 'Team',
    'name': 'Player',
    'position_txt': 'Position',
    'cost': 'Player Cost',
    'predicted_points': 'Predicted Gameweek Points',
    'team_name_oppo': 'Opposition Team Name',
    'was_home': 'Home or Away',
    'expected_goals': 'Xg',
    'expected_assists': 'Xa',
    'goal_scoring_probability': 'Goal Scoring Probability (%)'
}