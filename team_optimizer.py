'''
This script is a `TeamOptimizer` class designed to optimize the selection of a Fantasy Premier League (FPL) team for an upcoming game week. The optimizer uses linear programming to maximize predicted points while adhering to a set of constraints, such as budget limits and required player positions. Here’s a breakdown of the key functions:

### Key Components and Their Functions:

1. **Initialization (`__init__`)**:
   - Initializes the class but does not include any custom parameters at this stage.

2. **Team Selection Optimization (`optimize_team_selection`)**:
   - This function is responsible for selecting the optimal team of players based on predicted points and various constraints, such as budget and player positions.

   **Process Overview**:
   - **Inputs**:
     - `upcoming_game_week_data`: A dataframe containing player data, including costs, predicted points, and positions.
     - `budget`: The maximum budget for selecting players (default is 100).
   
   **Steps**:
   - Extracts relevant information from the data:
     - `player_costs`: Array of player costs.
     - `player_points`: Array of predicted points for each player.
     - `player_positions`: Array of player positions (GK, DEF, MID, FWD).
   - Sets up the objective function (`c`), which aims to **maximize predicted points** by minimizing the negative of the points.
   - Adds constraints:
     - **Budget constraint**: Ensures that the total cost of selected players does not exceed the budget.
     - **Position constraints**: Ensures the team adheres to specific position counts (e.g., 2 goalkeepers, 5 defenders, etc.). These are defined in `POSITION_COUNTS`.
   - Uses linear programming (`linprog`) to solve the problem:
     - `A_ub` and `b_ub` matrices define the upper bounds for the constraints (budget and position limits).
     - `bounds`: Each player can either be selected (1) or not selected (0).
   - Solves the linear programming problem using the "highs" method, which is efficient for this type of optimization.
   - Returns an array (`selected_players`) that indicates which players have been selected (1 for selected, 0 for not selected).

3. **Prepare Selected Team (`prepare_selected_team`)**:
   - This function takes the selected players and formats the data to display the final optimized team in a readable format.
   
   **Process Overview**:
   - **Inputs**:
     - `upcoming_game_week_data`: The same player data used in the optimization step.
     - `selected_players`: The binary array returned by `optimize_team_selection` indicating which players are selected.
     - `teams`: A dataframe containing team data, including team IDs and names.
   - Filters the `upcoming_game_week_data` to include only the selected players.
   - Merges this filtered data with the `teams` dataframe to include additional information, such as the team name and the opposing team's name for each selected player.
   - Maps position codes to text descriptions using `POSITION_MAPPING` (e.g., 'GK' for goalkeeper, 'DEF' for defender).
   - Converts home/away status to a readable format ('H' for home, 'A' for away).
   - Selects relevant columns to display using `DISPLAY_COLUMNS` and renames them according to `COLUMN_DISPLAY_MAPPING` to ensure consistency in the output format.
   - Returns the final formatted team dataframe sorted by predicted points in descending order.

4. **Optimize (`optimize`)**:
   - The main function that ties together the optimization and preparation steps.
   
   **Process Overview**:
   - **Inputs**:
     - `upcoming_game_week_data`: The data containing player predictions, costs, and positions.
     - `teams`: The team data used for merging and formatting.
     - `budget`: The total budget for selecting players (default 100).
   - Calls `optimize_team_selection` to perform the linear programming optimization and get the selected players.
   - Passes the selected players to `prepare_selected_team` to prepare the final team for display.
   - Returns the optimized team.

### Summary:
This class performs a **team optimization** process for FPL based on predicted points, budget, and position constraints. It uses **linear programming** to maximize the team's predicted points within the specified budget and ensures that the selected team meets the required position counts. The final team is formatted and returned in a readable format, including player details and their respective home/away status. This method allows for efficient, automated team selection for upcoming game weeks.
'''


import numpy as np
from scipy.optimize import linprog
import pandas as pd
from config import (POSITION_MAPPING, POSITION_COUNTS, DISPLAY_COLUMNS, 
                   COLUMN_DISPLAY_MAPPING, CURRENT_SEASON)

class TeamOptimizer:
    def __init__(self):
        pass
        
    def optimize_team_selection(self, upcoming_game_week_data, budget=100):
        """Optimize team selection based on predicted points and constraints."""
        # Prepare data
        player_costs = upcoming_game_week_data['cost'].values
        player_points = upcoming_game_week_data['predicted_points'].values
        player_positions = upcoming_game_week_data['position'].values
        num_players = len(player_costs)

        # Objective function: maximize points (minimize negative points)
        c = -player_points

        # Constraints
        A = [player_costs]  # Budget constraint
        b = [budget]

        # Position constraints
        for pos, (min_count, max_count) in POSITION_COUNTS.items():
            pos_indicator = (player_positions == [k for k, v in POSITION_MAPPING.items() 
                                               if v == pos][0]).astype(int)
            A.extend([pos_indicator, -pos_indicator])
            b.extend([max_count, -min_count])

        A = np.array(A)
        b = np.array(b)

        # Bounds for each player (0 or 1 - either selected or not)
        bounds = [(0, 1) for _ in range(num_players)]

        # Solve the linear programming problem
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        # Get the selected players
        selected_players = result.x.round().astype(int)
        return selected_players

    def prepare_selected_team(self, upcoming_game_week_data, selected_players, teams):
        """Prepare and format the selected team data for display."""
        # Filter for selected players
        selected_team = upcoming_game_week_data[selected_players == 1].copy()

        # Merge with team data
        current_season_teams = teams[teams['season'] == CURRENT_SEASON]
        selected_team = selected_team.merge(
            current_season_teams[['TeamUniqueID', 'team_name']], 
            on='TeamUniqueID', 
            how='left'
        )
        selected_team = selected_team.merge(
            current_season_teams[['TeamUniqueID', 'team_name']], 
            left_on='TeamUniqueID_oppo', 
            right_on='TeamUniqueID', 
            how='left', 
            suffixes=('', '_oppo')
        )

        # Map position codes to text
        selected_team['position_txt'] = selected_team['position'].map(POSITION_MAPPING)

        # Format home/away indicator
        selected_team['was_home'] = selected_team['was_home'].apply(
            lambda x: 'H' if x == 1 else 'A'
        )

        # Select and rename columns for display
        selected_team = selected_team[DISPLAY_COLUMNS].sort_values(
            'predicted_points', 
            ascending=False
        )
        selected_team.rename(columns=COLUMN_DISPLAY_MAPPING, inplace=True)

        return selected_team

    def optimize(self, upcoming_game_week_data, teams, budget=100):
        """Main optimization function that combines selection and preparation."""
        selected_players = self.optimize_team_selection(upcoming_game_week_data, budget)
        optimized_team = self.prepare_selected_team(upcoming_game_week_data, selected_players, teams)
        return optimized_team