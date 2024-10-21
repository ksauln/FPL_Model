# Load and Combine Seasons

'''
This code defines a class `FPLDataLoader` that loads and combines Fantasy Premier League (FPL) data for multiple seasons.

1. **Initialization**:
   - The `FPLDataLoader` class is initialized with an optional `seasons` list. If no list is provided, it uses a default list (`FPL_SEASONS`).
   - The base URL points to a GitHub repository containing FPL data files.

2. **Loading Data for a Single Season** (`load_season_data`):
   - The method loads various data files for a specified season, including:
     - `players_raw.csv` (player data)
     - `player_idlist.csv` (player ID mappings)
     - `fixtures.csv` (fixture information)
     - `teams.csv` (team data)
     - `gws/merged_gw.csv` (player gameweek history)
   - It stores the season information in the data for later reference.

3. **Loading All Seasons' Data** (`load_all_seasons_data`):
   - This method iterates through all the specified seasons, calling `load_season_data` to gather data for each season.
   - The data for each key (players, fixtures, etc.) is combined across seasons into a single dataset using `pd.concat`.

4. **Returning Combined Data** (`get_combined_data`):
   - Returns all the combined datasets as a dictionary containing:
     - `players`: player data across all seasons
     - `fixtures`: match fixture information
     - `player_ids`: player ID mappings
     - `player_history`: detailed gameweek performance data
     - `teams`: team data for each season

'''

import numpy as np
import pandas as pd

class FPLDataLoader:
    def __init__(self, seasons=None):
        """
        Initialize the FPL data loader
        Args:
            seasons (list, optional): List of seasons to load. If None, uses all seasons from config
        """
        self.base_url = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/'
        self.seasons = seasons if seasons is not None else FPL_SEASONS

    def load_season_data(self, season):
        """Load data for a single season."""
        data = {}
        files = {
            'players': 'players_raw.csv',
            'player_ids': 'player_idlist.csv',
            'fixtures': 'fixtures.csv',
            'teams': 'teams.csv',
            'player_history': 'gws/merged_gw.csv'
        }
        
        for key, file in files.items():
            if key == 'teams':
                data[key] = pd.read_csv(f'{self.base_url}{season}/{file}',
                                      usecols=['id', 'name'])
            else:
                data[key] = pd.read_csv(f'{self.base_url}{season}/{file}')
            data[key]['season'] = season
        return data

    def load_all_seasons_data(self):
        """Load and combine data for all configured seasons."""
        all_data = {key: [] for key in ['players', 'player_ids', 'fixtures',
                                       'teams', 'player_history']}
        for season in self.seasons:
            season_data = self.load_season_data(season)
            for key in all_data:
                all_data[key].append(season_data[key])
        return {key: pd.concat(value, ignore_index=True)
                for key, value in all_data.items()}

    def get_combined_data(self):
        """Get all combined datasets."""
        combined_data = self.load_all_seasons_data()
        return {
            'players': combined_data['players'],
            'fixtures': combined_data['fixtures'],
            'player_ids': combined_data['player_ids'],
            'player_history': combined_data['player_history'],
            'teams': combined_data['teams']
        }