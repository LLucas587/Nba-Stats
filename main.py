import pandas as pd
import numpy as np

def get_team_data(year) -> pd.DataFrame:
     df = pd.read_csv('DataSet/Team Summaries.csv') 
     df = df[df['lg'] == 'NBA']
     df = df[df['season'] == year]
     df = df.dropna(subset=['o_rtg'])
     df = df.dropna(subset=['abbreviation'])
     df = df[['season','abbreviation','o_rtg','d_rtg']]
     return df

def get_game_data(year) -> pd.DataFrame:
     """Cleans data on game per game basis for a certian year"""
     df = pd.read_csv('DataSet/game.csv')
     df = df[['team_abbreviation_home','team_abbreviation_away','wl_home','season_type','game_date']]
     df['game_date'] = df['game_date'].str.slice(0,4)
     df = df[df['season_type']!='All Star']
     df = df[df['season_type']!='All-Star']
     df['game_date'] = df['game_date'].astype(int)
     df = df[df['game_date'] == year]
     return df

if __name__ == "__main__":
     game = get_game_data(2023)
     team = get_team_data(2023)
     team_o_dict = dict(zip(team['abbreviation'], (team['o_rtg'])))
     team_d_dict = dict(zip(team['abbreviation'], (team['d_rtg'])))
     game['home_o_rtg'] = game.apply(lambda row: team_o_dict.get(row['team_abbreviation_home'], None), axis=1)
     game['away_o_rtg'] = game.apply(lambda row: team_o_dict.get(row['team_abbreviation_away'], None), axis=1)
     game['home_d_rtg'] = game.apply(lambda row: team_d_dict.get(row['team_abbreviation_home'], None), axis=1)
     game['away_d_rtg'] = game.apply(lambda row: team_d_dict.get(row['team_abbreviation_away'], None), axis=1)
     print(game)



     
     