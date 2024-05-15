import pandas as pd
import numpy as np

def get_team_data(year) -> pd.DataFrame:
     """Cleans team data for a certain year"""
     df = pd.read_csv('DataSet/Team Summaries.csv') 
     df = df[df['lg'] == 'NBA']
     df = df[df['season'] == year]
     df = df.dropna(subset=['o_rtg'])
     df = df.dropna(subset=['abbreviation'])
     df = df[['season','abbreviation','o_rtg','d_rtg']]
     return df

def get_game_data(year) -> pd.DataFrame:
     """Cleans data on game per game basis for a certain year"""
     df = pd.read_csv('DataSet/game.csv')
     df = df[['season_id','team_abbreviation_home','team_abbreviation_away','wl_home','season_type']]
     df['season_id'] = df['season_id'].astype(str)
     df['season_id'] = df['season_id'].str.slice(1,5)
     df = df[df['season_type']!='All Star']
     df = df[df['season_type']!='All-Star']
     df = df[df['season_type']!='Pre Season']
     df['season_id'] = df['season_id'].astype(int)
     df['season_id'] = df['season_id'] + 1 #correcting for end of season date instead of start of season to match second dataset
     df = df[df['season_id'] == year]
     return df

def get_playoff_game_data(year) -> pd.DataFrame:
     """Cleans data on game per game basis for a certain year"""
     df = pd.read_csv('DataSet/game.csv')
     df = df[['season_id','team_abbreviation_home','team_abbreviation_away','wl_home','season_type']]
     df['season_id'] = df['season_id'].astype(str)
     df['season_id'] = df['season_id'].str.slice(1,5)
     df = df[df['season_type']=='Playoffs']
     df['season_id'] = df['season_id'].astype(int)
     df['season_id'] = df['season_id'] + 1 #correcting for end of season date instead of start of season to match second dataset
     df = df[df['season_id'] == year]
     return df

def get_regular_game_data(year) -> pd.DataFrame:
     """Cleans data on game per game basis for a certain year"""
     df = pd.read_csv('DataSet/game.csv')
     df = df[['season_id','team_abbreviation_home','team_abbreviation_away','wl_home','season_type']]
     df['season_id'] = df['season_id'].astype(str)
     df['season_id'] = df['season_id'].str.slice(1,5)
     df = df[df['season_type']=='Regular Season']
     df['season_id'] = df['season_id'].astype(int)
     df['season_id'] = df['season_id'] + 1 #correcting for end of season date instead of start of season to match second dataset
     df = df[df['season_id'] == year]
     return df

def o_d_rtg_analysis(year,playoff) -> tuple[int, int, int, int]: # 
     """ *Only works for recent NBA due to abbreviation differences between data sets

     PARAMATERS:
     year = nba season year(end of season year)
     playoff == 1 for playoff, 0 for regular season, and 2 for all games

     First int returned is win percentage of greater o_rtg, second is win percentage of greater_d_rtg, third avg o_rtg_diff of winner, fourth avg d_rtg_diff of winner
     """
     #Getting data
     if playoff == 1:
          game = get_playoff_game_data(year)
     elif playoff == 0:
          game = get_regular_game_data(year)
     else:
          game = get_game_data(year)
     team = get_team_data(year)
     team_o_dict = dict(zip(team['abbreviation'], (team['o_rtg'])))
     team_d_dict = dict(zip(team['abbreviation'], (team['d_rtg'])))

     #Abbreviation Changes for compatbility between data sets
     if team_o_dict['CHO'] is not None:
          team_o_dict['CHA'] = team_o_dict['CHO']
          team_d_dict['CHA'] = team_d_dict['CHO']
     if team_o_dict['BRK'] is not None:
          team_o_dict['BKN'] = team_o_dict['BRK']
          team_d_dict['BKN'] = team_d_dict['BRK']
     if team_o_dict['PHO'] is not None:
          team_o_dict['PHX'] = team_o_dict['PHO']
          team_d_dict['PHX'] = team_d_dict['PHO']

     #Win Percentage of rating difference calculations
     game['home_o_rtg'] = game.apply(lambda row: team_o_dict.get(row['team_abbreviation_home'], None), axis=1)
     game['away_o_rtg'] = game.apply(lambda row: team_o_dict.get(row['team_abbreviation_away'], None), axis=1)
     game['home_d_rtg'] = game.apply(lambda row: team_d_dict.get(row['team_abbreviation_home'], None), axis=1)
     game['away_d_rtg'] = game.apply(lambda row: team_d_dict.get(row['team_abbreviation_away'], None), axis=1)
     game['greater_o_win'] = np.where( ((game['home_o_rtg'] > game['away_o_rtg']) & (game['wl_home'] == 'W')) | ((game['home_o_rtg'] < game['away_o_rtg']) & (game['wl_home'] == 'L')) ,1,0)
     game['greater_d_win'] = np.where((game['home_d_rtg'] > game['away_d_rtg']) & (game['wl_home'] == 'W') | ((game['home_d_rtg'] < game['away_d_rtg']) & (game['wl_home'] == 'L')),1,0)
     greater_o_win_percent = game['greater_o_win'].sum() / len(game)
     greater_d_win_percent = game['greater_d_win'].sum() / len(game)

     #Difference of ratings calculations
     game['o_diff_win'] = np.where((game['wl_home'] == 'W'),game['home_o_rtg']-game['away_o_rtg'],game['away_o_rtg']-game['home_o_rtg'])
     game['d_diff_win'] = np.where((game['wl_home'] == 'W'),game['home_d_rtg']-game['away_d_rtg'],game['away_d_rtg']-game['home_d_rtg'])
     o_diff_win_avg = game['o_diff_win'].sum() / len(game)
     d_diff_win_avg = game['d_diff_win'].sum() / len(game)


     print(game)
     return (greater_o_win_percent,greater_d_win_percent,o_diff_win_avg,d_diff_win_avg)


if __name__ == "__main__":
     print(o_d_rtg_analysis(2023,0))
