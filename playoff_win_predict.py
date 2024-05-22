import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

import seaborn as sns

def get_playoff_game_data() -> pd.DataFrame:
     """Cleans data on game per game basis for a certain year, helper function for combine_data"""
     df = pd.read_csv('DataSet/game.csv')
     df = df[['season_id','game_id','team_abbreviation_home','team_abbreviation_away','wl_home','season_type']]
     df['season_id'] = df['season_id'].astype(str)
     df['season_id'] = df['season_id'].str.slice(1,5)
     df = df[df['season_type']=='Playoffs']
     df['season_id'] = df['season_id'].astype(int)
     df['season_id'] = df['season_id'] + 1 #correcting for end of season date instead of start of season to match second dataset
     df = df[df['season_id'] >= 2000]

     df.rename(columns={'team_abbreviation_home':'abbreviation_h','team_abbreviation_away':'abbreviation_a','season_id':'season'},inplace=True)
     return df

def get_team_data() -> pd.DataFrame:
     """Cleans team data for a certain year, helper function for combine_data"""
     df = pd.read_csv('DataSet/Team Summaries.csv') 
     df = df[df['lg'] == 'NBA']
     df = df[df['season'] >= 2000]
     df = df.dropna(subset=['season','abbreviation','o_rtg','d_rtg','n_rtg','age','w','pace','ts_percent','e_fg_percent','opp_e_fg_percent','tov_percent'])
     df = df[['season','abbreviation','o_rtg','d_rtg','n_rtg','age','pace','ts_percent','e_fg_percent','opp_e_fg_percent','tov_percent','playoffs','w']]
     df['abbreviation'].replace({'CHO':'CHA','PHO':'PHX','BRK':'BKN'},inplace=True) # Changing Abbreviation to match other dataset
     df['abbreviation_a'] = df['abbreviation']
     df.rename(columns={'abbreviation':'abbreviation_h'},inplace=True)
     
     return df

def combine_data() -> pd.DataFrame:
    """ Merges Data Between Games Datasets and Team Stats Datasets"""
    games = get_playoff_game_data()
    teams = get_team_data()
    df1 = pd.merge(games, teams[['abbreviation_h', 'season', 'o_rtg','d_rtg','n_rtg','age','pace','ts_percent','e_fg_percent','opp_e_fg_percent','tov_percent','w']],
                     on=['abbreviation_h', 'season'], 
                     how='left')
    df1.rename(columns={'o_rtg':'o_rtg_h','d_rtg':'d_rtg_h','n_rtg':'n_rtg_h','age':'age_h','pace':'pace_h','ts_percent':'ts_percent_h','e_fg_percent':'e_fg_percent_h','opp_e_fg_percent':'opp_e_fg_percent_h','tov_percent':'tov_percent_h','w':'w_h'},inplace=True)
    df2 = pd.merge(games, teams[['abbreviation_a', 'season', 'o_rtg','d_rtg','n_rtg','age','pace','ts_percent','e_fg_percent','opp_e_fg_percent','tov_percent','w']],
                     on=['abbreviation_a', 'season'], 
                     how='left')
    df2.rename(columns={'o_rtg':'o_rtg_a','d_rtg':'d_rtg_a','n_rtg':'n_rtg_a','age':'age_a','pace':'pace_a','ts_percent':'ts_percent_a','e_fg_percent':'e_fg_percent_a','opp_e_fg_percent':'opp_e_fg_percent_a','tov_percent':'tov_percent_a','w':'w_a'},inplace=True)
    df = pd.merge(df1,df2)
    df.to_csv('output2.csv')
    df['wl_home'].replace({'W':1,'L':0},inplace=True)
    return df

def raw_stat_decision_tree():
    """Uses Decision Tree Classification to predict a winner in a playoff game"""
    df = combine_data()
    x = df.iloc[:,6:]
    y = df.iloc[:,4]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)
    dtc = DecisionTreeClassifier(criterion='gini',min_samples_split = 10,max_depth=5)
    dtc.fit(x_train,y_train)

    plt.figure(figsize=(12,8))
    tree.plot_tree(dtc.fit(x_train,y_train), feature_names = x_train.columns)
    plt.show()

    y_pred = dtc.predict(x_test)
    print(f" Accuracy score: {accuracy_score(y_test,y_pred)}")
    
    attribute_importance_dict= {}
    for col, val in sorted(zip(x_train.columns, dtc.feature_importances_),key=lambda x:x[1],reverse=True):
          attribute_importance_dict[col]=val
    
    keys = list(attribute_importance_dict.keys())
    values = list(attribute_importance_dict.values())

    plt.bar(keys,values)
    plt.show()

def stat_diff_decision_tree():
    """Uses Decision Tree Classification to build a model that predicts a winner in a playoff game based on stat difference between two teams"""
    df = combine_data()

    df['o_rtg_diff'] = df['o_rtg_h']-df['o_rtg_a']
    df['d_rtg_diff'] = df['d_rtg_h']-df['d_rtg_a']
    df['n_rtg_diff'] = df['n_rtg_h']-df['n_rtg_a']
    df['age_diff'] = df['age_h']-df['age_a']
    df['pace_diff'] = df['pace_h']-df['pace_a']
    df['ts_percent_diff'] = df['ts_percent_h']-df['ts_percent_a']
    df['e_fg_percent_diff'] = df['e_fg_percent_h']-df['e_fg_percent_a']
    df['opp_e_fg_percent_diff'] = df['opp_e_fg_percent_h']-df['opp_e_fg_percent_a']
    df['w_diff'] = df['w_h']-df['w_a']
    df['tov_percent_diff'] = df['tov_percent_h']-df['tov_percent_a']
    
    df = df[['season','abbreviation_h','abbreviation_a','wl_home','o_rtg_diff','d_rtg_diff','n_rtg_diff','age_diff','ts_percent_diff','e_fg_percent_diff','opp_e_fg_percent_diff','tov_percent_diff','w_diff']]
    x = df.iloc[:,4:] 
    y = df.iloc[:,3]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)
    dtc = DecisionTreeClassifier(criterion='gini',min_samples_split = 20,max_depth=6)
    dtc.fit(x_train,y_train)

    plt.figure(figsize=(12,8))
    tree.plot_tree(dtc.fit(x_train,y_train), feature_names = x_train.columns)
    plt.show()

    y_pred = dtc.predict(x_test)
    print(f" Accuracy score: {accuracy_score(y_test,y_pred)}")
    
    attribute_importance_dict= {}
    for col, val in sorted(zip(x_train.columns, dtc.feature_importances_),key=lambda x:x[1],reverse=True):
          attribute_importance_dict[col]=val
    
    keys = list(attribute_importance_dict.keys())
    values = list(attribute_importance_dict.values())

    plt.bar(keys,values)
    plt.show()

stat_diff_decision_tree()