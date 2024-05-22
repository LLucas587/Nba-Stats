import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import tree, linear_model
from sklearn.metrics import mean_squared_error, r2_score


import seaborn as sns

def get_data()-> pd.DataFrame:
    """ Obtains team data from dataset"""
    df = pd.read_csv('DataSet/Team Summaries.csv') 
    df = df[df['lg'] == 'NBA']
    # df = df[df['season'] >= 2003]
    df = df[df['season']!=2023] # season data not complete
    df = df.dropna(subset=['abbreviation']) # dropping league average rows
    df = df.dropna(subset=['season','abbreviation','o_rtg','d_rtg','n_rtg','age','w','pace','ts_percent','e_fg_percent','opp_e_fg_percent','tov_percent','playoffs']) # for compatability with older years where data is not recorded
    df = df[['season','abbreviation','o_rtg','d_rtg','n_rtg','age','pace','ts_percent','e_fg_percent','opp_e_fg_percent','tov_percent','playoffs','w']]
    df = df * 1 # converting playoffs column to 1's and 0's
    return df

def playoff_classifier():
    """ Uses Decision Tree Classifiers to create a model regarding playoff probability in terms of nba advanced stats"""
    df = get_data()
    df = df.drop('w',axis=1)
    df = df[df['season']!=2023] #playoff data not recorded yet
    df = df.drop('n_rtg',axis=1)
    x = df.iloc[:,2:10]
    y = df.iloc[:,10]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state=100)

    dtc = DecisionTreeClassifier(criterion='gini',min_samples_split = 3, max_depth = 3)
    dtc.fit(x_train,y_train)

    y_pred = dtc.predict(x_test)
    print(f" Accuracy score of playoff probability in terms of same season's final stats: {accuracy_score(y_test,y_pred)}")
    
    attribute_importance_dict= {}
    for col, val in sorted(zip(x_train.columns, dtc.feature_importances_),key=lambda x:x[1],reverse=True):
          attribute_importance_dict[col]=val
    
    keys = list(attribute_importance_dict.keys())
    values = list(attribute_importance_dict.values())

    plt.bar(keys,values)
    plt.show()

def win_regression():
    """ Uses Decision Tree Regression to create a model predicting the amount of wins with current season data"""
    df = get_data()
    df = df.drop('playoffs',axis=1)
    df = df.drop('tov_percent',axis=1)
    x = df.iloc[:,2:10]
    y = df.iloc[:,10]

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3 , random_state=100)
    clf = DecisionTreeRegressor(min_samples_split = 5, max_depth = 5)
    clf.fit(x_train,y_train)
    
    y_pred = clf.predict(x_test) 
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MSE of win_prediction with current data: ", mse)
    print("r2 of win_prediction with current data: ", r2)

    plt.scatter(y_test, y_pred, color='blue')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()  

def past_win_regression():
    """ Uses a simple linear regression to create a model predicting the amount of wins with past year wins
    *Not very good predictor
    """
    df = get_data()
    df['previous_season'] = df['season']-1
    df = df.merge(df[['season', 'abbreviation', 'w']], #adding new column w_last_year
              left_on=['previous_season', 'abbreviation'], 
              right_on=['season', 'abbreviation'], 
              suffixes=('', '_last_year'))
    df.drop(['previous_season', 'season_last_year'], axis=1, inplace=True)
    x = df['w_last_year'].values.reshape(-1,1) #put into 2d array for sklearn input
    y = df.iloc[:,-2]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=100)
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)

    y_pred = regr.predict(x_test) 
    print("Coefficients: \n", regr.coef_)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MSE: ", mse)
    print("r2: ", r2)

    plt.scatter(x_test,y_test,color="red")
    plt.plot(x_test,y_pred,color="blue",linewidth=3)
    plt.show()

def in_depth_past_win_regression():
    """ Uses Decision Tree Regression to create a model predicting the amount of wins with last season's data
    *Also not a very good predictor
    """
    df = get_data()
    df['previous_season'] = df['season']+1
    df = df.merge(df[['season', 'abbreviation', 'w']], #adding new column w_next_year
              left_on=['previous_season', 'abbreviation'], 
              right_on=['season', 'abbreviation'], 
              suffixes=('', '_next_year'))
    df.drop(['previous_season', 'season_next_year','o_rtg','d_rtg'], axis=1, inplace=True)
    x = df.iloc[:,2:9]
    y = df.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)
    clf = DecisionTreeClassifier(criterion='gini',min_samples_split = 3, max_depth = 3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test) 
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MSE: ", mse)
    print("r2: ", r2)


    attribute_importance_dict= {}
    for col, val in sorted(zip(x_train.columns, clf.feature_importances_),key=lambda x:x[1],reverse=True):
          attribute_importance_dict[col]=val
    
    keys = list(attribute_importance_dict.keys())
    values = list(attribute_importance_dict.values())

    plt.bar(keys,values)
    plt.show()
    
    # plt.scatter(y_test, y_pred, color='blue')
    # plt.xlabel('Actual')
    # plt.ylabel('Predicted')
    # plt.title('Actual vs Predicted')
    # plt.show()  

def past_playoff_classification():
    """ Uses Decision Tree Classifiers to create a model regarding playoff probability in terms of nba advanced stats from the season prior"""
    df = get_data()
    df['previous_season'] = df['season']+1
    df = df.merge(df[['season', 'abbreviation', 'playoffs']], #adding new column playoffs_next_year
              left_on=['previous_season', 'abbreviation'], 
              right_on=['season', 'abbreviation'], 
              suffixes=('', '_next_year'))
    df.drop(['previous_season', 'season_next_year'], axis=1, inplace=True)

    x = df.iloc[:,2:11]
    y = df.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=100)

    dtc = DecisionTreeClassifier(criterion='gini',min_samples_split = 3, max_depth = 3)
    dtc.fit(x_train,y_train)

    plt.figure(figsize=(12,8))
    tree.plot_tree(dtc.fit(x_train,y_train), feature_names = x_train.columns )
    plt.show()

    y_pred = dtc.predict(x_test)
    print(f" Accuracy score of past advanced stats to playoffs: {accuracy_score(y_test,y_pred)}")
    
    attribute_importance_dict= {}
    for col, val in sorted(zip(x_train.columns, dtc.feature_importances_),key=lambda x:x[1],reverse=True):
          attribute_importance_dict[col]=val
    
    keys = list(attribute_importance_dict.keys())
    values = list(attribute_importance_dict.values())

    plt.bar(keys,values)
    plt.show()

if __name__ == '__main__':
    playoff_classifier() # decision tree classification
    win_regression() # decision tree regression
    past_win_regression() # linear regression
    in_depth_past_win_regression() # decision tree regression
    past_playoff_classification() # decision tree classification
