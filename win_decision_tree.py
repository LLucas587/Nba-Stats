import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score


import seaborn as sns

def get_data()-> pd.DataFrame:
    """ Obtains team data for the last 20 years"""
    df = pd.read_csv('DataSet/Team Summaries.csv') 
    df = df[df['lg'] == 'NBA']
    df = df[df['season'] >= 2003]
    df = df.dropna(subset=['abbreviation']) # dropping league average rows
    df = df.dropna(subset=['season','abbreviation','o_rtg','d_rtg','n_rtg','age','w','pace','ts_percent','e_fg_percent','opp_e_fg_percent','tov_percent','playoffs']) # for compatability with older years where data is not recorded
    df = df[['season','abbreviation','o_rtg','d_rtg','n_rtg','age','pace','ts_percent','e_fg_percent','opp_e_fg_percent','tov_percent','playoffs','w']]
    df = df * 1 # converting playoffs column to 1's and 0's
    return df

def playoff_classifier():
    """ Uses Decision Tree Classifiers to create a model regarding playoff probability in terms of nba advanced stats"""
    df = get_data()
    df = df.drop('w',axis=1)
    x = df.iloc[:,2:10]
    y = df.iloc[:,10]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state=100)

    dtc = DecisionTreeClassifier(criterion='gini',min_samples_split = 3, max_depth = 3)
    dtc.fit(x_train,y_train)

    y_pred = dtc.predict(x_test)
    print(f" Accuracy score: {accuracy_score(y_test,y_pred)}")
    
    attribute_importance_dict= {}
    for col, val in sorted(zip(x_train.columns, dtc.feature_importances_),key=lambda x:x[1],reverse=True):
          attribute_importance_dict[col]=val
    
    keys = list(attribute_importance_dict.keys())
    values = list(attribute_importance_dict.values())

    plt.bar(keys,values)
    plt.show()

def win_regression():
    """ Uses Decision Tree Regression to create a model predicting the amount of wins"""
    df = get_data()
    df = df.drop('playoffs',axis=1)
    df = df.drop('tov_percent',axis=1)
    x = df.iloc[:,2:10]
    y = df.iloc[:,10]

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state=100)
    clf = DecisionTreeRegressor(min_samples_split = 5, max_depth = 5)
    clf.fit(x_train,y_train)
    
    y_pred = clf.predict(x_test) 
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(mse,r2)

    plt.scatter(y_test, y_pred, color='blue')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()  

win_regression()