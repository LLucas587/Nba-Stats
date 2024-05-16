import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

import seaborn as sns


def import_data() -> pd.DataFrame:
    """ Imports data from two datasets and adds a drafted column to the draft combine stats"""
    df = pd.read_csv('DataSet/draft_combine_stats.csv') 
    draft_history = pd.read_csv('DataSet/draft_history.csv')
    draft_history = draft_history[draft_history['season']>=2001]
    df = df[df['season']!=2023]
    df['drafted'] = df['player_name'].isin(draft_history['player_name'])
    df = df[['season','player_name','position','height_wo_shoes','weight','wingspan','standing_reach','body_fat_pct','standing_vertical_leap','max_vertical_leap','bench_press','lane_agility_time','three_quarter_sprint','drafted']]
    return df

def decision_tree():
    """
    Cleans data, splits data into train/test splits, and then makes decision trees using sklearn decision trees
    """
    df = import_data()
    x = df.iloc[:,3:13]
    y = df.iloc[:,13]
    x.dropna(inplace=True)#
    y=y[y.index.isin(x.index)]
    print(len(x),len(y))
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state=100)

    dtc = DecisionTreeClassifier(criterion='entropy',min_samples_split = 3, max_depth = 3)
    dtc.fit(x_train,y_train)
    

    y_pred = dtc.predict(x_test)
    print(f" Accuracy score: {accuracy_score(y_test,y_pred)}")

    plt.figure(figsize=(12,8))
    tree.plot_tree(dtc.fit(x_train,y_train), feature_names = x_train.columns )
    plt.show()

    attribute_importance_dict= {}
    for col, val in sorted(zip(x_train.columns, dtc.feature_importances_),key=lambda x:x[1],reverse=True):
          attribute_importance_dict[col]=val
    
    keys = list(attribute_importance_dict.keys())
    values = list(attribute_importance_dict.values())

    plt.bar(keys,values)
    plt.show()









decision_tree()
