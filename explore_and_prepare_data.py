__author__ = "Koren Gast"

# Assisting functions for preparing and exploring the data
# =======================================================

import pandas as pd
import numpy as np
from models.randomForest import RandomForest
from matplotlib import pyplot as plt

# Read csv file into dataframe, replace Nans with the average value and turn categorical features to dummies
def read_data_w_dummies(file_path, means=None):
    df = pd.read_csv(file_path)
    # print(df.isna().sum())
    categorical_features = df.columns[df.dtypes == 'object']
    if means is None:
        means = get_mean_values(file_path)
    df = df.fillna(means)
    dummies = pd.DataFrame(data=pd.get_dummies(df[categorical_features]))
    df = df.drop(categorical_features, axis=1)
    df = df.join(dummies)
    try:
        tags = df['tag']
        # print('Number of 1s in tags: %f' % sum(tags))
        df = df.drop(['tag'], axis=1)
        return np.array(df), np.array(tags)
    except:
        return np.array(df)

# Returns the averages values for each non categorical variable
def get_mean_values(file_path):
    df = pd.read_csv(file_path)
    categorical_features = df.columns[df.dtypes == 'object']
    return df.drop(categorical_features, axis=1).mean(axis=0)

# Comparing the distribution of the test and train
def compare_train_test_dist(train_path, test_path):
    train_df = pd.read_csv(train_path)
    train_df = train_df.drop('tag', axis=1)
    test_df = pd.read_csv(test_path)
    categorical_features = train_df.columns[train_df.dtypes == 'object']
    for cf in categorical_features:
        categories = list(train_df[cf].unique())
        train_df[cf] = train_df[cf].apply(categories.index)
        test_df[cf] = test_df[cf].apply(categories.index)
    train_df =train_df.dropna()
    test_df =test_df.dropna()
    for c in test_df.columns:
        plt.hist(train_df[c].dropna())
        plt.hist(test_df[c].dropna())
        plt.title(c)
        plt.show()

# Find the best value for the number of trees in random forest
def optimize_n_estimators(X_train, X_valid, y_train, y_valid, min_n, max_n, n_jump=1):
    train_acc = []
    valid_acc = []
    for n in range(min_n, max_n, n_jump):
        rf_model = RandomForest(n)
        rf_model.fit(X_train, y_train)
        train_acc.append(rf_model.evaluate(X_train, y_train))
        valid_acc.append(rf_model.evaluate(X_valid, y_valid))
    plt.plot(range(min_n, max_n, n_jump), train_acc, 'b-')
    plt.plot(range(min_n, max_n, n_jump), valid_acc, 'r-')
    plt.show()

