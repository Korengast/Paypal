__author__ = "Koren Gast"

# This script actions:
# 1. Finding a good number for n_estimators for the random forest (which had the best performances)
# 2. Compare histograms of the features of the train and test sets to ensure that the
#       test data distributing as the data we trained on
# 3. Predict on the test set after training on the train set
# ==================================================================================================

from models.randomForest import RandomForest
from explore_and_prepare_data import read_data_w_dummies, \
    optimize_n_estimators, compare_train_test_dist, get_mean_values
from sklearn.model_selection import train_test_split
import pandas as pd

TRAIN_PATH = 'csv_data/interview_dataset_train.csv'
TEST_PATH = 'csv_data/interview_dataset_test_no_tags.csv'

data_array, data_tags = read_data_w_dummies(TRAIN_PATH)
X_train, X_valid, y_train, y_valid = train_test_split(data_array, data_tags, test_size=0.25)

# Random forest is the best from the tried models. tune rf parameters:
optimize_n_estimators(X_train, X_valid, y_train, y_valid, min_n=1, max_n=200, n_jump=10)

# No accuracy imporvement after about 30.
optimize_n_estimators(X_train, X_valid, y_train, y_valid, min_n=10, max_n=30, n_jump=1)
# 20 looks like a good choice

rf_model = RandomForest(20)
rf_model.fit(data_array, data_tags)

# Verifying that the test is from the same distribution of the train
compare_train_test_dist(TRAIN_PATH, TEST_PATH)

means = get_mean_values(TRAIN_PATH)  # I'll assign the mean values from the train set to the Nans on the test set

test_array = read_data_w_dummies(TEST_PATH)
preds = rf_model.predict(test_array)
submission = pd.read_csv(TEST_PATH)
submission['Prediction'] = preds
submission.to_csv('Submission.csv')
