__author__ = "Koren Gast"

# This script is comparing 3 models:
# 1. Logistic regression
# 2. Random forest
# 3. MLP
# The model with the best accuracy would be chosen for parameters tuning and predicting
# =====================================================================================

from explore_and_prepare_data import read_data_w_dummies
from sklearn.model_selection import train_test_split
from models.logisticModel import LogisticModel
from models.randomForest import RandomForest
from models.neuralNetwork import MLP

FILE_PATH = 'csv_data/interview_dataset_train.csv'

data_array, data_tags = read_data_w_dummies(FILE_PATH)
X_train, X_valid, y_train, y_valid = train_test_split(data_array, data_tags, test_size=0.25)

##### Logistic model #####
log_model = LogisticModel()
log_model.fit(X_train, y_train)
eval_train = log_model.evaluate(X_train, y_train)
print('Logistic regression model')
print('Train accuracy: %f' % eval_train)
eval_valid = log_model.evaluate(X_valid, y_valid)
print('Validation accuracy: %f' % eval_valid)
print()
# Train/Valid accuracy: 0.83/0.829 - Underfit

##### Random forest model #####
rf_model = RandomForest(10)
rf_model.fit(X_train, y_train)
eval_train = rf_model.evaluate(X_train, y_train)
print('Random forest model')
print('Train accuracy: %f' % eval_train)
eval_valid = rf_model.evaluate(X_valid, y_valid)
print('Validation accuracy: %f' % eval_valid)
print()
# Train/Valid accuracy: 0.994/0.962

##### MLP model #####
nn_model = MLP()
nn_model.fit(X_train, y_train)
eval_train = nn_model.evaluate(X_train, y_train)
print('MLP model')
print('Train accuracy: %f' % eval_train)
eval_valid = nn_model.evaluate(X_valid, y_valid)
print('Validation accuracy: %f' % eval_valid)
print()
# Train/Valid accuracy: 0.928/0.926

