import math

import numpy as np
import pandas as pd
import xgboost
from openpyxl import load_workbook
from sklearn import linear_model
from sklearn.impute import KNNImputer
from sklearn.svm import SVC, SVR
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier, \
     RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from time import time, process_time

# load and summarize the dataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve

# for training dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# for feature reduction
from sklearn.feature_selection import SelectKBest, f_classif

def ridgeRegression(ridge, X_train, y_train, test_data, X_test):
     # Ridge Regression

     # Fit the model using training dataset
     ridge.fit(X_train, y_train)

     # calculating score
     print('Training Score_ridge = ', ridge.score(X_train, y_train))

     # calculating prediction
     prediction = ridge.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'Id': test_data['Id'],
          'SalePrice': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\House Price\result\ridge.csv', index=False)

def lassoRegression(lasso, X_train, y_train, test_data, X_test):
     # Lasso Regression

     # Fit the model using training dataset
     lasso.fit(X_train, y_train)

     # calculating score
     print('Training Score_lasso = ', lasso.score(X_train, y_train))

     # calculating prediction
     prediction = lasso.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'Id': test_data['Id'],
          'SalePrice': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\House Price\result\lasso.csv', index=False)

def elasticNetRegression(elasticNet, X_train, y_train, test_data, X_test):
     # ElasticNet Regression

     # Fit the model using training dataset
     elasticNet.fit(X_train, y_train)

     # calculating score
     print('Training Score_elasticNet = ', elasticNet.score(X_train, y_train))

     # calculating prediction
     prediction = elasticNet.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'Id': test_data['Id'],
          'SalePrice': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\House Price\result\elasticNet.csv', index=False)


def svRegression(svc_rbf, svc_linear, X_train, y_train, test_data, X_test):
     # Support Vector Regression
     # .............................

     # Fit the model using training dataset
     svr_rbf.fit(X_train, y_train)
     svr_linear.fit(X_train, y_train)

     # calculating score
     print('Training Score_SVRrbf = ', svr_rbf.score(X_train, y_train))
     print('Training Score_SVRlinear = ', svr_linear.score(X_train, y_train))

     # calculating prediction
     prediction_rbf = svc_rbf.predict(X_test)
     prediction_linear = svc_linear.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'Id': test_data['Id'],
          'SalePrice': prediction_rbf
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\House Price\result\svr_rbf.csv', index=False)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'Id': test_data['Id'],
          'SalePrice': prediction_linear
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\House Price\result\svr_linear.csv', index=False)


def xgbRegression(xgb, X_train, y_train, test_data, X_test):
     # XGBoosting Regression
     # .............................

     # Fit the model using training dataset
     xgb.fit(X_train, y_train)

     # calculating score
     print('Training Score_xgb = ', xgb.score(X_train, y_train))

     # calculating prediction
     prediction = xgb.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'Id': test_data['Id'],
          'SalePrice': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\House Price\result\xgb.csv', index=False)


def cbRegression(cb, X_train, y_train, test_data, X_test):
     # CatBoost Regression
     # .............................

     # Fit the model using training dataset
     cb.fit(X_train, y_train)

     # calculating score
     print('Training Score_cb = ', cb.score(X_train, y_train))

     # calculating prediction
     prediction = cb.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'Id': test_data['Id'],
          'SalePrice': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\House Price\result\cb.csv', index=False)


def lgbmRegression(lgbm, X_train, y_train, test_data, X_test):
     # lgbm Regression
     # .............................

     # Fit the model using training dataset
     lgbm.fit(X_train, y_train)

     # calculating score
     print('Training Score_lgbm = ', lgbm.score(X_train, y_train))

     # calculating prediction
     prediction = lgbm.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'Id': test_data['Id'],
          'SalePrice': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\House Price\result\lgbm.csv', index=False)


def rfRegression(rf, X_train, y_train, test_data, X_test):
     # Random Forest Regression
     # .........................

     # Fit the model using training dataset
     rf.fit(X_train, y_train)

     # calculating score
     print('Training Score_rf = ', rf.score(X_train, y_train))

     # calculating prediction
     prediction = rf.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'Id': test_data['Id'],
          'SalePrice': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\House Price\result\rf.csv', index=False)


def mlpRegression(mlp, X_train, y_train, test_data, X_test):
     # Multi-Layer Perceptron Regression
     # .............................

     # Fit the model using training dataset
     mlp.fit(X_train, y_train)

     # calculating score
     print('Training Score_mlp = ', mlp.score(X_train, y_train))

     # calculating prediction
     prediction = mlp.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'Id': test_data['Id'],
          'SalePrice': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\House Price\result\mlp.csv', index=False)


def gnbRegression(gnb, X_train, y_train, test_data, X_test):
     # gaussian Naive_Bayes Regression
     # .............................

     # Fit the model using training dataset
     gnb.fit(X_train, y_train)

     # calculating score
     print('Training Score_gnb = ', gnb.score(X_train, y_train))

     # calculating prediction
     prediction = gnb.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'Id': test_data['Id'],
          'SualePrice': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\House Price\result\gnb.csv', index=False)


def votingRegression(voting, X_train, y_train, test_data, X_test):
     # Voting Regression
     # .............................

     # Fit the model using training dataset
     voting.fit(X_train, y_train)

     # calculating score
     print('Training Score_voting = ', voting.score(X_train, y_train))

     # calculating prediction
     prediction = voting.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'Id': test_data['Id'],
          'SalePrice': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\House Price\result\voting.csv', index=False)


def stackingRegression(stacking_1, stacking_2, stacking_3, X_train, y_train, test_data, X_test):
     # Stacking Regression
     # .............................

     # Fit the model using training dataset
     print("Stacking.....using base model: XGB")
     stacking_1.fit(X_train, y_train)

     # calculating score
     print('Training Score_stacking1 = ', stacking_1.score(X_train, y_train))

     # calculating prediction
     prediction_stack1 = stacking_1.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'Id': test_data['Id'],
          'SalePrice': prediction_stack1
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\House Price\result\stacking_1.csv', index=False)

     # Fit the model using training dataset
     print("Stacking.....using base model: CB")
     stacking_2.fit(X_train, y_train)

     # calculating score
     print('Training Score_stacking2 = ', stacking_2.score(X_train, y_train))

     # calculating prediction
     prediction_stack2 = stacking_2.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'Id': test_data['Id'],
          'SalePrice': prediction_stack2
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\House Price\result\stacking_2.csv', index=False)

     # Fit the model using training dataset
     print("Stacking.....using base model: LGBM")
     stacking_3.fit(X_train, y_train)

     # calculating score
     print('Training Score_stacking3 = ', stacking_3.score(X_train, y_train))

     # calculating prediction
     prediction_stack3 = stacking_3.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'Id': test_data['Id'],
          'SalePrice': prediction_stack3
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\House Price\result\stacking_3.csv', index=False)


# calling main function
if __name__ == '__main__':
     # Load training data
     train_data = pd.read_csv(r"E:\Kaggle\House Price\Dataset\train.csv")

     X_train = train_data.drop(columns=["SalePrice"])
     y_train = train_data["SalePrice"]

     print("X_train:\n", X_train)

     # Load test data
     test_data = pd.read_csv(r"E:\Kaggle\House Price\Dataset\test.csv")

     X_test = test_data

     print("test_data:\n", test_data)

     # Print number of NaN or blank cell in column wise before encoding and scaling
     X_train = pd.DataFrame(X_train)
     X_test = pd.DataFrame(X_test)
     print("Before encoding and scaling:")
     print("NaN or blank cell in train data:\n", X_train.isna().sum())
     print("NaN or blank cell in test data:\n", X_test.isna().sum())

     # Fill numerical columns with mean
     for col in X_train.select_dtypes(include=['float64', 'int64']).columns:
          X_train[col].fillna(X_train[col].mean(), inplace=True)
     for col in X_test.select_dtypes(include=['float64', 'int64']).columns:
          X_test[col].fillna(X_test[col].mean(), inplace=True)

     # Fill categorical columns with ""Missing" word"
     for col in X_train.select_dtypes(include=['object']).columns:
          X_train[col].fillna("Missing")
     for col in X_test.select_dtypes(include=['object']).columns:
          X_test[col].fillna("Missing")

     # Filling NaN or blank cell with mean value for numerical and "Missing" word for categorical
     # X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())
     # X_train['Cabin'] = X_train['Cabin'].fillna("Missing")
     # X_train['Embarked'] = X_train['Embarked'].fillna("Missing")
     #
     # X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())
     # X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].mean())
     # X_test['Cabin'] = X_test['Cabin'].fillna("Missing")

     print("After filling:")
     print("NaN or blank cell in train data:\n", X_train.isna().sum())
     print("NaN or blank cell in test data:\n", X_test.isna().sum())

     # Conversion categorical data to continuous data using LabelEncoder

     X_train = pd.DataFrame(X_train)
     X_test = pd.DataFrame(X_test)

     # For non-numerical data
     lbl_encoder = LabelEncoder()

     for col in X_train.columns:
          if X_train[col].dtype == 'object':
               X_train[col] = lbl_encoder.fit_transform(X_train[col])

     for col in X_test.columns:
          if X_test[col].dtype == 'object':
               X_test[col] = lbl_encoder.fit_transform(X_test[col])


     # Normalization the dataset
     scaler = StandardScaler()
     X_train = scaler.fit_transform(X_train)
     X_test = scaler.fit_transform(X_test)

     print("After scaling_X_train:\n", X_train)
     print("After scaling_X_test:\n", X_test)

     total_testingData = len(X_test)
     print("\nLength of testing data: ", total_testingData)

     # Model Initialization
     ridge = linear_model.Ridge()
     lasso = linear_model.Lasso()
     elasticNet = linear_model.ElasticNet()

     svr_rbf = SVR(kernel='rbf')
     svr_linear = SVR(kernel='linear')
     svr_model = SVR()
     param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
     svr = GridSearchCV(estimator=svr_model, param_grid=param_grid, cv=5, n_jobs=-1)

     xgb = XGBRegressor(random_state=1, learning_rate=0.01, n_jobs=-1)
     cb = CatBoostRegressor(n_estimators=100, thread_count=-1)
     lgbm = LGBMRegressor(n_estimators=100, n_jobs=-1)
     dt = tree.DecisionTreeRegressor()
     rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)

     mlp = MLPRegressor(max_iter=300)

     # gnb = GaussianNB()
     voting = VotingRegressor(
          estimators=[('mlp', mlp), ('xgb', xgb), ('rf', rf)])
     stacking_1 = StackingRegressor(
          estimators=[('elasticNet', elasticNet), ('rf', rf), ('xgb', xgb)],
          final_estimator=mlp, cv=None)
     stacking_2 = StackingRegressor(
          estimators=[('elasticNet', elasticNet), ('rf', rf), ('cb', cb)],
          final_estimator=mlp, cv=None)
     stacking_3 = StackingRegressor(
          estimators=[('elasticNet', elasticNet), ('rf', rf), ('lgbm', lgbm)],
          final_estimator=mlp, cv=None)

     print("Execution started.............\n\n")

     # calling model functions
     ridgeRegression(ridge, X_train, y_train, test_data, X_test)
     lassoRegression(lasso, X_train, y_train, test_data, X_test)
     elasticNetRegression(elasticNet, X_train, y_train, test_data, X_test)
     svRegression(svr_rbf, svr_linear, X_train, y_train, test_data, X_test)
     xgbRegression(xgb, X_train, y_train, test_data, X_test)
     cbRegression(cb, X_train, y_train, test_data, X_test)
     lgbmRegression(lgbm, X_train, y_train, test_data, X_test)
     rfRegression(rf, X_train, y_train, test_data, X_test)
     mlpRegression(mlp, X_train, y_train, test_data, X_test)
     votingRegression(voting, X_train, y_train, test_data, X_test)
     stackingRegression(stacking_1, stacking_2, stacking_3, X_train, y_train, test_data, X_test)