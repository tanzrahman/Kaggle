import math

import numpy as np
import pandas as pd
import xgboost
from openpyxl import load_workbook
from sklearn import linear_model
from sklearn.impute import KNNImputer
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from time import time, process_time

# load and summarize the dataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve

# for training dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# for feature reduction
from sklearn.feature_selection import SelectKBest, f_classif

def logisticRegression(logistic_model, X_train, y_train, test_data, X_test):
     # Logistic Regression
     # .............................
     print("...........Logistic Model..........")

     # Fit the model using training dataset
     logistic_model.fit(X_train, y_train)

     # calculating score
     print('Training Score_logistic = ', logistic_model.score(X_train, y_train))

     # calculating defect prediction
     prediction = logistic_model.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\logistic.csv', index=False)

def svcClassifier(svc_rbf, svc_linear, X_train, y_train, test_data, X_test):
     # Support Vector Classifier
     # .............................
     # Fit the model using training dataset
     svc_rbf.fit(X_train, y_train)
     svc_linear.fit(X_train, y_train)

     # calculating score
     print('Training Score_SVMrbf = ', svc_rbf.score(X_train, y_train))
     print('Training Score_SVMlinear = ', svc_linear.score(X_train, y_train))

     # calculating disease prediction
     prediction_rbf = svc_rbf.predict(X_test)
     prediction_linear = svc_linear.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction_rbf
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\svm_rbf.csv', index=False)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction_linear
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\svm_linear.csv', index=False)


def xgbClassifier(xgb, X_train, y_train, test_data, X_test):
     # XGBoosting Classifiers
     # .............................
     print("start...")
     # Fit the model using training dataset
     xgb.fit(X_train, y_train)

     # calculating score
     print('Training Score_xgb = ', xgb.score(X_train, y_train))

     # calculating disease prediction
     prediction = xgb.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\xgb.csv', index=False)


def cbClassifier(cb, X_train, y_train, test_data, X_test):
     # CatBoost Classifiers
     # .............................
     # Fit the model using training dataset
     cb.fit(X_train, y_train)

     # calculating score
     print('Training Score_cb = ', cb.score(X_train, y_train))

     # calculating disease prediction
     prediction = cb.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\cb.csv', index=False)


def lgbmClassifier(lgbm, X_train, y_train, test_data, X_test):
     # lgbm Classifiers
     # .............................
     # Fit the model using training dataset
     lgbm.fit(X_train, y_train)

     # calculating score
     print('Training Score_lgbm = ', lgbm.score(X_train, y_train))

     # calculating disease prediction
     prediction = lgbm.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\lgbm.csv', index=False)


def rfClassifier(rf, X_train, y_train, test_data, X_test):
     # Random Forest Classifiers
     # .........................

     # Fit the model using training dataset
     rf.fit(X_train, y_train)

     # calculating score
     print('Training Score_rf = ', rf.score(X_train, y_train))

     # calculating disease prediction
     prediction = rf.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\rf.csv', index=False)


def mlpClassifier(mlp, X_train, y_train, test_data, X_test):
     # Multi-Layer Perceptron Classifiers
     # .............................

     # Fit the model using training dataset
     mlp.fit(X_train, y_train)

     # calculating score
     print('Training Score_mlp = ', mlp.score(X_train, y_train))

     # calculating disease prediction
     prediction = mlp.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\mlp.csv', index=False)


def gnbClassifier(gnb, X_train, y_train, test_data, X_test):
     # gaussian Naive_Bayes Classifiers
     # .............................

     # Fit the model using training dataset
     gnb.fit(X_train, y_train)

     # calculating score
     print('Training Score_gnb = ', gnb.score(X_train, y_train))

     # calculating disease prediction
     prediction = gnb.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\gnb.csv', index=False)


def votingClassifier(voting, X_train, y_train, test_data, X_test):
     # Voting Classifier
     # .............................

     # Fit the model using training dataset
     voting.fit(X_train, y_train)

     # calculating score
     print('Training Score_voting = ', voting.score(X_train, y_train))

     # calculating defect prediction
     prediction = voting.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\voting.csv', index=False)


def stackingClassifier(stacking_1, stacking_2, stacking_3, stacking_4, stacking_5, stacking_6, X_train, y_train, test_data, X_test):
     # Stacking Classifier
     # .............................

     # Fit the model using training dataset
     print("Stacking.....using base model: XGB")
     stacking_1.fit(X_train, y_train)

     # calculating score
     print('Training Score_stacking1 = ', stacking_1.score(X_train, y_train))

     # calculating disease prediction
     prediction_stack1 = stacking_1.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction_stack1
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\stacking_1.csv', index=False)

     # Fit the model using training dataset
     print("Stacking.....using base model: CB")
     stacking_2.fit(X_train, y_train)

     # calculating score
     print('Training Score_stacking2 = ', stacking_2.score(X_train, y_train))

     # calculating disease prediction
     prediction_stack2 = stacking_2.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction_stack2
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\stacking_2.csv', index=False)

     # Fit the model using training dataset
     print("Stacking.....using base model: LGBM")
     stacking_3.fit(X_train, y_train)

     # calculating score
     print('Training Score_stacking3 = ', stacking_3.score(X_train, y_train))

     # calculating disease prediction
     prediction_stack3 = stacking_3.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction_stack3
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\stacking_3.csv', index=False)

     # Fit the model using training dataset
     print("Stacking.....using base model: XGB")
     stacking_4.fit(X_train, y_train)

     # calculating score
     print('Training Score_stacking4 = ', stacking_4.score(X_train, y_train))

     # calculating disease prediction
     prediction_stack4 = stacking_4.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction_stack4
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\stacking_4.csv', index=False)

     # Fit the model using training dataset
     print("Stacking.....using base model: CB")
     stacking_5.fit(X_train, y_train)

     # calculating score
     print('Training Score_stacking5 = ', stacking_5.score(X_train, y_train))

     # calculating disease prediction
     prediction_stack5 = stacking_5.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction_stack5
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\stacking_5.csv', index=False)

     # Fit the model using training dataset
     print("Stacking.....using base model: LGBM")
     stacking_6.fit(X_train, y_train)

     # calculating score
     print('Training Score_stacking6 = ', stacking_6.score(X_train, y_train))

     # calculating disease prediction
     prediction_stack6 = stacking_6.predict(X_test)

     # ....................................................
     # .......Write the prediction data into csv.........
     # ....................................................

     data = {
          'PassengerId': test_data['PassengerId'],
          'Survived': prediction_stack6
     }

     # Convert numpy array to DataFrame for writing into excel
     df = pd.DataFrame(data)

     # Write into csv file
     df.to_csv(r'E:\Kaggle\Titanic\result\stacking_6.csv', index=False)


# def selfEnsemble(voting, stacking_1, stacking_2):
#      # Ensemble
#      # .............................
#
#      # Fit the model using training dataset
#      voting.fit(X_train, y_train)
#
#      # calculating score
#      print('Training Score = ', voting.score(X_train, y_train))
#
#      # calculating disease prediction
#      defect_predict = voting.predict(X_test)
#
#      score = accuracy_score(y_test, defect_predict)
#      print('Testing Score_Voting = ', score)
#
#      # Fit the model using training dataset
#      stacking_1.fit(X_train, y_train)
#
#      # calculating score
#      print('Training Score = ', stacking_1.score(X_train, y_train))
#
#      # calculating defect prediction
#      defect_predict = stacking_1.predict(X_test)
#
#      score = accuracy_score(y_test, defect_predict)
#      print('Testing Score_Stacking_1 = ', score)
#
#      # Fit the model using stacking classier_2
#      stacking_2.fit(X_train, y_train)
#
#      # calculating score
#      print('Training Score = ', stacking_2.score(X_train, y_train))
#
#      # calculating defect prediction
#      defect_predict = stacking_2.predict(X_test)
#
#      score = accuracy_score(y_test, defect_predict)
#      print('Testing Score_Stacking_2 = ', score)


# calling main function
if __name__ == '__main__':
     # Load training data
     train_data = pd.read_csv(r"E:\Kaggle\Titanic\Dataset\train.csv")

     X_train = train_data.drop(columns=["Survived"])
     y_train = train_data["Survived"]

     print("X_train:\n", X_train)

     # Load test data
     test_data = pd.read_csv(r"E:\Kaggle\Titanic\Dataset\test.csv")

     X_test = test_data

     print("test_data:\n", test_data)

     # Print number of NaN or blank cell in column wise before encoding and scaling
     X_train = pd.DataFrame(X_train)
     X_test = pd.DataFrame(X_test)
     print("Before encoding and scaling:")
     print("NaN or blank cell in train data:\n", X_train.isna().sum())
     print("NaN or blank cell in test data:\n", X_test.isna().sum())

     # Filling NaN or blank cell with mean value for numerical and "Missing" word for categorical
     X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())
     X_train['Cabin'] = X_train['Cabin'].fillna("Missing")
     X_train['Embarked'] = X_train['Embarked'].fillna("Missing")

     X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())
     X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].mean())
     X_test['Cabin'] = X_test['Cabin'].fillna("Missing")

     print("After filling:")
     print("NaN or blank cell in train data:\n", X_train.isna().sum())
     print("NaN or blank cell in test data:\n", X_test.isna().sum())

     # Conversion categorical data to continuous data using LabelEncoder

     # For non-numerical data
     lbl_encoder = LabelEncoder()
     X_train['Name'] = lbl_encoder.fit_transform(X_train['Name'])
     X_train['Sex'] = lbl_encoder.fit_transform(X_train['Sex'])
     X_train['Ticket'] = lbl_encoder.fit_transform(X_train['Ticket'])
     X_train['Cabin'] = lbl_encoder.fit_transform(X_train['Cabin'])
     X_train['Embarked'] = lbl_encoder.fit_transform(X_train['Embarked'])

     print("After label encoding_X_train:\n", X_train)

     X_test['Name'] = lbl_encoder.fit_transform(test_data['Name'])
     X_test['Sex'] = lbl_encoder.fit_transform(test_data['Sex'])
     X_test['Ticket'] = lbl_encoder.fit_transform(test_data['Ticket'])
     X_test['Cabin'] = lbl_encoder.fit_transform(test_data['Cabin'])
     X_test['Embarked'] = lbl_encoder.fit_transform(test_data['Embarked'])

     print("After label encoding_X_test:\n", X_test)

     # Normalization the dataset
     scaler = StandardScaler()
     X_train = scaler.fit_transform(X_train)
     X_test = scaler.fit_transform(X_test)

     print("After scaling_X_train:\n", X_train)
     print("After scaling_X_test:\n", X_test)

     total_testingData = len(X_test)
     print("\nLength of testing data: ", total_testingData)

     # Print number of NaN or blank cell in column wise after encoding and scaling
     X_train = pd.DataFrame(X_train)
     X_test = pd.DataFrame(X_test)
     print("After encoding and scaling number:")
     print("NaN or blank cell in train data:\n", X_train.isna().sum())
     print("NaN or blank cell in test data:\n", X_test.isna().sum())

     # Model Initialization
     logistic_model = linear_model.LogisticRegression()

     svc_rbf = SVC(kernel='rbf')
     svc_linear = SVC(kernel='linear')
     svc_model = SVC()
     param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
     svc = GridSearchCV(estimator=svc_model, param_grid=param_grid, cv=5, n_jobs=-1)

     xgb = XGBClassifier(random_state=1, learning_rate=0.01, n_jobs=-1)
     cb = CatBoostClassifier(n_estimators=100, thread_count=-1)
     lgbm = LGBMClassifier(n_estimators=100, n_jobs=-1)
     rf = RandomForestClassifier(n_estimators=100, criterion='entropy', n_jobs=-1)

     mlp_model = MLPClassifier(max_iter=300)
     param_grid = {'hidden_layer_sizes': [(50,), (100,)]}
     mlp = GridSearchCV(mlp_model, param_grid=param_grid, cv=5, n_jobs=-1)

     gnb = GaussianNB()
     voting = VotingClassifier(
          estimators=[('mlp', mlp), ('xgb', xgb), ('rf', rf)],
          voting='hard')
     stacking_1 = StackingClassifier(
          estimators=[('logistic_model', logistic_model), ('rf', rf), ('xgb', xgb)],
          final_estimator=mlp, cv=None)
     stacking_2 = StackingClassifier(
          estimators=[('logistic_model', logistic_model), ('rf', rf), ('cb', cb)],
          final_estimator=mlp, cv=None)
     stacking_3 = StackingClassifier(
          estimators=[('logistic_model', logistic_model), ('rf', rf), ('lgbm', lgbm)],
          final_estimator=mlp, cv=None)
     stacking_4 = StackingClassifier(
          estimators=[('mlp', mlp), ('rf', rf), ('xgb', xgb)],
          final_estimator=logistic_model, cv=None)
     stacking_5 = StackingClassifier(
          estimators=[('mlp', mlp), ('rf', rf), ('cb', cb)],
          final_estimator=logistic_model, cv=None)
     stacking_6 = StackingClassifier(
          estimators=[('mlp', mlp), ('rf', rf), ('lgbm', lgbm)],
          final_estimator=logistic_model, cv=None)

     print("Execution started.............\n\n")

     # calling model functions
     logisticRegression(logistic_model, X_train, y_train, test_data, X_test)
     svcClassifier(svc_rbf, svc_linear, X_train, y_train, test_data, X_test)
     xgbClassifier(xgb, X_train, y_train, test_data, X_test)
     cbClassifier(cb, X_train, y_train, test_data, X_test)
     lgbmClassifier(lgbm, X_train, y_train, test_data, X_test)
     rfClassifier(rf, X_train, y_train, test_data, X_test)
     mlpClassifier(mlp, X_train, y_train, test_data, X_test)

     gnbClassifier(gnb, X_train, y_train, test_data, X_test)
     votingClassifier(voting, X_train, y_train, test_data, X_test)
     stackingClassifier(stacking_1, stacking_2, stacking_3, stacking_4, stacking_5, stacking_6, X_train, y_train, test_data, X_test)
     # selfEnsemble(voting, stacking_1, stacking_2)