import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

def modeling(X_train, y_train, X_test, y_test):

    Logistic = LogisticRegression()
    AdaBoost = AdaBoostClassifier()
    GradientBoosting = GradientBoostingClassifier()
    RandomForest = RandomForestClassifier()
    #CatBoost = CatBoostClassifier()

    models = [Logistic, AdaBoost, GradientBoosting, RandomForest]

    for model in models:
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        print('{0} Accuracy: {1:0.3f}'.format(model.__class__.__name__, precision_score(y_test, y_predict)))
        print('{0} Precision: {1:0.3f}'.format(model.__class__.__name__, precision_score(y_test, y_predict)))
        print('{0} Recall: {1:0.3f}'.format(model.__class__.__name__, recall_score(y_test, y_predict)))
        print('{0} F1: {1:0.3f}'.format(model.__class__.__name__, f1_score(y_test, y_predict)))

def tuning(X_train, y_train):

    model = GradientBoostingClassifier()

    """parameters = {'learning_rate': [0.01, 0.1, 1],
              'max_features' : [7, 9, None],
              'max_depth' : [3, 5, 7],
              'loss' : ['exponential', 'deviance']}
    clf = GridSearchCV(model, parameters, n_jobs = 4, verbose=1, scoring = 'f1', cv = 5)
    clf.fit(X_train, y_train)
    print (clf.best_score_)
    print (clf.best_params_)"""

    parameters = {'learning_rate': [.1],
              'n_estimators': [100, 250],
              'subsample':[0.7, 0.8, 0.9],
              'min_samples_leaf' : [7, 9, 15],
              'max_features' : [7],
              'max_depth' : [5],
              'loss' : ['deviance']}
    clf2 = GridSearchCV(model, parameters, n_jobs = 4, verbose=1, scoring = 'f1', cv = 5)
    clf2.fit(X_train, y_train)
    print (clf2.best_score_)
    print (clf2.best_params_)

def final_model(X_train,y_train, X_test, y_test):

    model  = GradientBoostingClassifier(learning_rate= 0.1, loss= 'deviance', max_depth= 5, max_features= 7, min_samples_leaf= 9, n_estimators= 250, subsample= 0.9)

    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print('{0} Accuracy: {1:0.3f}'.format(model.__class__.__name__, precision_score(y_test, y_predict)))
    print('{0} Precision: {1:0.3f}'.format(model.__class__.__name__, precision_score(y_test, y_predict)))
    print('{0} Recall: {1:0.3f}'.format(model.__class__.__name__, recall_score(y_test, y_predict)))
    print('{0} F1: {1:0.3f}'.format(model.__class__.__name__, f1_score(y_test, y_predict)))




if __name__ == '__main__':

    X_train = pd.read_pickle('data/X_train.pkl')
    X_test = pd.read_pickle('data/X_test.pkl')
    y_train = pd.read_pickle('data/y_train.pkl')
    y_test = pd.read_pickle('data/y_test.pkl')

    #modeling(X_train, y_train, X_test, y_test)
    #tuning(X_train, y_train)
    final_model(X_train, y_train, X_test, y_test)
