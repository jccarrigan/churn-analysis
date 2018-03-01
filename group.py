import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm


class ChurnPredictor():
    def __init__(self, filepath, model, scaler=None, rating_filler=None):
        self.df = pd.read_csv(filepath)
        self.model = model
        self.df.last_trip_date = pd.to_datetime(self.df.last_trip_date)
        self.df.signup_date = pd.to_datetime(self.df.signup_date)
        self.df['churn'] = self.df.last_trip_date >= pd.to_datetime('2014-06-01')
        del self.df['last_trip_date']
        # use regression from training to predict on test
        self.rating_filler = rating_filler
        # scaler is set by the scaler method during training
        # and it should be resused during test
        if not scaler:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler

    def get_Xy(self):
        self.y = self.df.churn
        self.X = self.df.drop('churn', axis = 1)

    def get_dummies(self):
        columns = ['city', 'phone', 'luxury_car_user']
        for column in columns:
            dummies = pd.get_dummies(self.df[column],prefix=column[:3],dummy_na = True, drop_first = False)
            self.df = pd.concat([self.df, dummies], axis = 1)
            del self.df[column]
            try:
                del self.df[column[:3]+'_nan']
            except:
                pass

    def fill_rating(self,who):
        if who not in ['driver', 'rider']:
            print('who must be driver or rider')
            return
        if who == 'driver':
            column = 'avg_rating_of_driver'
        else:
            column = 'avg_rating_by_driver'
        mask = np.isnan(self.df[column])
        train_group = self.df[~mask]
        predict_group = self.df[mask]
        X = self.df.drop(['avg_rating_of_driver', 'avg_rating_by_driver'], axis = 1)
        X_train = X[~mask]
        X_predict = X[mask]
        y = train_group[column]
        y_train = y[~mask]
        # fit the model on the training
        # fill ratings in test data based on regression from training
        if not self.rating_filler:
            self.rating_filler = LinearRegression().fit(X_train,y_train)

        # apply it on the test data and don't refit
        y_predict = self.rating_filler.predict(X_predict)
        predict_group.is_copy = False
        predict_group[column] = y_predict
        self.df = pd.concat([train_group,predict_group], axis = 0)

    def scale(self):
        self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)

    def fit(self):
        self.model.fit(self.X,self.y)

    def predict(self):
        self.predictions = self.model.predict(self.X)
        self.score = model.score(self.X,self.y)

    def convert_signup_day(self):
        self.df['signup_weekday'] = 1*((self.df.signup_date.dt.dayofweek + 1) % 7 <= 4)
        self.df['signup_weekend'] = 1*((self.df.signup_date.dt.dayofweek + 1) % 7 >  4)
        self.df['signup_holiday'] = 1*(self.df.signup_date.dt.dayofyear == 1)
        del self.df['signup_date']


def calc_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return accuracy, precision, recall

def print_results(model_score, accuracy, precision, recall):
    num = 3
    print ('Scores:   ', round(model_score, num))
    print ('Accuracy: ', round(accuracy, num))
    print ('Precision:', round(precision, num))
    print ('Recall:   ', round(recall, num))

def main(cp):
    cp.convert_signup_day()
    cp.get_dummies()
    cp.fill_rating('driver')
    cp.fill_rating('rider')
    cp.get_Xy()
    cp.scale()
    cp.fit()
    cp.predict()


if __name__ == '__main__':
    models = {
        'Logistic Regression' : LogisticRegression(),
        'Ada Boost' : AdaBoostClassifier(learning_rate=0.2, n_estimators=100, random_state=1),
        'Random Forest'  : RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1),
        'Gradient Boosting': GradientBoostingClassifier(learning_rate=0.2, n_estimators=100, random_state=1)}
        # 'Support Vector Machine' : svm.SVC() }
    scenarios = {'Train': 'data/churn_train.csv', 'Test': 'data/churn_test.csv'}

    for name, model in models.items():
        X_Train, y_train, X_test, y_test, predictions = 0, 0, 0, 0, 0

        for phase, data_set in scenarios.items():
            print(f"{name.upper()}  --  {phase.upper()}: " + 50*'*')
            churn_prediction = ChurnPredictor(data_set, model)
            main(churn_prediction)

            if name == 'Train':
                X_train = churn_prediction.X
                y_train = churn_prediction.y
            else:
                X_test = churn_prediction.X
                y_test = churn_prediction.y
                predictions = churn_prediction.predictions
                model_score = churn_prediction.score

        accuracy, precision, recall = calc_metrics(y_test, predictions)
        print_results(model_score, accuracy, precision, recall)
        print()
