import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def cleaning(df):

    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['churn'] = np.where(df['last_trip_date'] >= '2014-06-01', 0, 1)
    df['avg_rating_by_driver'] = df['avg_rating_by_driver'].fillna(df['avg_rating_by_driver'].mean())
    df['avg_rating_of_driver'] = df['avg_rating_of_driver'].fillna(df['avg_rating_of_driver'].mean())
    df['phone'] = df['phone'].fillna('Other')
    columns = ['city', 'phone', 'luxury_car_user']
    df = pd.get_dummies(df, prefix=columns, columns=columns)
    df.drop('last_trip_date', axis=1, inplace=True)

    return df

def add_features(df):

    df['days_from_signup'] = (pd.to_datetime('2014-06-01') - df['signup_date']).astype(int)
    df['rating_diff'] = np.abs(df['avg_rating_by_driver'] - df['avg_rating_of_driver'])
    df.drop('signup_date', axis=1, inplace = True)

    return df



def scale(df):

    X = df.drop('churn', axis=1)
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y)

    ss = StandardScaler()

    columns = ['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver', 'avg_surge', 'surge_pct', 'trips_in_first_30_days', 'weekday_pct','days_from_signup','rating_diff']

    X_train[columns] = ss.fit_transform(X_train[columns])
    X_test[columns] = ss.transform(X_test[columns])

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    df = pd.read_csv('data/churn_train.csv')
    df_clean = cleaning(df)
    df_clean = add_features(df_clean)

    X_train, X_test, y_train, y_test = scale(df_clean)

    with open('data/X_train.pkl','wb') as p:
        pickle.dump(X_train,p)
    with open('data/X_test.pkl','wb') as p:
        pickle.dump(X_test,p)
    with open('data/y_train.pkl','wb') as p:
        pickle.dump(y_train,p)
    with open('data/y_test.pkl','wb') as p:
        pickle.dump(y_test,p)
