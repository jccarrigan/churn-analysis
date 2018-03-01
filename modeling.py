from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm


def simple_model(X_train, y_train, X_test, y_test):

    abc = AdaBoostClassifier(learning_rate=0.2,
                        n_estimators=100, random_state=1)

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1)

    gdbc = GradientBoostingClassifier(learning_rate=0.2,
                                     n_estimators=100, random_state=1)
    clf = svm.SVC()

    abc_model = abc.fit(X_train, y_train)
    y_pred = abc_model.predict(X_test)

    abc_model_score = abc_model.score(X_test, y_test)
    abc_accuracy = accuracy_score(y_test, y_pred)
    abc_precision = precision_score(y_test, y_pred)
    abc_recall = recall_score(y_test, y_pred)


    rf_model = rf.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    rf_model_score = rf_model.score(X_test, y_test)
    rf_accuracy = accuracy_score(y_test, y_pred)
    rf_precision = precision_score(y_test, y_pred)
    rf_recall = recall_score(y_test, y_pred)

    gdbc_model = gdbc.fit(X_train, y_train)
    y_pred = gdbc_model.predict(X_test)

    gdbc_model_score = gdbc_model.score(X_test, y_test)
    gdbc_accuracy = accuracy_score(y_test, y_pred)
    gdbc_precision = precision_score(y_test, y_pred)
    gdbc_recall = recall_score(y_test, y_pred)

    #clf_model = clf.fit(X_train, y_train)
    #clf_model_score = clf_model.score(X_test, y_test)

    print ("Scores: {:.2f}, {:.2f}, {:.2f}".format(abc_model_score, rf_model_score, gdbc_model_score))

    print ("Accuracy: {:.2f}, {:.2f}, {:.2f}".format(abc_accuracy, rf_accuracy, gdbc_accuracy))

    print ("Precision: {:.2f}, {:.2f}, {:.2f}".format(abc_precision, rf_precision, gdbc_precision))

    print ("Recall: {:.2f}, {:.2f}, {:.2f}".format(abc_recall, rf_recall, gdbc_recall))

def Model(X_train, y_train, X_test, y_test):

    abc = AdaBoostClassifier(learning_rate=0.1,
                        n_estimators=100, random_state=1)

    model = abc.fit(X_train, y_train)
    model_score = model.score(X_test, y_test)
    print ("Score: {}".format(model_score))

    # 12) Gridsearch best hyperparameters for GradientBoosting
    print ("\n12) AdaboostBoosting GridSearch")
    # note that learning rate has multiple values while n_estimators is set
    ada_boosting_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
                              'n_estimators': [500],
                              'random_state': [1]}
    abc_best_params, abc_best_model = gridsearch_with_output(AdaBoostClassifier(),
                                                               ada_boosting_grid,
                                                               X_train, y_train)
    print ("\nb & c) Comparing model with gridsearch params to initial model on Test set.")
    #abc.fit(X_train, y_train)
    #display_default_and_gsearch_model_results(abc, abc_best_model, X_test, y_test)
    model = abc_best_model.fit(X_train, y_train)
    model_score = model.score(X_test, y_test)
    print ("Score: {}".format(model_score))

def gridsearch_with_output(estimator, parameter_grid, X_train, y_train):
    '''
        Parameters: estimator: the type of model (e.g. RandomForestRegressor())
                    paramter_grid: dictionary defining the gridsearch parameters
                    X_train: 2d numpy array
                    y_train: 1d numpy array

        Returns:  best parameters and model fit with those parameters
    '''
    model_gridsearch = GridSearchCV(estimator,
                                    parameter_grid,
                                    n_jobs=-1,
                                    verbose=True,
                                    )
    model_gridsearch.fit(X_train, y_train)
    best_params = model_gridsearch.best_params_
    model_best = model_gridsearch.best_estimator_
    """print ("\nResult of gridsearch:")
    print ("{0:<20s} | {1:<8s} | {2}".format("Parameter", "Optimal", "Gridsearch values"))
    print ("-" * 55)
    for param, vals in parameter_grid.iteritems():
        print ("{0:<20s} | {1:<8s} | {2}".format(str(param),
                                                str(best_params[param]),
                                                str(vals)))
    """
    return best_params, model_best




"""def display_default_and_gsearch_model_results(model_default, model_gridsearch,
                                              X_test, y_test):
    '''
        Parameters: model_default: fit model using initial parameters
                    model_gridsearch: fit model using parameters from gridsearch
                    X_test: 2d numpy array
                    y_test: 1d numpy array
        Return: None, but prints out mse and r2 for the default and model with
                gridsearched parameters
    '''
    name = model_default.__class__.__name__.replace('Classifier', '') # for printing
    y_test_pred = model_gridsearch.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    print ("Results for {0}".format(name))
    print ("Gridsearched model accuracy: {0:0.3f} | precision: {1:0.3f} | recall: {2:0.3f}".format(accuracy, precision, recall))
    y_test_pred = model_default.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    print ("     Default model accuracy: {0:0.3f} | precision: {1:0.3f} | recall: {2:0.3f}".format(accuracy, precision, recall))"""
