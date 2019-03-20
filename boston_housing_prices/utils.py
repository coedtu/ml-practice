import pandas as pd
import sklearn
from sklearn.metrics import r2_score

import numpy as np
import visuals as vs

from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV


def loadData():
    # Load the Boston housing dataset
    data = pd.read_csv('housing.csv')
    prices = data['MEDV']
    features = data.drop('MEDV', axis=1)
    print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))
    return features, prices

def print_stats(prices):
    minimum_price = np.min(prices)
    maximum_price = np.max(prices)
    mean_price = np.mean(prices)
    median_price = np.median(prices)
    std_price = np.std(prices)

    # Show the calculated statistics
    print("Statistics for Boston housing dataset:\n")
    print("Minimum price: ${}".format(minimum_price))
    print("Maximum price: ${}".format(maximum_price))
    print("Mean price: ${}".format(mean_price))
    print("Median price ${}".format(median_price))
    print("Standard deviation of prices: ${}".format(std_price))


def r2_performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    score = r2_score(y_true, y_predict)

    return score




def r2_performance_metric_example():
    # Calculate the performance of this model
    score = r2_performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
    print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))


def visulaize_learning_curves(features, prices):
    vs.ModelLearning(features, prices)


def visulaize_model_complexity_curves(X_train, y_train):
    vs.ModelComplexity(X_train, y_train)

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor(random_state=0)

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(r2_performance_metric)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    grid = GridSearchCV(regressor, params, cv=cv_sets, scoring=scoring_fnc)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


def predict_house_price(reg, features_set_to_predict):
    # Show predictions
    for i, price in enumerate(reg.predict(features_set_to_predict)):
        print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i + 1, price))

