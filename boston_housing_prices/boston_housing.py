# Import libraries necessary for this project
import numpy as np
import utils
from sklearn.model_selection import train_test_split


# Import supplementary visualizations code visuals.py
import visuals as vs

"""Data load into data structures
"""



if __name__ == "__main__":
    features, prices = utils.loadData()
    # print(features)
    # print(prices)

    #Stats of data
    # Mean mode medium of data
    utils.print_stats(prices)

    utils.r2_performance_metric_example()

    X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)
    print("Training and testing split was successful.")

    utils.visulaize_learning_curves(features, prices)
    utils.visulaize_model_complexity_curves(X_train, y_train)

    # Fit the training data to the model using grid search
    decision_tree_reg = utils.fit_model(np.asarray(X_train), np.asarray(y_train))

    # # Produce the value for 'max_depth'
    print("Parameter 'max_depth' is {} for the optimal model.".format(decision_tree_reg.get_params()['max_depth']))

    features_set_to_predict = [[5, 17, 15],  # Client 1
                               [4, 32, 22],  # Client 2
                               [8, 3, 12]]  # Client 3
    utils.predict_house_price(decision_tree_reg, features_set_to_predict)
    # Produce a matrix for client data

    vs.PredictTrials(features, prices, utils.fit_model, features_set_to_predict)




