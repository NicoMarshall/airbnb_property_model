import itertools
import json
import joblib
import numpy as np
import os
import warnings
from sklearn import datasets, model_selection, linear_model,metrics, ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from tabular_data import load_airbnb


def makeGrid(my_dict):  
    """Generate a grid of combinations from a dictionary where all values are iterables eg lists"""
    keys=my_dict.keys()
    combinations=itertools.product(*my_dict.values())
    ds=[dict(zip(keys,cc)) for cc in combinations]
    return ds

def custom_tune_regression_hyperparameters(model_class, x_train, x_test, y_train,y_test, x_validation,y_validation, hyperparameter_dict):
    """custom grid search implementation. Not used in main code block,superceded by tune_regression_model_hyperparaemeters
    """
    param_combinations = makeGrid(hyperparameter_dict)
    best_params = None
    best_rmse = 1000000000
    for i in param_combinations:
        model = model_class(**i)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_validation)
        rmse_validation = metrics.mean_squared_error(y_pred,y_validation,squared = False)
        r_2_validation = metrics.r2_score(y_pred,y_validation)
        y_pred_train = model.predict(x_train)
        rmse_train = metrics.mean_squared_error(y_pred_train,y_train,squared=False)
        r_2_train = metrics.r2_score(y_pred_train,y_train)
        
        if rmse_validation < best_rmse:
            best_rmse = rmse_validation
            best_params = i
            r_2_score = r_2_validation
        else:
            pass
    hyperparameter_optimals = {"parameters": best_params,"rmse": best_rmse,"r_2": r_2_score}
    
    return hyperparameter_optimals
 
def tune_regression_model_hyperparameters(features,labels, model_class, hyperparameter_dict):
    """Uses sklearn's GridSearchCV to do a hyperparameter grid search

    Args:
        features (pd.Dataframe): features data
        labels (pd.Series): labels data
        model_class (class): sklearn regression class
        hyperparameter_dict (dict): dictionary of hyperparameters to try

    Returns:
        best_model and associated metrics ad hyperparameters
    """
    model = model_class()
    grid_search = GridSearchCV(model,hyperparameter_dict,scoring = "neg_root_mean_squared_error")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    grid_search.fit(features,labels)
    best_params = grid_search.get_params()
    best_params["estimator"] = str(best_params["estimator"])
    best_loss = np.abs(grid_search.best_score_)
    return grid_search,best_params,best_loss

def model_search(model_list: list, model_names:list, hyper_dicts: list):
    """Loops through model classes and performs hyperparameter grid search on each

    Args:
        model_list (list): list of model classes to try
        hyper_dicts (list): dictionary of hyperparameters to try for each model class
        model_names (list): list of model names as strings, which will be the folder names where they'll be saved
    """
    for model_class, model_name, hyperparams in zip(model_list, model_names, hyper_dicts):
        model, best_params, best_loss = tune_regression_model_hyperparameters(features, labels, model_class=model_class, hyperparameter_dict=hyperparams)
        save_model(folder=f"{root}\\models\\regression\\{model_name}", model=model, best_params=best_params, best_loss=best_loss)
    

def save_model(folder,model,best_params,best_loss):
    """Saves model and associated hyperparameters/metrics in file specified by folder.
    """
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    os.chdir(folder)
    joblib.dump(model,"model.joblib")
    with open("hyperparameters.json", "w") as outfile:
        json.dump(best_params, outfile)
    with open("metrics.json", "w") as outfile:
        json.dump(best_loss, outfile)
    os.chdir(root)
    
    
        
if __name__ == "__main__":
    root = os.getcwd()
    features, labels = load_airbnb("clean_tabular_data.csv","Price_Night")
    hyperparameter_dict_linear = {"loss":["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],"penalty":["l2", "l1", "elasticnet",None],"alpha":[0.0001,0.0005,0.001,0.1,0.001],"shuffle":[True,False],"learning_rate":["constant","optimal","invscaling"]}
    hyperparam_dict_random_forest = {"n_estimators":[50,100,150,200],"criterion":['absolute_error', 'poisson', 'friedman_mse', 'squared_error'],"max_depth":[None,10,20],"max_features":["sqrt", "log2", None]}
    hyperparam_dict_grad_boost = {"loss":['huber', 'squared_error', 'absolute_error', 'quantile'],"learning_rate":[0.001,0.01,0.1],"n_estimators":[100,150,200]}
    model_list = [linear_model.SGDRegressor, ensemble.GradientBoostingRegressor, ensemble.RandomForestRegressor]
    hyper_dicts = [hyperparameter_dict_linear, hyperparam_dict_grad_boost, hyperparam_dict_random_forest]
    model_names = ["Linear Regression", "Gradient Boosting", "Random Forest"]
    model_search(model_list=model_list, hyper_dicts=hyper_dicts)
    
   


    
    