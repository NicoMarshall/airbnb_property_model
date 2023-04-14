import numpy as np
import joblib
from sklearn import datasets, model_selection, linear_model,metrics
from tabular_data import load_airbnb
import itertools


def makeGrid(my_dict):  
    keys=my_dict.keys()
    combinations=itertools.product(*my_dict.values())
    ds=[dict(zip(keys,cc)) for cc in combinations]
    return ds

def custom_tune_regression_hyperparameters(model_class, x_train, x_test, y_train,y_test, x_validation,y_validation, hyperparameter_dict):
    param_combinations = makeGrid(hyperparameter_dict)
    best_params = None
    best_rmse = 1000000000
    for i in param_combinations:
        model=  model_class(loss =i["loss"],penalty=i["penalty"],alpha=i["alpha"],learning_rate=i["learning_rate"])
        model.fit(x_train,y_train)
        y_pred = model.predict(x_validation)
        rmse_validation = metrics.mean_squared_error(y_pred,y_validation,squared = False)
        r_2_validation = metrics.r2_score(y_pred,y_validation)
        if rmse_validation < best_rmse:
            best_rmse = rmse_validation
            best_params = i
            r_2_score = r_2_validation
        else:
            pass
    hyperparameter_optimals = {"parameters": best_params,"rmse": best_rmse,"r_2": r_2_score}
    
    return hyperparameter_optimals
   
    
        
if __name__ == "__main__":
    features, labels = load_airbnb("clean_tabular_data.csv","Price_Night")
    x_train ,x_test ,y_train ,y_test = model_selection.train_test_split(features,labels, test_size=0.3)
    x_validation, x_test, y_validation,y_test = model_selection.train_test_split(x_test,y_test,test_size = 0.5)
    hyperparameter_dict = {"loss":["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],"penalty":["l2", "l1", "elasticnet",None],"alpha":[0.0001,0.0005,0.001],"shuffle":[True,False],"learning_rate":["constant","optimal","invscaling"]}
    optimals = custom_tune_regression_hyperparameters(linear_model.SGDRegressor,x_train ,x_test ,y_train ,y_test,x_validation,y_validation,hyperparameter_dict)
    print(optimals)
   


    
    