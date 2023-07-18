import numpy as np
import itertools
import os
import joblib
import json
from sklearn import datasets,model_selection,linear_model,metrics, ensemble
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from tabular_data import load_airbnb
from sklearn.exceptions import ConvergenceWarning 



def evaluate_model(model, x, y, label_list):
    y_pred = model.predict(x)
    accuracy_score = metrics.accuracy_score(y,y_pred)
    #confusion_matrix = metrics.confusion_matrix(y_pred, y, labels=label_list)
    recall_scores = metrics.recall_score(y,y_pred, average=None, labels= label_list)
    recall_scores = dict(zip(label_list,recall_scores))
    precision_scores = metrics.precision_score(y, y_pred, average = None, labels=label_list)
    precision_scores= dict(zip(label_list, precision_scores))
    f_1_score = metrics.f1_score(y, y_pred, average="macro")
    #scores =  dict(zip(["accuracy_score", "recall_scores", "precision_scores", "f_1_score"], accuracy_score, recall_scores, precision_scores, f_1_score))
    return accuracy_score, recall_scores, precision_scores, f_1_score


    
def makeGrid(my_dict):  
    keys=my_dict.keys()
    combinations=itertools.product(*my_dict.values())
    ds=[dict(zip(keys,cc)) for cc in combinations]
    return ds

def save_model(folder,model,best_params,best_metrics):
    os.chdir(folder)
    joblib.dump(model,"model.joblib")
    with open("hyperparameters.json", "w") as outfile:
        json.dump(best_params, outfile)
    with open("metrics.json", "w") as outfile:
        json.dump(best_metrics, outfile)
    os.chdir(root)
    
def tune_hyperparameters(model_class, features, labels, hyperparam_dict):
    hyperparam_combs = makeGrid(hyperparam_dict)
    best_hyperparams = None
    best_validation_accuracy = 0
    best_model = None
    best_model_validation_metrics = None
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    x_train ,x_test ,y_train ,y_test = model_selection.train_test_split(features,labels, test_size=0.3)
    x_validation, x_test, y_validation,y_test = model_selection.train_test_split(x_test,y_test,test_size = 0.5)
    for i in hyperparam_combs:
        try:
            model = model_class(**i)
            model.fit(x_train,y_train)
            mod_metrics = evaluate_model(model, x_validation, y_validation, label_list)
            if mod_metrics[0] > best_validation_accuracy:
                best_validation_accuracy = mod_metrics[0]
                best_hyperparams = i
                best_model = model
                best_model__validation_metrics = mod_metrics
            else:
                pass
        except:
            ValueError
            pass

    y_pred = best_model.predict(x_validation)
    best_model_test_metrics = evaluate_model(best_model, x_test, y_test, label_list)
    best_model_train_metrics = evaluate_model(best_model, x_train, y_train, label_list)
    best_model_metrics = {"Train Set": best_model_train_metrics , "Validation Set": best_model__validation_metrics, "Test Set": best_model_test_metrics}
    confusion_matrix = metrics.confusion_matrix(y_pred, y_validation, labels = label_list)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=label_list)
    disp.plot()
    plt.show()
    return best_model, best_hyperparams, best_model_metrics 
 
def find_best_model(task_folder:str):
    files = [file for file in os.listdir(task_folder)]
    best_hyperparams = None
    best_test_accuracy = 0
    best_model = None
    best_model_metrics = None
    for file in files:
        metrics = f"{task_folder}\\{file}\\metrics.json"
        with open(metrics) as json_file:
            data = json.load(json_file)
            test_data = data[0] 
            accuracy = test_data["Accuracy"]
        if accuracy > best_test_accuracy:
                best_test_accuracy = accuracy
                hyperparams = f"{task_folder}\\{file}\\hyperparameters.json"
                with open(hyperparams) as json_file:
                    hyper_dict = json.load(json_file)
                best_hyperparams = hyper_dict
                best_model = joblib.load(f"{task_folder}\\{file}\\model.joblib")
                best_model_metrics = data
        else:
            pass
    return best_hyperparams, best_model_metrics, best_model     
            
            
if __name__ =="__main__":
    features, labels = load_airbnb("clean_tabular_data.csv","Category")
    label_list = list(set(labels))
    root = os.getcwd()
    hyperparam_dict_logistic = {"penalty":["l1", "l2", "elasticnet", None], "dual": [True, False], "C" : [0.5,1,1.5,2], "solver":["liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], "max_iter": [500]}
    hyperparam_dict_random_forest = {"n_estimators":[50,100,150,200],"criterion":["gini", "entropy", "log_loss"],"max_depth":[None,10,20],"max_features":["sqrt", "log2", None]}
    hyperparam_dict_grad_boost = {"loss":["log_loss","deviance","exponential"],"learning_rate":[0.001,0.01,0.1],"n_estimators":[100,150,200]}
    model, best_hyperparams, best_model_metrics = tune_hyperparameters(ensemble.GradientBoostingClassifier, features, labels, hyperparam_dict_grad_boost)
    file_dest = f"{root}\\models\\classification\\gradient_boosting"
    save_model(file_dest, model, best_hyperparams, best_model_metrics )
    task_folder = f"{root}\\models\\classification"
    #find_best_model(task_folder)
    
    
  