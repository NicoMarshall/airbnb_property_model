import numpy as np
import itertools
import os
import joblib
import json
from sklearn import datasets,model_selection,linear_model,metrics, ensemble
import matplotlib.pyplot as plt
import pandas as pd
from tabular_data import load_airbnb

def create_train_model(features,labels):
    model = linear_model.LogisticRegression(solver="newton-cg")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features,labels,test_size=0.3)
    model.fit(x_train,y_train)
    return model, x_test, y_test

def evaluate_model(model,x_test, y_test, label_list):
    y_pred = model.predict(x_test)
    accuracy_score = metrics.accuracy_score(y_test,y_pred)
    confusion_matrix = metrics.confusion_matrix(y_pred, y_test, labels=label_list)
    recall_scores = metrics.recall_score(y_test,y_pred, average=None, labels= label_list)
    recall_scores = dict(zip(label_list,recall_scores))
    precision_scores = metrics.precision_score(y_test, y_pred, average = None, labels=label_list)
    precision_scores= dict(zip(label_list, precision_scores))
    f_1_score = metrics.f1_score(y_test, y_pred, average="macro")
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
    os.chdir(f"C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model")
    
def tune_hyperparameters(model_class, features, labels, hyperparam_dict):
    hyperparam_combs = makeGrid(hyperparam_dict)
    best_hyperparams = None
    best_validation_accuracy = 0
    best_model = None
    best_model_metrics = None
    for i in hyperparam_combs:
        try:
            x_train ,x_test ,y_train ,y_test = model_selection.train_test_split(features,labels, test_size=0.3)
            x_validation, x_test, y_validation,y_test = model_selection.train_test_split(x_test,y_test,test_size = 0.5)
            model = model_class(**i)
            model.fit(x_train,y_train)
            metrics = evaluate_model(model, x_validation,y_validation, label_list)
            if metrics[0] > best_validation_accuracy:
                best_validation_accuracy = metrics[0]
                best_hyperparams = i
                best_model = model
                best_model_metrics = metrics
            else:
                pass
        except:
            ValueError
            pass
        
    return best_model, best_hyperparams, best_model_metrics 
 
def find_best_model(task_folder:str):
    files = [file for file in os.listdir(task_folder)]
    best_hyperparams = None
    best_validation_accuracy = 0
    best_model = None
    best_model_metrics = None
    for file in files:
        metrics = f"{task_folder}\\{file}\\metrics.json"
        with open(metrics) as json_file:
            data = json.load(json_file)
            accuracy = data["accuracy_score"] 
        if accuracy > best_validation_accuracy:
                best_validation_accuracy = accuracy
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
    hyperparam_dict_logistic = {"penalty":["l1", "l2", "elasticnet", None], "dual": [True, False], "C" : [0.5,1,1.5,2], "solver":["liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], "max_iter": [500]}
    hyperparam_dict_random_forest = {"n_estimators":[50,100,150,200],"criterion":["gini", "entropy", "log_loss"],"max_depth":[None,10,20],"max_features":["sqrt", "log2", None]}
    hyperparam_dict_grad_boost = {"loss":["log_loss","deviance","exponential"],"learning_rate":[0.001,0.01,0.1],"n_estimators":[100,150,200]}
    model, best_hyperparams, best_model_metrics = tune_hyperparameters(ensemble.RandomForestClassifier, features, labels, hyperparam_dict_random_forest)
    best_model_metrics = dict(zip(["accuracy_score", "recall_scores", "precision_scores", "f_1_score"],best_model_metrics))
    file_dest = f"C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model\\models\\classification\\random_forest"
    save_model(file_dest, model, best_hyperparams, best_model_metrics )
    task_folder = f"C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model\\models\\classification"
    find_best_model(task_folder)
    
    