import numpy as np
import itertools
import os
import joblib
import json
from sklearn import datasets,model_selection,linear_model,metrics
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
    #print(f"Accuracy is:{accuracy_score}")
    confusion_matrix = metrics.confusion_matrix(y_pred, y_test, labels=label_list)
    #print(confusion_matrix)
    recall_scores = metrics.recall_score(y_test,y_pred, average=None, labels= label_list)
    recall_scores = dict(zip(label_list,recall_scores))
    #print(f"Recall scores are {recall_scores}")
    precision_scores = metrics.precision_score(y_test, y_pred, average = None, labels=label_list)
    precision_scores= dict(zip(label_list, precision_scores))
    #print(f"Precision scores are {precision_scores}")
    f_1_score = metrics.f1_score(y_test, y_pred, average="macro")
    #print(f"f_1 score is {f_1_score}")
    return accuracy_score, recall_scores, precision_scores, f_1_score
    
def makeGrid(my_dict):  
    keys=my_dict.keys()
    combinations=itertools.product(*my_dict.values())
    ds=[dict(zip(keys,cc)) for cc in combinations]
    return ds

def save_model(folder,model,best_params,best_accuracy):
    os.chdir(folder)
    joblib.dump(model,"model.joblib")
    with open("hyperparameters.json", "w") as outfile:
        json.dump(best_params, outfile)
    with open("metrics.json", "w") as outfile:
        json.dump(best_accuracy, outfile)
    os.chdir(f"C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model")
    
def tune_hyperparameters(model_class, features, labels, hyperparam_dict):
    hyperparam_combs = makeGrid(hyperparam_dict)
    best_hyperparams = None
    best_validation_accuracy = 0
    best_model = None
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
            else:
                pass
        except:
            ValueError
            pass
        
    return best_model, best_hyperparams, best_validation_accuracy   
        
if __name__ =="__main__":
    features, labels = load_airbnb("clean_tabular_data.csv","Category")
    label_list = list(set(labels))
    hyperparam_dict = {"penalty":["l1", "l2", "elasticnet", None], "dual": [True, False], "C" : [0.5,1,1.5,2], "solver":["liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], "max_iter": [500]}
    model, best_hyperparams, best_accuracy = tune_hyperparameters(linear_model.LogisticRegression, features, labels, hyperparam_dict)
    file_dest = f"C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model\\models\\classification\\logistic_regression"
    save_model(file_dest, model, best_hyperparams, best_accuracy )
    