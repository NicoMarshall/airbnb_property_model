import numpy as np
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
    print(f"Accuracy is:{accuracy_score}")
    confusion_matrix = metrics.confusion_matrix(y_pred, y_test, labels=label_list)
    print(confusion_matrix)
    recall_scores = metrics.recall_score(y_test,y_pred, average=None, labels= label_list)
    recall_scores = dict(zip(label_list,recall_scores))
    print(f"Recall scores are {recall_scores}")
    precision_scores = metrics.precision_score(y_test, y_pred, average = None, labels=label_list)
    precision_scores= dict(zip(label_list, precision_scores))
    print(f"Precision scores are {precision_scores}")
    f_1_score = metrics.f1_score(y_test, y_pred, average="macro")
    print(f"f_1 score is {f_1_score}")
    
if __name__ =="__main__":
    features, labels = load_airbnb("clean_tabular_data.csv","Category")
    label_list = list(set(labels))
    classification_model, x_test, y_test = create_train_model(features,labels)
    evaluate_model(classification_model, x_test, y_test, label_list)
    