import numpy as np
import joblib
from sklearn import datasets, model_selection, linear_model,metrics
from tabular_data import load_airbnb


model = linear_model.SGDRegressor()
#model = linear_model.LinearRegression()

if __name__ == "__main__":
    features, labels = load_airbnb("clean_tabular_data.csv","Price_Night")
    model.fit(features, labels)
    print(model.predict(features)[0:10])
    print(labels[0:10])