import numpy as np
import joblib
from sklearn import datasets, model_selection, linear_model,metrics
from tabular_data import load_airbnb


model = linear_model.SGDRegressor()


if __name__ == "main":
    load_airbnb("clean_tabular_data.csv","price_night")
    