import numpy as np
import joblib
from sklearn import datasets, model_selection, linear_model,metrics
from tabular_data import load_airbnb


#model = linear_model.SGDRegressor() #doesn't work
model = linear_model.LinearRegression()

if __name__ == "__main__":
    features, labels = load_airbnb("clean_tabular_data.csv","Price_Night")
    x_train ,x_test ,y_train ,y_test = model_selection.train_test_split(features,labels, test_size=0.3)
    model.fit(x_train, y_train)
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    rmse_skl_train = metrics.mean_squared_error(y_pred_train, y_train, squared = False)
    rmse_skl_test = metrics.mean_squared_error(y_pred_test, y_test, squared = False)
    print("test loss",rmse_skl_test)
    print("training loss",rmse_skl_train)
    r_2_training = metrics.r2_score(y_train, y_pred_train)
    r_2_test = metrics.r2_score(y_test, y_pred_test)
    print("r_2_training",r_2_training)
    print("r_2_test",r_2_test)



    
    