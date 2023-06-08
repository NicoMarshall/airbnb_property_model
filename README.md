# airbnb_property_model
Training and evaluating multiple machine learning models on data from airbnbnb property listings in order to predict property features - price, category etc. Scripts written in Python, making use of various data science libraries for pre-processing and analysis; pandas, scikit-learn, matplotlib and PyTorch.

## Milestone 1: Data Cleaning
The first stage was cleaning and pre-processing the raw data to get it into the right format for ML. The raw property data in csv format was first loaded into a pandas dataframe. Then a script was written inside tabular_data.py to:
  * remove rows where ratings were missing 
  * format description strings 
  * set default values and split data into features, labels tuples 
  * save the cleaned data as a new csv file


Since the goal of this project is to train supervised learning models, we needed to be able to format the data into a features, labels tuple:
```
def load_airbnb(data, label: str):
    tab_df = pd.read_csv(data)
    labels = tab_df[label]
    tab_df = tab_df.drop(label, axis = 1)
    text_cols = tab_df.select_dtypes(include = object)
    tab_df = tab_df.drop(text_cols,axis=1).drop('Unnamed: 0',axis=1).drop('Unnamed: 19',axis=1)
    features = tab_df
    return features, labels
```
Additionally, a seperate script was written (prepare_image_data.py) to resize image data so that all images have the same height. Used the cv2 module for image augmentation.
 
## Milestone 2: Linear Regression modelling
The goal here was to train the best linear model to predict the nightly price of a property given its numerical features; beds, bedrooms, bathrooms, guests, cleanliness_rating, accuracy_rating, communication_rating, location_rating, check-in_rating, value_rating. Each model was an instance of the sklearn linear_model.SGDRegressor class, with a grid search implemented from scratch to find the best model hyperparameters;

```
def makeGrid(my_dict):  
    keys=my_dict.keys()
    combinations=itertools.product(*my_dict.values())
    ds=[dict(zip(keys,cc)) for cc in combinations]
    return ds #returns a list of all possible hyperparameter combinations

def custom_tune_regression_hyperparameters(model_class, x_train, x_test, y_train,y_test, x_validation,y_validation, hyperparameter_dict): 
    param_combinations = makeGrid(hyperparameter_dict)
    best_params = None
    best_rmse = np.inf
    for i in param_combinations: #loops through the combinations, returns the best performing model
        model=  model_class(**i)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_validation)
        rmse_validation = metrics.mean_squared_error(y_pred,y_validation,squared = False)
        r_2_validation = metrics.r2_score(y_pred,y_validation)
        y_pred_train = model.predict(x_train)
        rmse_train = metrics.mean_squared_error(y_pred_train,y_train,squared=False)
        r_2_train = metrics.r2_score(y_pred_train,y_train)
        
        if rmse_validation < best_rmse: #root mean squared error on validation set used as main metric of performance
            best_rmse = rmse_validation 
            best_params = i
            r_2_score = r_2_validation 
        else:
            pass
    hyperparameter_optimals = {"parameters": best_params,"rmse": best_rmse,"r_2": r_2_score,"rmse_train":rmse_train,"r_2_train":r_2_train,}
    
    return hyperparameter_optimals #returns dictionary of best hyperparameters and associated metrics
```
Once this was working well, I also made an alternative method to implement the same grid search using the in built GridSearchCV class of Sklearn;
```
    def tune_regression_model_hyperparameters(features,labels, hyperparameter_dict):
        model = linear_model.SGDRegressor(max_iter=1000)
        grid_search = GridSearchCV(model,hyperparameter_dict,scoring = "neg_root_mean_squared_error")
        grid_search.fit(features,labels)
        best_params = grid_search.get_params()
        best_params["estimator"] = str(best_params["estimator"])
        best_loss = np.abs(grid_search.best_score_)
        return grid_search,best_params,best_loss
    
```
The model and associated data were saved using the joblib module;
```
    def save_model(folder,model,best_params,best_loss):
        os.chdir(folder)
        joblib.dump(model,"model.joblib")
        with open("hyperparameters.json", "w") as outfile:
            json.dump(best_params, outfile)
        with open("metrics.json", "w") as outfile:
            json.dump(best_loss, outfile)
        os.chdir(f"C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model")
```
    
