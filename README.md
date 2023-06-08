# airbnb_property_model
Training and evaluating multiple machine learning models on data from airbnbnb property listings in order to predict property features - price, category etc. Scripts written in Python, making use of various data science libraries for pre-processing and ML; pandas, scikit-learn, matplotlib and PyTorch.

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
 
## Milestone 2: Linear Regression 
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
## Milestone 3: Classification 
Here we write a separate script (classification_modelling.py) for predicting the category each listing comes under eg. "chalets", "beachfront", "treehouse" etc, using the numerical data again as features. The strategy is to try three different model types from scikit-learn and compare their performance:

  * logistic regression (linear_model.LogisticRegression)
  * random forest (ensemble.RandomForestClassifier)
  * gradient boosting with decision trees as weak learners (ensemble.GradientBoostingClassifier)

As with regession modelling, a grid search is used on each model class to find the best hyperparameters and their performance metrics on the validation set. These are accuracy, precision, recall and f_1 score;
```
def evaluate_model(model, x_validation, y_validation, label_list):
    y_pred = model.predict(x_validation)
    accuracy_score = metrics.accuracy_score(y_validation, y_pred) #percent of correct predictions over all label classes
    confusion_matrix = metrics.confusion_matrix(y_pred, y_validation, labels=label_list) #optional visualisation 
    recall_scores = metrics.recall_score(y_validation, y_pred, average=None, labels= label_list) #list of recall scores for each label class
    recall_scores = dict(zip(label_list, recall_scores)) #zip into dictionary format for ease of comprehension
    precision_scores = metrics.precision_score(y_validation, y_pred, average = None, labels=label_list)
    precision_scores= dict(zip(label_list, precision_scores))
    f_1_score = metrics.f1_score(y_validation, y_pred, average="macro")
    
    return accuracy_score, recall_scores, precision_scores, f_1_score
```
The hyperparameter dictionaries used for the grid search are listed here below. Each value in the (key, value) pair is a list of the different hyperparameters to try:
```
hyperparam_dict_logistic = {"penalty":["l1", "l2", "elasticnet", None], "dual": [True, False], "C" : [0.5,1,1.5,2], "solver":["liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], "max_iter": [500]}
hyperparam_dict_random_forest = {"n_estimators":[50,100,150,200],"criterion":["gini", "entropy", "log_loss"],"max_depth":[None,10,20],"max_features":["sqrt", "log2", None]}
hyperparam_dict_grad_boost = {"loss":["log_loss","deviance","exponential"],"learning_rate":[0.001,0.01,0.1],"n_estimators":[100,150,200]}
```
