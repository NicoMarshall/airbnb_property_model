# airbnb_property_models
A project to train and evaluate different machine learning models for classification and regression on a dataset of details from 1000 property listings on the popular property rental website Airbnb. Linear regression and deep learning models predict the nightly price of a given property, while logistic regression, random forest and gradient boosting models predict a classification for the type of property. A hyperparameter grid search was used with each to find the best performer.
Scripts written in Python, making use of various data science libraries for pre-processing and machine learning;
  * Pandas
  * Numpy
  * Scikit-learn
  * Matplotlib
  * PyTorch

## Milestone 1: Data Cleaning
The first stage is cleaning and pre-processing the raw data to get it into the right format for ML. The raw property data in csv format was first loaded into a pandas dataframe. Then a script was written inside tabular_data.py to:
  * remove rows where ratings were missing 
  * format description strings 
  * set default values and split data into features, labels tuples 
  * save the cleaned data as a new csv file


Since the goal is to train supervised learning models, we need to be able to format the data into a (features, labels) tuple. The following method 
loads the (cleaned) data from a csv file and does the reformatting:
```
def load_airbnb(data, label: str):
    # load in the data
    features = pd.read_csv(data)
    # drop label column from features dataframe
    labels = features[label]
    features = features.drop(label, axis = 1)
    # remove text columns, since our models will only use numeric data
    text_cols = features.select_dtypes(include = object)
    features = features.drop(text_cols,axis=1).drop('Unnamed: 0',axis=1).drop('Unnamed: 19',axis=1)
    
    return features, labels
```
Additionally, a seperate script was written (prepare_image_data.py) to resize image data so that all images have the same height. Used the cv2 module for image augmentation.

We can now do some exploratory analysis on the cleaned data, to get a feel for how it looks. Our proposed target variable, price per night, is distributed as per the histogram below;

![price_night_hist](https://github.com/NicoMarshall/airbnb_property_model/assets/109066030/500cf223-4669-4a25-b257-e3ca2e9a6aec)

With a mean of £154, and a standard deviation of £129, we can see that the price exhibits slight right skewness.

The features data, meanwhile, is as follows:

![features_hists](https://github.com/NicoMarshall/airbnb_property_model/assets/109066030/51cc2fed-324f-4d93-be01-0ca0e66dd74e)

We can see that many of the variables are extremely concentrated in a small range, suggesting that these features may be of little predictive power.

## Milestone 2: Linear Regression 
The goal here was to train the best linear model to predict the nightly price of a property given its numerical features; beds, bedrooms, bathrooms, guests, cleanliness_rating, accuracy_rating, communication_rating, location_rating, check-in_rating, value_rating. Each model was an instance of the sklearn linear_model.SGDRegressor class, with a grid search implemented from scratch to find the best model hyperparameters;

```
def makeGrid(my_dict):  
    keys=my_dict.keys()
    combinations=itertools.product(*my_dict.values())
    ds=[dict(zip(keys,cc)) for cc in combinations]

    # return a list of dictionaries all possible hyperparameter combinations
    return ds 

def custom_tune_regression_hyperparameters(model_class, x_train, x_test, y_train,y_test, x_validation,y_validation, hyperparameter_dict): 
    param_combinations = makeGrid(hyperparameter_dict)
    best_params = None
    best_rmse = np.inf
    # loop through the possible combinations, return the best performing model
    for i in param_combinations: 
        model=  model_class(**i)
        # fit model to training set
        model.fit(x_train,y_train)
        y_pred = model.predict(x_validation)
        rmse_validation = metrics.mean_squared_error(y_pred,y_validation,squared = False)
        r_2_validation = metrics.r2_score(y_pred,y_validation)
        y_pred_train = model.predict(x_train)
        rmse_train = metrics.mean_squared_error(y_pred_train,y_train,squared=False)
        r_2_train = metrics.r2_score(y_pred_train,y_train)
        # root mean squared error on validation set used as main metric of performance
        if rmse_validation < best_rmse: 
            best_rmse = rmse_validation 
            best_params = i
            r_2_score = r_2_validation 
        else:
            pass
    hyperparameter_optimals = {"parameters": best_params,"rmse": best_rmse,"r_2": r_2_score}

    # return dictionary of best hyperparameters and associated metrics
    return hyperparameter_optimals 
```
Once this was working well, an alternative method was also used to implement the same grid search using the in built GridSearchCV class of Sklearn;
```
    def tune_regression_model_hyperparameters(features, labels, hyperparameter_dict):
        model = linear_model.SGDRegressor(max_iter=1000)
        grid_search = GridSearchCV(model,hyperparameter_dict,scoring = "neg_root_mean_squared_error")
        grid_search.fit(features,labels)
        best_params = grid_search.get_params()
        best_params["estimator"] = str(best_params["estimator"])
        best_loss = np.abs(grid_search.best_score_)
        
        return grid_search, best_params, best_loss
    
```
The model and associated data were saved using the joblib module;
```
    def save_model(folder, model, best_params, best_loss):
        working_dir = os.getcwd()
        # switch directory to the folder where we want to save model data
        os.chdir(folder)
        joblib.dump(model,"model.joblib")
        with open("hyperparameters.json", "w") as outfile:
            json.dump(best_params, outfile)
        with open("metrics.json", "w") as outfile:
            json.dump(best_loss, outfile)
        os.chdir(working_dir)
```
The best model found had an rmse value of £105 - a poor level of accuracy.

## Milestone 3: Classification 
Here we write a separate script (classification_modelling.py) for predicting the category each listing comes under eg. "chalets", "beachfront", "treehouse" etc, using the numerical data again as features. The strategy is to try three different model types from scikit-learn and compare their performance:

  * logistic regression (linear_model.LogisticRegression)
  * random forest (ensemble.RandomForestClassifier)
  * gradient boosting with decision trees as weak learners (ensemble.GradientBoostingClassifier)

As with regession modelling, a grid search is used on each model class to find the best hyperparameters and their performance metrics on the test set. These are accuracy, precision, recall and f_1 score;
```
def evaluate_model(model, x_test, y_test, label_list):
    y_pred = model.predict(x_test)
    # percent of correct predictions over all label classes
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    # generate confusion matrix as a useful visualisation
    confusion_matrix = metrics.confusion_matrix(y_pred, y_test, labels = label_list)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=label_list)
    disp.plot()
    plt.show()
    # list of recall scores for each label class
    recall_scores = metrics.recall_score(y_test, y_pred, average=None, labels= label_list) 
    # zip into dictionary format for ease of comprehension
    recall_scores = dict(zip(label_list, recall_scores)) #zip into dictionary format for ease of comprehension
    precision_scores = metrics.precision_score(y_test, y_pred, average = None, labels=label_list)
    precision_scores= dict(zip(label_list, precision_scores))
    f_1_score = metrics.f1_score(y_test, y_pred, average="macro")
    
    return accuracy_score, recall_scores, precision_scores, f_1_score
```
Overall the best model found, with an accuracy and f_1 score of 45% and 0.41 respectively (on test set), was a random forest with hyperparameters:
  * 200 estimators
  * max_depth = null
  * criterion = log_loss
  * max_features = log2

It should be noted that the best peforming logistic regression and gradient boosted models performed only slightly worse. It appears that 50% accuracy
on unseen data is an upper ceiling that can't be breached by the techniques tried here, at least with these 11 variables as our features. 

The confusion matrix for this model on the test set is shown here: 
![Figure_1](https://github.com/NicoMarshall/airbnb_property_model/assets/109066030/2cd6af91-a8dd-4493-a40f-51bcce47e00e)


The hyperparameter dictionaries used for the grid search are listed here below. Each value in the (key, value) pair is a list of the different hyperparameters to try:
```
hyperparam_dict_logistic = {"penalty":["l1", "l2", "elasticnet", None], "dual": [True, False], "C" : [0.5,1,1.5,2], "solver":["liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], "max_iter": [500]}
hyperparam_dict_random_forest = {"n_estimators":[50,100,150,200],"criterion":["gini", "entropy", "log_loss"],"max_depth":[None,10,20],"max_features":["sqrt", "log2", None]}
hyperparam_dict_grad_boost = {"loss":["log_loss","deviance","exponential"],"learning_rate":[0.001,0.01,0.1],"n_estimators":[100,150,200]}
```
## Milestone 4: Deep Learning
Here we use the Pytorch deep learning library to train multiple neural networks that predict the nightly price of a listing, as with the regression models in Milestone 1. The full Python script can be found in pytorch_model.py.

The first step is to load the data from a pandas dataframe into batches of Pytorch tensors; the required format. For this a data class is created that inherits from torch.utils.data.Dataset. This will later be used to create separate training, validation and test sets.
```
class AirbnbNightlyPriceRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        # init parent class
        super().__init__()
        # create features, labels tensor tuple
        self.features, self.labels = torch.from_numpy(np.array(x)).float(), torch.from_numpy(np.array(y)).float()
        
    def __getitem__(self, index):
        # return single datapoint
        return self.features[index], self.labels[index]
    
    def __len__(self):
        # size of dataset
        return len(self.labels)

# init train dataset
train_data = AirbnbNightlyPriceRegressionDataset(x_train, y_train)
 
```
The in-built torch Dataloader class is used to batch the data, ready to be ingested by a deep learning model:
```
 train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
```
We are now ready to begin experimenting with training different models. For the first attempt, we configure the hyperparameters 
of a neural network in a YAML file (nn_config.yaml). This describes a model with three hidden linear layers and relu activation functions;
```
network:
  name: nn_model
  # number of features for each datapoint
  input_size: 11 
  layers:
    - type: dense
      units: 20
      activation: relu
    - type: dense
      units: 10
      activation: relu
    - type: dense
      # final output needs to be a single scalar
      units: 1
      activation: relu
```
The optimiser type for backpropagation (stochastic gradient descent) and loss function (rmse) are also specified in this file.

We now need to create a model class that takes this  dictionary configuration as input and initialises a torch model. As is standard,
it inherits from the base nn.Module class. We use two types of layers; linear and batchnorm. 
```
class nn_config(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        input_size = config["input_size"]
        # loop through all layers and store in list
        for layer_config in config["layers"]:
            type = layer_config["type"]
            if type == "batchnorm":
                layers.append(nn.BatchNorm1d(input_size))
            else:
                # specify number of nodes in layer 
                width = layer_config["units"]
                activation = layer_config["activation"]
                if type == "dense":
                    layers.append(nn.Linear(input_size, width))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "softmax":
                    layers.append(nn.softmax)(dim =1)
                input_size = width
        # concatenate layers, activations into a callable function
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
```
All that's needed now is a method that will loop through the entire training dataset for the desired
number of epochs, and update the model parameters for each batch in the dataloader. The SummaryWriter tool from Tensorboard is useful
for visualising how model performance  varies with training time; every epoch on training set and every third epoch on validation we record the loss and store for later use.
```
def config_train(model_class, config, train_dataloader, val_dataloader):
    # record training time as a useful metric
    start_time = time.perf_counter()
    # init model
    model = model_class(config)
    optimiser = getattr(torch.optim, config["optim_name"])
    learning_rate = config["learning_rate"]
    optimiser = optimiser(model.parameters(), lr = learning_rate)
    epochs = config["epochs"]
    loss_function = getattr(F, config["loss"])
    writer = SummaryWriter()
    # loop through epochs
    for i in range(epochs):
        # loop through training set
        for features, labels in train_dataloader:
            # put model in training mode
            model.train()
            # call model on features, then re-size the tensor to the same dimensions as labels
            pred = model(features).view([len(labels)])
            # calculate rmse loss
            train_loss = torch.sqrt(loss_function(pred, labels))
            # backpropagation and gradient updates 
            optimiser.zero_grad()
            train_loss.backward()
            optimiser.step()
            writer.add_scalar("Training Loss", train_loss, i)

        # store validation loss every third epoch
        if i%3 == 0:
            # put model in evaluation mode   
            model.eval() 
            with torch.no_grad():
                
                val_loss = 0
                for features, labels in val_dataloader:
                        pred = model(features).view([len(labels)])
                        val_loss = torch.sqrt(loss_function(pred, labels))
                writer.add_scalar("Validation Loss", val_loss, i)
        else:
            pass
        
    train_time = time.perf_counter() - start_time         
    return model, train_time
```
Trying this a few times and tweaking the configurations lead to a best model with a rmse on the validation set of around £150;
much higher still than the best linear regression models. To try and improve on this we can modify the approach from previous 
milestones and implement a grid search to try find the best model hyperparameters. In other words, we train multiple models each time varying
the number of hidden layers, layer types, layer widths, activation functions, epochs and learning rate. The details for the implementation of this can be
found inside the following methods: 
  * generate_nn_config
  * find_best_nn
  * eval_model
  * save_model
   
The best performing model was selected based on the test set, and had the following hyperparameters:
  * learning_rate: 0.05 
  * training epochs: 50
  * number of layers : 11 (2 linear, 9 batchnorm)

The following schematic provides an overview of the model architecture; 
![nn](https://github.com/NicoMarshall/airbnb_property_model/assets/109066030/086ca6ee-5be4-465e-ae1f-a8f9ecdd6a13)

(tool source: https://alexlenail.me/NN-SVG/index.html)
 
Metrics:
  * training loss / R2 score: 101.59 / 0.40
  * validation loss / R2 score: 91.84 / 0.19
  * test loss / R2 score: 97.55 / 0.43

As we see, the grid search has yielded a better model - but still only one that is very similar in performance to the best linear regression model
from Milestone 1. We can observe that the smoothed loss on the training and validation sets plateaus at around £100; still a poor performance: 

![training_loss](https://github.com/NicoMarshall/airbnb_property_model/assets/109066030/763c3606-9e21-440f-894d-53d105c49148)
![validation_loss](https://github.com/NicoMarshall/airbnb_property_model/assets/109066030/1e7302d5-13ab-451e-86bf-ea2318078f9a)

## Conclusions
The overall conclusion looking across all the best models, and their relatively low predicitive power, is that the features used in these 
models are poorly correlated with price. A suggested explanation is that when most customers rate their stay, they do so without much thought and (so long as
they were at least relatively satisfied) assign ratings of 5 stars by default. This can be seen in the very low variance of the features histograms in Milestone 1. Furthermore, features such as bed and guest numbers might not be by themselves of much predictive use since the "density" and"quality" of these variables aren't captured. For example, a large mansion that fits two is likely to be more expensive than a cramped bunkhouse that fits 5, but our model might see the higher number of guests and predict a higher price. Thus for any future atempts to study this further, it might be worth changing the label to price per night per guest, and gathering the size of the property (eg in square feet) as another feature. 
