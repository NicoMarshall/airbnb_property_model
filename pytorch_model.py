import torch
import torch.nn as nn
import torch.nn.functional as F 
import torcheval
from torcheval.metrics import MeanSquaredError,  R2Score
from torcheval.metrics.functional import r2_score
import pandas as pd
import numpy as np
from tabular_data import load_airbnb
from modelling import makeGrid
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import json
import datetime
import time
import itertools
import random

class nn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(11,30)
        self.layer2 = nn.BatchNorm1d(30)
        self.layer3 = nn.Linear(30,30)
        self.layer4 = nn.BatchNorm1d(30)
        self.layer5 = nn.Linear(30,1)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x =  F.relu(self.layer2(x)) 
        x =  F.relu(self.layer3(x))
        x =  F.relu(self.layer4(x))
        x =  F.relu(self.layer5(x))
        
        return x
    
    
class AirbnbNightlyPriceRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.features, self.labels = torch.from_numpy(np.array(x)).float(), torch.from_numpy(np.array(y)).float()
        
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)
 
 
def train(model_class, train_dataloader, val_dataloader, learn_rate = 0.1 , num_epochs = 5 ): 
    model = model_class()
    writer = SummaryWriter()
    optimiser = torch.optim.SGD(model.parameters(), lr = learn_rate)
    torch.manual_seed(0)
    for i in range(num_epochs):
        for features, labels in train_dataloader:
            pred = model(features).view([len(labels)])
            loss = torch.sqrt(F.mse_loss(pred, labels))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            writer.add_scalar("Training Loss", loss, i)
        #if i%5 == 0:    
            with torch.no_grad():
                val_loss = 0
                for features, labels in val_dataloader:
                        pred = model(features).view([len(labels)])
                        val_loss = torch.sqrt(F.mse_loss(pred, labels))
                #print(f"Loss in epoch#{i} was {val_loss}")
                writer.add_scalar("Validation Loss", val_loss, i)
        #else:
            #pass          
    return model

class nn_config(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        input_size = config["input_size"]
        for layer_config in config["layers"]:
            type = layer_config["type"]
            if type == "batchnorm":
                layers.append(nn.BatchNorm1d(input_size))
            else:
                width = layer_config["units"]
                activation = layer_config["activation"]
                if type == "dense":
                    layers.append(nn.Linear(input_size, width))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "softmax":
                    layers.append(nn.softmax)(dim =1)
                input_size = width
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def get_nn_config(config_file: yaml):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config

def generate_nn_config(n_hidden:int):
    config_layers = makeGrid({"type":["relu","batchnorm"],"units":[1,10,20],"activation":["relu","Softmax"]})
    config_lr_epochs = makeGrid({"learning_rate":[0.1, 0.01, 0.001],"epochs":[100]})
    layers= random.choices(config_layers, k=n_hidden)
    layers.append({"type":"dense","units":1,"activation":"relu"})
    hyperparams = random.choice(config_lr_epochs)
    hyperparams.update([("layers",layers),("input_size",11),("n_hidden",n_hidden),("optim_name","SGD"),("loss","mse_loss")]) 
    return hyperparams
    
def find_best_nn(num_combinations:int, n_hidden):
    for i in range(num_combinations):
        config = generate_nn_config(n_hidden)
        model, train_time = config_train(nn_config, config, train_dataloader, validation_dataloader)        
        metrics = eval_model(model, train_dataloader, validation_dataloader, test_dataloader, train_size)
        metrics["train time"] = train_time
        dest_folder="C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model\\models\\neural_networks\\regression"
        try:
            save_model(dest_folder,model,config,metrics)
        except FileExistsError:
            save_model(dest_folder,model,config,metrics,duplicate=True)

def config_train(model_class, config, train_dataloader, val_dataloader):
    start_time = time.perf_counter()
    model = model_class(config)
    optimiser = getattr(torch.optim, config["optim_name"])
    learning_rate = config["learning_rate"]
    optimiser = optimiser(model.parameters(), lr = learning_rate)
    epochs = config["epochs"]
    loss_function = getattr(F, config["loss"])
    for i in range(epochs):
        for features, labels in train_dataloader:
            pred = model(features).view([len(labels)])
            train_loss = torch.sqrt(loss_function(pred, labels))
            optimiser.zero_grad()
            train_loss.backward()
            optimiser.step()
            #writer.add_scalar("Training Loss", loss, i)
        """
        if i%5 == 0:    
            with torch.no_grad():
                val_loss = 0
                for features, labels in val_dataloader:
                        pred = model(features).view([len(labels)])
                        val_loss = torch.sqrt(loss_function(pred, labels))
                print(f"Loss in epoch#{i} was {val_loss}")
                #writer.add_scalar("Validation Loss", val_loss, i)
        else:
            pass
        """ 
    train_time = time.perf_counter() - start_time         
    return model, train_time
    
def eval_model(model,train_dataloader,validation_dataloader,test_dataloader,train_size):
    with torch.no_grad():
        loss = MeanSquaredError()
        r2_score = R2Score()
        inference_latency = 0
        for features, labels in train_dataloader:
            start_time = time.perf_counter()
            pred = model(features).view(len(labels))
            end_time = time.perf_counter()
            inference_latency += end_time-start_time
            loss.update(pred, labels)
            r2_score.update(pred, labels)
        train_loss = torch.sqrt(loss.compute())
        train_r2_score = r2_score.compute()
        inference_latency /= train_size
        for features, labels in test_dataloader:
            pred = model(features).view(len(labels))
            test_loss = torch.sqrt(F.mse_loss(pred, labels))
            test_r2_score = torcheval.metrics.functional.r2_score(pred, labels)
        for features, labels in validation_dataloader:
            pred = model(features).view(len(labels))
            validation_loss = torch.sqrt(F.mse_loss(pred, labels))
            validation_r2_score = torcheval.metrics.functional.r2_score(pred, labels)
        metrics = [train_loss, test_loss, validation_loss, train_r2_score, test_r2_score, validation_r2_score, inference_latency]
        metrics = np.array(metrics)
        metric_labels = ["train_loss", "test_loss", "validation_loss", "train_r2_score", "test_r2_score", "validation_r2_score", "inference_latency"]
        metric_dict = dict(zip(metric_labels,metrics))
        return metric_dict

def save_model(folder,model,config,metrics, duplicate = False):
    working_dir = os.getcwd()
    if isinstance(model, nn.Module):
        date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
        if duplicate:
            date_time = datetime.datetime.now().strftime("%H%M%S")
        os.chdir(folder)
        os.mkdir(date_time)
        os.chdir(date_time)
        torch.save(model,"model.pt")
        with open("hyperparameters.json", "w") as outfile:
            json.dump(config, outfile)
        with open("metrics.json", "w") as outfile:
            json.dump(metrics, outfile)
        os.chdir(working_dir)
    else:
        print("Not a Pytorch model")
        
      
    
if __name__ == "__main__":
    features, labels = load_airbnb("clean_tabular_data.csv","Price_Night")
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size= 0.2)
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size= 0.5)
    train_data = AirbnbNightlyPriceRegressionDataset(x_train, y_train)
    train_size = len(train_data)
    test_data = AirbnbNightlyPriceRegressionDataset(x_test, y_test)
    validation_data = AirbnbNightlyPriceRegressionDataset(x_validation, y_validation)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size = len(validation_data), shuffle = True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = len(test_data), shuffle = True)
    #train (nn_model, train_dataloader, validation_dataloader, num_epochs= 31, learn_rate=0.1)
    """
    config = get_nn_config("nn_config.yaml")
    model, train_time = config_train(nn_config, config, train_dataloader, validation_dataloader)        
    metrics = eval_model(model, train_dataloader, validation_dataloader, test_dataloader, train_size)
    metrics["train time"] = train_time
    dest_folder="C:\\Users\\nicom\\OneDrive\\Υπολογιστής\\airbnb_property_model\\models\\neural_networks\\regression"
    save_model(dest_folder,model,config,metrics)
    """
    find_best_nn(num_combinations=5, n_hidden = 20)