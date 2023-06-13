import torch
import torch.nn as nn
import torch.nn.functional as F 
import pandas as pd
import numpy as np
from tabular_data import load_airbnb
from sklearn.model_selection import train_test_split

class nn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(11,1)
        
    def forward(self, features):    
        return self.layer(features)
    
    
class AirbnbNightlyPriceRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.features, self.labels = torch.from_numpy(np.array(x)).float(), torch.from_numpy(np.array(y)).float()
        
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)
 
 
def train(model_class, dataloader, num_epochs = 5): 
    model = model_class()
    for i in range(num_epochs):
        for batch in train_dataloader:
            features, labels = batch
            print(model(features))
            break
        break
    
    
if __name__ == "__main__":
    features, labels = load_airbnb("clean_tabular_data.csv","Price_Night")
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size= 0.2)
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size= 0.5)
    train_data = AirbnbNightlyPriceRegressionDataset(x_train, y_train)
    test_data = AirbnbNightlyPriceRegressionDataset(x_test, y_test)
    validation_data = AirbnbNightlyPriceRegressionDataset(x_validation, y_validation)
    #print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
    train (nn_model, train_dataloader)
              