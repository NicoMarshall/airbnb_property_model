import torch
import torch.nn as nn
import torch.nn.functional as F 
import pandas as pd
import numpy as np
from tabular_data import load_airbnb
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

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
    
if __name__ == "__main__":
    features, labels = load_airbnb("clean_tabular_data.csv","Price_Night")
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size= 0.2)
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size= 0.5)
    train_data = AirbnbNightlyPriceRegressionDataset(x_train, y_train)
    test_data = AirbnbNightlyPriceRegressionDataset(x_test, y_test)
    validation_data = AirbnbNightlyPriceRegressionDataset(x_validation, y_validation)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size = len(validation_data), shuffle = True)
    train (nn_model, train_dataloader, validation_dataloader, num_epochs= 31, learn_rate=0.1)
              