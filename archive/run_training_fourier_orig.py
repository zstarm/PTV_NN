import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import math
import NN_models

from NN_models import scaleTensors, unscaleTensors, gradScalings, write_data_file, fourier_transform_time

# define a mean squared error loss function -> NN regression of the coarse data
def loss_mse(x,y,NN):
    y_pred = NN(x)
    return torch.mean((y-y_pred)**2 ) #+ torch.mean((y[0]-y_pred[0])**2)  #add extra weight for initial condition

def closure():
    optimizer.zero_grad()
    l = loss_mse(grid_, velocity_, myNN)
    myNN.error = torch.sqrt(l).item()
    l.backward()

    return l

#Set Device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.autograd.set_detect_anomaly(True)
#device = "cpu"
torch.set_default_dtype(torch.double)
device

feats = torch.load("Training Data/unfiltered4DPTV_features_subgrid_fourier_202815_042724.pth")
velocity = torch.load("Training Data/unfiltered4DPTV_velocity_subgrid_fourier_202815_042724.pth")  

velocity, velocity_scales = scaleTensors(velocity)
feats, feats_scales = scaleTensors(feats)

grid_ = feats
velocity_ = velocity

if device == torch.device('cuda:0'):
    print("Using GPU")
    grid_ = grid_.cuda()
    #grid_trans = grid_trans.cuda()
    velocity_ = velocity_.cuda()
else:
    print("Using CPU")


myNN = NN_models.PTV_NN_TRANS_v2(111, 2, 250).to(device)

#path = "Model Parameters/unfiltered_model_2L_200N_fourier444_050124.pth"
#myNN.load_state_dict(torch.load(path , map_location=torch.device(device)))

optimizer = torch.optim.LBFGS(myNN.parameters(), lr=1, max_iter=25,tolerance_grad=1e-200, tolerance_change=1e-200)

prev_error = 0
epochs = 1001
tol = 0.0001
skip = 10
print("STARTING TRAINING: ")
for i in range(epochs):
    optimizer.step(closure)
    change = abs(prev_error - myNN.error)
    if not (i%skip):
        print("Iteration: ", i+1)
        print("Change in Error: ", change)
        print("Current Error: ",myNN.error,"\n")
    prev_error = myNN.error
    if myNN.error <= tol:
        break;
    #if change <= 1e-8:
        #break;
        
print("FINISHED")

with torch.no_grad():
    pred = myNN(grid_)

percentError = torch.sqrt( torch.mean( (velocity_ - pred)**2 )/ torch.mean(velocity_**2) ) * 100
print("%Error: ",percentError.item())

torch.save(myNN.state_dict(), "Model Parameters/unfiltered_model_2L_250N_fourier111_050324.pth")
