import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import math
import NN_models

from NN_models import scaleTensors, unscaleTensors, gradScalings, write_data_file

#Set Device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

feats = torch.load("Training Data/unfiltered4DPTV_features_subgrid_fourier_202815_042724.pth")
velocity = torch.load("Training Data/unfiltered4DPTV_velocity_subgrid_fourier_202815_042724.pth")  

velocity, velocity_scales = scaleTensors(velocity)
feats, feats_scales = scaleTensors(feats)

grid_ = feats
velocity_ = velocity

if device == torch.device('cuda:0'):
    print("Using GPU: coarse grid prediction")
    grid_ = grid_.cuda()
    #grid_trans = grid_trans.cuda()
    velocity_ = velocity_.cuda()
else:
    print("Using CPU: coarse grid prediction")


myNN = NN_models.PTV_NN_TRANS_v2(111, 4, 250).to(device)
path = "Model Parameters/unfiltered_model_4L_250N_fourier111_050324.pth"
myNN.load_state_dict(torch.load(path , map_location=torch.device(device)))


with torch.no_grad():
    pred = myNN(grid_)

percentError = torch.sqrt( torch.mean( (velocity_ - pred)**2 )/ torch.mean(velocity_**2) ) * 100
print("Final model percent error: ",percentError.item())

b = 100
timePred = pred.reshape(888,117,3).permute(1,2,0)
timeVel = velocity_.reshape(888,117,3).permute(1,2,0)
u_pred = timePred[b,0,:].cpu().numpy()
v_pred = timePred[b,1,:].cpu().numpy()
w_pred = timePred[b,2,:].cpu().numpy()
u_actual = timeVel[b,0,:].cpu().numpy()
v_actual = timeVel[b,1,:].cpu().numpy()
w_actual = timeVel[b,2,:].cpu().numpy()

time_vect = (feats.reshape(888,117,3).permute(1,2,0)[0,2,:].cpu().numpy()*feats_scales[5]+feats_scales[4])# / 444.2 * 1.531/3.048

fig, axes = plt.subplots(3, sharex=True, figsize = [10,10])
#U Velocity
axes[0].set_title("U Velocity")
axes[0].plot(time_vect ,(u_actual*velocity_scales[1]+velocity_scales[0]),'r')
axes[0].plot(time_vect ,(u_pred*velocity_scales[1]+velocity_scales[0]),'b')
axes[0].legend(["4DPTV/Measured", "NN Model/Prediction"])

#V Velocity
axes[1].set_title("V Velocity")
axes[1].plot(time_vect ,v_actual*velocity_scales[3]+velocity_scales[2],'r')
axes[1].plot(time_vect ,v_pred*velocity_scales[3]+velocity_scales[2],'b')

#W Velocity
axes[2].set_title("W Velocity")
axes[2].plot(time_vect ,w_actual*velocity_scales[5]+velocity_scales[4],'r')
axes[2].plot(time_vect ,w_pred*velocity_scales[5]+velocity_scales[4],'b')

plt.savefig("velocity(timeseries)_measured_v_model_comparison.png")


fine_res_grid = torch.load("Training Data/50x50_subgrid_scaled_fourier_042724.pth")
core_fine = torch.load("Training Data/core_sup_fine_time_resolution.pth")

if device == torch.device('cuda:0'):
    print("Using GPU: fine grid predictions")
    fine_res_grid = fine_res_grid.cuda()
    core_fine = core_fine.cuda()
else:
    print("Using CPU: fine grid predictions")

with torch.no_grad():
    fine_res_pred = myNN(fine_res_grid)
    core_velocity_fine = myNN(core_fine)
print("Finished with fine grid predictions")


time_vect_core = (core_fine[:,2].cpu().detach().numpy()*feats_scales[5]+feats_scales[4])
time_vect = (fine_res_grid.reshape(888,2500,3).permute(1,2,0)[0,2,:].cpu().detach().numpy()*feats_scales[5]+feats_scales[4])
u_core = core_velocity_fine[:,0].cpu().detach().numpy()
v_core = core_velocity_fine[:,1].cpu().detach().numpy()
w_core = core_velocity_fine[:,2].cpu().detach().numpy()

u_fine = fine_res_pred.reshape(888,2500,3).permute(1,2,0)[1582,0,:].cpu().detach().numpy()
v_fine = fine_res_pred.reshape(888,2500,3).permute(1,2,0)[1582,1,:].cpu().detach().numpy()
w_fine = fine_res_pred.reshape(888,2500,3).permute(1,2,0)[1582,2,:].cpu().detach().numpy()

plt.plot(time_vect,u_fine*velocity_scales[1]+velocity_scales[0],'ro')
plt.plot(time_vect_core ,u_core*velocity_scales[1]+velocity_scales[0],'b')
plt.axis([0, 100, 0.2, 1.4])
plt.legend(["Coarse (Original) Time Step", "Finer Time Step"])
plt.savefig("vortexCore_comparison_fineTime_vs_coarseTime_nnModels_0_100")

print("Writing results to data files")

#clone the grid tensors
out_grid_coarse = grid_.detach().clone()
out_grid_fine = fine_res_grid.detach().clone()
out_grid_core = core_fine.detach().clone()

#clone the velocity tensors
org_velocity = velocity_.detach().clone()
out_velocity_coarse = pred.detach().clone()
out_velocity_fine = fine_res_pred.detach().clone()
out_velocity_core = core_velocity_fine.detach().clone()

#recale the outputs
out_grid_coarse = unscaleTensors(out_grid_coarse, feats_scales)
out_grid_fine = unscaleTensors(out_grid_fine, feats_scales)
out_grid_core = unscaleTensors(out_grid_core, feats_scales)
out_velocity_coarse = unscaleTensors(out_velocity_coarse, velocity_scales)
out_velocity_fine = unscaleTensors(out_velocity_fine, velocity_scales)
out_velocity_core = unscaleTensors(out_velocity_core, velocity_scales)
org_velocity = unscaleTensors(org_velocity, velocity_scales)


fine_resolution = torch.cat((out_grid_fine.cpu(),out_velocity_fine.cpu()),1)
coarse_resolution = torch.cat((out_grid_coarse.cpu(),out_velocity_coarse.cpu()),1)
core = torch.cat((out_grid_core.cpu(), out_velocity_core.cpu()),1)
original_4DPTV = torch.cat((out_grid_coarse.cpu(),org_velocity.cpu()),1)

fine_df = pd.DataFrame(fine_resolution.numpy(), columns=["Y", "Z", "TIME", "U", "V","W"])
coarse_df = pd.DataFrame(coarse_resolution.numpy(), columns=["Y", "Z", "TIME", "U", "V","W"])
core_df = pd.DataFrame(core.numpy(), columns=["Y", "Z", "TIME", "U", "V","W"])
original_df = pd.DataFrame(original_4DPTV.numpy(), columns=["Y", "Z", "TIME", "U", "V","W"])

fgs = 50
write_data_file("original_data_050324.dat", "original 4DTPV data", original_df, 9, 13, 888)
write_data_file("coarse_grid_prediction_050324.dat", "coarse_grid_prediction", coarse_df, 9, 13, 888)
write_data_file("fine_grid_"+str(fgs)+"x"+str(fgs)+"_prediction_050324.dat", "fine_grid_prediction", fine_df, fgs, fgs, 888)
write_data_file("vorx_core_predicition_050324.dat", "vorx core NN velocity", core_df, 1, 1, 16384)

print("FINSIHED!")