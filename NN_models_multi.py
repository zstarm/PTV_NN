import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datautils import PTV_dataset
import numpy as np
import math

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
import os

def loss_mse(y_pred, y):
    return torch.mean((y-y_pred)**2 ) #+ torch.mean((y[0]-y_pred[0])**2)  #add extra weight for initial condition


def ddp_setup(rank, world_size):
    """
    Args:
        rank (_type_): unique identifer of each process
        world_size (_type_): total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, source, targets):
        '''
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = loss_mse(output, targets)
        loss.backward()
        '''
        def closure():
            self.optimizer.zero_grad()
            output = self.model(source)
            loss = loss_mse(output, targets)
            loss.backward()
            return loss
        
        self.optimizer.step(closure)

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[Device:{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = PTV_dataset("unfiltered4DPTV_velocity_subgrid_fourier_202815_042724.pth", "unfiltered4DPTV_features_subgrid_fourier_202815_042724.pth", "Training Data")  # load your dataset
    model = PTV_NN_multi(111, 4, 250)  # load your model
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=25,tolerance_grad=1e-200, tolerance_change=1e-200)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


#MODEL CLASS
class PTV_NN_multi(nn.Module):
    def __init__(self, num_terms,num_hidden_layers,hidden_size):
        super().__init__()
        
        #simple ANN architecture
        self.fourier_sin_layer = nn.Linear(1, num_terms, False)
        self.fourier_cos_layer = nn.Linear(1, num_terms, False)
        self.input_layer = nn.Linear(2+(2*num_terms), hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_size, 3)

        with torch.no_grad():
            for t in range(0,num_terms):
                self.fourier_cos_layer.weight[t] = nn.Parameter(torch.tensor(2 * t * math.pi))
                self.fourier_sin_layer.weight[t] = nn.Parameter(torch.tensor(2 * (t+1) * math.pi))
        
        #self.fourier_cos_layer.bias = nn.Parameter(torch.tensor(0.0))
        self.fourier_cos_layer.weight.requires_grad = False
        self.fourier_sin_layer.weight.requires_grad = False
                
    
    def forward(self, x):
        fc = torch.cos(self.fourier_cos_layer(x[:,2].reshape(-1,1)))
        fs = torch.sin(self.fourier_sin_layer(x[:,2].reshape(-1,1)))
        x = torch.cat((x[:,0:2], fc, fs), dim=1)
        
        x = torch.tanh(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        
        output = self.output_layer(x)
        return output
    

def scaleTensors(t):
    s = []
    for i in range(0,3):
        shift = torch.min(t[:,i]).item()
        s.append(shift)
        t[:,i] -= shift
        scale = torch.max(t[:,i]).item()
        s.append(scale)
        t[:,i] /= scale
    return t,s





def unscaleTensors(t,s):
    ut = torch.empty_like(t)
    for i in range(0,3):
        ut[:,i] = s[2*i+1]*t[:,i] + s[2*i]
    return ut
def gradScalings(x,v):
    s = []
    for i in range(0,3):
        for j in range(0,3):
            s.append(v[2*i+1]/x[2*j+1])
    return s

def write_data_file(fname, title, df, J, K, tf):
    f = open(fname, "w")
    index = 0
    f.write("TITLE = \"" + title + "\"\n")
    f.write("VARIABLES = Y,Z,U,V,W\n")
    for t in range(0,tf):
        zonestr = "Zone T = \"" + title + "\" SOLUTIONTIME = " +  str(t) + " I = " + str(J) + " J = " + str(K) + "\n"
        f.write(zonestr)
        for z in range(0,K):
            for y in range(0,J):
                linestr = str(df["Y"][index])+"\t"+str(df["Z"][index])+"\t"+str(df["U"][index])+"\t"+str(df["V"][index])+"\t"+str(df["W"][index])+"\n"
                f.write(linestr)       
                index += 1
