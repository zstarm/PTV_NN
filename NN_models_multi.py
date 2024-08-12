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
    return torch.mean((y-y_pred)**2 )  #add extra weight for initial condition


def ddp_setup(rank, world_size):
    """
    Args:
        rank (_type_): unique identifer of each process
        world_size (_type_): total number of processes
    """
    #os.environ["MASTER_ADDR"] = "localhost"
    #os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl") #, rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int, 
        snapshot_path = str,
    ) -> None:
        self.dev_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.dev_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        if(os.path.exists(snapshot_path):
           print("Loading snapsho")
           self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.dev_id])

    def _load_snapshot(self, snapshot_path):
           torch.load(snapshot_path)
           self.model.load_state_dict(snapshot["MODEL_STATE"])
           self.epochs_run = snapshot["EPOCHS_RUN"]
           print(f"Resuming Training from snapshot at epoch {self.epochs_run}")
        

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

    def _save_snapshot(self, epoch):
        snapshot = {}
        shapshot["MODEL_STATE"]  = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, "snapshot.pt")
        print(f"Epoch {epoch} | Training snapshot saved at snapshot.pt")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = PTV_dataset("unfiltered4DPTV_velocity_subgrid_fourier_202815_042724.pth", "unfiltered4DPTV_features_subgrid_fourier_202815_042724.pth", "Training Data")  # load your dataset
    model = PTV_NN_multi(222, 10, 3, 24)  # load your model
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


#CUSTOM LAYERS
class fourier_basis_function(nn.Module):
    def __init__(self, size_in: int, num_terms: int, phase_lag: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype':dtype}
        super().__init__()
        if (num_terms % 2):
            print("Odd number of terms. Using N-1 terms instead")
            num_terms = num_terms-1
        
        self.in_features = size_in
        self.out_features = num_terms
        self.freq = nn.Parameter(torch.empty((2, math.floor(num_terms/2)), **factory_kwargs))
        self.amp = nn.Parameter(torch.empty((1,num_terms * size_in), **factory_kwargs))
        if phase_lag:
            self.phase = nn.Parameter(torch.empty((2, size_in * math.floor(num_terms/2)), **factory_kwargs))
        else:
            self.register_parameter('phase', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for t in range(0,math.floor(self.out_features/2)):
                self.freq[0,t] = nn.Parameter(torch.tensor(2 * t * math.pi))
                self.freq[1,t] = nn.Parameter(torch.tensor(2 * (t+1) * math.pi))
        
        self.freq.requires_grad = False
        
        nn.init.kaiming_uniform_(self.amp, a=math.sqrt(5))
        #nn.init.uniform_(self.amp,1,1)

        if self.phase is not None:
            fan_in, _  = nn.init._calculate_fan_in_and_fan_out(self.amp)
            bound = 1/math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.phase, -bound, bound)
            #nn.init.uniform_(self.phase,1,1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if(input.size(dim=1) != self.in_features):
            print("Incorrect Number of features in input")
            return None
        else:
            for i in range(0, self.in_features):
                cos_tmp = torch.mm(input[:,i].reshape(-1,1), self.freq[0,:].reshape(1,-1))
                sin_tmp = torch.mm(input[:,i].reshape(-1,1), self.freq[0,:].reshape(1,-1))
                if i > 0:
                    cos_inner = torch.cat((cos_inner, cos_tmp), dim = 1)
                    sin_inner = torch.cat((sin_inner, sin_tmp), dim = 1)
                else:
                    cos_inner = cos_tmp
                    sin_inner = sin_tmp

            del cos_tmp
            del sin_tmp
            if self.phase is not None:
                cos_inner.add_(self.phase[0,:])
                sin_inner.add_(self.phase[1,:])

        return torch.mul(torch.cat((torch.cos(cos_inner), torch.sin(sin_inner)), dim = 1), self.amp)
        
class radial_basis_function(nn.Module):
    def __init__(self, size_in: int, num_terms: int, sep: float = 0.5,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype':dtype}
        super().__init__()
        self.in_features = size_in
        self.out_features = num_terms
        self.N = num_terms - 1
        self.l = 1 / (2*self.N*math.sqrt(-2*math.log(sep)))
        self.weight = nn.Parameter(torch.empty((1,num_terms * size_in), **factory_kwargs))
        self.shift = nn.Parameter(torch.empty((num_terms), **factory_kwargs)) 

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #nn.init.uniform_(self.weight,1,1)
        
        with torch.no_grad():
            for i in range(0,self.out_features):
                self.shift[i] = nn.Parameter(torch.tensor(i/self.N))
        
        self.shift.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if(input.size(dim=1) != self.in_features):
            print("Incorrect Number of features in input")
            return None
        else:
            for i in range(0, self.out_features):
                tmp = (torch.subtract(input, self.shift[i]))**2
                if i > 0:
                    a = torch.cat((a, tmp), dim = 1)
                else:
                    a = tmp

            del tmp
            #a = (torch.subtract(input, self.shift))**2
            return torch.mul(torch.exp(-a/(2*(self.l)**2)), self.weight)


#MODEL CLASS
class prev_PTV_NN_multi(nn.Module):
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

class PTV_NN_multi(nn.Module):
    def __init__(self, num_FBT: int, num_RBT: int, num_HL: int = 0, num_HLT: int = 8):
        super().__init__()

        self.fourier_layer = fourier_basis_function(1,num_FBT, False)
        self.radial_layer = radial_basis_function(2, num_RBT, sep = 0.5) 
        
        if num_HL:
            self.in_HL = nn.Linear(2*num_RBT + num_FBT, num_HLT)
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(num_HLT, num_HLT) for _ in range(num_HL-1)]
            )
            self.output_layer = nn.Linear(num_HLT,3) 

        else:
            self.register_parameter = ('in_HL', None)
            self.register_parameter = ('hidden_layers', None)

            self.output_layer = nn.Linear(2*num_RBT + num_FBT, 3)

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        FT = self.fourier_layer(input[:,2].reshape(-1,1))
        RT = self.radial_layer(input[:,0:2])

        input = torch.cat((RT, FT), dim=1)

        if self.in_HL is not None:
            input = torch.tanh(self.in_HL(input))
            for layer in self.hidden_layers:
                input = torch.tanh(layer(input))

        return self.output_layer(input)

class test_FBF_layer(nn.Module):
    def __init__(self, num_FBT: int):
        super().__init__()

        self.fourier_layer = fourier_basis_function(2,num_FBT, False)
        self.layer1 = nn.Linear(2*num_FBT, 8)
        self.layer2 = nn.Linear(8,8)
        self.output_layer = nn.Linear(8, 1)

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.fourier_layer(input)
        input = torch.tanh(self.layer1(input))
        input = torch.tanh(self.layer2(input))
        return self.output_layer(input)


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
