import torch
import torch.nn as nn
import numpy as np
import math

class FourierBasis(object):
    """
    A set of linear basis functions.
    
    Arguments:
    num_terms  -  The number of Fourier terms.
    L          -  The period of the function.
    """
    def __init__(self, num_terms, L):
        self.num_terms = num_terms
        self.L = L
        self.num_basis = 2 * num_terms
    def __call__(self, x):
        #res = np.ndarray((self.num_basis,))
        res = torch.empty(self.num_basis,)
        for i in range(self.num_terms):
            res[2 * i] = torch.cos(2 * i * math.pi / self.L * x[0])
            res[2 * i + 1] = torch.sin(2 * (i+1) * math.pi / self.L * x[0])
        return res

def compute_design_matrix(X, phi):   # predictions
    """
    Arguments:
    
    X   -  The observed inputs (1D array)
    phi -  The basis functions.
    """
    num_observations = X.shape[0]
    num_basis = phi.num_basis
    #Phi = np.ndarray((num_observations, num_basis))
    Phi = torch.empty(num_observations, num_basis)
    for i in range(num_observations):
        Phi[i, :] = phi(X[i, :])
    return Phi


#MODEL CLASS
class PTV_NN_TRANS(nn.Module):
    def __init__(self, num_terms, num_hidden_layers,hidden_size):
        super().__init__()
        
        #simple ANN architecture
        self.input_layer = nn.Linear(2+(2*num_terms), hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_size, 3)
    
    def forward(self, x):
        x = torch.tanh(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        
        output = self.output_layer(x)
        return output

class PTV_NN_TRANS_v2(nn.Module):
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

def fourier_transform_time(x, num_terms):
    #compute Fourier transformation for time input
    phi = FourierBasis(num_terms, 1)
    t = x[:,2].reshape(-1,1)
    time_tran = compute_design_matrix(t,phi)
    return torch.cat((x[:,0:2], time_tran), dim=1)


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
