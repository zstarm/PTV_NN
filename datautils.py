import torch
from torch.utils.data import Dataset
import os

class PTV_dataset(Dataset):
    def __init__(self, out_fname: str, in_fname: str, root_dir="Training Data"):
        self.feats = torch.load(os.path.join(root_dir,in_fname))
        self.outputs = torch.load(os.path.join(root_dir,out_fname))
        
        
        #normalize the dataset
        self.out_scales = []
        self.feats_scales = []
        for i in range(0,3):
            
            out_shift = torch.min(self.outputs[:,i]).item()
            in_shift = torch.min(self.feats[:,i]).item()
            
            self.out_scales.append(out_shift)
            self.feats_scales.append(in_shift)
            
            self.outputs[:,i] -= out_shift
            self.feats[:,i] -= in_shift
            
            out_scale = torch.max(self.outputs[:,i]).item()
            in_scale = torch.max(self.feats[:,i]).item()
            
            self.out_scales.append(out_scale)
            self.feats_scales.append(in_scale)
            
            self.outputs[:,i] /= out_scale
            self.feats[:,i] /= in_scale
            

    def __len__(self):
        return len(self.feats)
    
    def __getitem__(self, idx):
        coord = self.feats[idx, :]
        velocity = self.outputs[idx, :]
        return coord, velocity