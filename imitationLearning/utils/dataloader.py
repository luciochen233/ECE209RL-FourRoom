from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch

to_tensor = transforms.Compose([
    transforms.ToTensor()
])

class LoadData(Dataset):

  def __init__(self, expertdata_path = './expert_vis15_il_vis15.pt'):

    self.filename = expertdata_path
    self.expertdata = torch.load(expertdata_path)
    self.states = self.expertdata['states']
    self.actions = self.expertdata['actions']
    self.dataset_size = len(self.states)
    self.max_len = 0
    for i in range(len(self.states)):
      self.max_len = max(self.max_len,len(self.states[i]))
    self.states = np.asarray(self.states)
    self.actions = np.asarray(self.actions)

    self.masks = []
    for i in range(self.states.shape[0]):
      pad_len = self.max_len - len(self.states[i])
      
      self.states[i] = np.asarray(self.states[i], dtype=np.float32)
      self.states[i] = np.pad(self.states[i],((0,pad_len),(0,0),(0,0),(0,0)), constant_values=(0, 0))
      self.actions[i] = np.asarray(self.actions[i], dtype=np.float32)
      self.masks.append(np.ones_like(self.actions[i]))
      self.actions[i] = np.pad(self.actions[i],(0,pad_len), constant_values=(0, 0))
      self.masks[i] = np.pad(self.masks[i],(0,pad_len), constant_values=(0, 0))
    self.masks = np.asarray(self.masks)


  def __len__(self):
    return self.dataset_size
  def __getitem__(self, idx):
    state = torch.from_numpy(self.states[idx])

    action = torch.from_numpy(self.actions[idx]).long()

    mask = torch.from_numpy(self.masks[idx])

    return state,action,mask
