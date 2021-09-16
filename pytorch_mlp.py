import torch
import torch.nn as nn
import torch.nn.functional as F  # 激励函数的库



class PYTORCH_MLP(torch.nn.Module):  
    def __init__(self):
        super(PYTORCH_MLP,self).__init__()  
        self.fc1 = torch.nn.Linear(5,4) 
        self.fc2 = torch.nn.Linear(4,4) 
        self.fc3 = torch.nn.Linear(4,3)  
     
    def forward(self,din):
        dout = torch.sigmoid(self.fc1(din))  
        dout = torch.sigmoid(self.fc2(dout))
        dout = torch.softmax(self.fc3(dout), dim=1) 
        return dout
 
