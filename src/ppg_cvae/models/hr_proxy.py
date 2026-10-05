"""Frozen differentiable normalized-HR proxy."""
from torch import nn
class HRProxy(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(1,16,7,2,3),nn.ReLU(),nn.Conv1d(16,32,7,2,3),nn.ReLU(),nn.Conv1d(32,64,7,2,3),nn.ReLU())
        self.head=nn.Linear(64,1)
    def forward(self,x): return self.head(self.net(x).mean(dim=-1)).squeeze(-1)
    def freeze(self):
        self.eval()
        for p in self.parameters(): p.requires_grad=False
