import torch
from ppg_cvae.models.cvae import ConditionalVAE

def test_shapes():
 m=ConditionalVAE(); x=torch.randn(4,1,512); c=torch.ones(4); y,mu,lv=m(x,c); assert y.shape==(4,1,512); assert mu.shape==lv.shape==(4,16)
