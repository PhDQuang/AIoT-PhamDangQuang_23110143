import torch
from ppg_cvae.training.losses import kl_loss,reconstruction_loss
from ppg_cvae.models.hr_proxy import HRProxy
def test_losses_and_proxy_grad():
 x=torch.zeros(2,1,512); y=torch.ones_like(x); mu=torch.zeros(2,16); lv=torch.zeros_like(mu); assert reconstruction_loss(x,y).item()==1.; assert kl_loss(mu,lv).item()==0.; p=HRProxy(); p.freeze(); y.requires_grad_(); p(y).sum().backward(); assert y.grad is not None and any(q.grad is None for q in p.parameters())
