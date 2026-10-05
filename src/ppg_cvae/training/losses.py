"""Loss terms kept separate for auditability."""
import torch

def reconstruction_loss(x,x_hat): return ((x-x_hat)**2).mean(dim=(1,2)).mean()
def kl_loss(mu,logvar): return (0.5*(torch.exp(logvar)+mu**2-1-logvar).sum(dim=1)/mu.shape[1]).mean()
def hr_reconstruction_loss(proxy,x_hat,c): return torch.nn.functional.l1_loss(proxy(x_hat),c.reshape(-1))
def cvae_losses(x,x_hat,mu,logvar,c,proxy,beta,lambda_hr,lambda_prior=0.0,x_prior=None):
    l_rec=reconstruction_loss(x,x_hat); l_kl=kl_loss(mu,logvar); l_hr=hr_reconstruction_loss(proxy,x_hat,c)
    l_prior=torch.zeros((),device=x.device)
    if lambda_prior and x_prior is not None: l_prior=hr_reconstruction_loss(proxy,x_prior,c)
    total=l_rec+beta*l_kl+lambda_hr*l_hr+lambda_prior*l_prior
    return {"L_rec":l_rec,"L_KL":l_kl,"L_HR_rec":l_hr,"L_HR_prior":l_prior,"L_total":total}
