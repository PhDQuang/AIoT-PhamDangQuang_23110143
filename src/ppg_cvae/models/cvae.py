"""Exact 1D conditional VAE architecture specified by the proposal."""
from __future__ import annotations
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, latent_dim: int=16):
        super().__init__(); self.latent_dim=latent_dim
        self.features=nn.Sequential(
            nn.Conv1d(2,16,7,2,3), nn.ReLU(),
            nn.Conv1d(16,32,7,2,3), nn.ReLU(),
            nn.Conv1d(32,64,7,2,3), nn.ReLU())
        self.fc=nn.Sequential(nn.Linear(64*64,64),nn.ReLU())
        self.mu=nn.Linear(64,latent_dim); self.logvar=nn.Linear(64,latent_dim)
    def forward(self,x:torch.Tensor,c:torch.Tensor):
        if x.ndim!=3 or x.shape[1:]!=(1,512): raise ValueError(f"x must be [B,1,512], got {tuple(x.shape)}")
        c=torch.as_tensor(c, device=x.device, dtype=x.dtype).reshape(-1,1,1).expand(-1,1,x.shape[-1]); h=self.features(torch.cat([x,c],dim=1)); h=self.fc(h.flatten(1)); return self.mu(h),self.logvar(h)

class Decoder(nn.Module):
    def __init__(self,latent_dim:int=16):
        super().__init__(); self.latent_dim=latent_dim
        self.fc=nn.Sequential(nn.Linear(latent_dim+1,4096),nn.ReLU())
        self.deconv=nn.Sequential(nn.ConvTranspose1d(64,32,4,2,1),nn.ReLU(),nn.ConvTranspose1d(32,16,4,2,1),nn.ReLU(),nn.ConvTranspose1d(16,1,4,2,1))
    def forward(self,z:torch.Tensor,c:torch.Tensor):
        if z.ndim!=2 or z.shape[1]!=self.latent_dim: raise ValueError(f"z must be [B,{self.latent_dim}]")
        c=torch.as_tensor(c, device=z.device, dtype=z.dtype).reshape(-1,1); h=self.fc(torch.cat([z,c],dim=1)).reshape(-1,64,64); y=self.deconv(h)
        if y.shape[-1]!=512: raise RuntimeError(f"decoder output length is {y.shape[-1]}")
        return y

class ConditionalVAE(nn.Module):
    def __init__(self,latent_dim:int=16):
        super().__init__(); self.latent_dim=latent_dim; self.encoder=Encoder(latent_dim); self.decoder=Decoder(latent_dim)
    @staticmethod
    def reparameterize(mu,logvar): return mu + torch.exp(0.5*logvar)*torch.randn_like(mu)
    def forward(self,x,c,sample:bool=True):
        mu,logvar=self.encoder(x,c); z=self.reparameterize(mu,logvar) if sample else mu; return self.decoder(z,c),mu,logvar
    def reconstruct(self,x,c):
        mu,logvar=self.encoder(x,c); return self.decoder(mu,c),mu,logvar
    def generate(self,c,z=None):
        c=torch.as_tensor(c, device=self.decoder.fc[0].weight.device, dtype=self.decoder.fc[0].weight.dtype).reshape(-1)
        if z is None: z=torch.randn(c.shape[0],self.latent_dim,device=c.device,dtype=c.dtype)
        return self.decoder(z,c)

def parameter_counts(model: ConditionalVAE):
    return {"encoder":sum(p.numel() for p in model.encoder.parameters()),"decoder":sum(p.numel() for p in model.decoder.parameters()),"cvae":sum(p.numel() for p in model.parameters())}
