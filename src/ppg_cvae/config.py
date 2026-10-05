from dataclasses import dataclass,asdict
@dataclass
class CVAEConfig:
    latent_dim:int=16; beta:float=.01; lambda_hr:float=1.; lambda_prior:float=0.; lr:float=1e-3; batch_size:int=128; max_epochs:int=100; patience:int=12; beta_warmup_epochs:int=10; seed:int=42; fold:int=1; condition_mode:str='hr'
    def resolved(self): return asdict(self)
