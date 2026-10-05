import os, random, numpy as np, torch
def seed_everything(seed:int, deterministic:bool=True):
    if deterministic: os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ["PYTHONHASHSEED"]=str(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
