import torch
from ppg_cvae.models.cvae import Decoder
def test_free_generation_without_encoder():
 d=Decoder(); y=d(torch.randn(3,16), torch.tensor([1., 4/3, 2.])); assert y.shape==(3,1,512)
