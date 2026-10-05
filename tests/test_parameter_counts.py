from ppg_cvae.models.cvae import ConditionalVAE,parameter_counts
from ppg_cvae.models.hr_proxy import HRProxy
def test_counts():
 assert parameter_counts(ConditionalVAE())=={'encoder':282544,'decoder':84081,'cvae':366625}; assert sum(p.numel() for p in HRProxy().parameters())==18209
