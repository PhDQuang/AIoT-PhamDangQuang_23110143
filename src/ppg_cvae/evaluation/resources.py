from pathlib import Path
import copy, platform, os, time, numpy as np, torch
from ..models.cvae import parameter_counts
from ..utils.io import environment_info
def measure_resources(cvae,proxy=None,decoder_path=None,iterations=1000,warmup=100):
    if iterations < 1 or warmup < 0: raise ValueError('iterations must be positive and warmup nonnegative')
    old_threads=torch.get_num_threads()
    dec=copy.deepcopy(cvae.decoder).cpu().float().eval()
    z=torch.randn(1,cvae.latent_dim); c=torch.ones(1)
    try:
        torch.set_num_threads(1)
        with torch.inference_mode():
            for _ in range(warmup): dec(z,c)
            vals=[]
            for _ in range(iterations):
                t=time.perf_counter(); dec(z,c); vals.append((time.perf_counter()-t)*1000)
    finally:
        torch.set_num_threads(old_threads)
    out={'parameter_counts':parameter_counts(cvae),'decoder_latency_ms_median':float(np.median(vals)),'decoder_latency_ms_p95':float(np.percentile(vals,95))}
    out.update(device='cpu', dtype='float32', num_threads=1, batch_size=1,
               warmup=warmup, iterations=iterations, cpu=platform.processor() or os.environ.get('PROCESSOR_IDENTIFIER','unknown'),
               os=platform.platform(), torch=str(torch.__version__))
    environment = environment_info()
    out.update(cpu=environment['cpu'], ram_bytes=environment.get('ram_bytes'))
    if proxy is not None: out['proxy_parameters']=sum(p.numel() for p in proxy.parameters())
    if decoder_path and Path(decoder_path).exists(): out['decoder_state_dict_bytes']=Path(decoder_path).stat().st_size; out['decoder_state_dict_mb']=Path(decoder_path).stat().st_size/1e6
    return out
