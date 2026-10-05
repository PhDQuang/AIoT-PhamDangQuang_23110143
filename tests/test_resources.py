import torch
from ppg_cvae.models.cvae import ConditionalVAE
from ppg_cvae.evaluation.resources import measure_resources


def test_resource_benchmark_restores_model_mode_and_threads(tmp_path):
    model=ConditionalVAE().train()
    threads=torch.get_num_threads()
    path=tmp_path/'decoder.pt'
    torch.save(model.decoder.state_dict(),path)
    metrics=measure_resources(model,decoder_path=path,iterations=3,warmup=1)
    assert model.decoder.training
    assert torch.get_num_threads()==threads
    assert metrics['device']=='cpu' and metrics['dtype']=='float32'
    assert metrics['decoder_state_dict_bytes']==path.stat().st_size
    assert metrics['decoder_latency_ms_p95']>0
    assert metrics['cpu']
    assert 'ram_bytes' in metrics
