import torch
from ppg_cvae.config import CVAEConfig
from ppg_cvae.models.hr_proxy import HRProxy
import ppg_cvae.training.train_cvae as training_module


def test_cpu_best_checkpoint_does_not_follow_later_epochs(monkeypatch):
    x = torch.randn(4,1,512)
    c = torch.ones(4)
    snapshots = []
    original_step = torch.optim.Adam.step
    original_loss = training_module.cvae_losses
    validation_count = 0

    def capture_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        snapshots.append([p.detach().clone() for group in optimizer.param_groups for p in group['params']])
        return result

    def controlled_validation(*args, **kwargs):
        nonlocal validation_count
        losses = original_loss(*args, **kwargs)
        if not torch.is_grad_enabled():
            validation_count += 1
            losses['L_total'] = torch.tensor(float(validation_count))
        return losses

    monkeypatch.setattr(torch.optim.Adam, 'step', capture_step)
    monkeypatch.setattr(training_module, 'cvae_losses', controlled_validation)
    config = CVAEConfig(max_epochs=2, batch_size=4, beta_warmup_epochs=0)
    model, history = training_module.train_cvae(x,c,x,c,HRProxy(),config)
    assert len(history) == 2
    for epoch in history:
        dimension_average = sum(epoch[f'val_kl_dim_{i}'] for i in range(config.latent_dim)) / config.latent_dim
        assert abs(dimension_average - epoch['val_L_KL']) < 1e-5
    assert any(not torch.equal(a,b) for a,b in zip(snapshots[0],snapshots[1]))
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),snapshots[0]))
