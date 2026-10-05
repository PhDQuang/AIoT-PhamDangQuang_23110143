"""Train cVAE with fixed-weight, batched validation and independent checkpoints."""
from pathlib import Path
import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
from .losses import cvae_losses
from ..models.cvae import ConditionalVAE
from ..utils.seed import seed_everything
from ..utils.io import save_json, environment_info


def train_cvae(train_x, train_c, val_x, val_c, proxy, config, device='cpu', output_dir=None, validation_monitor=None):
    if not len(train_x) or not len(val_x) or config.max_epochs < 1:
        raise ValueError('Nonempty train/validation arrays and positive max_epochs required')
    seed_everything(config.seed)
    model = ConditionalVAE(config.latent_dim).to(device)
    proxy = proxy.to(device)
    proxy.freeze()
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    def loader(x, c, shuffle):
        return DataLoader(TensorDataset(torch.as_tensor(x).float(), torch.as_tensor(c).float().reshape(-1)),
                          batch_size=config.batch_size, shuffle=shuffle)
    tr, va = loader(train_x, train_c, True), loader(val_x, val_c, False)
    best, bad, hist, state = float('inf'), 0, [], None
    for epoch in range(config.max_epochs):
        model.train()
        b = config.beta*min(1., (epoch+1)/max(1, config.beta_warmup_epochs))
        sums = {}
        for x, c in tr:
            x, c = x.to(device), c.to(device)
            model_c = torch.zeros_like(c) if config.condition_mode == 'zero' else c
            xhat, mu, logvar = model(x, model_c)
            prior = model.generate(model_c) if config.lambda_prior else None
            losses = cvae_losses(x, xhat, mu, logvar, c, proxy, b, config.lambda_hr,
                                 config.lambda_prior, x_prior=prior)
            if not torch.isfinite(losses['L_total']): raise ValueError('Non-finite training loss')
            opt.zero_grad()
            losses['L_total'].backward()
            opt.step()
            for k, v in losses.items(): sums[k] = sums.get(k, 0.)+float(v.detach())*len(x)
        model.eval()
        val_sums = {}
        val_kl_dimensions = torch.zeros(config.latent_dim, device=device)
        # Fixed prior samples keep optional prior validation comparable across epochs.
        generator = torch.Generator(device=device).manual_seed(config.seed+10000)
        with torch.no_grad():
            for x, c in va:
                x, c = x.to(device), c.to(device)
                model_c = torch.zeros_like(c) if config.condition_mode == 'zero' else c
                xhat, mu, logvar = model(x, model_c, sample=False)
                val_kl_dimensions += (0.5*(mu.square()+logvar.exp()-1-logvar)).sum(dim=0)
                prior = None
                if config.lambda_prior:
                    z = torch.randn(len(x), config.latent_dim, device=device, generator=generator)
                    prior = model.generate(model_c, z)
                losses = cvae_losses(x, xhat, mu, logvar, c, proxy, config.beta,
                                     config.lambda_hr, config.lambda_prior, x_prior=prior)
                for k, v in losses.items(): val_sums[k] = val_sums.get(k, 0.)+float(v)*len(x)
        val = val_sums['L_total']/len(val_x)
        if not torch.isfinite(torch.tensor(val)): raise ValueError('Non-finite validation loss')
        row = {'epoch': epoch+1, 'beta_effective': b}
        row.update({f'train_{k}': v/len(train_x) for k, v in sums.items()})
        row.update({f'val_{k}': v/len(val_x) for k, v in val_sums.items()})
        row.update({f'val_kl_dim_{i}': float(value)/len(val_x)
                    for i, value in enumerate(val_kl_dimensions)})
        if validation_monitor is not None:
            row.update(validation_monitor(model))
        hist.append(row)
        if val < best:
            best, bad = val, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch+1 > config.beta_warmup_epochs:
            bad += 1
        if bad >= config.patience: break
    model.load_state_dict(state)
    if output_dir:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), p/'best_model.pt')
        torch.save(model.decoder.state_dict(), p/'decoder_state_dict.pt')
        import yaml
        (p/'resolved_config.yaml').write_text(yaml.safe_dump(config.resolved()), encoding='utf-8')
        save_json({'best_val_L_total': best, 'status': 'completed'}, p/'validation_metrics.json')
        save_json(environment_info(), p/'environment.json')
        with (p/'training_history.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=hist[0].keys())
            w.writeheader()
            w.writerows(hist)
    return model, hist
