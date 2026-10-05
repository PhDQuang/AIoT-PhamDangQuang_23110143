from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..models.hr_proxy import HRProxy
from ..utils.seed import seed_everything
from ..utils.io import save_json, environment_info


def train_proxy(train_x, train_c, val_x, val_c, epochs=50, batch_size=128, lr=1e-3,
                device='cpu', output_dir=None, seed=42):
    if not len(train_x) or not len(val_x) or epochs < 1:
        raise ValueError('Nonempty train/validation arrays and positive epochs required')
    seed_everything(seed)
    model = HRProxy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, history = float('inf'), None, []
    def loader(x, c, shuffle):
        return DataLoader(TensorDataset(torch.as_tensor(x).float(), torch.as_tensor(c).float().reshape(-1)),
                          batch_size=batch_size, shuffle=shuffle)
    tr, va = loader(train_x, train_c, True), loader(val_x, val_c, False)
    for epoch in range(epochs):
        model.train()
        train_sum = 0.
        for x, c in tr:
            pred = model(x.to(device))
            loss = torch.nn.functional.l1_loss(pred, c.to(device))
            if not torch.isfinite(loss): raise ValueError('Non-finite proxy training loss')
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_sum += float(loss.detach())*len(x)
        model.eval()
        val_sum = 0.
        with torch.no_grad():
            for x, c in va:
                val_sum += float(torch.nn.functional.l1_loss(model(x.to(device)), c.to(device), reduction='sum'))
        val = val_sum/len(val_x)
        if not np.isfinite(val): raise ValueError('Non-finite proxy validation loss')
        history.append({'epoch': epoch+1, 'train_mae_bpm': train_sum/len(train_x)*60, 'val_mae_bpm': val*60})
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.freeze()
    if output_dir:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), p/'hr_proxy.pt')
        save_json({'best_val_mae_normalized': best, 'best_val_mae_bpm': best*60}, p/'hr_proxy_validation_metrics.json')
        save_json({'seed': seed, 'epochs': epochs, 'batch_size': batch_size, 'lr': lr}, p/'resolved_config.json')
        save_json(environment_info(), p/'environment.json')
        pd.DataFrame(history).to_csv(p/'training_history.csv', index=False)
    return model, history
