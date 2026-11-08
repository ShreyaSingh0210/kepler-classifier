import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from src.models import TwoViewCNN
from src.splits import get_folds
from src.utils import set_seed

PROCESSED_DIR = Path("data") / "processed"
RESULTS_DIR = Path("results")


def load_processed():
    files = sorted(PROCESSED_DIR.glob("*.npz"))
    G, L, y, kepids = [], [], [], []
    for f in files:
        d = np.load(f)
        G.append(d["global_view"])
        L.append(d["local_view"])
        y.append(int(d["label"]))
        kepids.append(int(f.stem))
    G = torch.tensor(np.stack(G)).unsqueeze(1).float()
    L = torch.tensor(np.stack(L)).unsqueeze(1).float()
    return G, L, np.array(y), np.array(kepids)


def augment(g, l):
    """Random time flip (same for both views), small noise, small shift."""
    flip = torch.rand(g.shape[0], device=g.device) < 0.5
    g = torch.where(flip[:, None, None], g.flip(-1), g)
    l = torch.where(flip[:, None, None], l.flip(-1), l)
    g = g + 0.02 * torch.randn_like(g)
    l = l + 0.02 * torch.randn_like(l)
    g = torch.roll(g, int(torch.randint(-10, 11, (1,))), -1)
    l = torch.roll(l, int(torch.randint(-3, 4, (1,))), -1)
    return g, l


@torch.no_grad()
def predict(model, G, L, device, batch=512):
    model.eval()
    out = []
    for i in range(0, len(G), batch):
        g = G[i:i + batch].to(device)
        l = L[i:i + batch].to(device)
        out.append(torch.sigmoid(model(g, l)).cpu())
    return torch.cat(out).numpy()


def metrics(y_true, scores):
    preds = (scores >= 0.5).astype(int)
    k = min(100, len(y_true) // 2)
    top = np.argsort(scores)[::-1][:k]
    return {
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "pr_auc": average_precision_score(y_true, scores),
        "precision_at_k": y_true[top].mean(),
    }


def train_one_fold(args, G, L, y, kepids, train_idx, test_idx, seed, device):
    # hold out some training stars for early stopping (split by star)
    inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr, va = next(inner.split(train_idx, groups=kepids[train_idx]))
    tr_idx, va_idx = train_idx[tr], train_idx[va]

    y_t = torch.tensor(y).float()
    loader = DataLoader(
        TensorDataset(G[tr_idx], L[tr_idx], y_t[tr_idx]),
        batch_size=args.batch, shuffle=True,
    )

    model = TwoViewCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    pos_weight = None
    if args.weights:
        n_pos = max(1, int((y[tr_idx] == 1).sum()))
        n_neg = int((y[tr_idx] == 0).sum())
        pos_weight = torch.tensor(n_neg / n_pos, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_score, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for g, l, t in loader:
            g, l, t = g.to(device), l.to(device), t.to(device)
            if args.aug:
                g, l = augment(g, l)
            opt.zero_grad()
            loss = loss_fn(model(g, l), t)
            loss.backward()
            opt.step()

        val_scores = predict(model, G[va_idx], L[va_idx], device)
        val_pr = average_precision_score(y[va_idx], val_scores)
        if val_pr > best_score:
            best_score = val_pr
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    model.load_state_dict(best_state)
    test_scores = predict(model, G[test_idx], L[test_idx], device)
    result = metrics(y[test_idx], test_scores)
    result["epochs_run"] = epoch + 1
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="cnn")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weights", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aug", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G, L, y, kepids = load_processed()
    print(f"Device: {device} | stars: {len(y)} | planets: {y.sum()} | run: {args.name}")
    df = pd.DataFrame({"kepid": kepids, "label": y})

    rows = []
    for seed in range(args.seeds):
        for fold, train_idx, test_idx in get_folds(df, seed):
            if fold >= args.folds:
                break
            set_seed(seed * 100 + fold)
            r = train_one_fold(args, G, L, y, kepids, train_idx, test_idx, seed, device)
            r.update({"run": args.name, "seed": seed, "fold": fold})
            rows.append(r)
            print(f"seed {seed} fold {fold}: F1 {r['f1']:.3f}, PR-AUC {r['pr_auc']:.3f}, epochs {r['epochs_run']}")

    runs = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "cnn_runs.csv"
    if path.exists():
        runs = pd.concat([pd.read_csv(path), runs])
    runs.to_csv(path, index=False)

    cols = ["precision", "recall", "f1", "pr_auc", "precision_at_k"]
    mine = runs[runs["run"] == args.name]
    print(mine[cols].agg(["mean", "std"]).round(3).T)


if __name__ == "__main__":
    main()