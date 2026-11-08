import torch
import torch.nn as nn


def make_branch(channels):
    """1D conv blocks with BatchNorm. Ends with a fixed output size of 4."""
    layers = []
    in_ch = 1
    for out_ch in channels:
        layers += [
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
        ]
        in_ch = out_ch
    layers.append(nn.AdaptiveAvgPool1d(4))
    return nn.Sequential(*layers)


class TwoViewCNN(nn.Module):
    """Global branch + local branch, joined (and optionally tabular features), then dense layers."""

    def __init__(self, n_tabular=0, dropout=0.3):
        super().__init__()
        self.global_branch = make_branch([16, 32, 64, 128])
        self.local_branch = make_branch([16, 32, 64])
        self.head = nn.Sequential(
            nn.Linear(128 * 4 + 64 * 4 + n_tabular, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def embed(self, g, l):
        g = self.global_branch(g).flatten(1)
        l = self.local_branch(l).flatten(1)
        return torch.cat([g, l], dim=1)

    def forward(self, g, l, tab=None):
        x = self.embed(g, l)
        if tab is not None:
            x = torch.cat([x, tab], dim=1)
        return self.head(x).squeeze(1)  # raw score; sigmoid gives a probability