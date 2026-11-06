"""Combined 1D-CNN + LSTM network for fall / ADL classification (PyTorch).

    input  [B, C, T]  normalised sensor window (C channels, T = 400 samples @ 200 Hz)
      -> 3 x { Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d }   local movement patterns
      -> Dropout
      -> transpose to [B, T', F]                                CNN feature sequence
      -> LSTM                                                   temporal dependencies
      -> final hidden state -> Dropout -> Linear -> ReLU -> Linear(1)
    output [B]  one logit; sigmoid(logit) = P(fall)   (label 1 = fall, 0 = ADL, as in the RF)

Raw logits are returned so the loss can be BCEWithLogits (numerically stable, supports
pos_weight for class imbalance).
"""
from __future__ import annotations

import torch
from torch import nn


class CNNLSTM(nn.Module):
    def __init__(self, in_channels: int = 6, conv_channels=(32, 64, 64), kernel_sizes=(7, 5, 3),
                 pool_size: int = 2, lstm_hidden: int = 64, lstm_layers: int = 1,
                 bidirectional: bool = False, fc_hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have the same length")
        blocks, prev = [], in_channels
        for out_c, k in zip(conv_channels, kernel_sizes):
            blocks += [nn.Conv1d(prev, out_c, kernel_size=k, padding=k // 2),
                       nn.BatchNorm1d(out_c), nn.ReLU(inplace=True), nn.MaxPool1d(pool_size)]
            prev = out_c
        self.cnn = nn.Sequential(*blocks)
        self.cnn_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=prev, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.bidirectional = bidirectional
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, fc_hidden),
                                  nn.ReLU(inplace=True), nn.Linear(fc_hidden, 1))
        self.config = dict(in_channels=in_channels, conv_channels=list(conv_channels),
                           kernel_sizes=list(kernel_sizes), pool_size=pool_size,
                           lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
                           bidirectional=bidirectional, fc_hidden=fc_hidden, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected input [B, C, T], got {tuple(x.shape)}")
        f = self.cnn_dropout(self.cnn(x))          # [B, F, T']
        _, (h_n, _) = self.lstm(f.transpose(1, 2))  # h_n: [layers*dirs, B, H]
        last = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        return self.head(last).squeeze(1)


def build_model(model_cfg: dict, in_channels: int) -> CNNLSTM:
    return CNNLSTM(in_channels=in_channels, **model_cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- checkpoints
CHECKPOINT_NAME = "cnn_lstm_fall_detector.pt"


def save_checkpoint(path, model: CNNLSTM, *, channels, window_size, sampling_rate, mean, std,
                    threshold: float, data_source: str, seed: int, extra: dict | None = None) -> None:
    """Self-describing checkpoint (plain tensors/lists/strings only, so it can be loaded with
    torch.load(weights_only=True)). Never writes anywhere but `path`."""
    from datetime import datetime, timezone
    from pathlib import Path
    payload = dict(
        model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
        model_config=model.config,
        channels=list(channels), window_size=int(window_size), sampling_rate=int(sampling_rate),
        normalization=dict(mean=[float(m) for m in mean], std=[float(s) for s in std]),
        threshold=float(threshold),
        label_map={"0": "adl", "1": "fall"},
        data_source=data_source,          # "sisfall" or "synthetic" - travels with the weights
        seed=int(seed), torch_version=str(torch.__version__),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        extra=extra or {},
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device: str = "cpu") -> tuple[CNNLSTM, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = dict(ckpt["model_config"])
    in_channels = cfg.pop("in_channels")
    model = CNNLSTM(in_channels=in_channels, **cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt
