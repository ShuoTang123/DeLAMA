import copy
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models import Phi, init_model, resnet


class AgentModel(nn.Module):
    """
    Unified agent model supporting two fair comparison modes:
      - is_backbone=True:  Frozen pre-trained feature extractor phi + trainable linear head (feature_dim -> n_classes).
                           Matches DeLAMA's representation capacity exactly.
      - is_backbone=False: End-to-end trainable backbone phi + linear head.
    """

    def __init__(
        self,
        dataset_type: str,
        n_classes: int = 5,
        feature_dim: int = 50,
        is_backbone: bool = True,
        backbone_state_dict: Optional[ dict ] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dataset_type = dataset_type
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.is_backbone = is_backbone
        self.device = device
        self.dtype = dtype

        mode = 0 if dataset_type == "mnist" else 1
        if not is_backbone:
            self.backbone = (Phi(feature_dim) if mode == 0 else resnet(feature_dim)).to(device=device, dtype=dtype)
            if backbone_state_dict is not None:
                self.backbone.load_state_dict(backbone_state_dict)
        else:
            self.backbone = None

        self.head = nn.Linear(feature_dim, n_classes, bias=True, device=device, dtype=dtype)

    def extract_features(self, x: torch.Tensor, shared_backbone: Optional[nn.Module] = None) -> torch.Tensor:
        if self.is_backbone:
            if x.dim() == 2 and x.shape[1] == self.feature_dim:
                return x.to(device=self.device, dtype=self.dtype)
            assert shared_backbone is not None, "shared_backbone required when input is raw images in is_backbone=True mode"
            with torch.no_grad():
                if x.shape[0] > 256:
                    chunks = [
                        shared_backbone(x[i : i + 256].to(device=self.device, dtype=self.dtype))
                        for i in range(0, x.shape[0], 256)
                    ]
                    return torch.cat(chunks, dim=0)
                return shared_backbone(x.to(device=self.device, dtype=self.dtype))
        else:
            if not torch.is_grad_enabled() and x.shape[0] > 256:
                chunks = [
                    self.backbone(x[i : i + 256].to(device=self.device, dtype=self.dtype))
                    for i in range(0, x.shape[0], 256)
                ]
                return torch.cat(chunks, dim=0)
            return self.backbone(x.to(device=self.device, dtype=self.dtype))

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features.to(device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor, shared_backbone: Optional[nn.Module] = None) -> torch.Tensor:
        feat = self.extract_features(x, shared_backbone=shared_backbone)
        return self.head(feat)

    def trainable_parameters(self) -> List[nn.Parameter]:
        if self.is_backbone:
            return list(self.head.parameters())
        return list(self.parameters())

    def get_flat_params(self) -> torch.Tensor:
        """Return 1D flattened tensor of all trainable parameters (detached)."""
        return torch.cat([p.detach().reshape(-1) for p in self.trainable_parameters()])

    def set_flat_params(self, flat: torch.Tensor) -> None:
        """Load 1D flattened tensor into trainable parameters."""
        offset = 0
        with torch.no_grad():
            for p in self.trainable_parameters():
                numel = p.numel()
                p.copy_(flat[offset : offset + numel].view_as(p).to(device=p.device, dtype=p.dtype))
                offset += numel

    def param_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.trainable_parameters())


def load_shared_backbone(
    dataset_type: str,
    feature_dim: int = 50,
    model_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Load the pre-trained DeLAMA feature extractor phi."""
    if model_path is None:
        model_path = os.path.join(
            SRC_DIR, "model", f"collaboration_phi_[0.0001, 0.001]_2_{dataset_type}_100.pt"
        )
    mode = 0 if dataset_type == "mnist" else 1
    phi = init_model(mode=mode, feature_dim=feature_dim, device=device, float_type=dtype)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
    phi.load_state_dict(state_dict)
    phi.eval()
    for p in phi.parameters():
        p.requires_grad_(False)
    return phi


def preprocess_episode_data(
    episode_data: list,
    shared_backbone: Optional[nn.Module],
    is_backbone: bool,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[List[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]], List[List[List[int]]]]:
    """
    Pre-process one episode from Classification_Scheduler.get_data(...).
    If `is_backbone=True`, pre-extracts features once using `shared_backbone` so that
    multi-epoch / multi-round baseline loops do not redundantly re-run ResNet/CNN forward passes.
    Deduplicates shared group evaluation tensors (`gt_imgs`, `gt_labels`) by storage pointer.

    Returns:
      - processed_agents: list of length agent_num, each is (train_x_list, train_y_list, test_x_list, test_y_list)
      - seen_classes_per_step: [t][j] -> sorted list of unique class indices seen by agent j at step t
    """
    agent_num = len(episode_data)
    T = len(episode_data[0][0][0])
    processed_agents = []
    seen_classes_per_step: List[List[List[int]]] = [[[] for _ in range(agent_num)] for _ in range(T)]
    gt_x_cache: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], torch.Tensor] = {}
    gt_y_cache: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], torch.Tensor] = {}

    def _extract_img_batch(img_t: torch.Tensor) -> torch.Tensor:
        if is_backbone and shared_backbone is not None and img_t.dim() > 2:
            with torch.no_grad():
                if img_t.shape[0] > 256:
                    chunks = [
                        shared_backbone(img_t[i : i + 256].to(device=device, dtype=dtype))
                        for i in range(0, img_t.shape[0], 256)
                    ]
                    return torch.cat(chunks, dim=0)
                return shared_backbone(img_t.to(device=device, dtype=dtype))
        return img_t.to(device=device, dtype=dtype)

    for j in range(agent_num):
        (train_imgs, train_labels), (gt_imgs, gt_labels) = episode_data[j]
        tx_list, ty_list, gx_list, gy_list = [], [], [], []
        for t in range(T):
            x_tr = _extract_img_batch(train_imgs[t])
            y_tr = train_labels[t].to(device=device, dtype=torch.long)

            gx_raw = gt_imgs[t]
            gx_key = (gx_raw.data_ptr(), tuple(gx_raw.shape), tuple(gx_raw.stride()))
            x_gt = gt_x_cache.get(gx_key)
            if x_gt is None:
                x_gt = _extract_img_batch(gx_raw)
                gt_x_cache[gx_key] = x_gt

            gy_raw = gt_labels[t]
            gy_key = (gy_raw.data_ptr(), tuple(gy_raw.shape), tuple(gy_raw.stride()))
            y_gt = gt_y_cache.get(gy_key)
            if y_gt is None:
                y_gt = gy_raw.to(device=device, dtype=torch.long)
                gt_y_cache[gy_key] = y_gt

            tx_list.append(x_tr)
            ty_list.append(y_tr)
            gx_list.append(x_gt)
            gy_list.append(y_gt)
            seen_classes_per_step[t][j] = sorted(torch.unique(y_tr).cpu().tolist())

        processed_agents.append((tx_list, ty_list, gx_list, gy_list))

    return processed_agents, seen_classes_per_step


def split_train_val(
    x: torch.Tensor, y: torch.Tensor, val_ratio: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split a local batch (x, y) into local train and local validation sets (used by DPFL and L2C).
    If the sample size is small (e.g., 10 samples), retains all samples for training and uses a
    held-out slice (or full local batch if < 5) for validation evaluation.
    """
    n = len(x)
    if n < 5:
        return x, y, x, y
    n_val = max(2, int(round(n * val_ratio)))
    return x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:]
