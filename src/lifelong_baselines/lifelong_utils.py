from avalanche.training.supervised import EWC, GEM, LwF, Naive, MAS, OnlineNaive, AGEM, ICaRL, BiC, MIR, MER
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ''))
import torch.nn as nn
import torch
from models import Phi, init_model
from avalanche.models.base_model import BaseModel



class LinearClassifier(nn.Module, BaseModel):
    def __init__(self, feature_dim, n_classes):
        super().__init__()
        self.classifier = nn.Linear(in_features=feature_dim, out_features=n_classes)

    def forward(self, feature):
        out = self.classifier(feature)
        return out
    
    def get_features(self, feature):
        return feature
    
class ImageClassifier(nn.Module, BaseModel):
    def __init__(self, dataset_type, n_classes) -> None:
        super().__init__()
        assert dataset_type in ['mnist', 'cifar10']
        mode = 0 if dataset_type == 'mnist' else 1
        self.backbone = init_model(mode, feature_dim=50, device='cpu', float_type=torch.float32)

        self.classifier = nn.Linear(in_features=50, out_features=n_classes)

    def forward(self, img):
        x = self.backbone(img)
        out = self.classifier(x)

        return out
    
    def get_features(self, img):

        return self.backbone(img)

def naive_strategy(model, optimizer, criterion, num_epochs, device, *args):
    return Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=100,
        eval_mb_size=100,
        train_epochs=num_epochs,
        device=device
    )


def online_naive_strategy(model, optimizer, criterion, num_epochs, device, *args):
    return OnlineNaive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=100,
        eval_mb_size=100,
        train_passes=10,
        device=device
    )


def ewc_strategy(model, optimizer, criterion, num_epochs, device, *args):
    assert len(args) == 1
    ewc_lambda = args[0]
    print(ewc_lambda)
    return EWC(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        ewc_lambda=ewc_lambda,
        train_epochs=num_epochs,
        train_mb_size=100,
        eval_mb_size=100,
        device=device
    )


def lwf_strategy(model, optimizer, criterion, num_epochs, device, *args):
    assert len(args) == 2
    alpha, temperature = args
    return LwF(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_epochs=num_epochs,
        train_mb_size=100,
        eval_mb_size=100,
        alpha=alpha,
        temperature=temperature,
        device=device
    )


def gem_strategy(model, optimizer, criterion, num_epochs, device, *args):
    assert len(args) == 1
    pattern_num = args[0]
    return GEM(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        patterns_per_exp=pattern_num,
        device=device
    )

def online_gem_strategy(model, optimizer, criterion, num_epochs, device, *args):
    assert len(args) == 1
    pattern_num = args[0]
    return AGEM(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        patterns_per_exp=pattern_num,
        device=device
    )

def icarl_strategy(model: ImageClassifier, optimizer, criterion, num_epochs, device, *args):
    memory_size = args[0]
    return ICaRL(
        feature_extractor=model.backbone,
        classifier=model.classifier,
        optimizer=optimizer,
        buffer_transform=torch.nn.Identity(),
        fixed_memory=True,
        memory_size=memory_size,
        train_epochs=num_epochs,
        device=device
    )


def mas_strategy(model, optimizer, criterion, num_epochs, device, *args):
    return MAS(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=100,
        eval_mb_size=100,
        train_epochs=num_epochs,
        device=device
    )

def bic_strategy(model, optimizer, criterion, num_epochs, device, *args):
    assert len(args) == 1
    mem_size = args[0]
    return BiC(
        model=model, 
        optimizer=optimizer,
        criterion=criterion,
        mem_size=mem_size,
        train_epochs=num_epochs,
        train_mb_size=100,
        eval_mb_size=100,
        device=device
    )

def mir_strategy(model, optimizer, criterion, num_epochs, device, *args):
    assert len(args) == 1
    mem_size = args[0]
    return MIR(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        mem_size=mem_size,
        train_mb_size=100,
        eval_mb_size=100,
        train_epochs=num_epochs,
        device=device
    )

def mer_strategy(model, optimizer, criterion, num_epochs, device, *args):
    assert len(args) == 1
    mem_size = args[0]
    return MER(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        mem_size=mem_size,
        train_mb_size=100,
        eval_mb_size=100,
        train_epochs=num_epochs,
        device=device
    )