"""Model registry + checkpoint I/O shared by the whole team.

Register a model once and every script can build it by name:

    from src.models.registry import register

    @register("scratch_cnn_v1", head=("classifier",))
    def scratch_cnn_v1(num_classes, pretrained=False):
        return MyCNN(num_classes)

`head` lists the attribute prefixes of the classifier layer(s). It is what `freeze`
and the trainer's `lr_backbone` treat as "head"; everything else is "backbone".
`input_size` documents a fixed input resolution (e.g. a CNN whose fc expects a fixed
feature-map size); leave it None when any reasonable size works.
Modules anywhere in `src/models/` are imported automatically the first time an
unknown name is requested, so `load_ckpt` works without extra imports.

Models must flatten feature maps with torch.flatten(x, 1) / nn.Flatten / reshape, not
.view: on GPU the trainer runs channels_last, where .view on a conv map fails.
"""
import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import models as tvm

FREEZE_MODES = ("none", "backbone", "partial")


@dataclass(frozen=True)
class ModelSpec:
    builder: callable
    head: tuple
    partial: tuple  # extra prefixes left trainable when freeze="partial"
    input_size: int | None = None


_REGISTRY: dict[str, ModelSpec] = {}


def register(name, head=("fc",), partial=(), input_size=None):
    """Decorator: `builder(num_classes, pretrained=False) -> nn.Module` returning logits."""
    def deco(builder):
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        _REGISTRY[name] = ModelSpec(builder, tuple(head), tuple(partial), input_size)
        return builder
    return deco


def available_models():
    _autodiscover()
    return sorted(_REGISTRY)


def model_input_size(name):
    """Fixed input resolution declared at registration, or None if the model accepts any size."""
    if name not in _REGISTRY:
        _autodiscover()
    return _REGISTRY[name].input_size


def _autodiscover():
    import src.models as pkg
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name != "registry":
            importlib.import_module(f"{pkg.__name__}.{info.name}")


def _matches(name, prefixes):
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def build_model(name, num_classes, pretrained=False, freeze="none"):
    if name not in _REGISTRY:
        _autodiscover()
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_REGISTRY)}")
    if freeze not in FREEZE_MODES:
        raise ValueError(f"freeze must be one of {FREEZE_MODES}, got {freeze!r}")
    spec = _REGISTRY[name]
    model = spec.builder(num_classes, pretrained=pretrained)
    model.model_name = name
    model.head_prefixes = spec.head
    if freeze != "none":
        if not spec.head:
            raise ValueError(f"model {name!r} declares no head, so freeze={freeze!r} is undefined")
        keep = spec.head + (spec.partial if freeze == "partial" else ())
        for pname, p in model.named_parameters():
            p.requires_grad = _matches(pname, keep)
    return model


def split_parameters(model):
    """Return (backbone_params, head_params) using the model's `head_prefixes`."""
    prefixes = getattr(model, "head_prefixes", ())
    if not prefixes:
        raise ValueError("model has no head_prefixes; build it with build_model or set model.head_prefixes")
    backbone, head = [], []
    for pname, p in model.named_parameters():
        (head if _matches(pname, prefixes) else backbone).append(p)
    return backbone, head


# ---------- built-in models (torchvision) ----------

@register("resnet18", head=("fc",), partial=("layer4",))
def _resnet18(num_classes, pretrained=False):
    model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@register("resnet50", head=("fc",), partial=("layer4",))
def _resnet50(num_classes, pretrained=False):
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@register("efficientnet_b0", head=("classifier",), partial=("features.7", "features.8"))
def _efficientnet_b0(num_classes, pretrained=False):
    model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


@register("cifar_resnet18", head=("fc",), partial=("layer4",), input_size=32)
def _cifar_resnet18(num_classes, pretrained=False):
    """ResNet-18 for 32x32 inputs: 3x3 stride-1 stem, no max-pool.
    With pretrained=True every layer except the new stem starts from ImageNet weights."""
    model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------- checkpoints ----------

def save_ckpt(path, model, *, model_name, num_classes, **meta):
    """Save weights plus everything needed to rebuild the model from this file alone.
    Typical meta: cfg, class_to_idx, img_size, norm_mean, norm_std, val_acc, epoch, seed, git_commit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model_name": model_name, "num_classes": num_classes, "state_dict": state, **meta}, path)


def load_ckpt(path, map_location="cpu"):
    """Return (model, ckpt_dict). The model is rebuilt unfrozen and in eval mode."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = build_model(ckpt["model_name"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt
