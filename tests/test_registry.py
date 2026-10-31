import sys

import pytest
import torch
from torch import nn

import src.models
from src.models.registry import (
    available_models, build_model, load_ckpt, model_input_size, register, save_ckpt, split_parameters,
)


@pytest.mark.parametrize("num_classes", [2, 10])
@pytest.mark.parametrize("name,size", [
    ("resnet18", 64), ("resnet50", 64), ("efficientnet_b0", 64), ("cifar_resnet18", 32),
])
def test_builtin_models_output_one_logit_per_class(name, size, num_classes):
    model = build_model(name, num_classes, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, size, size))
    assert out.shape == (2, num_classes)


def test_cifar_resnet18_stem_keeps_32px_resolution():
    model = build_model("cifar_resnet18", 10)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)


def test_pretrained_resnet18_loads_imagenet_weights():
    a = build_model("resnet18", 2, pretrained=True)
    b = build_model("resnet18", 2, pretrained=True)
    assert torch.equal(a.conv1.weight, b.conv1.weight)  # same downloaded weights, not random init
    assert a.fc.out_features == 2


def test_build_model_tags_name_and_head():
    model = build_model("resnet18", 2)
    assert model.model_name == "resnet18"
    assert model.head_prefixes == ("fc",)


def test_unknown_model_error_lists_available_names():
    with pytest.raises(KeyError, match="resnet18"):
        build_model("no_such_model", 2)


class _Lin(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.body = nn.Linear(4, 4)
        self.fc = nn.Linear(4, k)

    def forward(self, x):
        return self.fc(self.body(x))


def test_register_makes_model_buildable():
    @register("reg_test_lin", head=("fc",))
    def _build(num_classes, pretrained=False):
        return _Lin(num_classes)

    model = build_model("reg_test_lin", 3)
    assert "reg_test_lin" in available_models()
    assert model(torch.randn(1, 4)).shape == (1, 3)


def test_register_rejects_duplicate_name():
    with pytest.raises(ValueError, match="resnet18"):
        register("resnet18")(lambda num_classes, pretrained=False: None)


def test_freeze_backbone_leaves_only_head_trainable():
    model = build_model("resnet18", 2, freeze="backbone")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}


def test_freeze_partial_trains_last_stage_and_head():
    model = build_model("resnet18", 2, freeze="partial")
    trainable = {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"layer4", "fc"}


def test_freeze_rejects_unknown_mode():
    with pytest.raises(ValueError, match="freeze"):
        build_model("resnet18", 2, freeze="everything")


def test_freeze_requires_declared_head():
    @register("reg_test_headless", head=())
    def _build(num_classes, pretrained=False):
        return _Lin(num_classes)

    with pytest.raises(ValueError, match="head"):
        build_model("reg_test_headless", 2, freeze="backbone")


def test_split_parameters_separates_head_from_backbone():
    model = build_model("resnet18", 2)
    backbone, head = split_parameters(model)
    assert {id(p) for p in head} == {id(p) for p in model.fc.parameters()}
    assert len(backbone) + len(head) == len(list(model.parameters()))


def test_ckpt_roundtrip_rebuilds_identical_model(tmp_path):
    torch.manual_seed(0)
    model = build_model("cifar_resnet18", 10).eval()
    path = tmp_path / "m.pt"
    save_ckpt(path, model, model_name="cifar_resnet18", num_classes=10, epoch=3, cfg={"lr": 0.1})
    rebuilt, ckpt = load_ckpt(path)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(x), rebuilt.eval()(x))
    assert ckpt["epoch"] == 3
    assert ckpt["cfg"] == {"lr": 0.1}


def test_unknown_name_triggers_autodiscovery_of_models_package(tmp_path, monkeypatch):
    (tmp_path / "autodisc_mod.py").write_text(
        "from torch import nn\n"
        "from src.models.registry import register\n"
        "@register('autodisc_test_net')\n"
        "def build(num_classes, pretrained=False):\n"
        "    return nn.Linear(4, num_classes)\n"
    )
    monkeypatch.setattr(src.models, "__path__", list(src.models.__path__) + [str(tmp_path)])
    monkeypatch.delitem(sys.modules, "src.models.autodisc_mod", raising=False)
    model = build_model("autodisc_test_net", 5)
    assert model.out_features == 5


def test_registered_models_accept_channels_last_input():
    """Guards models added later (e.g. A's scratch CNN): flatten with torch.flatten / reshape, never .view."""
    for name in available_models():
        if "test" in name:
            continue
        size = model_input_size(name) or 64
        model = build_model(name, 3, pretrained=False).to(memory_format=torch.channels_last)
        x = torch.randn(2, 3, size, size).contiguous(memory_format=torch.channels_last)
        model(x).sum().backward()


def test_register_records_input_size():
    assert model_input_size("cifar_resnet18") == 32
    assert model_input_size("resnet18") is None
