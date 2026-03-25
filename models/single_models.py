import torch
import torch.nn as nn
import timm
import torchvision.models as models


SUPPORTED_MODELS = {
    "resnet50": "resnet50",
    "efficientnet": "efficientnet_b0",
    "efficientnet_b4": "efficientnet_b4",
    "densenet": "densenet121",
    "inception": "inception_v3",
    "vit": "vit_base_patch16_224",
    "convnext": "convnext_tiny"
}


# -------------------------------
# TIMM MODELS (FIXED)
# -------------------------------
def build_model(model_name: str, num_classes: int = 3, pretrained: bool = True):

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes   # 🔥 IMPORTANT FIX
    )

    return model


# -------------------------------
# VGG16 (OK)
# -------------------------------
def build_vgg16(num_classes=3):

    backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    for param in backbone.features.parameters():
        param.requires_grad = False

    in_features = backbone.classifier[-1].in_features

    backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    return backbone


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def get_single_model(name: str, num_classes: int = 3, pretrained: bool = True):

    if name == "vgg16":
        return build_vgg16(num_classes)

    if name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{name}' not supported. Choose from {list(SUPPORTED_MODELS.keys()) + ['vgg16']}"
        )

    return build_model(SUPPORTED_MODELS[name], num_classes, pretrained)


# -------------------------------
# FREEZE BACKBONE (FIXED)
# -------------------------------
def freeze_backbone(model):

    for name, param in model.named_parameters():
        if not any(k in name for k in ["classifier", "fc", "head"]):
            param.requires_grad = False

    return model


# -------------------------------
# PARTIAL UNFREEZE
# -------------------------------
def unfreeze_last_layers(model, num_layers=1):

    if hasattr(model, "blocks"):  # ViT / ConvNeXt
        for block in model.blocks[-num_layers:]:
            for param in block.parameters():
                param.requires_grad = True

    elif hasattr(model, "layer4"):  # ResNet
        for param in model.layer4.parameters():
            param.requires_grad = True

    return model