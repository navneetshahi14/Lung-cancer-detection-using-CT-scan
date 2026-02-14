import torch.nn as nn
import timm
import torchvision.models as models


def build_model(model_name: str, num_classes: int = 3, pretrained:bool = True):

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes = 0
    )

    in_features = model.num_features

    classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    model = nn.Sequential(
        model,
        classifier
    )

    return model


SUPPORTED_MODELS = {
    "resnet50": "resnet50",
    "vgg16": "vgg16",
    "efficientnet": "efficientnet_b0",
    "inception": "inception_v3",
    "vit": "vit_base_patch16_224",
    "densenet":"densenet121",
    "b4":"efficientnet_b4",
    "convextiny":"convnext_tiny"
}

# cnn_models = [
#     "resnet50",
#     "densenet121",
#     "efficientnet_b0",
#     "inception_v3",
#     "efficientnet_b4",
#     "convnext_tiny"
# ]


def get_single_model(name: str, num_classes: int = 3):

    if name == "vgg16":
        backbone = models.vgg16(pretrained=True)

        for param in backbone.features.parameters():
            param.requires_grad = False

        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)

        return backbone

    if name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{name}' not supported. Choose from {list(SUPPORTED_MODELS.keys()) + ['vgg16']}"
        )

    return build_model(SUPPORTED_MODELS[name], num_classes=num_classes)
