import torch
import torch.nn as nn
import timm


class CNNTransformerHybrid(nn.Module):

    def __init__(
        self,
        cnn_name: str = "resnet50",
        vit_name: str = "vit_base_patch16_224",
        num_classes: int = 3,
        pretrained: bool = True
    ):
        super().__init__()

        self.cnn = timm.create_model(
            cnn_name,
            pretrained=pretrained,
            num_classes=0
        )

        self.vit = timm.create_model(
            vit_name,
            pretrained=pretrained,
            num_classes=0
        )

        cnn_features = self.cnn.num_features
        vit_features = self.vit.num_features

        fusion_dim = cnn_features + vit_features

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f1 = self.cnn(x)
        f2 = self.vit(x)

        fused = torch.cat((f1, f2), dim=1)
        return self.classifier(fused)


class MultiCNNTransformerHybrid(nn.Module):

    def __init__(
        self,
        cnn_list=("resnet50", "efficientnet_b0"),
        vit_name="vit_base_patch16_224",
        num_classes=3,
        pretrained=True
    ):
        super().__init__()

        self.cnns = nn.ModuleList([
            timm.create_model(cnn, pretrained=pretrained, num_classes=0)
            for cnn in cnn_list
        ])

        self.vit = timm.create_model(
            vit_name,
            pretrained=pretrained,
            num_classes=0
        )

        cnn_features = sum(cnn.num_features for cnn in self.cnns)
        vit_features = self.vit.num_features

        fusion_dim = cnn_features + vit_features

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_feats = [cnn(x) for cnn in self.cnns]
        vit_feat = self.vit(x)

        fused = torch.cat(cnn_feats + [vit_feat], dim=1)
        return self.classifier(fused)



def get_hybrid_model(
    cnn_name="resnet50",
    vit_name="vit_base_patch16_224",
    num_classes=3
):
    return CNNTransformerHybrid(
        cnn_name=cnn_name,
        vit_name=vit_name,
        num_classes=num_classes
    )


def get_multi_hybrid_model(
    cnn_list=("resnet50", "efficientnet_b0"),
    vit_name="vit_base_patch16_224",
    num_classes=3
):
    return MultiCNNTransformerHybrid(
        cnn_list=cnn_list,
        vit_name=vit_name,
        num_classes=num_classes
    )