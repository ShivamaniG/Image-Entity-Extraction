import torch
import torch.nn as nn
import torchvision.models as models


class EntityExtractor(nn.Module):
    def __init__(self, num_classes, image_size=(224, 224), pretrained=True):
        super(EntityExtractor, self).__init__()

        self.backbone = models.resnet50(pretrained=pretrained)

        num_features = self.backbone.fc.in_features
        self.fc = nn.Linear(num_features, num_classes)


    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x