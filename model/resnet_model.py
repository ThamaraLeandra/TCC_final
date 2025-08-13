import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_resnet18_model(pretrained: bool = True, num_classes: int = 4) -> nn.Module:
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
