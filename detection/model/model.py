import torch.nn as nn
import torchvision


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1000, 1)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x
