# %%
import torch
from torch import nn, relu
from .vision_transformer import VisionTransformerHead
from .config import *

class SceneClassifier(nn.Module):
    def __init__(self, arch, final_dim) -> None:
        super(SceneClassifier, self).__init__()
        # self.net = torch.hub.load("facebookresearch/swag", model=arch)
        self.net = torch.hub.load("/home/tb5zhh/.cache/torch/hub/facebookresearch_swag_main/", model=arch, source='local')
        self.head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, final_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x


def get_network(arch, final_dim) -> nn.Module:
    return SceneClassifier(arch, final_dim)
        
