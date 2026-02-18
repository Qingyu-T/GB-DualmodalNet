import torch
import torch.nn as nn

class SegmentationModel(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.model = torch.load(checkpoint_path, map_location="cpu")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            mask = self.model(x)
        return mask