import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel

class CrossModalAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, img_feat, txt_feat):
        fused, _ = self.attn(img_feat, txt_feat, txt_feat)
        return self.norm(fused + img_feat)

class GBDualmodalNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(vgg.features.children()), nn.AdaptiveAvgPool2d((7,7)))
        self.img_fc = nn.Linear(512*7*7, 768)
        self.text_encoder = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        self.attn = CrossModalAttention(768)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.cnn(image)
        img_feat = img_feat.view(img_feat.size(0), -1)
        img_feat = self.img_fc(img_feat).unsqueeze(1)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = text_outputs.last_hidden_state
        fused = self.attn(img_feat, txt_feat)
        fused = fused.mean(dim=1)
        out = self.classifier(fused)
        return out