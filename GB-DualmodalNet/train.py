import torch
from torch.utils.data import DataLoader
from models.gb_dualmodal import GBDualmodalNet
from data.dataset import GBPLDataset
from sklearn.metrics import roc_auc_score
import numpy as np

def train_model(device):
    records = []
    images = []
    labels = []
    dataset = GBPLDataset(records, images, labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = GBDualmodalNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(10):
        for img, ids, mask, label in loader:
            img = img.to(device)
            ids = ids.to(device)
            mask = mask.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out = model(img, ids, mask)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), "gb_dualmodal.pth")