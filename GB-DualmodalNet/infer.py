import torch
import numpy as np
from models.gb_dualmodal import GBDualmodalNet

def run_inference(device):
    model = GBDualmodalNet().to(device)
    model.load_state_dict(torch.load("gb_dualmodal.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        image = torch.randn(1,3,224,224).to(device)
        input_ids = torch.randint(0,1000,(1,128)).to(device)
        attention_mask = torch.ones(1,128).to(device)
        output = model(image, input_ids, attention_mask)
        prob = torch.softmax(output, dim=1).cpu().numpy()
    return prob