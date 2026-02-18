import torch
from train import train_model
from infer import run_inference

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(device)
    run_inference(device)