import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils.clinical_prompt import build_prompt

class GBPLDataset(Dataset):
    def __init__(self, records, images, labels):
        self.records = records
        self.images = images
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

    def __len__(self):
        return len(self.labels)

    def preprocess(self, img):
        img = cv2.resize(img, (224,224))
        img = img.astype(np.float32)/255.0
        img = np.transpose(img, (2,0,1))
        return torch.tensor(img)

    def __getitem__(self, idx):
        img = self.preprocess(self.images[idx])
        prompt = build_prompt(self.records[idx])
        tokens = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        label = torch.tensor(self.labels[idx])
        return img, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0), label