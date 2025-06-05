import torch
from transformers import AutoTokenizer, AutoModel

class PretrainedAssets:
    def __init__(self, model_name_or_path: str, device: torch.device):
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model.to(device)

        