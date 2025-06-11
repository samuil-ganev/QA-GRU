import torch
from transformers import AutoTokenizer, AutoModel

class PretrainedAssets:
    def __init__(self, model_name_or_path: str, device: torch.device):
        self.model_name = model_name_or_path
        model = AutoModel.from_pretrained(model_name_or_path)
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model.to(device)

        if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'word_embeddings'):
            self.embedding_weights = model.embeddings.word_embeddings.weight.detach().clone().to(device)
        elif hasattr(model, 'shared'):
            self.embedding_weights = model.shared.weight.detach().clone().to(device)
        elif hasattr(model, 'wte'):
            self.embedding_weights = model.wte.weight.detach().clone().to(device)
        else:
            raise AttributeError('не е намерена ембединг матрица')

        self.embedding_dim = self.embedding_weights.size(1)
        self.vocab_size = self.embedding_weights.size(0)
        self.pad_token_id = self.tokenizer.pad_token_id

        self.sos_token_id = self.tokenizer.cls_token_id
        if self.sos_token_id is None:
            self.sos_token_id = self.tokenizer.bos_token_id
        
        self.eos_token_id = self.tokenizer.sep_token_id
        if self.eos_token_id is None:
            self.eos_token_id = self.tokenizer.eos_token_id
        
        self.embedding_weights.requires_grad_(False)

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_weights[input_ids.long()]

    def tokenize_and_pad(self, texts: list[str], max_length: int, add_special_tokens: bool=True) -> dict:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.pad_token_id
        
        tokenized_output = self.tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return tokenized_output
