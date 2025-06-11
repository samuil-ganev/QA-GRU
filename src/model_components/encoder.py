import torch
from src.layers import GRULayer

class Encoder:
    def __init__(self,
                 embedding_dim: int,
                 encoder_hidden_dim: int,
                 decoder_hidden_dim: int,
                 n_layers: int=1,
                 bias_gru: bool=True,
                 batch_first_gru: bool=True,
                 device=None,
                 dtype=None):
        
        n_layers = 1
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        self.gru = GRULayer(
            input_size=embedding_dim,
            hidden_size=encoder_hidden_dim,
            bias=bias_gru,
            batch_first=batch_first_gru,
            device=device,
            dtype=dtype
        )

    def forward(self, embedded_src_seq: torch.Tensor, src_length: torch.Tensor=None, h_initial: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_outputs, final_hidden_state = self.gru(embedded_src_seq, h_initial)
        return encoder_outputs, final_hidden_state
        
    def __call__(self, embedded_src_seq: torch.Tensor,
                 src_lengths: torch.Tensor = None,
                 h_initial: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(embedded_src_seq, src_lengths, h_initial)

    def parameters(self) -> list[torch.Tensor]:
        params = self.gru.parameters()
        return params
        
    def to(self, device):
        self.gru.to(device)
        return self
