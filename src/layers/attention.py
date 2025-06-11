import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention:
    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int, attention_dim: int=None, device=None, dtype=None):
        '''
        score(h_enc, h_dec) := v_a.T @ tanh(W_a @ h_enc + U_a @ h_dec)
        '''
        factor_kwargs = {'device': device,
                         'dtype': dtype
                         }
        self.attention_dim = attention_dim if attention_dim is not None else decoder_hidden_dim
        self.W_enc = nn.Linear(encoder_hidden_dim, self.attention_dim, bias=False, **factor_kwargs)
        self.W_dec = nn.Linear(decoder_hidden_dim, self.attention_dim, bias=False, **factor_kwargs)
        self.v_att = nn.Linear(self.attention_dim, 1, bias=False, **factor_kwargs)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, encoder_output_mask: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len_enc, _ = encoder_outputs.shape
        
        proj_dec = self.W_dec(decoder_hidden)
        # proj_dec.shape = (batch_size, attn_dim)
        proj_dec_expanded = proj_dec.unsqueeze(1)
        # proj_dec_expanded.shape = (batch_size, 1, attn_dim)

        proj_enc = self.W_enc(encoder_outputs)
        # proj_enc.shape = (batch_size, seq_len_enc, attn_dim)

        scores = self.v_att(proj_enc + proj_dec_expanded)
        # scores.shape = (batch_size, seq_len_enc, 1)
        scores = scores.squeeze(2)

        if encoder_output_mask is not None:
            scores.masked_fill_(encoder_output_mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=1)
        # attention_weights.shape = (batch_size, seq_len_enc)

        context_vector = (attention_weights.unsqueeze(2) * encoder_outputs).sum(dim=1)
        # context_vector.shape = (batch_size, enc_hid_dim)

        return context_vector, attention_weights

    def parameters(self) -> list[torch.Tensor]:
        params = []
        params.extend(list(self.W_enc.parameters()))
        params.extend(list(self.W_dec.parameters()))
        params.extend(list(self.v_att.parameters()))
        return params
    
    def __call__(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, encoder_output_mask: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(decoder_hidden, encoder_outputs, encoder_output_mask)

    def to(self, device):
        self.W_enc.to(device)
        self.W_dec.to(device)
        self.v_att.to(device)
        return self
