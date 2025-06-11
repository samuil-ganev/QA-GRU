import torch
import torch.nn as nn
from src.layers import GRULayer, Attention

class Decoder:
    def __init__(self,
                 output_vocab_size: int,
                 embedding_dim: int,
                 encoder_hidden_dim: int,
                 decoder_hidden_dim: int,
                 attention_dim: int=None,
                 n_layers_gru: int=1,
                 bias_gru: bool=True,
                 batch_first_gru: bool=True,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device,
                          'dtype': dtype
                          }
        self.n_layers_gru = 1
        self.decoder_hidden_dim = decoder_hidden_dim
        self.output_vocab_size = output_vocab_size
        self.batch_first_gru = batch_first_gru
        self.attention = Attention(
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            attention_dim=attention_dim,
            **factory_kwargs
        )

        gru_input_dim = embedding_dim + encoder_hidden_dim
        self.gru = GRULayer(
            input_size=gru_input_dim,
            hidden_size=decoder_hidden_dim,
            bias=bias_gru,
            batch_first=batch_first_gru,
            **factory_kwargs
        )

        self.fc_out = nn.Linear(decoder_hidden_dim, output_vocab_size, **factory_kwargs)

    def forward(self,
                embedded_input_token: torch.Tensor,
                decoder_hidden_state: torch.Tensor,
                encoder_outputs: torch.Tensor,
                encoder_output_mask: torch.Tensor=None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_vector, attention_weights = self.attention(
            decoder_hidden=decoder_hidden_state,
            encoder_outputs=encoder_outputs,
            encoder_output_mask=encoder_output_mask
        )
        gru_input = torch.cat((embedded_input_token, context_vector), dim=1)
        # gru_input.shape = (batch_size, embedding_dim + encoder_hidden_dim)

        if self.batch_first_gru:
            gru_input_for_layer = gru_input.unsqueeze(1)
            # gru_input_for_layer.shape = (batch_size, 1, embedding_dim + encoder_hidden_dim)

            gru_output_seq, new_decoder_hidden_state = self.gru(gru_input_for_layer, decoder_hidden_state)
            gru_output_single_step = gru_output_seq.squeeze(1)
        else:
            gru_input_for_layer = gru_input.unsqueeze(0)
            gru_output_seq, new_decoder_hidden_state = self.gru(gru_input_for_layer, decoder_hidden_state)
            gru_output_single_step = gru_output_seq.squeeze(0)

        output_logits = self.fc_out(gru_output_single_step)

        return output_logits, new_decoder_hidden_state, attention_weights

    def __call__(self,
                 embedded_input_token: torch.Tensor,
                 decoder_hidden_state: torch.Tensor,
                 encoder_outputs: torch.Tensor,
                 encoder_output_mask: torch.Tensor = None
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(
            embedded_input_token=embedded_input_token,
            decoder_hidden_state=decoder_hidden_state,
            encoder_outputs=encoder_outputs,
            encoder_output_mask=encoder_output_mask
        )
    
    def parameters(self) -> list[torch.Tensor]:
        params = []
        params.extend(self.attention.parameters())
        params.extend(self.gru.parameters())
        params.extend(list(self.fc_out.parameters()))
        return params
    
    def to(self, device):
        self.attention.to(device)
        self.gru.to(device)
        self.fc_out.to(device)
        return self
