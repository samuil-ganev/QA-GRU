import torch
import random
from src.pretrained_utils import PretrainedAssets
from encoder import Encoder
from decoder import Decoder

class Seq2Seq:
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 pretrained_assets,
                 device: torch.device):
        self.encoder = encoder
        self.decoder = decoder
        self.pretrained_assets = pretrained_assets
        self.device = device
        
        self.encoder.to(device)
        self.decoder.to(device)

    def create_target_mask(self, trg_input_ids: torch.Tensor) -> torch.Tensor:
        trg_pad_idx = self.pretrained_assets.pad_token_id
        trg_mask = (trg_input_ids != trg_pad_idx)
        # trg_mark.shape = (batch_size, trg_len)
        return trg_mask
    
    def create_encoder_output_mask(self, src_input_ids: torch.Tensor) -> torch.Tensor:
        src_pad_idx = self.pretrained_assets.pad_token_id
        src_mask = (src_input_ids != src_pad_idx)
        # src_mask.shape = (batch_size, src_len)
        return src_mask
    
    def forward(self,
                src_input_ids: torch.Tensor,
                trg_input_ids: torch.Tensor=None,
                teacher_forcing_ratio: float=0.5,
                max_output_len: int=50
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = src_input_ids.shape[0]
        is_training = trg_input_ids is not None
        output_seq_len = trg_input_ids.shape[1]-1 if is_training else max_output_len

        vocab_size = self.decoder.output_vocab_size
        output_logits = torch.zeros(batch_size, output_seq_len, vocab_size).to(self.device)
        all_attention_weights = torch.zeros(batch_size, output_seq_len, src_input_ids.shape[1]).to(self.device)

        embedded_src = self.pretrained_assets.get_embeddings(src_input_ids)
        encoder_mask = self.create_encoder_output_mask(src_input_ids)
        encoder_outputs, decoder_hidden_state = self.encoder(embedded_src)

        decoder_input_ids_t = torch.full((batch_size, 1), self.pretrained_assets.sos_token_id, dtype=torch.long, device=self.device)

        for t in range(output_seq_len):
            embedded_decoder_input_t = self.pretrained_assets.get_embeddings(decoder_input_ids_t.squeeze(1))
            # embedded_decoder_input_ids.shape = (batch_size, embedding_dim)

            output_logits_t, decoder_hidden_state, attention_weights_t = self.decoder(
                                                                                        embedded_input_token=embedded_decoder_input_t,
                                                                                        decoder_hidden_state=decoder_hidden_state,
                                                                                        encoder_outputs=encoder_outputs,
                                                                                        encoder_output_mask=encoder_mask
                                                                                    )
            output_logits[:, t, :] = output_logits_t
            if all_attention_weights is not None:
                all_attention_weights[:, t, :] = attention_weights_t
            if is_training and random.random() < teacher_forcing_ratio:
                decoder_input_ids_t = trg_input_ids[:, t+1].unsqueeze(1)
            else:
                top1_predicted_token_ids = output_logits_t.argmax(1)
                decoder_input_ids_t = top1_predicted_token_ids.unsqueeze(1)
        
        return output_logits, all_attention_weights
    
    def parameters(self) -> list[torch.Tensor]:
        params = self.encoder.parameters() + self.decoder.parameters()
        return params

    def train_mode(self):
        pass

    def eval_mode(self):
        pass