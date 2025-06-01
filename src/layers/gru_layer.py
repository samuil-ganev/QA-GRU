import torch
from .gru_cell import GRUCell

class GRULayer:
    def __init__(self, input_size: int, hidden_size: int, bias: bool=True, device=None, dtype=None):
        factory_kwargs = {'device': device,
                          'dtype': dtype
                          }
        self.cell = GRUCell(input_size, hidden_size, bias=bias, **factory_kwargs)
        self.hidden_size = hidden_size

    def forward(self, input_seq: torch.Tensor, h_initial: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = input_seq.shape
        
        if h_initial is None:
            h_initial = torch.zeros(batch_size, self.hidden_size, device=input_seq.device, dtype=input_seq.dtype)
        h_t = h_initial

        outputs = []
        for t in range(seq_len):
            h_t = self.cell(input_seq[t], h_t)
            outputs.append(h_t)
        
        output_seq = torch.stack(outputs, dim=1)
        # output_seq.shape = (batch, seq_len, hidden_size)
        return output_seq, h_t
    
    def parameters(self) -> list[torch.Tensor]:
        return self.cell.parameters()
    
    def __call__(self, input_seq: torch.Tensor, h_initial: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(input_seq, h_initial)
        
    def to(self, device):
        self.cell.to(device)
        return self
