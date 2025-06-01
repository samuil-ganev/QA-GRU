import torch
import torch.nn as nn
import math

class GRUCell:
    def __init__(self, input_size: int, hidden_size: int, bias: bool=True, device=None, dtype=None):
        '''
        h_t = u_t * h~_t + (1 - u_t) * h_{t-1}
        h~_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_h))
        u_t = sigmoid(W_iu @ x_t + b_iz + W_hu @ h_{t-1} + b_z)
        r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_r)
        '''

        factory_kwargs = {'device': device,
                          'dtype': dtype
                          }
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_i = nn.Linear(input_size, 3 * hidden_size, bias=bias, **factory_kwargs)
        # W_i = [W_ir, W_iu, W_in]
        self.W_h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias, **factory_kwargs)
        # W_h = [W_hr, W_hu, W_hn]
    
    def forward(self, input_tensor: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros(input_tensor.size(0), self.hidden_size, device=input_tensor.device, dtype=input_tensor.dtype)
        
        gi = self.W_i(input_tensor)
        # gi.shape = (batch, 3 * hidden_size)
        gh = self.W_h(h_prev)
        # gh.shape = (batch, 3 * hidden_size)

        i_r, i_u, i_n = gi.chunk(3, 1)
        h_r, h_u, h_n = gh.chunk(3, 1)

        r_t = torch.sigmoid(i_r + h_r)
        u_t = torch.sigmoid(i_u + h_u)
        _h_t = torch.tanh(i_n + r_t * h_n)
        h_t = u_t * _h_t + (1 - u_t) * h_prev

        return h_t
    
    def parameters(self) -> list[torch.Tensor]:
        params = []
        params.extend(list(self.W_i.parameters()))
        params.extend(list(self.W_h.parameters()))
        return params

    def __call__(self, input_tensor: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return self.forward(input_tensor, h_prev)

    def to(self, device):
        self.W_i.to(device)
        self.W_h.to(device)
        return self
