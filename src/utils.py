import torch
import os
import json

def save_parameters(model_parameters: list[torch.Tensor], path: str):
    torch.save([param.data.clone() for param in model_parameters], path)
    print(f'Моделът е запазен в {path}')

def load_parameters(model_parameters_to_load_into: list[torch.Tensor], path: str, device: torch.device):
    if not os.path.exists(path):
        print(f'Не е намерен модел за зареждане в {path}')
        return False
    
    loaded_params_data = torch.load(path, map_location=device)

    with torch.no_grad():
        for i, param_to_load in enumerate(model_parameters_to_load_into):
            loaded_data = loaded_params_data[i]
            if param_to_load.data.shape != loaded_data.shape:
                return False
            param_to_load.data.copy_(loaded_data)
    
    return True

def save_config(config_dict: dict, path: str):
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=4)

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        config_dict = json.load(f)
    return config_dict

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def count_parameters(model_params: list[torch.Tensor]) -> int:
    return sum(p.numel() for p in model_params if p.requires_grad)

def format_time(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f'{h:02d}:{m:02d}:{s:02d}'
