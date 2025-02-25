import torch
import numpy as np
from typing import Optional, Any

class TorchConverter:

    def __init__(self, device: str='cpu', dtype: Optional[Any]=None):

        self.device = device
        self.dtype = dtype

    def __call__(self, x: Any) -> torch.Tensor:
        
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x, device=self.device, dtype=self.dtype)
        
        # Only apply type conversion if dtype is not None
        if self.dtype is not None:
            x = x.type(self.dtype)

        return x.to(self.device)

class NumpyConverter:

    def __init__(self):
        pass

    def __call__(self, x: Any) -> np.ndarray:

        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, torch.Tensor):
            return x.cpu().detach().numpy()
        else:
            return np.ndarray(x)
        

class RealConverter:

    def __init__(self):
        pass

    def __call__(self, x: Any) -> Any:
        return x.real
    
class IdentityFunction:

    def __init__(self):
        pass

    def __call__(self, x: Any) -> Any:
        return x

