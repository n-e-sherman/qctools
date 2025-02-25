import torch
import mlflow
import inspect
import numpy as np
import quimb.tensor as qtn
from quimb.tensor.optimize import TNOptimizer
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional

class TNLoss(ABC):

    def __init__(self, callback_metric: str='loss'):

        self.callback_metric=callback_metric
        self._kwargs = {}

    @abstractmethod
    def __call__(self, x: Any) -> torch.Tensor:
        ''' Abstract call method for actual loss. '''
        pass

    def callback(self, opt: TNOptimizer):

        mlflow.log_metric(self.callback_metric, opt.loss, opt.nevals)

    def get_log_params(self, tag: Optional[str]=None):
        
        if tag:
            tag += '-'
        else:
            tag = ''
        return {f"{tag}{k}": v for k,v in self._kwargs.items()}
    
class GaugeTensorTNLoss(TNLoss):

    def __init__(self, target, 
                 gauge: bool=True,
                 gauge_inds: Tuple[int] = (0, 0),
                 alpha_thresh: float=1e-7,
                 callback_metric: str='loss'
    ):
            super().__init__(callback_metric=callback_metric)

            self.target = target
            self.gauge = gauge
            self.gauge_inds = gauge_inds
            self.alpha_thresh = alpha_thresh

            frame = inspect.currentframe()
            args, vargs, varkw, locals = inspect.getargvalues(frame)
            skip_kwargs = ['self', 'frame','__class__', 'target', 'callback_metric']
            self._kwargs = {key: locals[key] for key in locals if key not in skip_kwargs}

    def __call__(self, circ: qtn.Circuit):
         
        U = circ.get_uni().to_dense()
        if self.gauge:
            alpha = U[*self.gauge_inds]
            beta = self.target[*self.gauge_inds]
            if abs(alpha) > self.alpha_thresh:
                U = U * (beta / alpha) * (abs(alpha) / abs(beta))
        return (abs(U - self.target)**2).sum()
    

class BitstringTNLoss(TNLoss):

    def __init__(self, target, callback_metric: str='loss', **amplitude_kwargs):
        super().__init__(callback_metric=callback_metric)

        self.target = target
        self.amplitude_kwargs = amplitude_kwargs

        frame = inspect.currentframe()
        args, vargs, varkw, locals = inspect.getargvalues(frame)
        skip_kwargs = ['self', 'frame', '__class__', 'callback_metric', 'amplitude_kwargs']
        self._kwargs = {key: locals[key] for key in locals if key not in skip_kwargs}

    def __call__(self, circ: qtn.Circuit):

        return -torch.abs(circ.amplitude(self.target, **self.amplitude_kwargs))**2
    
class PeakTargetBitstringTNLoss(TNLoss):

    def __init__(self, target, peak_target=1., gamma: float=2, callback_metric: str='loss', **amplitude_kwargs):
        super().__init__(callback_metric=callback_metric)

        self.target=target
        self.peak_target=peak_target
        self.gamma=gamma
        self.amplitude_kwargs = amplitude_kwargs

        frame = inspect.currentframe()
        args, vargs, varkw, locals = inspect.getargvalues(frame)
        skip_kwargs = ['self', 'frame', '__class__', 'callback_metric', 'amplitude_kwargs']
        self._kwargs = {key: locals[key] for key in locals if key not in skip_kwargs}

    def __call__(self, circ: qtn.Circuit) -> torch.Tensor:

        P = torch.abs(circ.amplitude(self.target, **self.amplitude_kwargs))**2
        return (torch.abs(P-self.peak_target))**self.gamma
    
    def callback(self, opt: TNOptimizer):

        super().callback(opt)
        peak = self.peak_target - (opt.loss)**(1/self.gamma)
        mlflow.log_metric('peak', peak, opt.nevals)
        
        peak_best = self.peak_target - (opt.loss_best)**(1/self.gamma)
        msg = f"{peak:+.12f} [best: {peak_best:+.12f}]"
        opt._pbar.set_description(msg)

class PeakTargetBitstringPsiTNLoss(TNLoss):

    def __init__(self, target, peak_target=1., gamma: float=2, callback_metric: str='loss'):
        super().__init__(callback_metric=callback_metric)

        self.target=target
        self.peak_target=peak_target
        self.gamma=gamma

        frame = inspect.currentframe()
        args, vargs, varkw, locals = inspect.getargvalues(frame)
        skip_kwargs = ['self', 'frame', '__class__', 'callback_metric', 'amplitude_kwargs']
        self._kwargs = {key: locals[key] for key in locals if key not in skip_kwargs}

    def __call__(self, circ: qtn.Circuit) -> torch.Tensor:

        psi = circ.psi
        
        for i, b in enumerate(self.target):
            psi.isel_({psi.site_ind(i): int(b)})

        amplitude = psi.contract(backend="torch")
        P = torch.abs(amplitude)**2
        return (torch.abs(P-self.peak_target))**self.gamma
    
    def callback(self, opt: TNOptimizer):

        super().callback(opt)
        peak = self.peak_target - (opt.loss)**(1/self.gamma)
        mlflow.log_metric('peak', peak, opt.nevals)
        
        peak_best = self.peak_target - (opt.loss_best)**(1/self.gamma)
        msg = f"{peak:+.12f} [best: {peak_best:+.12f}]"
        opt._pbar.set_description(msg)
        
class PeakTargetGeneralTNLoss(TNLoss):

    def __init__(self, tn_loss: TNLoss, peak_target=1., callback_metric: str='loss'):
        super().__init__(callback_metric=callback_metric)

        self.tn_loss = tn_loss
        self.peak_target = peak_target
        self._kwargs = {**tn_loss._kwargs, **{'peak_target': peak_target}}

    def __call__(self, x: Any) -> torch.Tensor:
        
        # Assume tn_loss returns -P
        L = self.tn_loss(x)
        return (torch.abs(L+self.peak_target))**2

    def callback(self, opt: TNOptimizer):

        super().callback(opt)
        mlflow.log_metric('peak', self.peak_target-np.sqrt(opt.loss), opt.nevals)


class LocalSumTNLoss(TNLoss):

    def __init__(self, target, qubits: int=1, overlap: Optional[int]=None, device: str='cpu', callback_metric: str='loss'):
        
        super().__init__(callback_metric=callback_metric)

        self.target=target
        self.qubits=qubits
        self.overlap=overlap if overlap is not None else qubits-1
        self.device=device
        self._kwargs = {'target': self.target, 'qubits': self.qubits, 'overlap': self.overlap}

        assert self.qubits > self.overlap, f"overlap: {overlap} must be less than qubits: {qubits}"

    def __call__(self, circ):

        res = torch.zeros(1, device=self.device)
        N = circ.N
        for i in range(0, N, self.qubits-self.overlap):
            sites = [(i+x)%N for x in range(self.qubits)]
            local_target = ""
            for site in sites:
                local_target += self.target[site]
            rho = circ.partial_trace(sites)
            ind = int(local_target, 2)
            res += torch.abs(rho[ind, ind])
        return res
            