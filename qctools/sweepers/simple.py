import inspect
import numpy as np
import quimb.tensor as qtn

from typing import Optional, Callable, List, Dict, Any

from qctools.utils import TorchConverter
from ._base import DeformationSweeper


class SimpleDeformationSweeper(DeformationSweeper):

    def __init__(self, attempts: int=10, to_backend: Callable=TorchConverter(), 
                 epochs: int=1000, rel_tol: float=1e-8, tol: float=1e-3, progbar: bool=True,
                 noise_dist: str='normal', noise_mean: float=0, noise_std: float=0.4,
                 round_start: int=0, round_end: Optional[int]=None, depth: int=4, stride: Optional[int]=None, shift_last_patch: bool=True,
                 early_stopping: bool=True,
                 **opt_args
    ):
        super().__init__(attempts=attempts, to_backend=to_backend, 
                         epochs=epochs, rel_tol=rel_tol, tol=tol, progbar=progbar, early_stopping=early_stopping,
                         **opt_args
        )
        _noise_func = {
                'normal': np.random.normal,
                'uniform': np.random.uniform
        }.get(noise_dist, np.random.normal)
        self.noise_func = lambda x: _noise_func(noise_mean, noise_std, x.shape)

        self.round_start = round_start
        self.round_end = round_end
        self.depth = depth
        self.stride = stride
        self.shift_last_patch = shift_last_patch
        self.early_stopping = early_stopping

        frame = inspect.currentframe()
        args, vargs, varkw, locals = inspect.getargvalues(frame)
        skip_kwargs = ['self', 'to_backend', 'progbar', 'opt_args',  '_noise_func', 'frame']
        self._kwargs = {key: locals[key] for key in locals if key not in skip_kwargs}
        self._kwargs.update(opt_args)

    def get_log_params(self):

        return self._kwargs

    def get_schedule(self, qc_full: qtn.Circuit) -> List[Dict[Any, Any]]:
        
        all_rounds = list(set([gate.round for gate in qc_full.gates]))
        round_start = self.round_start
        round_end = self.round_end if self.round_end else max(all_rounds)
        depth = self.depth
        stride = self.stride if self.stride else depth

        sweep_schedule = []
        for round_patch in range(round_start, round_end-depth, stride):
            patch_args = {}
            patch_args['patch_target'] = {'round_start': round_patch, 'depth': depth}
            sweep_schedule.append(patch_args)
        
        # add last patch
        patch_args = {}
        if self.shift_last_patch:
            patch_args['patch_target'] = {'round_start': round_end-depth, 'depth': depth}
        else:
            patch_args['patch_target'] = {'round_start': round_patch+stride, 'depth': round_end-(round_patch+stride)}
        sweep_schedule.append(patch_args)

        return sweep_schedule

    def get_patch_target(self, qc: qtn.Circuit, round_start: int=0, depth: int=4) -> qtn.Circuit:

        qc_target = qtn.Circuit(qc.N)
        qc_target.apply_to_arrays(self.to_backend)
        rounds = [round_start + i for i in range(depth)]
        for gate in qc.gates:
            if gate.round in rounds:
                qc_target.apply_gate(gate)
        return qc_target
    
    def get_patch_train(self, qc_target: qtn.Circuit):

        qc_train = qtn.Circuit(qc_target.N)
        qc_train.apply_to_arrays(self.to_backend)

        for gate in qc_target.gates:
            params = self.to_backend(gate.params)
            noise = self.to_backend(self.noise_func(params))
            qc_train.apply_gate(gate.label, params=params+noise, qubits=gate.qubits, parametrize=True, gate_round=gate.round)
        
        return qc_train