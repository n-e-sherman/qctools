import warnings
import torch
import inspect
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize import TNOptimizer
from typing import Union, Tuple, Callable, Iterable, Optional, Dict

from .echo_mosaic import EchoMosaicCircuitManager
from qctools.gates import Gate, Haar4Gate

class EnhancedEchoMosaicCircuitManager(EchoMosaicCircuitManager):

    def __init__(self, N, 
                 tau_r, 
                 tau_o: int=1, # defaults to 1
                 tau_i: Optional[int]=None, # defaults to tau_r
                 tau_e: int=0, # defaults to 0
                 tau_s: int=2,
                 tau_p: int=2,
                 gate: Gate=Haar4Gate(), 
                 gate_o: Optional[Gate]=None,
                 gate_i: Optional[Gate]=None,
                 gate_e: Optional[Gate]=None,
                 gate_s: Optional[Gate]=None,
                 gate_p: Optional[Gate]=None,
                 pbc: bool=False, 
                 pbc_o: Optional[bool]=None, 
                 pbc_i: Optional[bool]=None, 
                 pbc_e: Optional[bool]=None,
                 pbc_s: Optional[bool]=None,
                 pbc_p: Optional[bool]=None,
                 shift: int=0,
                 epochs: int=500,
                 tol: float=1e-8,
                 identity_tol: float=1e-7,
                 identity_attempts: int=20,
                 oqc_tol: Optional[float] = None,
                 oqc_identity_tol: Optional[float]=None,
                 oqc_identity_attempts: Optional[int]=None, 
                 gauge: bool=True,
                 gauge_inds: Tuple[int]=(0, 0), 
                 alpha_thresh:float=1e-7,
                 progbar: bool=True,
                 qasm: Optional[str] = None,
                 to_backend: Optional[Callable]=None,
                 to_frontend: Optional[Callable]=None,
    ):

        
        super().__init__(N, tau_r, 
                         tau_o=tau_o, tau_i=tau_i, tau_e=tau_e,
                         gate=gate, gate_o=gate_o, gate_i=gate_i, gate_e=gate_e,
                         pbc=pbc, pbc_o=pbc_o, pbc_i=pbc_i, pbc_e=pbc_e, shift=shift,
                         epochs=epochs, tol=tol, identity_tol=identity_tol, identity_attempts=identity_attempts,
                         oqc_tol=oqc_tol, oqc_identity_tol=oqc_identity_tol, oqc_identity_attempts=oqc_identity_attempts,
                         gauge=gauge, gauge_inds=gauge_inds, alpha_thresh=alpha_thresh, progbar=progbar,
                         qasm=qasm, to_backend=to_backend, to_frontend=to_frontend)

        self.tau_s = tau_s
        self.tau_p = tau_p

        self.pbc_s = pbc_s if pbc_s is not None else pbc
        self.pbc_p = pbc_p if pbc_p is not None else pbc

        # shift handling, shift corresponds to the shift of RQC, others have prefix
        self.shift_s = (shift+tau_s)%2
        self.shift_p = (shift+1)%2
        self.shift_e = (self.shift_p + tau_p)%2

        self.gate_s = gate_s if gate_s is not None else gate
        self.gate_p = gate_p if gate_p is not None else gate

        frame = inspect.currentframe()
        args, vargs, varkw, locals = inspect.getargvalues(frame)
        skip_kwargs = ['self', 'frame', 'to_backend', 'to_frontend', 'gate', 'gate_o', 'gate_i', 'gate_e', 'progbar']
        self._kwargs = {key: locals[key] for key in locals if key not in skip_kwargs}

        self.sqc = self.rqc = self.oqc = self.iqc = self.pqc = self.eqc = None

    def build(self, end: str='back', which: Optional[Union[str|Iterable]]=None, parametrize: Union[bool, Dict[str, bool]]=False, **kwargs) -> None:
        """Abstract build method to construct specified circuit parts."""
        which_list, parametrize_map = self._process_which_and_parametrize(which, parametrize)

        for circ_name in which_list:
            if circ_name.lower() == 'sqc':
                self._build_sqc(parametrize=parametrize_map[circ_name], end=end, **kwargs)
            elif circ_name.lower() == 'rqc':
                self._build_rqc(parametrize=parametrize_map[circ_name], end=end, **kwargs)
            elif circ_name.lower() == 'oqc':
                self._build_oqc(parametrize=parametrize_map[circ_name], end=end, **kwargs)
            elif circ_name.lower() == 'iqc':
                self._build_iqc(parametrize=parametrize_map[circ_name], end=end, **kwargs)
            elif circ_name.lower() == 'pqc':
                self._build_pqc(parametrize=parametrize_map[circ_name], end=end, **kwargs)
            elif circ_name.lower() == 'eqc':
                self._build_eqc(parametrize=parametrize_map[circ_name], end=end, **kwargs)
            else:
                raise RuntimeError(f"circ_name: {circ_name} is not a valid circuit option to build.")

    def get_log_params(self):

        return {
            **self._kwargs,
            'gate': self.gate.name,
            'gate_class': self.gate.__class__.__name__,
            'gate_o': self.gate_o.name,
            'gate_o_class': self.gate_o.__class__.__name__,
            'gate_i': self.gate_i.name,
            'gate_i_class': self.gate_i.__class__.__name__,
            'gate_e': self.gate_e.name,
            'gate_e_class': self.gate_e.__class__.__name__,
            'gate_s': self.gate_s.name,
            'gate_s_class': self.gate_s.__class__.__name__,
            'gate_p': self.gate_p.name,
            'gate_p_class': self.gate_p.__class__.__name__,
        }
    
    @property
    def default_which(self):
        
        return ["SQC", "RQC", "OQC", "IQC", "PQC", "EQC"]
    
    @property
    def gates(self):
        """Derived classes must define the default circuit combination order."""
        return  {
                    self.gate.name: self.gate, self.gate_o.name: self.gate_o, self.gate_i.name: self.gate_i, 
                    self.gate_e.name: self.gate_e, self.gate_s.name: self.gate_s, self.gate_p.name: self.gate_p
                }
    
    def _build_sqc(self, parametrize: bool=False, end: str='back', **kwargs):

        to_end = self._get_to_end(end)
        self.gate_s.set_backend(to_end)

        self.sqc = qtn.Circuit(self.N, **kwargs)
        self.sqc.apply_to_arrays(to_end)
        for d in range(self.tau_s):
            for i in range((d+self.shift_s)%2, self.N-1+int(self.pbc_s), 2):
                params = self.gate_s.random_params()
                qubits = [i, (i+1)%self.N]
                self.sqc.apply_gate(self.gate_s.name, params=params, qubits=qubits, gate_round=d, parametrize=parametrize)

    def _build_pqc(self, parametrize: bool=True, end: str='back', **kwargs):

        to_end = self._get_to_end(end)
        self.gate_p.set_backend(to_end)

        self.pqc = qtn.Circuit(self.N, **kwargs)
        self.pqc.apply_to_arrays(to_end)
        for d in range(self.tau_p):
            for i in range((d+self.shift_p)%2, self.N-1+int(self.pbc_p), 2):
                params = self.gate_p.random_params()
                qubits = [i, (i+1)%self.N]
                self.pqc.apply_gate(self.gate_p.name, params=params, qubits=qubits, gate_round=d, parametrize=parametrize)
