import warnings
import torch
import inspect
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize import TNOptimizer
from typing import Union, Tuple, Callable, Iterable, Optional, Dict

from ._base import CircuitManager

from qctools.gates import Gate, Haar4Gate
from qctools.utils import GaugeTensorTNLoss


class EchoMosaicCircuitManager(CircuitManager):

    def __init__(self, N, 
                 tau_r, 
                 tau_o: int=1, # defaults to 1
                 tau_i: Optional[int]=None, # defaults to tau_r
                 tau_e: int=0, # defaults to 0
                 gate: Gate=Haar4Gate(), 
                 gate_o: Optional[Gate]=None,
                 gate_i: Optional[Gate]=None,
                 gate_e: Optional[Gate]=None,
                 pbc: bool=False, 
                 pbc_o: Optional[bool]=None, 
                 pbc_i: Optional[bool]=None, 
                 pbc_e: Optional[bool]=None,
                 shift: int=0,
                 epochs: int=500,
                 tol: float=1e-8,
                 identity_tol: float=1e-7,
                 identity_attempts: int=20,
                 gauge: bool=True,
                 gauge_inds: Tuple[int]=(0, 0), 
                 alpha_thresh:float=1e-7,
                 progbar: bool=True,
                 qasm: Optional[str] = None,
                 to_backend: Optional[Callable]=None,
                 to_frontend: Optional[Callable]=None,
    ):

        super().__init__(qasm=qasm, to_backend=to_backend, to_frontend=to_frontend)

        self.N = N

        self.tau_r = tau_r
        self.tau_o = tau_o
        self.tau_i = tau_i if tau_i is not None else tau_r
        self.tau_e = tau_e
        self.tau_p = self.tau_o + self.tau_i + self.tau_e
        if not (tau_o % 2) == 1:
            warnings.warn(f"brickwall pattern is broken when tau_o is even, and tau_o={tau_o} was given.")

        self.pbc = pbc
        self.pbc_o = pbc_o if pbc_o is not None else pbc
        self.pbc_i = pbc_i if pbc_i is not None else pbc
        self.pbc_e = pbc_e if pbc_e is not None else pbc

        self.shift = shift
        self.shift_o = (shift+tau_r)%2
        self.shift_e = (self.shift_o + tau_r + 1)%2

        self.gate = gate
        self.gate_o = gate_o if gate_o is not None else gate
        self.gate_i = gate_i if gate_i is not None else gate
        self.gate_e = gate_e if gate_e is not None else gate

        self.epochs=epochs
        self.tol=tol
        self.identity_tol=identity_tol
        self.identity_attempts=identity_attempts
        self.gauge=gauge
        self.gauge_inds=gauge_inds
        self.alpha_thresh=alpha_thresh
        self.progbar=progbar

        frame = inspect.currentframe()
        args, vargs, varkw, locals = inspect.getargvalues(frame)
        skip_kwargs = ['self', 'frame', 'to_backend', 'to_frontend', 'gate', 'gate_o', 'gate_i', 'gate_e', 'progbar']
        self._kwargs = {key: locals[key] for key in locals if key not in skip_kwargs}

        self.rqc = self.oqc = self.iqc = self.eqc = None

    def build(self, end: str='back', which: Optional[Union[str|Iterable]]=None, parametrize: Union[bool, Dict[str, bool]]=False, **kwargs) -> None:
        """Abstract build method to construct specified circuit parts."""
        which_list, parametrize_map = self._process_which_and_parametrize(which, parametrize)

        for circ_name in which_list:
            if circ_name.lower() == 'rqc':
                self._build_rqc(parametrize=parametrize_map[circ_name], end=end, **kwargs)
            elif circ_name.lower() == 'oqc':
                self._build_oqc(parametrize=parametrize_map[circ_name], end=end, **kwargs)
            elif circ_name.lower() == 'iqc':
                self._build_iqc(parametrize=parametrize_map[circ_name], end=end, **kwargs)
            elif circ_name.lower() == 'eqc':
                self._build_eqc(parametrize=parametrize_map[circ_name], end=end, **kwargs)


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
        }
    
    @property
    def default_which(self):
        
        return ["RQC", "OQC", "IQC", "EQC"]
    
    @property
    def gates(self):
        """Derived classes must define the default circuit combination order."""
        return {self.gate.name: self.gate, self.gate_o.name: self.gate_o, self.gate_i.name: self.gate_i, self.gate_e.name: self.gate_e}

    def _build_rqc(self, parametrize: bool=False, end: str='back', **kwargs):

        to_end = self._get_to_end(end)
        self.gate.set_backend(to_end)

        self.rqc = qtn.Circuit(self.N, **kwargs)
        self.rqc.apply_to_arrays(to_end)
        for d in range(self.tau_r):
            for i in range((d+self.shift)%2, self.N-1+int(self.pbc), 2):
                params = self.gate.random_params()
                qubits = [i, (i+1)%self.N]
                self.rqc.apply_gate(self.gate.name, params=params, qubits=qubits, gate_round=d, parametrize=parametrize)

    def _build_oqc(self, parametrize: bool=True, end: str='back', **kwargs):

        to_end = self._get_to_end(end)
        self.gate_o.set_backend(to_end)

        self.oqc = qtn.Circuit(self.N, **kwargs)
        self.oqc.apply_to_arrays(to_end)
        print("Training OQC")
        for d in range(self.tau_o):
            for i in range((d+self.shift_o)%2, self.N-1+int(self.pbc_o), 2):

                for attempt in range(self.identity_attempts):
                    params = self.gate_o.random_params()
                    _qc = qtn.Circuit(2)
                    _qc.apply_gate(self.gate_o.name, params=params, qubits=[0, 1], parametrize=True)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings only

                        #TODO: Make more general
                        loss = GaugeTensorTNLoss(torch.eye(2**_qc.N))
                        msg = f"gate: {len(self.oqc.gates)+1}/{(self.N//2)*self.tau_o + 1}"
                        opt = TNOptimizer(
                            _qc,
                            loss,
                            autodiff_backend="torch",
                            progbar=self.progbar,
                            callback=lambda opt: self._update_progbar_callback(opt, msg)
                        )
                        _qc_opt = opt.optimize(self.epochs, self.tol)

                    if opt.loss < self.identity_tol:
                        break
                    elif attempt == self.identity_attempts:
                        raise RuntimeError(f"gate {i+1} in OQC did not converge in {self.identity_attempts} attempts.")

                _qc_gate = _qc_opt.gates[0]
                qubits = [i, (i+1)%self.N]
                self.oqc.apply_gate(_qc_gate.label, params=_qc_gate.params, qubits=qubits, gate_round=d, parametrize=parametrize)


    def _build_iqc(self, parametrize: bool=True, reorder: bool=True, end: str='back', **kwargs):

        # NOTE: Only does gate by gate inversion currently

        to_end = self._get_to_end(end)
        self.gate_i.set_backend(to_end)

        _iqc = qtn.Circuit(self.N, **kwargs)
        _iqc.apply_to_arrays(to_end)

        print("Training IQC")
        for i, gate in enumerate(self.rqc.gates[::-1]):
            if not self.progbar:
                print(i, end='\r')
            round = self.tau_r - gate.round - 1
            if round >= self.tau_i:
                break
            
            for attempt in range(self.identity_attempts):
                qc = qtn.Circuit(2)
                qc.apply_gate(gate.label, params=gate.params, qubits=[0, 1], gate_round=gate.round)
                qc.apply_gate(self.gate_i.name, params=self.gate_i.random_params(), qubits=[0, 1], gate_round=round, parametrize=True)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings only

                    #TODO: Make more general
                    loss = GaugeTensorTNLoss(torch.eye(2**qc.N))
                    opt = TNOptimizer(
                        qc,
                        loss,
                        autodiff_backend="torch",
                        progbar=self.progbar,
                        callback=lambda opt: self._update_progbar_callback(opt, f"gate: {i+1}/{len(self.rqc.gates)}")
                    )
                    Iqc_opt = opt.optimize(self.epochs, self.tol)
                    if opt.loss < self.identity_tol:
                        break
                    elif attempt == self.identity_attempts:
                        raise RuntimeError(f"gate {i+1} in IQC did not converge in {self.identity_attempts} attempts.")

            iqc_gate = Iqc_opt.gates[-1]
            _iqc.apply_gate(iqc_gate.label, params=iqc_gate.params, qubits=gate.qubits, gate_round=round, parametrize=parametrize)

        # reorder the gates
        if not reorder:
            self.iqc = _iqc
            return
        
        self.iqc = qtn.Circuit(self.N, **kwargs)
        self.iqc.apply_to_arrays(to_end)

        for round in range(self.tau_i):
            gates = {}
            for gate in _iqc.gates:
                if gate.round < round:
                    continue
                if gate.round > round:
                    break
                gates[gate.qubits[0]] = {'gate_id': gate.label, 'params': gate.params, 'qubits': gate.qubits, 'parametrize': parametrize, 'gate_round': round}
            
            inds = list(gates.keys())
            inds = np.sort(inds)
            for i in inds:
                self.iqc.apply_gate(**gates[i])

    def _build_eqc(self, parametrize: bool=True, end: str='back', **kwargs):

        to_end = self._get_to_end(end)
        self.gate_e.set_backend(to_end)

        self.eqc = qtn.Circuit(self.N, **kwargs)
        self.eqc.apply_to_arrays(to_end)
        for d in range(self.tau_e):
            for i in range((d+self.shift_e)%2, self.N-1+int(self.pbc_e), 2):
                params = self.gate_e.random_params()
                qubits = [i, (i+1)%self.N]
                self.eqc.apply_gate(self.gate_e.name, params=params, qubits=qubits, gate_round=d, parametrize=parametrize)

    @staticmethod
    def _update_progbar_callback(opt, postfix_str: str):
        if hasattr(opt, '_pbar') and opt._pbar is not None:
            opt._pbar.set_postfix_str(postfix_str)  # Directly set the postfix string
        return False  # Returning False means "continue optimization"
