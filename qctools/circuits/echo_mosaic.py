import warnings
import torch
import inspect
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize import TNOptimizer
from qctools.gates import Gate, Haar4Gate, HaarCZMinimalStackGate
from qctools.utils import GaugeTensorTNLoss

from typing import Union, Tuple, Callable, Iterable, Optional, Dict, Any, List

from ._base import CircuitManager
from qctools.mosaics import Mosaic

class EchoMosaicCircuitManager(CircuitManager):

    def __init__(self, N, echo_mosaic: Mosaic, oqc_mosaic: Optional[Mosaic]=None,
                 tau_r: Optional[int]=None, 
                 tau_p: Optional[int]=None,

                 gate: Gate=HaarCZMinimalStackGate(), 
                 gate_r: Optional[Gate]=None,
                 gate_p: Optional[Gate]=None,
                 gate_echo: Optional[Gate]=None,
                 gate_echo_r: Optional[Gate]=None,
                 gate_echo_p: Optional[Gate]=None,
                 gate_o: Optional[Gate]=None,

                 pbc: bool=True, 
                 pbc_r: Optional[bool]=None, 
                 pbc_p: Optional[bool]=None,

                 echo_epochs: int=2000,
                 echo_rel_tol: float=1e-10,
                 echo_tol: float=1e-8,
                 echo_attempts: int=50,
                 echo_build_options: Dict[Any, Any]={},
                 
                 oqc_epochs: int=1000,
                 oqc_rel_tol: float=1e-7,
                 oqc_tol: float=1e-5,
                 oqc_attempts: int=20,
                 oqc_build_options: Dict[Any, Any]={},

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
        self.echo_mosaic = echo_mosaic
        self.oqc_mosaic = oqc_mosaic

        self.tau_r = tau_r if tau_r is not None else 0
        self.tau_p = tau_p if tau_p is not None else 0

        if (tau_r % 2) == 1:
            warnings.warn(f"brickwall pattern is broken when tau_r is odd, and {tau_r} was given.")

        if oqc_mosaic and (not (oqc_mosaic.T % 2) == 1):
            warnings.warn(f"brickwall pattern is broken when oqc_mosaic depth is even, and {oqc_mosaic.T} was given.")

        self.pbc = pbc
        self.pbc_r = pbc_r if pbc_r is not None else pbc
        self.pbc_p = pbc_p if pbc_p is not None else pbc
        
        
        self.shift_echo_r = echo_mosaic.shift_r # might not be needed
        self.shift_echo_p = echo_mosaic.shift
        self.shift_o = oqc_mosaic.shift
        
        self.shift_r = (self.shift_echo_r - (self.tau_r%2))%2
        self.shift_p = 1 - self.shift_echo_r

        self.gate = gate
        self.gate_r = gate_r if gate_r is not None else gate
        self.gate_echo = gate_echo if gate_echo is not None else gate
        self.gate_echo_r = gate_echo_r if gate_echo_r is not None else self.gate_echo
        self.gate_echo_p = gate_echo_p if gate_echo_p is not None else self.gate_echo
        self.gate_o = gate_o if gate_o is not None else gate
        self.gate_p = gate_p if gate_p is not None else gate

        self.echo_epochs = echo_epochs
        self.echo_rel_tol = echo_rel_tol
        self.echo_tol = echo_tol
        self.echo_attempts = echo_attempts

        self.oqc_epochs = oqc_epochs
        self.oqc_rel_tol = oqc_rel_tol
        self.oqc_tol = oqc_tol
        self.oqc_attempts = oqc_attempts

        self.gauge=gauge
        self.gauge_inds=gauge_inds
        self.alpha_thresh=alpha_thresh
        self.progbar=progbar

        frame = inspect.currentframe()
        args, vargs, varkw, locals = inspect.getargvalues(frame)
        skip_kwargs = ['self', 'frame', 'to_backend', 'to_frontend', 'gate', 'gate_o', 'gate_i', 'gate_e', 'progbar', 'echo_mosaic', 'oqc_mosaic']
        self._kwargs = {key: locals[key] for key in locals if key not in skip_kwargs}

        self.rqc = self.echo_eqc = self.oqc = self.echo_pqc = self.pqc = None

        # build mosaics if not built
        self.echo_build_options = echo_build_options
        self.oqc_build_options=oqc_build_options
        build_echo = True if echo_mosaic.N_patches == 0 else False
        build_oqc = True if (oqc_mosaic and oqc_mosaic.N_patches == 0) else False
        self.build_mosaics(echo=build_echo, oqc=build_oqc)

    def build(self, end: str='back', which: Optional[Union[str|Iterable]]=None, parametrize: Union[bool, Dict[str, bool]]=False, **kwargs) -> None:
        """Abstract build method to construct specified circuit parts."""
        which_list, parametrize_map = self._process_which_and_parametrize(which, parametrize)

        echo_built = False
        for circ_name in which_list:
            if circ_name.lower() == 'rqc':
                self._build_rqc(parametrize=parametrize_map[circ_name], end=end, **kwargs)
            elif not echo_built and 'echo' in circ_name.lower():
                self._build_echo(parametrize=parametrize_map[circ_name], end=end, **kwargs)
                echo_built=True
            elif circ_name.lower() == 'oqc':
                self._build_oqc(parametrize=parametrize_map[circ_name], end=end, **kwargs)
            elif circ_name.lower() == 'pqc':
                self._build_pqc(parametrize=parametrize_map[circ_name], end=end, **kwargs)

    def build_mosaics(self, echo: bool=True, oqc: bool=True):
        
        if echo:
            self.echo_mosaic.build(**self.echo_build_options)
        if self.oqc_mosaic and oqc:
            self.oqc_mosaic.build(**self.oqc_build_options)

    def get_log_params(self):

        res = {
            **self._kwargs,
            'gate': self.gate.name,
            'gate_class': self.gate.__class__.__name__,
            'gate_r': self.gate_r.name,
            'gate_r_class': self.gate_r.__class__.__name__,
            'gate_echo_r': self.gate_echo_r.name,
            'gate_echo_r_class': self.gate_echo_r.__class__.__name__,
            'gate_o': self.gate_o.name,
            'gate_o_class': self.gate_o.__class__.__name__,
            'gate_echo_p': self.gate_echo_p.name,
            'gate_echo_p_class': self.gate_echo_p.__class__.__name__,
            'gate_p': self.gate_p.name,
            'gate_p_class': self.gate_p.__class__.__name__,
            **self.echo_mosaic.get_log_params(tag='echo')
        }
        if self.oqc_mosaic is not None:
            res.update(self.oqc_mosaic.get_log_params(tag='oqc'))
        return res


    @property
    def default_which(self):
        
        return ["RQC", "ECHO_RQC", "OQC", "ECHO_PQC", "PQC"]
    
    @property
    def gates(self):
        """Derived classes must define the default circuit combination order."""
        return  {
                    self.gate.name: self.gate, self.gate_r.name: self.gate_r, self.gate_echo_r.name: self.gate_echo_r, 
                    self.gate_o.name: self.gate_o, self.gate_echo_p.name: self.gate_echo_p, self.gate_p.name: self.gate_p
                }
    
    def _build_rqc(self, parametrize: bool=False, end: str='back', **kwargs):

        to_end = self._get_to_end(end)
        self.gate_r.set_backend(to_end)

        self.rqc = qtn.Circuit(self.N, **kwargs)
        self.rqc.apply_to_arrays(to_end)
        for d in range(self.tau_r):
            for i in range((d+self.shift_r)%2, self.N-1+int(self.pbc_r), 2):
                params = self.gate_r.random_params()
                qubits = [i, (i+1)%self.N]
                self.rqc.apply_gate(self.gate_r.name, params=params, qubits=qubits, gate_round=d, parametrize=parametrize)

    def _build_echo(self, parametrize: bool=False, end: str='back', **kwargs):
        
        to_end = self._get_to_end(end)
        self.gate_echo_r.set_backend(to_end)
        self.gate_echo_p.set_backend(to_end)

        print("training echo circuits")
        full_qc_gates = {}
        self.echo_patch_qcs = {}
        for i, (k, _patch) in enumerate(self.echo_mosaic.get_patches().items()):
            patch = sorted(_patch, key=lambda x: x[-1])
            patch_to_full, full_to_patch = self._get_patch_to_full_mappings(patch)
            N_patch = np.max(list(full_to_patch.values()))+1
            # create qc_patch
            for attempt in range(self.echo_attempts):

                # build patch qc
                pqc_patch = qtn.Circuit(N_patch)
                qc_patch = qtn.Circuit(N_patch)
                qc_patch.apply_to_arrays(to_end)
                for gate in patch:
                    qubits = [full_to_patch[q] for q in gate[:2]]
                    pqc_patch.apply_gate(self.gate_echo_p.name, params=self.gate_echo_p.random_params(), qubits=qubits, parametrize=True, gate_round=gate[-1])
                for gate in pqc_patch.gates[::-1]:
                    qubits = gate.qubits
                    qc_patch.apply_gate(self.gate_echo_r.name, params=self.gate_echo_r.random_params(), qubits=qubits, parametrize=False, gate_round=-1-gate.round)    
                qc_patch.apply_gates(pqc_patch.gates)

                # optimize qc_patch
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings only
                    loss = GaugeTensorTNLoss(torch.eye(2**qc_patch.N), gauge=self.gauge, gauge_inds=self.gauge_inds, alpha_thresh=self.alpha_thresh)
                    msg = f"patch: {i+1}/{self.echo_mosaic.N_patches}"
                    opt = TNOptimizer(
                        qc_patch,
                        loss,
                        autodiff_backend="torch",
                        progbar=self.progbar,
                        callback=lambda opt: self._update_progbar_callback(opt, msg)
                    )
                    qc_opt = opt.optimize(self.echo_epochs, tol=self.echo_rel_tol)
                    qc_opt.apply_to_arrays(to_end)
                    self.echo_patch_qcs[k] = qc_opt.copy()

                    if opt.loss < self.echo_tol:
                        break

                    elif attempt == self.echo_attempts-1:
                        raise RuntimeError(f"patch {i+1} in echo_mosaic did not converge in {self.echo_attempts} attempts.")

            # store gates in to build full_qc
            for _gate in qc_opt.gates:
                qubits = [patch_to_full[q] for q in _gate.qubits]
                gate = qtn.Gate(_gate.label, _gate.params, qubits=qubits, round=_gate.round, parametrize=_gate.parametrize)
                full_qc_gates[gate.round] = sorted(full_qc_gates.get(gate.round, []) + [gate], key=lambda x: x.qubits[0])

        # Build full echo qc
        self.echo_rqc = qtn.Circuit(self.N, **kwargs)
        self.echo_pqc = qtn.Circuit(self.N, **kwargs)
        self.echo_rqc.apply_to_arrays(to_end)
        self.echo_pqc.apply_to_arrays(to_end)

        for i in np.sort(list(full_qc_gates.keys())):
            
            for gate in full_qc_gates[i]:
                if i < 0:
                    self.echo_rqc.apply_gate(gate.label, params=gate.params, qubits=gate.qubits, parametrize=parametrize, gate_round=gate.round)
                else:
                    self.echo_pqc.apply_gate(gate.label, params=gate.params, qubits=gate.qubits, parametrize=parametrize, gate_round=gate.round)

    def _build_oqc(self, parametrize: bool=False, end: str='back', **kwargs):
        
        to_end = self._get_to_end(end)
        self.gate_o.set_backend(to_end)
        

        if self.oqc_mosaic is None:
            self.oqc = qtn.Circuit(self.N)
            self.oqc.apply_to_arrays(to_end)
            return

        print("training OQC")
        full_qc_gates = {}
        self.oqc_patch_qcs = {} # for testing
        for i, (k, _patch) in enumerate(self.oqc_mosaic.get_patches().items()):
            patch = sorted(_patch, key=lambda x: x[-1])
            patch_to_full, full_to_patch = self._get_patch_to_full_mappings(patch)
            N_patch = np.max(list(full_to_patch.values()))+1
            # create qc_patch
            for attempt in range(self.oqc_attempts):

                # build patch qc
                oqc_patch = qtn.Circuit(N_patch)
                oqc_patch.apply_to_arrays(to_end)
                for gate in patch:
                    qubits = [full_to_patch[q] for q in gate[:2]]
                    oqc_patch.apply_gate(self.gate_echo_p.name, params=self.gate_echo_p.random_params(), qubits=qubits, parametrize=True, gate_round=gate[-1])

                # optimize qc_patch
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings only
                    loss = GaugeTensorTNLoss(torch.eye(2**oqc_patch.N), gauge=self.gauge, gauge_inds=self.gauge_inds, alpha_thresh=self.alpha_thresh)
                    
                    msg = f"patch: {i+1}/{self.oqc_mosaic.N_patches}"
                    opt = TNOptimizer(
                        oqc_patch,
                        loss,
                        autodiff_backend="torch",
                        progbar=self.progbar,
                        callback=lambda opt: self._update_progbar_callback(opt, msg)
                    )
                    qc_opt = opt.optimize(self.oqc_epochs, tol=self.oqc_rel_tol)
                    qc_opt.apply_to_arrays(to_end)
                    self.oqc_patch_qcs[k] = qc_opt.copy()


                    if opt.loss < self.oqc_tol:
                        break

                    elif attempt == self.oqc_attempts-1:
                        raise RuntimeError(f"patch {i+1} in echo_mosaic did not converge in {self.echo_attempts} attempts.")

            # store gates in to build full_qc
            for _gate in qc_opt.gates:
                qubits = [patch_to_full[q] for q in _gate.qubits]
                gate = qtn.Gate(_gate.label, _gate.params, qubits=qubits, round=_gate.round, parametrize=_gate.parametrize)
                full_qc_gates[gate.round] = sorted(full_qc_gates.get(gate.round, []) + [gate], key=lambda x: x.qubits[0])

        # Build full echo qc
        self.oqc = qtn.Circuit(self.N, **kwargs)
        self.oqc.apply_to_arrays(to_end)

        for i in np.sort(list(full_qc_gates.keys())):
            for gate in full_qc_gates[i]:
                self.oqc.apply_gate(gate.label, params=gate.params, qubits=gate.qubits, parametrize=parametrize, gate_round=gate.round)
        
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

    def _get_patch_to_full_mappings(self, patch: List[Tuple[int]]):

        # find qubit mapping to/from patch
        bonds = [g[:2] for g in patch]
        qs = np.sort(np.unique(np.array(bonds).ravel()))
        if np.any(np.diff(qs)> 1): # periodic bonds
            dq = 0
            patch_qs = qs
            while np.any(np.diff(patch_qs) > 1):
                patch_qs = [(q+1)%self.N for q in patch_qs]
                dq += 1
                if dq == 2*self.N:
                    raise RuntimeError("Patch is disconnected.")
            q_sub = self.N-dq
        else:
            q_sub = np.min(qs)
        patch_to_full = {(q-q_sub)%self.N: q for q in qs}
        full_to_patch = {q:(q-q_sub)%self.N for q in qs}
        return patch_to_full, full_to_patch
    
    @staticmethod
    def _update_progbar_callback(opt, postfix_str: str):
        if hasattr(opt, '_pbar') and opt._pbar is not None:
            opt._pbar.set_postfix_str(postfix_str)  # Directly set the postfix string
        return False  # Returning False means "continue optimization"

