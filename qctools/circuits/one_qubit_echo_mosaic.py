import warnings
import torch
import inspect
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize import TNOptimizer
from qctools.gates import Gate, Haar4Gate, HaarCZMinimalStackGate
from qctools.utils import GaugeTensorTNLoss, TraceFidelityTNLoss, early_stopping_callback, EarlyStopException

from typing import Union, Tuple, Callable, Iterable, Optional, Dict, Any, List

from qctools.mosaics import Mosaic
from .echo_mosaic import EchoMosaicCircuitManager

import numpy as np

def get_prior_patches(patches, wires):
    """
    Computes the prior patch for each qubit in every patch.

    Parameters:
        patches: Dictionary {patch_id: [(q0, q1, time), ...]} describing each patch.
        wires: 2D matrix of shape (N, T) where wires[q, t] contains the patch_id at time t.

    Returns:
        Dictionary mapping patch_id to another dictionary {qubit: prior_patch_id or None}.
    """
    prior_patches = {}

    for patch_id, gates in patches.items():
        prior_patch_info = {}

        for q0, q1, time in gates:
            # Initialize prior patches for q0 and q1 as None
            prior_patch_q0, prior_patch_q1 = None, None

            # Search backward in time to find the most recent patch acting on q0 and q1
            for t in range(time - 1, -1, -1):  # Iterate backward in time
                if prior_patch_q0 is None and wires[q0, t] != patch_id:
                    prior_patch_q0 = wires[q0, t]
                if prior_patch_q1 is None and wires[q1, t] != patch_id:
                    prior_patch_q1 = wires[q1, t]
                if prior_patch_q0 is not None and prior_patch_q1 is not None:
                    break

            # Store the prior patches for this patch
            prior_patch_info[q0] = prior_patch_q0
            prior_patch_info[q1] = prior_patch_q1
        prior_patches[patch_id] = prior_patch_info
    return prior_patches

class OneQubitEchoMosaicCircuitManager(EchoMosaicCircuitManager):

    def build(self, end: str='back', which: Optional[Union[str|Iterable]]=None, parametrize: Union[bool, Dict[str, bool]]=False, **kwargs) -> None:
        """Abstract build method to construct specified circuit parts."""
        which_list, parametrize_map = self._process_which_and_parametrize(which, parametrize)

        reorder = False
        for w in which_list:
            if w.lower() == 'oqc':
                reorder = True
        if reorder:
            which_list = ['OQC'] + [w for w in which_list if not w.lower() == 'OQC'.lower()]

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

    def _build_echo(self, parametrize: bool=False, end: str='back', **kwargs):

        if self.oqc is None:
            raise RuntimeError("oqc must be built before the echo can be built.")
        
        to_end = self._get_to_end(end)
        self.gate_echo_r.set_backend(to_end)
        self.gate_echo_p.set_backend(to_end)

        # build patch dependency and target unitaries
        patches = self.echo_mosaic.get_patches()
        wires = self.echo_mosaic.wires
        prior_patches = get_prior_patches(patches, wires)
        unitaries = {}
        for k, patch in patches.items():
            qs = np.unique(np.array([g[:2] for g in patch]).ravel())
            _unitaries = {q: qtn.Gate('U3', to_end(np.random.uniform(0, 2*np.pi, 3)), qubits=[q], round=None, parametrize=False) for q in qs}
            unitaries[k] = _unitaries
        self.final_unitaries = {}
        for q, patch_id in enumerate(wires[:, -1]):
            gate = unitaries[patch_id][q]
            self.final_unitaries[q] = gate

        full_qc_gates = {}
        self.echo_patch_qcs = {}
        print("training echo circuits")
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

                # add the layer of 1-qubit unitaries
                for q, prior_patch_id in prior_patches[k].items():
                    if prior_patch_id is None:
                        gate = self.oqc_final_unitaries[q]
                    else:
                        gate = unitaries[prior_patch_id][q]
                    qubits = [full_to_patch[q] for q in gate.qubits]
                    qc_patch.apply_gate(gate.label, params=gate.params, qubits=qubits, parametrize=False, gate_round=None)
                qc_patch.apply_gates(pqc_patch.gates)

                # optimize qc_patch
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings only

                    qc_target = qtn.Circuit(qc_patch.N)
                    for q, gate in unitaries[k].items():
                        qubits = [full_to_patch[q]]
                        qc_target.apply_gate(gate.label, params=gate.params, qubits=qubits, parametrize=False, gate_round=None)

                    if self.echo_loss == 'l2':
                        U_target = qc_target.get_uni().to_dense()
                        I = torch.argmax(torch.abs(U_target))
                        gauge_inds = (I//len(U_target), I%len(U_target))
                        loss = GaugeTensorTNLoss(U_target, gauge_inds=gauge_inds, alpha_thresh=self.alpha_thresh)
                    elif self.echo_loss == 'trace':
                        loss = TraceFidelityTNLoss(qc_target.get_uni())
                    else:
                        raise RuntimeError(f"echo_loss: {self.echo_loss} is not a valid option.")

                    msg = f"patch: {i+1}/{self.echo_mosaic.N_patches}"
                    opt = TNOptimizer(
                        qc_patch,
                        loss,
                        autodiff_backend="torch",
                        progbar=self.progbar,
                        callback=lambda opt: self._update_progbar_callback(opt, msg, self.echo_tol)
                    )
                    try:
                        qc_opt = opt.optimize(self.echo_epochs, tol=self.echo_rel_tol)
                    except EarlyStopException:
                        qc_opt = opt.get_tn_opt()  # Return the last optimized quantum circuit state
                    qc_opt.apply_to_arrays(to_end)
                    self.echo_patch_qcs[k] = qc_opt.copy()
                    if opt.loss < self.echo_tol:
                        break
                    elif attempt == self.echo_attempts-1:
                        raise RuntimeError(f"patch {i+1} in echo_mosaic did not converge in {self.echo_attempts} attempts.")
                
            # store gates in to build full_qc
            for _gate in qc_opt.gates:
                if _gate.round is None:
                    continue
                
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

        # build simplified circuit
        self.echo_full = qtn.Circuit(self.N)
        self.echo_full.apply_to_arrays(to_end)
        for q, gate in self.final_unitaries.items():
            self.echo_full.apply_gate(gate.label, params=gate.params, qubits=[q], parametrize=parametrize)

    def _build_oqc(self, parametrize: bool=False, end: str='back', **kwargs):
        
        to_end = self._get_to_end(end)
        self.gate_o.set_backend(to_end)
        
        if self.oqc_mosaic is None:
            self.oqc = qtn.Circuit(self.N)
            self.oqc.apply_to_arrays(to_end)
            return
        
        # build patch dependency and target unitaries
        patches = self.oqc_mosaic.get_patches()
        wires = self.oqc_mosaic.wires
        prior_patches = get_prior_patches(patches, wires)
        unitaries = {}
        for k, patch in patches.items():
            qs = np.unique(np.array([g[:2] for g in patch]).ravel())
            _unitaries = {q: qtn.Gate('U3', to_end(np.random.uniform(0, 2*np.pi, 3)), qubits=[q], round=None, parametrize=False) for q in qs}
            unitaries[k] = _unitaries
        self.oqc_final_unitaries = {}
        for q, patch_id in enumerate(wires[:, -1]):
            gate = unitaries[patch_id][q]
            self.oqc_final_unitaries[q] = gate

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
                # add 1q-layer
                for q, prior_patch_id in prior_patches[k].items():
                    if prior_patch_id is None:
                        continue
                    gate = unitaries[prior_patch_id][q]
                    qubits = [full_to_patch[q] for q in gate.qubits]
                    oqc_patch.apply_gate(gate.label, params=gate.params, qubits=qubits, parametrize=False, gate_round=None)
                # add gates in patch
                for gate in patch:
                    qubits = [full_to_patch[q] for q in gate[:2]]
                    oqc_patch.apply_gate(self.gate_echo_p.name, params=self.gate_echo_p.random_params(), qubits=qubits, parametrize=True, gate_round=gate[-1])
                # optimize qc_patch
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings only
                    qc_target = qtn.Circuit(oqc_patch.N)
                    for q, gate in unitaries[k].items():
                        qubits = [full_to_patch[q]]
                        qc_target.apply_gate(gate.label, params=gate.params, qubits=qubits, parametrize=False, gate_round=None)

                    if self.oqc_loss == 'l2':
                        U_target = qc_target.get_uni().to_dense()
                        I = torch.argmax(torch.abs(U_target))
                        gauge_inds = (I//len(U_target), I%len(U_target))
                        loss = GaugeTensorTNLoss(U_target, gauge_inds=gauge_inds, alpha_thresh=self.alpha_thresh)
                    elif self.oqc_loss == 'trace':
                        loss = TraceFidelityTNLoss(qc_target.get_uni())
                    else:
                        raise RuntimeError(f"oqc_loss: {self.oqc_loss} is not a valid option.")
                    
                    msg = f"patch: {i+1}/{self.oqc_mosaic.N_patches}"
                    opt = TNOptimizer(
                        oqc_patch,
                        loss,
                        autodiff_backend="torch",
                        progbar=self.progbar,
                        callback=lambda opt: self._update_progbar_callback(opt, msg, self.oqc_tol)
                    )
                    try:
                        qc_opt = opt.optimize(self.oqc_epochs, tol=self.oqc_rel_tol)
                    except EarlyStopException:
                        qc_opt = opt.get_tn_opt()  # Return the last optimized quantum circuit state
                    qc_opt.apply_to_arrays(to_end)
                    self.oqc_patch_qcs[k] = qc_opt.copy()
                    if opt.loss < self.oqc_tol:
                        break
                    elif attempt == self.oqc_attempts-1:
                        raise RuntimeError(f"patch {i+1} in echo_mosaic did not converge in {self.echo_attempts} attempts.")

            # store gates in to build full_qc
            for _gate in qc_opt.gates:
                if _gate.round is None:
                    continue
                qubits = [patch_to_full[q] for q in _gate.qubits]
                gate = qtn.Gate(_gate.label, _gate.params, qubits=qubits, round=_gate.round, parametrize=_gate.parametrize)
                full_qc_gates[gate.round] = sorted(full_qc_gates.get(gate.round, []) + [gate], key=lambda x: x.qubits[0])

        # Build full echo qc
        self.oqc = qtn.Circuit(self.N, **kwargs)
        self.oqc.apply_to_arrays(to_end)

        for i in np.sort(list(full_qc_gates.keys())):
            for gate in full_qc_gates[i]:
                self.oqc.apply_gate(gate.label, params=gate.params, qubits=gate.qubits, parametrize=parametrize, gate_round=gate.round)
        















