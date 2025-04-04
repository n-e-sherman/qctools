import random
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
from ._base import CircuitManager


import numpy as np

def permutation_to_cycles(perm):
    """Convert a permutation (dict) into disjoint cycles."""
    seen = set()
    cycles = []

    for start in perm:
        if start in seen:
            continue
        cycle = []
        i = start
        while i not in seen:
            seen.add(i)
            cycle.append(i)
            i = perm[i]
        if len(cycle) > 1:
            cycles.append(cycle)
    return cycles

def cycle_to_transpositions(cycle):
    """Convert a single cycle into a list of transpositions."""
    a = cycle[0]
    return [(a, b) for b in reversed(cycle[1:])]

def cycle_to_transpositions_balanced(cycle):
    """Convert a cycle into transpositions using adjacent pairs."""
    return [(cycle[i], cycle[i + 1]) for i in reversed(range(len(cycle) - 1))][::-1]

def permutation_to_transpositions(perm, balanced=True):
    """Convert a full permutation (dict) into a list of transpositions."""
    transpositions = []
    for cycle in permutation_to_cycles(perm):
        if balanced:
            transpositions.extend(cycle_to_transpositions_balanced(cycle))
        else:
            transpositions.extend(cycle_to_transpositions(cycle))
    return transpositions

def generate_oqc_patches(N, max_subset, balanced=True):
    """
    Given a list of qubit indices, break into subsets and
    return a list of patch specs with local permutations and transpositions.
    """
    qubits = list(range(N))
    # qubits = list(qubits)
    random.shuffle(qubits)
    patches = []

    while qubits:
        group = qubits[:max_subset]
        qubits = qubits[max_subset:]

        # Build local index maps
        local_to_global = {i: q for i, q in enumerate(group)}
        global_to_local = {q: i for i, q in enumerate(group)}
        n = len(group)

        # Build random local permutation
        perm_list = list(range(n))
        random.shuffle(perm_list)
        local_permutation = {i: perm_list[i] for i in range(n)}

        # Decompose to transpositions
        local_transpositions = permutation_to_transpositions(local_permutation, balanced=balanced)

        patch = {
            'global_qubits': group,
            'local_to_global': local_to_global,
            'global_to_local': global_to_local,
            'local_permutation': local_permutation,
            'local_transpositions': local_transpositions,
        }

        patches.append(patch)

    return patches

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

def assign_transpositions_to_two_layers(transpositions, max_attempts=1000):
    """
    Assigns transpositions into two disjoint layers.
    Ensures no qubit appears more than once per layer.
    Returns any unassigned transpositions due to layer conflicts.
    
    Returns:
        layer1, layer2: lists of (i, j) tuples (disjoint in each layer)
        unassigned: list of (i, j) transpositions that could not be placed
    """
    attempt = 0
    while True:
        if attempt == max_attempts:
            raise RuntimeError(f"was unable to find 2 layers with all transpositions with {max_attempts} attempts")
        layer1, layer2 = [], []
        used_1, used_2 = set(), set()
        unassigned = []

        shuffled = list(transpositions)
        random.shuffle(shuffled)

        for a, b in shuffled:
            if a not in used_1 and b not in used_1:
                layer1.append((a, b))
                used_1.update([a, b])
            elif a not in used_2 and b not in used_2:
                layer2.append((a, b))
                used_2.update([a, b])
            else:
                unassigned.append((a, b))
        if len(unassigned) == 0:
            break
        attempt += 1

    return layer1, layer2

def generate_ansatz_layers(patch):

    trans = patch['local_transpositions'].copy()
    qubits = list(range(len(patch['global_qubits'])))
    layer1, layer2 = assign_transpositions_to_two_layers(trans)
    unassigned_qubits_layer1 = [q for q in qubits if not q in np.array(layer1).ravel()]
    unassigned_qubits_layer2 = [q for q in qubits if not q in np.array(layer2).ravel()]
    np.random.shuffle(unassigned_qubits_layer1)
    np.random.shuffle(unassigned_qubits_layer2)
    additional_pairs_layer1 = [(unassigned_qubits_layer1[i], unassigned_qubits_layer1[i+1]) for i in range(0, len(unassigned_qubits_layer1)-1, 2)]
    additional_pairs_layer2 = [(unassigned_qubits_layer2[i], unassigned_qubits_layer2[i+1]) for i in range(0, len(unassigned_qubits_layer2)-1, 2)]
    full_layer1 = layer1 + additional_pairs_layer1
    full_layer2 = layer2 + additional_pairs_layer2

    return full_layer1, full_layer2

class AllToAllPermutedOneQubitEchoMosaicCircuitManager(CircuitManager):

    def __init__(self, N, mosaic: Mosaic, mosaic_build_options: Dict[Any, Any]={},
                 tau_r: Optional[int]=None, 
                 tau_p: Optional[int]=None,
                 tau_o: int=3,

                 gate: Gate=HaarCZMinimalStackGate(), 
                 gate_r: Optional[Gate]=None,
                 gate_p: Optional[Gate]=None,
                 gate_echo: Optional[Gate]=None,
                 gate_echo_r: Optional[Gate]=None,
                 gate_echo_p: Optional[Gate]=None,
                 gate_o: Optional[Gate]=None,

                 echo_epochs: int=2000,
                 echo_rel_tol: float=1e-10,
                 echo_tol: float=1e-8,
                 echo_attempts: int=50,
                 echo_loss: str='L2',
                 
                 oqc_max_width: int=6,
                 oqc_double_swap_round: bool=True,
                 oqc_epochs: int=1000,
                 oqc_rel_tol: float=1e-7,
                 oqc_tol: float=1e-5,
                 oqc_attempts: int=20,
                 oqc_loss: str='L2',

                 gauge: bool=True,
                 gauge_inds: Tuple[int]=(0, 0), 
                 alpha_thresh:float=1e-7,
                 progbar: bool=True,
                 early_stopping: bool=True,
                 qasm: Optional[str] = None,
                 to_backend: Optional[Callable]=None,
                 to_frontend: Optional[Callable]=None,
    ):
        
        super().__init__(qasm=qasm, to_backend=to_backend, to_frontend=to_frontend)

        self.N = N
        self.mosaic = mosaic

        self.tau_r = tau_r if tau_r is not None else 0
        self.tau_p = tau_p if tau_p is not None else 0
        self.tau_o = tau_o
        

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
        self.echo_loss = echo_loss.lower()

        self.oqc_max_width=oqc_max_width
        self.oqc_double_swap_round = oqc_double_swap_round
        self.oqc_epochs = oqc_epochs
        self.oqc_rel_tol = oqc_rel_tol
        self.oqc_tol = oqc_tol
        self.oqc_attempts = oqc_attempts
        self.oqc_loss = oqc_loss.lower()

        self.gauge=gauge
        self.gauge_inds=gauge_inds
        self.alpha_thresh=alpha_thresh
        self.progbar=progbar
        self.early_stopping = early_stopping

        if self.mosaic.N_patches == 0:
            self.mosaic.build(**mosaic_build_options)

        frame = inspect.currentframe()
        args, vargs, varkw, locals = inspect.getargvalues(frame)
        skip_kwargs = ['self', 'frame', 'to_backend', 'to_frontend', 'gate', 'gate_o', 'gate_i', 'gate_e', 'progbar', 'mosaic', 'mosaic_build_options']
        self._kwargs = {key: locals[key] for key in locals if key not in skip_kwargs}

        self.rqc = self.echo_rqc = self.oqc = self.echo_pqc = self.pqc = None

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
            **self.mosaic.get_log_params(tag='echo')
        }
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

    def _build_echo(self, parametrize: bool=False, end: str='back', **kwargs):

        if self.oqc is None:
            raise RuntimeError("oqc must be built before the echo can be built.")
        
        to_end = self._get_to_end(end)
        self.gate_echo_r.set_backend(to_end)
        self.gate_echo_p.set_backend(to_end)

        # build patch dependency and target unitaries
        patches = self.mosaic.get_patches()
        wires = self.mosaic.wires
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
        for i, (k, _patch) in enumerate(self.mosaic.get_patches().items()):
            patch = sorted(_patch, key=lambda x: x[-1])
            qs_patch = np.sort(np.unique([gate[:2] for gate in patch]))
            patch_to_full = {i:q for i, q in enumerate(qs_patch)}
            full_to_patch = {q:i for i, q in enumerate(qs_patch)}
            N_patch = len(qs_patch)

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
                        gate = self.oqc_final_unitaries[q] #
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

                    msg = f"patch: {i+1}/{self.mosaic.N_patches}"
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
                    qubits = [self.oqc_permutation_inv[q] for q in gate.qubits]
                    self.echo_rqc.apply_gate(gate.label, params=gate.params, qubits=qubits, parametrize=parametrize, gate_round=gate.round)
                else:
                    self.echo_pqc.apply_gate(gate.label, params=gate.params, qubits=gate.qubits, parametrize=parametrize, gate_round=gate.round)

        # build simplified circuit
        self.echo_simplified = qtn.Circuit(self.N)
        self.echo_simplified.apply_to_arrays(to_end)
        for q, gate in self.final_unitaries.items():
            self.echo_simplified.apply_gate(gate.label, params=gate.params, qubits=[q], parametrize=parametrize)

    def _build_oqc(self, parametrize: bool=False, end: str='back', **kwargs):
        
        to_end = self._get_to_end(end)
        self.gate_o.set_backend(to_end)

        ansatz_layers = (self.tau_o+1) // 3
        if 3*ansatz_layers != self.tau_o:
            warnings.warn(f"tau_o should be divisible by 3, but {self.tau_o} was given. Increasing to {ansatz_layers * 3}")
            self.tau_o = 3 * ansatz_layers
            
        
        patches = generate_oqc_patches(self.N, max_subset=self.oqc_max_width)
        self.oqc_patches = patches
        global_permutation = {}
        for patch in patches:
            global_permutation.update({patch['local_to_global'][k]: patch['local_to_global'][v] for k,v in patch['local_permutation'].items()})
        self.oqc_permutation = global_permutation
        self.oqc_permutation_inv = {v:k for k,v in global_permutation.items()}
        full_unitaries = {}

        self.oqc = qtn.Circuit(self.N, **kwargs)
        self.oqc.apply_to_arrays(to_end)
        print("training OQC circuit")
        for patch_idx, patch in enumerate(patches):
            for attempt in range(self.oqc_attempts):
                
                # oqc_patch_target
                qc_target = qtn.Circuit(len(patch['global_qubits']))
                for swap in patch['local_transpositions'][::-1]:
                    qc_target.swap(*swap)
                unitaries = {q: qtn.Gate('U3', to_end(np.random.uniform(0, 2*np.pi, 3)), qubits=[q], round=None, parametrize=False) for q in range(qc_target.N)}
                for q, gate in unitaries.items():
                    qc_target.apply_gate(gate)
                qc_target.apply_to_arrays(to_end)

                # oqc_patch_train
                # TODO: Rethink how to do this part, probably a more elegant solution

                qc_train = qtn.Circuit(qc_target.N)
                all_qubits = list(range(qc_train.N))
                for t in range(ansatz_layers):
                    layer1, layer2 = generate_ansatz_layers(patch)
                    for qubits in layer1:
                        qc_train.apply_gate(self.gate_o.name, params=self.gate_o.random_params(), qubits=[*qubits], parametrize=True, gate_round=2*t)
                    for qubits in layer2:
                        qc_train.apply_gate(self.gate_o.name, params=self.gate_o.random_params(), qubits=[*qubits], parametrize=True, gate_round=2*t+1)
                for t in range(ansatz_layers):
                    np.random.shuffle(all_qubits)
                    for i in range(0, len(all_qubits)-1, 2):
                        qubits = [all_qubits[i], all_qubits[i+1]]
                        qc_train.apply_gate(self.gate_o.name, params=self.gate_o.random_params(), qubits=[*qubits], parametrize=True, gate_round=2*ansatz_layers+t)

                # qc_train = qtn.Circuit(qc_target.N)
                # swap_rounds = [np.random.randint(0, self.tau_o) for _ in patch['local_transpositions']]
                # swap_rounds.sort()
                # for t in range(self.tau_o):
                #     _qubits = list(range(qc_train.N))
                #     random.shuffle(_qubits)
                #     for i in range(0, len(_qubits)-1, 2):
                #         qubits = [_qubits[i], _qubits[i+1]]
                #         qc_train.apply_gate(self.gate_o.name, params=self.gate_o.random_params(), qubits=qubits, parametrize=True, gate_round=t)
                #     for swap_round, swap in zip(swap_rounds, patch['local_transpositions']):
                #         if (swap_round == t) or ((self.tau_o-swap_round == t) and self.oqc_double_swap_round):
                #             qc_train.apply_gate(self.gate_o.name, params=self.gate_o.random_params(), qubits=[*swap], parametrize=True, gate_round=t)

                qc_train.apply_to_arrays(to_end)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings only

                    if self.oqc_loss == 'l2':
                        U_target = qc_target.get_uni().to_dense()
                        I = torch.argmax(torch.abs(U_target))
                        gauge_inds = (I//len(U_target), I%len(U_target))
                        loss = GaugeTensorTNLoss(U_target, gauge_inds=gauge_inds, alpha_thresh=self.alpha_thresh)
                    elif self.oqc_loss == 'trace':
                        loss = TraceFidelityTNLoss(qc_target.get_uni())
                    else:
                        raise RuntimeError(f"oqc_loss: {self.oqc_loss} is not a valid option.")
                    
                    msg = f"patch: {patch_idx+1}/{len(patches)}"
                    opt = TNOptimizer(
                        qc_train,
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
                    if opt.loss < self.oqc_tol:
                        break
                    elif attempt == self.oqc_attempts-1:
                        raise RuntimeError(f"patch {patch_idx+1} in echo_mosaic did not converge in {self.echo_attempts} attempts.")

            _full_unitaries = {k: qtn.Gate(gate.label, gate.params, qubits=[patch['local_to_global'][k]], parametrize=False, round=None) for k, gate in unitaries.items()}
            full_unitaries.update({patch['local_to_global'][k]: v for k,v in _full_unitaries.items()})

            for gate in qc_opt.gates:
                qubits = [patch['local_to_global'][q] for q in gate.qubits]
                self.oqc.apply_gate(gate.label, params=gate.params, qubits=qubits, gate_round=gate.round, parametrize=parametrize)

        self.oqc.apply_to_arrays(to_end)
        self.oqc_final_unitaries = full_unitaries
        
    def _build_rqc(self, parametrize: bool=False, end: str='back', **kwargs):

        to_end = self._get_to_end(end)
        self.gate_r.set_backend(to_end)

        self.rqc = qtn.Circuit(self.N, **kwargs)
        self.rqc.apply_to_arrays(to_end)
        for d in range(self.tau_r):
            _qubits = list(range(self.N))
            np.random.shuffle(_qubits)
            for i in range(0, len(_qubits)-1, 2):
                qubits = list(np.sort([_qubits[i], _qubits[i+1]]))
                params = self.gate_r.random_params()
                self.rqc.apply_gate(self.gate_r.name, params=params, qubits=qubits, gate_round=d, parametrize=parametrize)

    def _build_pqc(self, parametrize: bool=True, end: str='back', **kwargs):

        to_end = self._get_to_end(end)
        self.gate_p.set_backend(to_end)

        self.pqc = qtn.Circuit(self.N, **kwargs)
        self.pqc.apply_to_arrays(to_end)
        for d in range(self.tau_p):
            _qubits = list(range(self.N))
            np.random.shuffle(_qubits)
            for i in range(0, len(_qubits)-1, 2):
                qubits = list(np.sort([_qubits[i], _qubits[i+1]]))
                params = self.gate_p.random_params()
                self.pqc.apply_gate(self.gate_p.name, params=params, qubits=qubits, gate_round=d, parametrize=parametrize)

    def _update_progbar_callback(self, opt, postfix_str: str, loss_threshold: float):
        if hasattr(opt, '_pbar') and opt._pbar is not None:
            opt._pbar.set_postfix_str(postfix_str)  # Directly set the postfix string
        if self.early_stopping:
            early_stopping_callback(opt, loss_threshold)
        return False  # Returning False means "continue optimization"




