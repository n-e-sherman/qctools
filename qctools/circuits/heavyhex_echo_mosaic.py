import random
import copy
import warnings
import torch
import inspect
import numpy as np
import quimb as qu
import quimb.tensor as qtn
import networkx as nx
from quimb.tensor.optimize import TNOptimizer
from qctools.gates import Gate, Haar4Gate, HaarCZMinimalStackGate, HaarCZHaarGate
from qctools.utils import GaugeTensorTNLoss, TraceFidelityTNLoss, BitstringTNLoss, early_stopping_callback, EarlyStopException

from typing import Union, Tuple, Callable, Iterable, Optional, Dict, Any, List

from qctools.mosaics import HeavyHexMosaic
from ._base import CircuitManager

def get_prior_patches(patches, wires, missing_id=-2):
    """
    Computes the prior patch for each qubit in every patch, skipping missing_id values.

    Parameters:
        patches: Dictionary {patch_id: [(q0, q1, time), ...]} describing each patch.
        wires: 2D array of shape (N, T) where wires[q, t] contains the patch_id or missing_id.
        missing_id: The value representing an inactive qubit (no patch).

    Returns:
        Dictionary mapping patch_id to another dictionary {qubit: prior_patch_id or None}.
    """
    prior_patches = {}

    for patch_id, gates in patches.items():
        prior_patch_info = {}

        for q0, q1, time in gates:
            prior_patch_q0, prior_patch_q1 = None, None

            # Search backward in time for the last non-missing, non-self patch
            for t in range(time - 1, -1, -1):
                if prior_patch_q0 is None:
                    candidate_q0 = wires[q0, t]
                    if candidate_q0 != patch_id and candidate_q0 != missing_id:
                        prior_patch_q0 = candidate_q0
                if prior_patch_q1 is None:
                    candidate_q1 = wires[q1, t]
                    if candidate_q1 != patch_id and candidate_q1 != missing_id:
                        prior_patch_q1 = candidate_q1
                if prior_patch_q0 is not None and prior_patch_q1 is not None:
                    break

            prior_patch_info[q0] = prior_patch_q0
            prior_patch_info[q1] = prior_patch_q1

        prior_patches[patch_id] = prior_patch_info

    return prior_patches

def get_final_patch_ids(wires, missing_id=-2):
    """
    For each qubit, return the final non-missing patch_id.
    If no valid patch is found, assign None.

    Parameters:
        wires: 2D array of shape (N, T)
        missing_id: value representing "inactive" (to ignore)

    Returns:
        List of final patch_id or None per qubit
    """
    N, T = wires.shape
    final_patches = []

    for q in range(N):
        final_patch = None
        for t in range(T - 1, -1, -1):
            patch_id = wires[q, t]
            if patch_id != missing_id:
                final_patch = patch_id
                break
        final_patches.append(final_patch)

    return final_patches


def get_disconnected_clusters_from_layers(layers):
    """Given a list of sets of edges (layers), return connected components."""
    combined_edges = set().union(*layers)
    G_sub = nx.Graph()
    G_sub.add_edges_from(combined_edges)

    # Get connected components (as sets of nodes)
    components = list(nx.connected_components(G_sub))
    return components

def get_cluster_local_layers(layers, clusters):
    """
    Given layers (list of sets of edges) and node clusters (sets of nodes),
    return a list of cluster-specific layers.
    
    Output: list of tuples (cluster_nodes, [layer1_edges, layer2_edges])
    """
    cluster_layers = []

    for cluster in clusters:
        sublayers = []
        for layer in layers:
            # Keep only edges fully within this cluster
            sublayer = {e for e in layer if e[0] in cluster and e[1] in cluster}
            sublayers.append(sublayer)
        cluster_layers.append((cluster, sublayers))

    return cluster_layers

def get_cluster_layers(covering):

    clusters = get_disconnected_clusters_from_layers(covering)
    return get_cluster_local_layers(covering, clusters)

class HeavyHexEchoMosaicCircuitManager(CircuitManager):

    def __init__(self, mosaic: HeavyHexMosaic, seed: Optional[int]=None, max_trials: int=1000, mosaic_build_options: Dict[Any, Any]={},
                 rqc_coverings: List[str]=['3'],
                 pqc_coverings: List[str] = ['3', '2'],
                 pqc_match_rqc_connectivity: bool=False,
                 pqc_reverse_rqc_connectivity: bool=False,
                #  tau_r: Optional[int]=None, 
                #  tau_p: Optional[int]=None,
                 oqc_blocks: int=1,
                 oqc_repeat: bool=True, # used for HaarCZHaar gates to train simplification, makes each block twice as big
                 oqc_covering: str='2',

                 gate: Gate=HaarCZHaarGate(), 
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
                 echo_loss: str='trace',
                 
                 oqc_epochs: int=1000,
                 oqc_rel_tol: float=1e-7,
                 oqc_tol: float=1e-5,
                 oqc_attempts: int=20,
                 oqc_loss: str='trace',

                 peak_epochs: int=2000,
                 peak_rel_tol: float=1e-1000,
                 peak_tol: float=0.01,
                 peak_attempts: int=50,
                 target: Optional[str]=None,
                 simplify_sequence: str='ADCR',


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

        
        self.mosaic = mosaic
        self.N = mosaic.G.order()
        self.seed = seed
        self.max_trials = max_trials

        self.oqc_blocks = oqc_blocks
        self.oqc_repeat = oqc_repeat

        # OQC error handling and setup
        if not oqc_blocks == 1:
            raise RuntimeError(f"oqc_blocks: {oqc_blocks} is not a valid option.")
        if oqc_covering == '2':
            tau_block = 2
            self.generate_oqc_covering = lambda : mosaic.generate_2layer_covering(mosaic.G, seed=seed, max_trials=max_trials)
        elif oqc_covering == '3':
            tau_block = 3
            self.generate_oqc_covering = lambda : mosaic.generate_3layer_covering(mosaic.G, seed=seed, max_trials=max_trials)
        else:
            raise RuntimeError(f"covering type: {oqc_covering} is not a valid option")
        self.tau_o = int((tau_block * oqc_blocks) * (1 + int(oqc_repeat)))
        
        # RQC-PQC error handling
        tau_r = 0
        for rqc_covering in rqc_coverings:
            if rqc_covering == '2':
                tau_r += 2
            elif rqc_covering == '3':
                tau_r += 3
            else:
                raise RuntimeError(f"rqc_covering type: {rqc_covering} is not a valid option")
        self.tau_r = tau_r
            
        tau_p = 0
        for pqc_covering in pqc_coverings:
            if pqc_covering == '2':
                tau_p += 2
            elif pqc_covering == '3':
                tau_p += 3
            else:
                raise RuntimeError(f"rqc_covering type: {rqc_covering} is not a valid option")
        self.tau_p = tau_p
        self.rqc_layers = self.pqc_layers = None

        if pqc_match_rqc_connectivity:
            assert pqc_coverings[:len(rqc_coverings)] == rqc_coverings, "pqc_match_rqc_connectivty is true, but their coverings don't match."
        if pqc_reverse_rqc_connectivity:
            assert pqc_coverings[:len(rqc_coverings)][::-1] == rqc_coverings, "pqc_match_rqc_connectivty is true, but their coverings don't match in reverse."

        self.rqc_coverings = rqc_coverings
        self.pqc_coverings = pqc_coverings
        self.pqc_match_rqc_connectivity = pqc_match_rqc_connectivity
        self.pqc_reverse_rqc_connectivity = pqc_reverse_rqc_connectivity

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

        self.oqc_epochs = oqc_epochs
        self.oqc_rel_tol = oqc_rel_tol
        self.oqc_tol = oqc_tol
        self.oqc_attempts = oqc_attempts
        self.oqc_loss = oqc_loss.lower()

        self.peak_epochs = peak_epochs
        self.peak_rel_tol = peak_rel_tol
        self.peak_tol = peak_tol
        self.peak_attempts = peak_attempts
        self.target = target
        self.new_random_target = target is None
        self.simplify_sequence = simplify_sequence

        self.gauge=gauge
        self.gauge_inds=gauge_inds
        self.alpha_thresh=alpha_thresh
        self.progbar=progbar
        self.early_stopping = early_stopping

        if self.mosaic.N_patches == 0:
            self.mosaic.build(**mosaic_build_options)

        frame = inspect.currentframe()
        args, vargs, varkw, locals = inspect.getargvalues(frame)
        skip_kwargs = ['self', 'frame', 'to_backend', 'to_frontend', 'gate', 'gate_o', 'gate_i', 'gate_e', 'progbar', 'mosaic', 'mosaic_build_options', 'layers']
        self._kwargs = {key: locals[key] for key in locals if key not in skip_kwargs}

        self.rqc = self.echo_rqc = self.oqc = self.echo_pqc = self.pqc = None
        self.peak_converged = False
        self.peak = None

    def build(self, end: str='back', which: Optional[Union[str|Iterable]]=None, parametrize: Union[bool, Dict[str, bool]]=False, **kwargs) -> None:
        
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

    def train_peak(self):
        
        if (not self.rqc) or (not self.pqc): # maybe raise a warning, or even an exception?
            self.generate_rqc_pqc_layers()
            self.build(which=['RQC','PQC'], parametrize={'RQC': False, 'PQC': True})

        target = self.target
        for attempt in range(self.peak_attempts):

            qc_train = self.get(which=['RQC', 'ECHO_SIMPLIFIED','PQC'], 
                                parametrize={'RQC': False, 'ECHO_SIMPLIFIED': False, 'PQC': True})
            if (self.target is None) or (self.new_random_target):
                self.target = ''.join(random.choice('01') for _ in range(self.N))
            loss = BitstringTNLoss(self.target, simplify_sequence=self.simplify_sequence)
            opt = TNOptimizer(
                qc_train,
                loss,
                autodiff_backend='torch',
                callback=lambda opt: early_stopping_callback(opt, -self.peak_tol)
            )
            try:
                qc_opt = opt.optimize(self.peak_epochs, tol=self.peak_rel_tol)
            except EarlyStopException:
                print("Optimization stopped early due to reaching target loss.")
                qc_opt = opt.get_tn_opt()  # Return the last optimized quantum circuit state
            if opt.loss < -self.peak_tol:
                self.peak_converged = True
                break
            else:
                self.generate_rqc_pqc_layers()
                self.build(which=['RQC','PQC'], parametrize={'RQC': False, 'PQC': True})

        qc_opt.apply_to_arrays(self.to_backend)
        self.peak = abs(qc_opt.amplitude(self.target).item())**2
        
        new_params, _ = qtn.pack(qc_opt)
        self.update(pqc=new_params)
        
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
            'peak_converged': self.peak_converged,
            'target': self.target,
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
        prior_patches = get_prior_patches(patches, wires, missing_id=self.mosaic.missing_id)
        final_patch_ids = get_final_patch_ids(wires, missing_id=self.mosaic.missing_id)

        # for testing
        self.prior_patches = prior_patches

        unitaries = {}
        for k, patch in patches.items():
            qs = np.unique(np.array([g[:2] for g in patch]).ravel())
            _unitaries = {q: qtn.Gate('U3', to_end(np.random.uniform(0, 2*np.pi, 3)), qubits=[q], round=None, parametrize=False) for q in qs}
            unitaries[k] = _unitaries
        self.final_unitaries = {}
        for q, patch_id in enumerate(final_patch_ids):
            gate = unitaries[patch_id][int(q)]
            self.final_unitaries[q] = gate
        
        # for testing purposes
        self.echo_unitaries = unitaries

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
                    self.echo_rqc.apply_gate(gate.label, params=gate.params, qubits=gate.qubits, parametrize=parametrize, gate_round=gate.round)
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

        # NOTE: Only implementing a single block for now
        #           If we wanted multiple blocks, need to generate all unitaries ahead, and use prior_patches etc.

        covering = self.generate_oqc_covering()
        cluster_layers = get_cluster_layers(covering)
        self.oqc_final_unitaries = {}

        # For testing
        self.oqc_patch_qcs = {} # for testing
        self.oqc_cluster_layers = cluster_layers
        self.oqc_covering = covering

        print("training OQC")
        full_qc_gates = {}
        for idx, (cluster, layers) in enumerate(cluster_layers):
            msg = f"patch: {idx+1}/{len(cluster_layers)}"
            global_qubits = list(cluster)
            patch_to_full = {i: x for i,x in enumerate(global_qubits)}
            full_to_patch = {x: i for i,x in enumerate(global_qubits)}

            for attempt in range(self.oqc_attempts):
                
                # build target
                qc_target = qtn.Circuit(len(global_qubits))
                for q in range(qc_target.N):
                    params = to_end(np.random.uniform(0, 2*np.pi, 3))
                    self.oqc_final_unitaries[patch_to_full[q]] = qtn.Gate('U3', params, qubits=[patch_to_full[q]], round=None, parametrize=False)
                    qc_target.apply_gate('U3', params=params, qubits=[q], gate_round=None, parametrize=False)

                # build patch to train
                oqc_patch = qtn.Circuit(len(global_qubits))
                r = 0
                for _ in range(1+int(self.oqc_repeat)):
                    for layer in layers:
                        for bond in layer:
                            qubits = [full_to_patch[q] for q in bond]
                            oqc_patch.apply_gate(self.gate_o.name, params=self.gate_o.random_params(), qubits=qubits, parametrize=True, gate_round=r)
                        r += 1

                # set loss function
                if self.oqc_loss == 'l2':
                    U_target = qc_target.get_uni().to_dense()
                    I = torch.argmax(torch.abs(U_target))
                    gauge_inds = (I//len(U_target), I%len(U_target))
                    loss = GaugeTensorTNLoss(U_target, gauge_inds=gauge_inds, alpha_thresh=self.alpha_thresh)
                elif self.oqc_loss == 'trace':
                    loss = TraceFidelityTNLoss(qc_target.get_uni())
                else:
                    raise RuntimeError(f"oqc_loss: {self.oqc_loss} is not a valid option.")

                # train patch
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings only
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
                    
                    if opt.loss < self.oqc_tol:
                        break
                    elif attempt == self.oqc_attempts-1:
                        raise RuntimeError(f"patch {i+1} in echo_mosaic did not converge in {self.echo_attempts} attempts.")
                    
            # for testing
            self.oqc_patch_qcs[idx] = qc_opt.copy()

            # store gates to build full_qc
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
        
    def _build_rqc(self, parametrize: bool=False, end: str='back', **kwargs):

        to_end = self._get_to_end(end)
        self.gate_r.set_backend(to_end)

        self.rqc = qtn.Circuit(self.N, **kwargs)
        self.rqc.apply_to_arrays(to_end)

        if not self.rqc_layers:
            self.generate_rqc_pqc_layers()

        for t, layer in enumerate(self.rqc_layers):
            for bond in layer:
                self.rqc.apply_gate(self.gate_r.name, params=self.gate_r.random_params(), qubits=bond, gate_round=t, parametrize=parametrize)

    def _build_pqc(self, parametrize: bool=True, end: str='back', **kwargs):

        to_end = self._get_to_end(end)
        self.gate_p.set_backend(to_end)

        self.pqc = qtn.Circuit(self.N, **kwargs)
        self.pqc.apply_to_arrays(to_end)

        if not self.pqc_layers:
            self.generate_rqc_pqc_layers()

        for t, layer in enumerate(self.pqc_layers):
            for bond in layer:
                self.pqc.apply_gate(self.gate_p.name, params=self.gate_p.random_params(), qubits=bond, gate_round=t, parametrize=parametrize)

    def generate_rqc_pqc_layers(self):

        self.rqc_layers = []
        for rqc_covering in self.rqc_coverings:
            if rqc_covering == '2':
                layers = self.mosaic.generate_2layer_covering(self.mosaic.G, seed=self.seed, max_trials=self.max_trials)
            elif rqc_covering == '3':
                layers = self.mosaic.generate_3layer_covering(self.mosaic.G, seed=self.seed, max_trials=self.max_trials)
            else:
                raise RuntimeError(f"rqc_covering type: {rqc_covering} is not a valid option")
            self.rqc_layers += layers

        self.pqc_layers = []
        pqc_coverings = self.pqc_coverings
        if self.pqc_match_rqc_connectivity:
            self.pqc_layers = copy.deepcopy(self.rqc_layers)
            pqc_coverings = self.pqc_coverings[len(self.rqc_coverings):]
        if self.pqc_reverse_rqc_connectivity:
            self.pqc_layers = copy.deepcopy(self.rqc_layers[::-1])
            pqc_coverings = self.pqc_coverings[len(self.rqc_coverings):]
        for pqc_covering in pqc_coverings:
            if pqc_covering == '2':
                layers = self.mosaic.generate_2layer_covering(self.mosaic.G, seed=self.seed, max_trials=self.max_trials)
            elif pqc_covering == '3':
                layers = self.mosaic.generate_3layer_covering(self.mosaic.G, seed=self.seed, max_trials=self.max_trials)
            else:
                raise RuntimeError(f"pqc_covering type: {pqc_covering} is not a valid option.")
            self.pqc_layers += layers

    def _update_progbar_callback(self, opt, postfix_str: str, loss_threshold: float):
        if hasattr(opt, '_pbar') and opt._pbar is not None:
            opt._pbar.set_postfix_str(postfix_str)  # Directly set the postfix string
        if self.early_stopping:
            early_stopping_callback(opt, loss_threshold)
        return False  # Returning False means "continue optimization"
