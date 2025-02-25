import random
import torch
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize import TNOptimizer
from typing import Optional, List

from qctools.gates import *
from qctools.utils import TorchConverter, NumpyConverter

# NOTE: This is sloppy, maybe clean up later


class PeakedCircuit(qtn.Circuit):
    def __init__(self, circuit: qtn.Circuit, target: str, peak: float, 
                 tau_r: Optional[int]=None, 
                 tau_p: Optional[int]=None, 
                 D: Optional[int]=None,
                 approximate: bool=True, 
                 **kwargs
    ):
        """
        Initialize a PeakedCircuit.

        Parameters:
        - circuit (qtn.Circuit): The base quantum circuit.
        - target: The target bitstring for peaking.
        - peak: The peak value associated with the target.
        """
        # Copy the circuit's internal structure to maintain full functionality
        super().__init__(N=circuit.N, **kwargs)
        self.__dict__.update(circuit.__dict__)

        # Store the additional metadata
        self.peak = peak
        self.target = target
        self.tau_r = tau_r
        self.tau_p = tau_p
        self._D = D
        self.approximate = approximate

    # def copy(self):

    #     new = super().copy()
    #     new.peak = self.peak
    #     new.target = self.target
    #     new.tau_r = self.tau_r
    #     new.tau_p = self.tau_p
    #     new._D = self._D
    #     new.approximate = self.approximate

    @property
    def D(self):
        
        if self._D is not None:
            return self._D
        try:
            return self.tau_r + self.tau_p
        except:
            return None
        
    def __repr__(self):
        return f"PeakedCircuit(N={self.N}, target={self.target}, peak={self.peak})"


def embed_gate(U, to_backend=TorchConverter()):

    converter = NumpyConverter()
    return to_backend(qu.kron(qu.identity(2), converter(U), qu.identity(2)))

def build_tower(D, gate):

    gate.register_gate()

    tower = qtn.Circuit(2)
    tower.apply_to_arrays(gate.to_backend)

    for d in range(D):
        tower.apply_gate(gate.name, params=gate.random_params(), qubits=[0, 1], parametrize=True, gate_round=d)
    return tower

def train_tower(tower, target=None, epochs=500, tol=1e-10, hessp=True):

    return_target=False
    if target is None:
        return_target = True
        target=''.join(random.choice('01') for _ in range(2))
    def loss(circ):

        return -torch.abs(circ.amplitude(target))**2

    opt = TNOptimizer(
        tower,
        loss,
        autodiff_backend="torch"
    )

    tower_opt = opt.optimize(epochs, hessp=hessp, tol=tol)
    return {"tower": tower_opt, "target": target, "loss": opt.loss}

def build_patch_segment(N, gate, depth=2, shifted=True):

    gate.register_gate()
    
    patch = qtn.Circuit(N)
    for d in range(depth):
        for i in range((d+int(shifted))%2, N-1, 2):
            patch.apply_gate(gate.name, params=gate.random_params(), qubits=[i, i+1], parametrize=True, gate_round=d)

    return patch

def train_patch_segment(patch, target, gauge=True, alpha_thresh=1e-7, epochs=500, hessp=True, tol=1e-10):

    I = torch.argmax(torch.abs(target))
    gauge_inds = (I // target.shape[0], I % target.shape[0])

    def loss(circ):
        U = circ.get_uni().to_dense()
        if gauge:
            alpha = U[*gauge_inds]
            beta = target[*gauge_inds]
            if abs(alpha) > alpha_thresh:
                U = U * (beta / alpha) * (abs(alpha) / abs(beta))
        return (abs(U - target)**2).sum()
    
    opt = TNOptimizer(
        patch,
        loss,
        autodiff_backend="torch"
    )

    patch_opt = opt.optimize(epochs, hessp=hessp, tol=tol)
    return {"patch": patch_opt, "loss": opt.loss}
    

def train_patch(D, gate, target=None, epochs=500, hessp=True, tol=1e-10, patch_depth=2, shifted=True, gauge=True, alpha_thresh=1e-7):

    _tower = build_tower(D//2, gate)
    while True:
        _tower = build_tower(D//2, gate)
        res = train_tower(_tower, target=target, epochs=epochs, hessp=hessp, tol=tol)
        if abs(res["loss"] + 1) < tol:
            break
    tower = res["tower"]
    target = res["target"]

    qc_patch = qtn.Circuit(4)
    round = 0
    for G in tower.gates:
        U = G.array.data.reshape(4, 4)
        U_embed = embed_gate(U, to_backend=gate.to_backend)
        while True:
            _patch = build_patch_segment(4, gate, depth=patch_depth, shifted=shifted)
            res = train_patch_segment(_patch, U_embed, gauge=gauge, alpha_thresh=alpha_thresh, epochs=epochs, hessp=hessp, tol=tol)
            if abs(res["loss"]) < 10*tol:
                break
        patch = res["patch"]

        for G in patch.gates:
            qc_patch.apply_gate(G.label, params=G.params, qubits=G.qubits, gate_round=G.round+round)
        round += patch_depth
    return qc_patch, target

import quimb.tensor as qtn
from typing import Optional

from typing import Optional, List

def stitch_peaked_circuits(qcs: List[PeakedCircuit], fills: Optional[List[bool]]=None, keep_parametrize=True, pbc=False):

    if fills is None:
        fills = [True, False]*(len(qcs)//2)
    assert len(fills) == len(qcs)

    # make target and N
    target_res = ''
    N = 0
    for qc,fill in zip(qcs, fills):
        target = qc.target
        if fill:
            target_res += target
            N += qc.N
        else:
            target_res += target[1:-1]
            N += qc.N-2
    if pbc:
        if (not fills[0]) and fills[-1]:
            target_res = qcs[-1].target[-1] + target_res[:-1]
    else: # NOTE: This might not work in general
        N += 1
    
    rounds = [qc.gates[-1].round for qc in qcs]
    assert all(rounds)

    qc_res = qtn.Circuit(N)
    for round in range(rounds[0]+1):

        q_shift = 0
        for qc in qcs:
            for gate in qc.gates:
                if gate.round == round:
                    parametrize = keep_parametrize and gate.parametrize
                    qubits = [(q+q_shift)%N for q in gate.qubits]
                    qc_res.apply_gate(gate.label, params=gate.params, qubits=qubits, parametrize=parametrize, gate_round=round)
            q_shift += (qc.N-1)
    
    approximate = all([qc.approximate for qc in qcs])
    peak = 1.
    for qc in qcs:
        peak *= qc.peak

    return PeakedCircuit(qc_res, target_res, peak, D=rounds[0]+1, approximate=approximate)

def stitch_circuits(qc1, qc2, target1, target2, fill_target='first', keep_parametrize=True, pbc=False):

    assert fill_target in ['first', 'last', 'random'], f"fill_target: {fill_target} is not a valid option"

    _t1 = target1[:-1]
    _t2 = target2[1:]
    _tmid = '0'
    if fill_target == 'first':
        _tmid = target1[-1]
    elif fill_target == 'last':
        _tmid = target2[0]
    else:
        _tmid = random.choice('01')
    target = _t1 + _tmid + _t2

    N = qc1.N + qc2.N - 1 - int(pbc)
    qc = qtn.Circuit(N)

    assert qc1.gates[-1].round == qc2.gates[-1].round
    rounds = qc1.gates[-1].round

    for round in range(rounds + 1):
        for gate in qc1.gates:
            if gate.round == round:
                parametrize = keep_parametrize and gate.parametrize
                qc.apply_gate(gate.label, params=gate.params, qubits=gate.qubits, parametrize=parametrize, gate_round=round)
        
        for gate in qc2.gates:
            if gate.round == round:
                qubits = [(qc1.N-1+q)%N for q in gate.qubits]
                parametrize = keep_parametrize and gate.parametrize
                qc.apply_gate(gate.label, params=gate.params, qubits=qubits, parametrize=parametrize, gate_round=round)

    return qc, target

import numpy as np

def shift_peaked_circuit(qc, shift, keep_parametrize=True, reorder_gates=True):

    # build circuit
    qc_res = qtn.Circuit(qc.N)
    for round in range(qc.gates[-1].round+1):
        gates = {}
        for gate in qc.gates:
            if gate.round < round:
                continue
            if gate.round > round:
                break
            qubits = [(q+shift)%qc_res.N for q in gate.qubits]
            parametrize = keep_parametrize and gate.parametrize
            gates[qubits[0]] = {'gate_id': gate.label, 'params': gate.params, 'qubits': qubits, 'parametrize': parametrize, 'gate_round' : round}
            
        inds = list(gates.keys())
        if reorder_gates:
            inds = np.sort(inds)
        for i in inds:
            qc_res.apply_gate(**gates[i])
            # qc_res.apply_gate(gates[i]['label'], params=gates[i]['params'], qubits=gates[i]['qubits'], gate_round=round)

    # make target
    target_res = qc.target[-shift:] + qc.target[:-shift]

    return PeakedCircuit(qc_res, target_res, qc.peak, tau_r=qc.tau_r, tau_p=qc.tau_p, D=qc.D, approximate=qc.approximate)
    
    
