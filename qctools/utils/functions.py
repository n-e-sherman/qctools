import copy
import torch
import qiskit
import numpy as np
import pandas as pd
import quimb.tensor as qtn
from typing import List, Tuple
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagnode import DAGOpNode
from collections import defaultdict

from .lambdas import TorchConverter, NumpyConverter

def i_to_b(index, num_qubits, order="big"):
    """Convert an integer index to a bitstring of a given length, with optional ordering.

    Args:
        index (int): The integer to convert.
        num_qubits (int): The number of qubits (length of the bitstring).
        order (str): The bit ordering, either "little" or "big".
                     "little" gives the least significant bit first (default).
                     "big" gives the most significant bit first.

    Returns:
        str: The bitstring representation of the index.
    """
    if order not in ["little", "big"]:
        raise ValueError("Order must be either 'little' or 'big'")

    # Convert to a bitstring
    bitstring = format(index, f"0{num_qubits}b")

    # Reverse the bitstring for little-endian order
    if order == "little":
        bitstring = bitstring[::-1]

    return bitstring



def b_to_i(bitstring, order="big"):
    """Convert a bitstring to an integer index, with optional ordering.

    Args:
        bitstring (str): The bitstring to convert.
        order (str): The bit ordering, either "little" or "big".
                     "little" treats the bitstring as little-endian (default).
                     "big" treats the bitstring as big-endian.

    Returns:
        int: The integer index represented by the bitstring.
    """
    if order not in ["little", "big"]:
        raise ValueError("Order must be either 'little' or 'big'")

    # Reverse the bitstring for little-endian order
    if order == "little":
        bitstring = bitstring[::-1]

    # Convert the bitstring to an integer
    return int(bitstring, 2)




def gauge_tensor_loss(circ: qtn.Circuit, target: torch.Tensor, 
                      gauge: bool=True, 
                      gauge_inds: Tuple[int]=(0, 0), 
                      alpha_thresh: float=1e-7
):

        U = circ.get_uni().to_dense()
        if gauge:
            alpha = U[*gauge_inds]
            beta = target[*gauge_inds]
            if abs(alpha) > alpha_thresh:
                U = U * (beta / alpha)
        return (abs(U - target)**2).sum()



def bitstring_loss(circ: qtn.Circuit, target: str):

    return -torch.abs(circ.amplitude(target))**2

def mlflow_runs_to_df(all_runs):

    df_all = pd.DataFrame()

    print("total runs:", len(all_runs))
    for i, run in enumerate(all_runs):
        print(i, end='\r')
        
        run_dict = run.to_dictionary()
        run_dict.keys()
        
        res_dict = {}
        res_dict.update(run_dict['info'])
        
        # res_dict.update(run_dict['inputs'])
        for key, data in run_dict['data'].items():
            if key == "params":
                if "loss" in data:
                    data["loss_fn"] = data.pop("loss")
            res_dict.update(data)
            
        
        _df = pd.DataFrame([res_dict])
        df_all = pd.concat([df_all, _df])


    # update dtypes
    for column in df_all.columns:
        if column == 'target':
            continue
        try:
            df_all[column] = pd.to_numeric(df_all[column])
        except:
            pass
    df_all = df_all.convert_dtypes()
    return df_all


def qiskit_to_quimb(qc_qiskit, converter = TorchConverter()):

    QISKIT_TO_QUIMB = {
        'cz': 'CZ',
        'x': 'X',
        'sx': 'X_1_2',
        'rz': 'RZ',
        'u3': 'U3',
        'cx': 'CX'
    }

    qc = qtn.Circuit(qc_qiskit.num_qubits)
    qc.apply_to_arrays(converter)
    q_to_i = {q:i for i,q in enumerate(qc_qiskit.qubits)}
    for i, instruction in enumerate(qc_qiskit.data):
            
        if instruction.name in ['measure', 'barrier']: # ignore measure instructions
            continue
    
        gate = QISKIT_TO_QUIMB.get(instruction.operation.name, instruction.operation.name.upper())
        qubits = [q_to_i[q] for q in instruction.qubits]
        params = converter(instruction.operation.params)
        qc.apply_gate(gate, params=params, qubits=qubits)

    return qc

def qiskit_to_quimb_raw(qc_qiskit, **circuit_args):

    qc_quimb = qtn.Circuit(qc_qiskit.num_qubits, **circuit_args)
    q_to_i = {q: i for i, q in enumerate(qc_qiskit.qubits)}

    for instruction in qc_qiskit.data:
        U = Operator(instruction.operation).data
        qubits = [q_to_i[q] for q in instruction.qubits]

        if len(qubits) == 2:
            U = U.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)

        qc_quimb.apply_gate_raw(U, where=qubits)
    return qc_quimb

def quimb_to_qiskit(qc_quimb: qtn.Circuit) -> qiskit.QuantumCircuit:
        
        QISKIT_TO_QUIMB = {
            'cz': 'CZ',
            'x': 'X',
            'sx': 'X_1_2',
            'rz': 'RZ',
            'u': 'U3',
            'cx': 'CX'
        }
        QUIMB_TO_QISKIT = {v:k for k,v in QISKIT_TO_QUIMB.items()}

        to_end = NumpyConverter()
        qc_qiskit = qiskit.QuantumCircuit(qc_quimb.N)

        for gate in qc_quimb.gates:
            label = gate.label
            qiskit_label = QUIMB_TO_QISKIT.get(label, label.upper())
            try:
                getattr(qc_qiskit, qiskit_label)(*to_end(gate.params), *gate.qubits)
            except AttributeError: # use raw matrix
                if gate.parametrize:
                    U = gate.array.data
                else:
                    U = gate.array
                U = U.reshape(int(np.sqrt(len(U.ravel()))), -1)
                ugate = UnitaryGate(to_end(U), label='SU4')
                qc_qiskit.append(ugate, gate.qubits)
                
                 

        return qc_qiskit

def set_rounds(qc: qtn.Circuit, parametrize: bool=False, round_start: int=0):

    rounds = np.zeros(qc.N) + round_start
    qc_res = qtn.Circuit(qc.N)

    for gate in qc.gates:
        qubits = gate.qubits
        round = np.max(rounds[[qubits]].ravel())
        for q in qubits:
            rounds[q] = round + 1
        qc_res.apply_gate(gate.label, params=gate.params, qubits=gate.qubits, parametrize=parametrize, gate_round=round)
    return qc_res

def permute_qubits(qc, return_perm=False):

    qc_perm = qtn.Circuit(qc.N)
    qubits = list(range(qc.N))
    np.random.shuffle(qubits)
    perm = {i:q for i,q in enumerate(qubits)}

    for gate in qc.gates:
        qubits = [perm[q] for q in gate.qubits]
        qc_perm.apply_gate(gate.label, params=gate.params, qubits=qubits)
    res = qc_perm
    if return_perm:
        res = (qc_perm, perm)
    return res

def reorder_gates(qc):

    qc_rounded = set_rounds(qc)
    qc_reordered = qtn.Circuit(qc.N)
    full_qc_gates = {}
    for gate in qc_rounded.gates:
        full_qc_gates[gate.round] = sorted(full_qc_gates.get(gate.round, []) + [gate], key=lambda x: x.qubits[0])
    rounds = np.sort(list(full_qc_gates))
    for round in rounds:
        qc_reordered.apply_gates(full_qc_gates[round])
    return qc_reordered

def permute_peaked_circuit(qc, target):

    qubits = list(range(qc.N))
    np.random.shuffle(qubits)
    perm = {i: q for i, q in enumerate(qubits)}
    perm_inv = {q: i for i, q in enumerate(qubits)}

    qc_perm = qtn.Circuit(qc.N)
    for gate in qc.gates:
        qubits = [perm[q] for q in gate.qubits]
        qc_perm.apply_gate(gate.label, params=gate.params, qubits=qubits)

    target_perm = ''
    for i in range(len(target)):
        target_perm += target[perm_inv[i]]

    return qc_perm, target_perm

def permute_peaked_circuit_qiskit(qc, target_bitstring):
    num_qubits = qc.num_qubits
    qubit_indices = list(range(num_qubits))
    np.random.shuffle(qubit_indices)

    # Mapping: old_index → new_index
    perm = {i: q for i, q in enumerate(qubit_indices)}
    # Inverse mapping: new_index → old_index
    perm_inv = {q: i for i, q in enumerate(qubit_indices)}

    # Create a new permuted quantum register and circuit
    new_qr = QuantumRegister(num_qubits)
    qc_permuted = QuantumCircuit(new_qr)

    for instr, qargs, cargs in qc.data:
        # Use find_bit(q) to get the original index
        permuted_qargs = [new_qr[perm[qc.find_bit(q).index]] for q in qargs]
        qc_permuted.append(instr, permuted_qargs, cargs)

    # Permute the target bitstring using the inverse map
    target_perm = ''.join(target_bitstring[perm_inv[i]] for i in range(num_qubits))

    return qc_permuted, target_perm

def fuse_2q_gates_qiskit(circ):
    dag = circuit_to_dag(circ)
    new_dag = dag.copy_empty_like()

    qubit_indices = {q: i for i, q in enumerate(circ.qubits)}
    instructions = list(dag.topological_op_nodes())

    pending_1q = defaultdict(list)  # qubit -> list of pending 1q ops
    active_blocks = dict()  # (q0, q1) -> list of gates

    def flush_block(pair):
        block = active_blocks.pop(pair, [])
        if not block:
            return
        q0, q1 = pair
        index_map = {q0: 0, q1: 1}
        sub_qc = QuantumCircuit(2)
        for node in block:
            sub_qc.append(node.op, [index_map[q] for q in node.qargs])
        fused_op = Operator(sub_qc)
        new_dag.apply_operation_back(
            fused_op.to_instruction(),
            [q0, q1]
        )

    def flush_all_blocks():
        for pair in list(active_blocks):
            flush_block(pair)

    for node in instructions:
        qubits = node.qargs

        if node.op.num_qubits == 1:
            q = qubits[0]
            found_block = False
            for (q0, q1) in active_blocks:
                if q in (q0, q1):
                    active_blocks[(q0, q1)].append(node)
                    found_block = True
                    break
            if not found_block:
                pending_1q[q].append(node)

        elif node.op.num_qubits == 2:
            q0, q1 = sorted(qubits, key=lambda q: qubit_indices[q])
            pair = (q0, q1)

            # Flush any overlapping blocks that share only one qubit
            overlapping = [p for p in active_blocks if len(set(p).intersection(pair)) == 1]
            for p in overlapping:
                flush_block(p)

            if pair in active_blocks:
                active_blocks[pair].append(node)
            else:
                block = pending_1q[q0] + pending_1q[q1] + [node]
                pending_1q[q0].clear()
                pending_1q[q1].clear()
                active_blocks[pair] = block

        else:
            flush_all_blocks()
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

    flush_all_blocks()

    # flush any remaining 1Q gates not absorbed
    for q, ops in pending_1q.items():
        for node in ops:
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

    return dag_to_circuit(new_dag)

def reverse_bits_statevector(vec, num_qubits):
    """Reverse bitstrings of statevector indices."""
    N = len(vec)
    reversed_vec = np.empty_like(vec)
    for i in range(N):
        reversed_index = int(f"{i:0{num_qubits}b}"[::-1], 2)
        reversed_vec[reversed_index] = vec[i]
    return reversed_vec

def get_gate_tuples(qc: qtn.Circuit) -> List[Tuple[int]]:

    gates = []
    for gate in qc.gates:
        gates.append((*gate.qubits, gate.round))
    return gates

def trace_metric(qc1, qc2):

    assert qc1.N == qc2.N
    N = qc1.N

    return abs((qc1.get_uni().H & qc2.get_uni()).contract(all, optimize='auto-hq')) / (2**N)

def overlap_metric(qc1, qc2):

    return abs((qc1.psi.H & qc2.psi).contract(all, optimize='auto-hq'))

class EarlyStopException(Exception):
    """Custom exception to stop optimization early."""
    pass

def early_stopping_callback(opt, loss_threshold):
    """Callback function to stop training early when loss is below threshold."""
    if opt.loss < loss_threshold:
        raise EarlyStopException  # Forcefully terminate optimization