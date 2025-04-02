import copy
import torch
import qiskit
import numpy as np
import pandas as pd
import quimb.tensor as qtn
from typing import Tuple

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

def quimb_to_qiskit(qc_quimb: qtn.Circuit) -> qiskit.QuantumCircuit:
        
        QISKIT_TO_QUIMB = {
            'cz': 'CZ',
            'x': 'X',
            'sx': 'X_1_2',
            'rz': 'RZ',
            'u3': 'U3',
            'cx': 'CX'
        }
        QUIMB_TO_QISKIT = {v:k for k,v in QISKIT_TO_QUIMB.items()}

        to_end = NumpyConverter()
        qc_qiskit = qiskit.QuantumCircuit(qc_quimb.N)

        for gate in qc_quimb.gates:
            label = gate.label
            qiskit_label = QUIMB_TO_QISKIT.get(label, label.upper())
            getattr(qc_qiskit, qiskit_label)(*to_end(gate.params), *gate.qubits)

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

class EarlyStopException(Exception):
    """Custom exception to stop optimization early."""
    pass

def early_stopping_callback(opt, loss_threshold):
    """Callback function to stop training early when loss is below threshold."""
    if opt.loss < loss_threshold:
        raise EarlyStopException  # Forcefully terminate optimization