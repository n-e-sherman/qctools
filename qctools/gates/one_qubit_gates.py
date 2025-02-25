import numpy as np
import quimb.tensor as qtn
from typing import Iterable, Optional, Tuple
from quimb.tensor.tensor_core import Tensor
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_unitary
from qiskit.circuit.library import UnitaryGate

from ._base import Gate

class OneQubitGate(Gate):

    _name = "1QBASE"
    _num_qubits = 1
    _num_params = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Optional[Iterable]=None) -> qtn.Tensor:

        raise NotImplementedError(f"The __call__ function most be overwritten by derived classes.")

    def apply(self, circuit: qtn.Circuit, qubit: int, params: Optional[Iterable]=None, **gate_args):

        raise NotImplementedError(f"The __call__ function most be overwritten by derived classes.")

class I2Gate(OneQubitGate):

    _name = "I2"
    _num_params = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Optional[Iterable]=None, inds=["a1", "a0"]) -> qtn.Tensor:

        I = self._constant_gates['IDEN']

        U = I
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubit: int, params: Optional[Tuple]=None, **gate_args):

        return

class SXGate(OneQubitGate):

    _name = "SX"
    _num_params = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Optional[Iterable]=None, inds=["a1", "a0"]) -> qtn.Tensor:

        SX = self._constant_gates['X_1_2']

        U = SX
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubit: int, params: Optional[Tuple]=None, **gate_args):

        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)


class RZGate(OneQubitGate):

    _name = "RZ"
    _num_params = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Iterable, inds=["a1", "a0"]) -> qtn.Tensor:

        RZ = self._param_gates['RZ']

        U = RZ(params[0].reshape(-1))
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubit: int, params: Optional[Tuple]=None, **gate_args):

        circuit.apply_gate('RZ', params=params[0].reshape(-1), qubits=[qubit], **gate_args)


class V3Gate(OneQubitGate):

    _name = "V3"
    _num_params = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Iterable, inds=["a1", "a0"]) -> qtn.Tensor:

        RZ = self._param_gates['RZ']
        SX = self._constant_gates['X_1_2']

        U = RZ(params[2].reshape(-1)) @ SX @ RZ(params[1].reshape(-1)) @ SX @ RZ(params[0].reshape(-1))
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubit: int, params: Iterable, **gate_args):

        assert len(params) >= 3, "need 3 parameters for a V3 gate."

        circuit.apply_gate('RZ', params=params[0].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[1].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[2].reshape(-1), qubits=[qubit], **gate_args)


class V2Gate(OneQubitGate):

    _name = "V2"
    _num_params = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Iterable, inds=["a1", "a0"]) -> qtn.Tensor:

        RZ = self._param_gates['RZ']
        SX = self._constant_gates['X_1_2']

        U = RZ(params[1].reshape(-1)) @ SX @ RZ(params[0].reshape(-1))
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubit: int, params: Iterable, **gate_args):

        assert len(params) >= 2, "need 2 parameters for a V3 gate."

        circuit.apply_gate('RZ', params=params[0].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[1].reshape(-1), qubits=[qubit], **gate_args)

class V1Gate(RZGate):

    _name = 'V1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class S3Gate(OneQubitGate):

    _name = "S3"
    _num_params = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Iterable, inds=["a1", "a0"]) -> qtn.Tensor:

        RZ = self._param_gates['RZ']
        SX = self._constant_gates['X_1_2']

        U = SX @ RZ(params[2].reshape(-1)) @ SX @ RZ(params[1].reshape(-1)) @ SX @ RZ(params[0].reshape(-1)) @ SX
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubit: int, params: Iterable, **gate_args):

        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[0].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[1].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[2].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)

class S2Gate(OneQubitGate):

    _name = "S2"
    _num_params = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Iterable, inds=["a1", "a0"]) -> qtn.Tensor:

        RZ = self._param_gates['RZ']
        SX = self._constant_gates['X_1_2']

        U = SX @ RZ(params[1].reshape(-1)) @ SX @ RZ(params[0].reshape(-1)) @ SX
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubit: int, params: Iterable, **gate_args):

        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[0].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[1].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)

class S1Gate(OneQubitGate):

    _name = "S1"
    _num_params = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Iterable, inds=["a1", "a0"]) -> qtn.Tensor:

        RZ = self._param_gates['RZ']
        SX = self._constant_gates['X_1_2']

        U = SX @ RZ(params[0].reshape(-1)) @ SX
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubit: int, params: Iterable, **gate_args):

        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[0].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)

class A3Gate(OneQubitGate):

    _name = "A3"
    _num_params = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Iterable, inds=["a1", "a0"]) -> qtn.Tensor:

        RZ = self._param_gates['RZ']
        SX = self._constant_gates['X_1_2']

        U = RZ(params[2].reshape(-1)) @ SX @ RZ(params[1].reshape(-1)) @ SX @ RZ(params[0].reshape(-1)) @ SX
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubit: int, params: Iterable, **gate_args):

        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[0].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[1].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[2].reshape(-1), qubits=[qubit], **gate_args)


class A2Gate(OneQubitGate):

    _name = "A2"
    _num_params = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Iterable, inds=["a1", "a0"]) -> qtn.Tensor:

        RZ = self._param_gates['RZ']
        SX = self._constant_gates['X_1_2']

        U = RZ(params[1].reshape(-1)) @ SX @ RZ(params[0].reshape(-1)) @ SX
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubit: int, params: Iterable, **gate_args):

        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[0].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[1].reshape(-1), qubits=[qubit], **gate_args)


class A1Gate(OneQubitGate):

    _name = "A1"
    _num_params = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Iterable, inds=["a1", "a0"]) -> qtn.Tensor:

        RZ = self._param_gates['RZ']
        SX = self._constant_gates['X_1_2']

        U = RZ(params[0].reshape(-1)) @ SX
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubit: int, params: Iterable, **gate_args):

        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[0].reshape(-1), qubits=[qubit], **gate_args)


class W3Gate(OneQubitGate):

    _name = "W3"
    _num_params = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Iterable, inds=["a1", "a0"]) -> qtn.Tensor:

        RZ = self._param_gates['RZ']
        SX = self._constant_gates['X_1_2']

        U =  SX @ RZ(params[2].reshape(-1)) @ SX @ RZ(params[1].reshape(-1)) @ SX @ RZ(params[0].reshape(-1))
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubit: int, params: Iterable, **gate_args):

        circuit.apply_gate('RZ', params=params[0].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[1].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[2].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        
class W2Gate(OneQubitGate):

    _name = "W2"
    _num_params = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Iterable, inds=["a1", "a0"]) -> qtn.Tensor:

        RZ = self._param_gates['RZ']
        SX = self._constant_gates['X_1_2']

        U =  SX @ RZ(params[1].reshape(-1)) @ SX @ RZ(params[0].reshape(-1))
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubit: int, params: Iterable, **gate_args):

        circuit.apply_gate('RZ', params=params[0].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        circuit.apply_gate('RZ', params=params[1].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)
        
class W1Gate(OneQubitGate):

    _name = "W1"
    _num_params = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Iterable, inds=["a1", "a0"]) -> qtn.Tensor:

        RZ = self._param_gates['RZ']
        SX = self._constant_gates['X_1_2']

        U =  SX @ RZ(params[0].reshape(-1))
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubit: int, params: Iterable, **gate_args):

        circuit.apply_gate('RZ', params=params[0].reshape(-1), qubits=[qubit], **gate_args)
        circuit.apply_gate('X_1_2', qubits=[qubit], **gate_args)



class Haar2Gate(V3Gate):

    _name = "H2"
    _num_params = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def random_params(self):

        su2_matrix = random_unitary(2).to_matrix()

        su2_gate = UnitaryGate(su2_matrix, label="SU4")
        qc = QuantumCircuit(1)  # 2-qubit circuit
        qc.append(su2_gate, [0])  # Apply SU(4) gate to qubits 0 and 1

        # Step 3: Transpile into a desired gateset
        basis_gates = ["rz", "sx", "cz"]  # IBM's native gateset
        qct = transpile(qc, basis_gates=basis_gates, optimization_level=3)

        phase = qct.global_phase
        params = []
        for instruction in qct.data:
            _params = instruction.operation.params
            if len(_params) > 0:
                params.append(_params[0])
        params = np.array(params)

        return self.to_backend(params)
