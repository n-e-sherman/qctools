import quimb.tensor as qtn
from typing import Iterable, Optional, Tuple
from quimb.tensor.tensor_core import Tensor

from ._base import Gate

class TwoQubitGate(Gate):

    _name = "2QBASE"
    _num_qubits = 2
    _num_params = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Iterable) -> qtn.Tensor:

        raise NotImplementedError(f"The __call__ function most be overwritten by derived classes.")

    def apply(self, circuit: qtn.Circuit, qubits: Tuple[int], params: Optional[Iterable]=None, **gate_args):

        raise NotImplementedError(f"The __call__ function most be overwritten by derived classes.")
    
class I4Gate(TwoQubitGate):

    _name = "I4"
    _num_params = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Optional[Iterable]=None, inds=["a1", "b1", "a0", "b0"]) -> qtn.Tensor:

        I = self._constant_gates['I4']

        U = I
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubit: int, params: Optional[Tuple]=None, **gate_args):

        return

class CZGate(TwoQubitGate):

    _name = "CZ"
    _num_params = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __call__(self, params: Optional[Iterable]=None, inds=["a1", "b1", "a0", "b0"]) -> qtn.Tensor:

        CZ = self._constant_gates['CZ']
        U = CZ.reshape(2, 2, 2, 2)
        return Tensor(U, inds=inds)

    def apply(self, circuit: qtn.Circuit, qubits: Tuple[int], params: Optional[Iterable]=None, **gate_args):

        circuit.apply_gate('CZ', qubits=qubits, **gate_args)

class RZZGate(TwoQubitGate):

    _name = "RZZ"
    _num_params = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Iterable, inds=["a1", "b1", "a0", "b0"]) -> qtn.Tensor:

        RZZ = self._param_gates['RZZ']

        U = RZZ(params[0].reshape(-1)).reshape(2, 2, 2, 2)
        return Tensor(U, inds=inds)
    
    def apply(self, circuit: qtn.Circuit, qubits: Tuple[int], params: Optional[Tuple]=None, **gate_args):

        circuit.apply_gate('RZZ', params=params[0].reshape(-1), qubits=qubits, **gate_args)