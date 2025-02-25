import numpy as np
import quimb.tensor as qtn
from typing import Iterable, Tuple, Callable, Optional, Any
from quimb.tensor.circuit import PARAM_GATES, CONSTANT_GATES

from qctools.utils import IdentityFunction

class Gate:

    _name = "BASE"
    _num_qubits = 0
    _num_params = 0

    _param_gates = PARAM_GATES
    _constant_gates = CONSTANT_GATES.copy() # copy so can change backend if needed
    _constant_gates['I4'] = np.eye(4)

    def __init__(self, distribution: str="uniform", dist_params: Tuple=[0.0, 2*np.pi], to_backend: Optional[Callable] = None):

        self._random_func = {
            'normal': np.random.normal,
            'uniform': np.random.uniform
        }.get(distribution)
        self._dist_params = dist_params
        
        if to_backend is None:
            to_backend = IdentityFunction()

        self.set_backend(to_backend)
        # self.register_gate()

    def set_backend(self, to_backend: Callable):
        self.to_backend = to_backend
        for k,v in self._constant_gates.items():
            self._constant_gates[k] = to_backend(v)

    def __call__(self, params: Iterable) -> qtn.Tensor:
        
        raise NotImplementedError(f"The __call__ function most be overwritten by derived classes.")
    
    def apply(self, *args, **kwargs) -> Any:

        raise NotImplementedError(f"The apply function most be overwritten by derived classes.")
    
    def validate(self) -> Any:

        params = self.random_params()
        UC = self(params)
        if isinstance(UC, qtn.Tensor):
            UC = UC.data

        circ = qtn.Circuit(self.num_qubits)
        qubits = [i for i in range(self.num_qubits)]
        if len(qubits) == 1:
            qubits = qubits[0]
        self.apply(circ, qubits, params)
        circ.apply_to_arrays(self.to_backend)
        UE = circ.get_uni().to_dense().reshape(*UC.shape)
        return (UC-UE).sum()
    
    def random_params(self) -> qtn.Tensor:
        return self.to_backend(self._random_func(*self._dist_params, self.num_params))
    
    def register_gate(self):
        
        qtn.circuit.register_param_gate(self.name, self.__call__, self.num_qubits)
    
    @property
    def name(self):
        return f"{self._name}"
    
    @property
    def num_params(self):
        return self._num_params
    
    @property
    def num_qubits(self):
        return self._num_qubits