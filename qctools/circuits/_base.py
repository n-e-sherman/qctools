import inspect
import warnings
import qiskit
import numpy as np
import quimb.tensor as qtn
from qiskit import qasm, qasm2, qasm3
from abc import ABC, abstractmethod, update_abstractmethods
from typing import Callable, Optional, Union, Iterable, Dict

from qctools.utils import TorchConverter, NumpyConverter


class CircuitManager(ABC):
     
    QISKIT_TO_QUIMB = {
        'cz': 'CZ',
        'x': 'X',
        'sx': 'X_1_2',
        'rz': 'RZ',
        'u3': 'U3',
        'cx': 'CX'
    }
    QUIMB_TO_QISKIT = {v:k for k,v in QISKIT_TO_QUIMB.items()}

    def __init__(self, 
                  qasm: Optional[str]=None, 
                  to_backend: Optional[Callable]=None,
                  to_frontend: Optional[Callable]=None,
    ):

        self.to_backend = to_backend if to_backend is not None else TorchConverter()
        self.to_frontend = to_frontend if to_frontend is not None else NumpyConverter()
        self._set_qasm(qasm)
           

    @abstractmethod
    def build(self, end: str='back', which: Optional[Union[str|Iterable]]=None, parametrize: Union[bool, Dict[str, bool]]=False, **kwargs) -> None:
        """Abstract build method to construct specified circuit parts."""
        pass

    @abstractmethod
    def get_log_params(self):
        """Derived classes must define the default circuit combination order."""
        pass
    
    @property
    @abstractmethod
    def default_which(self):
        """Derived classes must define the default circuit combination order."""
        pass

    @property
    @abstractmethod
    def gates(self):
        """Derived classes must define the default circuit combination order."""
        pass

    def get(self, 
            build: bool=True, 
            parametrize: Union[bool, Dict[str, bool]]=False, 
            end: str='back',
            which: Optional[Union[str|Iterable[str]]]=None, 
            **kwargs
    ) -> qtn.Circuit:
        
        """Combine specified circuits in the provided order using getattr."""

        # which circuits to construct
        which_list, parametrize_map = self._process_which_and_parametrize(which, parametrize)
        
        # Build missing circuits if needed
        missing = [name for name in which_list if getattr(self, name.lower(), None) is None]
        parametrize_missing = {k: parametrize_map[k] for k in missing}
        if len(missing) > 0:
            if build:
                self.build(which=missing, parametrize=parametrize_missing, end=end)
            else:
                raise ValueError(f"Missing circuits: {missing}. Use build_if_missing=True to build them.")

        # get the qcs and maybe convert parametrization
        qcs = []
        for circuit_name in which_list:
            qc = getattr(self, circuit_name.lower(), None) #.lower ensures consistent naming convention
            if qc is None:
                raise ValueError(f"Circuit '{circuit_name}' has not been built.")
            qc = self._maybe_convert_parametrization(qc, parametrize_map[circuit_name])
            qcs.append(qc)

        # combine qcs
        qc_res = qtn.Circuit(np.max([_qc.N for _qc in qcs]))
        gate_round = 0
        for qc in qcs:
            for gate in qc.gates:
                qc_res.apply_gate(gate.label, params=gate.params, qubits=gate.qubits, 
                                  gate_round=gate_round+gate.round, parametrize=gate.parametrize)
            gate_round += (gate.round + 1)

        # convert backend
        to_end = self._get_to_end(end)
        qc_res = self._convert_backend(qc_res, to_end)    
            
        return qc_res

    def update(self, allow_new: bool=False, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    # update based on new params rather than the circuit directly
                    old = getattr(self, key)
                    if isinstance(old, dict):
                        setattr(self, key, value)
                    elif isinstance(old, qtn.Circuit): # update circuit with dict of params
                        old_qc = self.get(which=key, parametrize=True)
                        old_params, old_skeleton = qtn.pack(old_qc)
                        assert len(old_params) == len(value), f"trying to update parameters of circuit {key}, which has {len(old_params)} parameters while new value has {len(value)} params"
                        new_params = {k:v for k,v in zip(old_params.keys(), value.values())}
                        new_qc = qtn.unpack(new_params, old_skeleton)
                        setattr(self, key, new_qc)
                    else:
                        raise RuntimeError(f"trying to update member {key} with a dict, but it is neither a circuit or a dict")
                else:
                    setattr(self, key, value)
            elif allow_new:
                warnings.warn(f"Adding new attribute '{key}'.")
                setattr(self, key, value)
            else:
                raise AttributeError(f"{self.__class__.__name__} has no attribute '{key}'. Set `allow_new=True` to add it.")
    
    def expand_circuit(self, circ: qtn.Circuit) -> qtn.Circuit:
 
        qc_expanded = qtn.Circuit(circ.N)
        for gate in circ.gates:
            self.gates[gate.label].apply(qc_expanded, gate.qubits, gate.params, parametrize=gate.parametrize)
        
        return qc_expanded

    def to_qiskit(self, end: str="front", which: Optional[Union[str|Iterable]]=None, build: bool=False) -> qiskit.QuantumCircuit:

        to_end = self._get_to_end(end)
        qc_quimb = self.get(which=which, build=build, parametrize=False)
        qc_quimb = self.expand_circuit(qc_quimb)
        qc_qiskit = qiskit.QuantumCircuit(qc_quimb.N)

        for gate in qc_quimb.gates:
            label = gate.label
            qiskit_label = self.QUIMB_TO_QISKIT[label]
            getattr(qc_qiskit, qiskit_label)(*to_end(gate.params), *gate.qubits)

        return qc_qiskit
    
    def to_qasm(self, end: str="front", which: Optional[Union[str|Iterable]]=None, build: bool=False, file_path: Optional[str]=None) -> str:

        qc = self.to_qiskit(end=end, which=which, build=build)
        qasm_str = self._qasm_obj.dumps(qc)

        if file_path is not None:
            self._qasm_obj.dump(qc, file_path)

        return qasm_str
    
    def convert_backend(self, end: Union[str|Callable]='front'):
        
        if isinstance(end, str):
            to_end = self._get_to_end(end)
        for circuit_name in self.default_which:
            qc = getattr(self, circuit_name.lower(), None) #.lower ensures consistent naming convention
            if qc:
                qc = self._convert_backend(qc, to_end)
            setattr(self, circuit_name, qc)

    def _get_to_end(self, end: str) -> Callable:

        if end == 'back':
            to_end = self.to_backend
        elif end == 'front':
            to_end = self.to_frontend
        else:
            raise RuntimeError(f"to_end with choice end={end} is not valid.")
        return to_end

    def _set_qasm(self, which: Optional[str]=None): 

        # NOTE: Issue with qasm2 and qasm that I do not understand
        if which is None:
            self._qasm_obj = qasm2
        elif which == "1":
            self._qasm_obj = qasm
        elif which == "3":
            self._qasm_obj = qasm3
        else:
            raise RuntimeError(f"qasm = {which} is not a valid choice.")
        
    def _process_which_and_parametrize(self, which: Optional[Union[str|Iterable[str]]]=None, parametrize: Union[bool, Dict[str, bool]]=False, ):

        which_list = (
            [which] if isinstance(which, str)
            else list(which) if which is not None
            else self.default_which
        )

        # Normalize 'parametrize' to a dict, each circuit in which_list gets a parametrize value
        if isinstance(parametrize, bool):
            parametrize_map = {name: parametrize for name in which_list}
        elif isinstance(parametrize, dict):
            parametrize_map = {name: parametrize.get(name, False) for name in which_list}
        else:
            raise TypeError("'parametrize' must be a bool or a dict of circuit names to bools.")
        assert len(which_list) == len(parametrize_map), "must have a parametrize value for each circuit in which_list."

        return which_list, parametrize_map
    
    def _maybe_convert_parametrization(self, qc: qtn.Circuit, parametrize: bool=False):

        # already correct parametrize
        if (len(qc.gates) == 0) or (parametrize == qc.gates[0].parametrize):
            return qc
        
        qc_res = qtn.Circuit(qc.N)
        if isinstance(qc._psi.tensors[0].data, self.to_frontend(qc._psi.tensors[0].data).__class__): 
            qc_res.apply_to_arrays(self.to_frontend)
        else:
            qc_res.apply_to_arrays(self.to_backend)

        for gate in qc.gates:
            qc_res.apply_gate(gate.label, params=gate.params, qubits=gate.qubits, gate_round=gate.round, parametrize=parametrize)
        return qc_res
    
    def _convert_backend(self, qc: qtn.Circuit, to_end: Callable=TorchConverter()):

        qc_res = qtn.Circuit(qc.N)
        qc_res.apply_to_arrays(to_end)

        if len(qc.gates) == 0:
            return qc_res

        gates = list(set([gate.label for gate in qc.gates]))
        for gate in gates:
            self.gates[gate].set_backend(to_end)

        for gate in qc.gates:
            qc_res.apply_gate(gate.label, params=to_end(gate.params), qubits=gate.qubits, gate_round=gate.round, parametrize=gate.parametrize)

        return qc_res