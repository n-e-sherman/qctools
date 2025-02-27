import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_unitary
from qiskit.circuit.library import UnitaryGate

from ._base import *
from .one_qubit_gates import *
from .two_qubit_gates import *
from .composite_gates import *

class SV4Gate(FourUThreeGGate):

    def __init__(self, distribution: str="uniform", dist_params: Tuple=[0.0, 2*np.pi], to_backend: Optional[Callable] = None, rename: bool=True):
        
        U1 = V3Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U2 = V3Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        G12 = CZGate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U3 = S1Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U4 = A1Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        G34 = CZGate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U5 = S1Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U6 = SXGate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        G56 = CZGate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U7 = V3Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U8 = V3Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)

        super().__init__(U1, U2, G12, U3, U4, G34, U5, U6, G56, U7, U8, distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        
        if rename:
            self._name = "SV4"
            self.register_gate()

class SV4CutGate(ThreeUThreeGGate):

    def __init__(self, distribution: str="uniform", dist_params: Tuple=[0.0, 2*np.pi], to_backend: Optional[Callable] = None, rename: bool=True):
        
        U1 = V3Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U2 = V3Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        G12 = CZGate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U3 = S1Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U4 = A1Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        G34 = CZGate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U5 = S1Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U6 = SXGate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        G56 = CZGate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)

        super().__init__(U1, U2, G12, U3, U4, G34, U5, U6, G56, distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        
        if rename:
            self._name = "SV4CUT"
            self.register_gate()

class HaarCZGate(OneUOneGGate):

    def __init__(self, distribution: str="uniform", dist_params: Tuple=[0.0, 2*np.pi], to_backend: Optional[Callable] = None, rename: bool=True):

        U1 = Haar2Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U2 = Haar2Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        G12 = CZGate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        super().__init__(U1, U2, G12, distribution=distribution, dist_params=dist_params, to_backend=to_backend)

        if rename:
            self._name = 'HAARCZSTACK'
            self.register_gate()

class HaarCZStackGate(ThreeUTwoGGate):

    def __init__(self, distribution: str="uniform", dist_params: Tuple=[0.0, 2*np.pi], to_backend: Optional[Callable] = None, rename: bool=True):

        U1 = Haar2Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U2 = Haar2Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        G12 = CZGate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U3 = Haar2Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U4 = Haar2Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        G34 = CZGate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U5 = Haar2Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U6 = Haar2Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        super().__init__(U1, U2, G12, U3, U4, G34, U5, U6, distribution=distribution, dist_params=dist_params, to_backend=to_backend)

        if rename:
            self._name = 'HAARCZSTACK'
            self.register_gate()

class HaarCZMinimalStackGate(ThreeUTwoGGate):

    def __init__(self, distribution: str="uniform", dist_params: Tuple=[0.0, 2*np.pi], to_backend: Optional[Callable] = None, rename: bool=True):

        U1 = Haar2Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U2 = Haar2Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        G12 = CZGate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U3 = S1Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U4 = S1Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        G34 = CZGate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U5 = Haar2Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U6 = Haar2Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        super().__init__(U1, U2, G12, U3, U4, G34, U5, U6, distribution=distribution, dist_params=dist_params, to_backend=to_backend)

        if rename:
            self._name = 'HAARCZMINSTACK'
            self.register_gate()

class Haar4Gate(SV4Gate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._phase = 1
        self._name = 'HAAR4'
        self.register_gate()

    def random_params(self):
        su4_matrix = random_unitary(4).to_matrix()
        su4_gate = UnitaryGate(su4_matrix, label="SU4")
        qc = QuantumCircuit(2)  # 2-qubit circuit
        qc.append(su4_gate, [1, 0])  # apply in reverse order for equivalence

        # Step 3: Transpile into a desired gateset
        basis_gates = ["rz", "sx", "cz"]  # IBM's native gateset
        qct = transpile(qc, basis_gates=basis_gates, optimization_level=3)
        params = []
        for instruction in qct.data:
            _params = instruction.operation.params
            if len(_params) > 0:
                params.append(_params[0])
        params = np.array(params)

        phase = qct.global_phase
        self._phase = np.exp(1j*phase - 1j*np.pi/2)
        return self.to_backend(params)

class DoubleV3CZGate(TwoUTwoGGate):

    def __init__(self, distribution: str="uniform", dist_params: Tuple=[0.0, 2*np.pi], to_backend: Optional[Callable] = None, rename: bool=True):
        U1 = V3Gate(distribution=distribution, dist_params=dist_params)
        U2 = V3Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        G12 = CZGate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U3 = V3Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        U4 = V3Gate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        G34 = CZGate(distribution=distribution, dist_params=dist_params, to_backend=to_backend)

        super().__init__(U1, U2, G12, U3, U4, G34, distribution=distribution, dist_params=dist_params, to_backend=to_backend)
        if rename:
            self._name = 'DOUBLEV3CZ'
            self.register_gate()
    
    def identity_params(self):

        params = np.zeros(self.num_params) - np.pi/2
        params[1] = params[4] = params[7] = params[10] = np.pi
        return self.to_backend(params)
