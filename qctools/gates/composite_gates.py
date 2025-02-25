import quimb.tensor as qtn
import numpy as np
from typing import Iterable, Optional, Tuple, Callable
from quimb.tensor.tensor_core import Tensor, tensor_contract

from ._base import Gate
from .one_qubit_gates import OneQubitGate
from .two_qubit_gates import TwoQubitGate

class CompositeGate(Gate):

    _name = "COMPBASE"
    _num_qubits = 2
    _num_params = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, params: Iterable) -> qtn.Tensor:

        raise NotImplementedError(f"The __call__ function most be overwritten by derived classes.")

    def apply(self, circuit: qtn.Circuit, qubits: Tuple[int], params: Optional[Iterable]=None, **gate_args):

        raise NotImplementedError(f"The apply function most be overwritten by derived classes.")
    
    def set_num_params(self):

        self._num_params = 0
        for attr in dir(self):
            if isinstance(self.__getattribute__(attr), Gate):
                gate = self.__getattribute__(attr)
                self._num_params += gate.num_params
    
class OneUOneGGate(CompositeGate):

    def __init__(self, 
            U1: OneQubitGate,
            U2: OneQubitGate,     
            G12: TwoQubitGate, 
            distribution: str="uniform",
            dist_params: Tuple[int]=[0, 2*np.pi],
            to_backend: Optional[Callable]=None,
    ):
        super().__init__(distribution=distribution, dist_params=dist_params, to_backend=to_backend)

        self.U1 = U1
        U1.set_backend(self.to_backend)
        self.U2 = U2
        U2.set_backend(self.to_backend)
        self.G12 = G12
        G12.set_backend(self.to_backend)

        self._name = f"-{U1.name}-|{G12.name}|-\n-{U2.name}-|{G12.name}|-"
        
        self.set_num_params()

    def __call__(self, params: Iterable, inds=["a2", "b2", "a0", "b0"]):

        s = 0
        T1 = self.U1(params[s:s+self.U1.num_params], inds=["a1", inds[2]])
        s += self.U1.num_params
        T2 = self.U2(params[s:s+self.U2.num_params], inds=["b1", inds[3]])
        s += self.U2.num_params
        TG12 = self.G12(params[s:s+self.G12.num_params], inds=[inds[0], inds[1], "a1", "b1"])

        return tensor_contract(
            T1,
            T2,
            TG12,
            output_inds=inds,
            optimize="auto-hq",
        ).data
        
        
    def apply(self, circuit: qtn.Circuit, qubits: Tuple[int], params: Optional[Iterable]=None, **gate_args):

        q0, q1 = qubits
        
        s = 0
        self.U1.apply(circuit, q0, params[s:s+self.U1.num_params], **gate_args)
        s += self.U1.num_params
        self.U2.apply(circuit, q1, params[s:s+self.U2.num_params], **gate_args)
        s += self.U2.num_params
        self.G12.apply(circuit, qubits, params[s:s+self.G12.num_params], **gate_args)

class TwoUOneGGate(CompositeGate):

    def __init__(self, 
            U1: OneQubitGate,
            U2: OneQubitGate,     
            G12: TwoQubitGate, 
            U3: OneQubitGate,
            U4: OneQubitGate,
            distribution: str="uniform",
            dist_params: Tuple[int] = [0, 2*np.pi],
            to_backend: Optional[Callable]=None,
    ):
        super().__init__(distribution=distribution, dist_params=dist_params, to_backend=to_backend)

        self.U1 = U1
        U1.set_backend(self.to_backend)
        self.U2 = U2
        U2.set_backend(self.to_backend)
        self.G12 = G12
        G12.set_backend(self.to_backend)
        self.U3 = U3
        U3.set_backend(self.to_backend)
        self.U4 = U4
        U4.set_backend(self.to_backend)

        self._name = f"-{U1.name}-|{G12.name}|-{U3.name}-\n-{U2.name}-|{G12.name}|-{U4.name}-"

        self.set_num_params()

    def __call__(self, params: Iterable, inds=["a3", "b3", "a0", "b0"]):

        s = 0
        T1 = self.U1(params[s:s+self.U1.num_params], inds=["a1", inds[2]])
        s += self.U1.num_params
        T2 = self.U2(params[s:s+self.U2.num_params], inds=["b1", inds[3]])
        s += self.U2.num_params
        TG12 = self.G12(params[s:s+self.G12.num_params], inds=["a2", "b2", "a1", "b1"])
        s += self.G12.num_params
        T3 = self.U3(params[s:s+self.U3.num_params], inds=[inds[0], "a2"])
        s += self.U3.num_params
        T4 = self.U4(params[s:s+self.U4.num_params], inds=[inds[1], "b2"])

        return tensor_contract(
            T1,
            T2,
            TG12,
            T3,
            T4,
            output_inds=inds,
            optimize="auto-hq",
        ).data
        
    def apply(self, circuit: qtn.Circuit, qubits: Tuple[int], params: Optional[Iterable]=None, **gate_args):

        q0, q1 = qubits
        
        s = 0
        self.U1.apply(circuit, q0, params[s:s+self.U1.num_params], **gate_args)
        s += self.U1.num_params
        self.U2.apply(circuit, q1, params[s:s+self.U2.num_params], **gate_args)
        s += self.U2.num_params
        self.G12.apply(circuit, qubits, params[s:s+self.G12.num_params], **gate_args)
        s += self.G12.num_params
        self.U3.apply(circuit, q0, params[s:s+self.U3.num_params], **gate_args)
        s += self.U3.num_params
        self.U4.apply(circuit, q1, params[s:s+self.U4.num_params], **gate_args)
        
class TwoUTwoGGate(CompositeGate):

    def __init__(self, 
            U1: OneQubitGate,
            U2: OneQubitGate,     
            G12: TwoQubitGate,
            U3: OneQubitGate,
            U4: OneQubitGate,     
            G34: TwoQubitGate,    
            distribution: str="uniform",
            dist_params: Tuple[int] = [0, 2*np.pi],
            to_backend: Optional[Callable]=None,
    ):
        super().__init__(distribution=distribution, dist_params=dist_params, to_backend=to_backend)

        self.U1 = U1
        U1.set_backend(self.to_backend)
        self.U2 = U2
        U2.set_backend(self.to_backend)
        self.G12 = G12
        G12.set_backend(self.to_backend)
        self.U3 = U3
        U3.set_backend(self.to_backend)
        self.U4 = U4
        U4.set_backend(self.to_backend)
        self.G34 = G34
        G34.set_backend(self.to_backend)

        self._name = f"-{U1.name}-|{G12.name}|-{U3.name}-|{G34.name}|-\n-{U2.name}-|{G12.name}|-{U4.name}-|{G34.name}|-"
        
        self.set_num_params()
        

    def __call__(self, params: Iterable, inds=["a4", "b4", "a0", "b0"]):

        s = 0
        T1 = self.U1(params[s:s+self.U1.num_params], inds=["a1", inds[2]])
        s += self.U1.num_params
        T2 = self.U2(params[s:s+self.U2.num_params], inds=["b1", inds[3]])
        s += self.U2.num_params
        TG12 = self.G12(params[s:s+self.G12.num_params], inds=["a2", "b2", "a1", "b1"])
        s += self.G12.num_params
        T3 = self.U3(params[s:s+self.U3.num_params], inds=["a3", "a2"])
        s += self.U3.num_params
        T4 = self.U4(params[s:s+self.U4.num_params], inds=["b3", "b2"])
        s += self.U4.num_params
        TG34 = self.G34(params[s:s+self.G34.num_params], inds=[inds[0], inds[1], "a3", "b3"])

        return tensor_contract(
            T1,
            T2,
            TG12,
            T3,
            T4,
            TG34,
            output_inds=inds,
            optimize="auto-hq",
        ).data
        
        
    def apply(self, circuit: qtn.Circuit, qubits: Tuple[int], params: Optional[Iterable]=None, **gate_args):

        q0, q1 = qubits
        
        s = 0
        self.U1.apply(circuit, q0, params[s:s+self.U1.num_params], **gate_args)
        s += self.U1.num_params
        self.U2.apply(circuit, q1, params[s:s+self.U2.num_params], **gate_args)
        s += self.U2.num_params
        self.G12.apply(circuit, qubits, params[s:s+self.G12.num_params], **gate_args)
        s += self.G12.num_params
        self.U3.apply(circuit, q0, params[s:s+self.U3.num_params], **gate_args)
        s += self.U3.num_params
        self.U4.apply(circuit, q1, params[s:s+self.U4.num_params], **gate_args)
        s += self.U4.num_params
        self.G34.apply(circuit, qubits, params[s:s+self.G34.num_params], **gate_args)
        
class ThreeUTwoGGate(CompositeGate):

    def __init__(self, 
            U1: OneQubitGate,
            U2: OneQubitGate,     
            G12: TwoQubitGate,
            U3: OneQubitGate,
            U4: OneQubitGate,     
            G34: TwoQubitGate,
            U5: OneQubitGate,
            U6: OneQubitGate,        
            distribution: str="uniform",
            dist_params: Tuple[int] = [0, 2*np.pi],
            to_backend: Optional[Callable]=None,
    ):
        super().__init__(distribution=distribution, dist_params=dist_params, to_backend=to_backend)

        self.U1 = U1
        U1.set_backend(self.to_backend)
        self.U2 = U2
        U2.set_backend(self.to_backend)
        self.G12 = G12
        G12.set_backend(self.to_backend)
        self.U3 = U3
        U3.set_backend(self.to_backend)
        self.U4 = U4
        U4.set_backend(self.to_backend)
        self.G34 = G34
        G34.set_backend(self.to_backend)
        self.U5 = U5
        U5.set_backend(self.to_backend)
        self.U6 = U6
        U6.set_backend(self.to_backend)

        self._name = f"-{U1.name}-|{G12.name}|-{U3.name}-|{G34.name}|-{U5.name}-\n-{U2.name}-|{G12.name}|-{U4.name}-|{G34.name}|-{U6.name}-"
        
        self.set_num_params()
        
        
    def __call__(self, params: Iterable, inds=["a5", "b5", "a0", "b0"]):

        s = 0
        T1 = self.U1(params[s:s+self.U1.num_params], inds=["a1", inds[2]])
        s += self.U1.num_params
        T2 = self.U2(params[s:s+self.U2.num_params], inds=["b1", inds[3]])
        s += self.U2.num_params
        TG12 = self.G12(params[s:s+self.G12.num_params], inds=["a2", "b2", "a1", "b1"])
        s += self.G12.num_params
        T3 = self.U3(params[s:s+self.U3.num_params], inds=["a3", "a2"])
        s += self.U3.num_params
        T4 = self.U4(params[s:s+self.U4.num_params], inds=["b4", "b2"])
        s += self.U4.num_params
        TG34 = self.G34(params[s:s+self.G34.num_params], inds=["a4", "b4", "a3", "b3"])
        s += self.G34.num_params
        T5 = self.U5(params[s:s+self.U5.num_params], inds=[inds[0], "a4"])
        s += self.U5.num_params
        T6 = self.U6(params[s:s+self.U6.num_params], inds=[inds[1], "b4"])

        return tensor_contract(
            T1,
            T2,
            TG12,
            T3,
            T4,
            TG34,
            T5,
            T6,
            output_inds=inds,
            optimize="auto-hq",
        ).data
        
        
    def apply(self, circuit: qtn.Circuit, qubits: Tuple[int], params: Optional[Iterable]=None, **gate_args):

        q0, q1 = qubits
        
        s = 0
        self.U1.apply(circuit, q0, params[s:s+self.U1.num_params], **gate_args)
        s += self.U1.num_params
        self.U2.apply(circuit, q1, params[s:s+self.U2.num_params], **gate_args)
        s += self.U2.num_params
        self.G12.apply(circuit, qubits, params[s:s+self.G12.num_params], **gate_args)
        s += self.G12.num_params
        self.U3.apply(circuit, q0, params[s:s+self.U3.num_params], **gate_args)
        s += self.U3.num_params
        self.U4.apply(circuit, q1, params[s:s+self.U4.num_params], **gate_args)
        s += self.U4.num_params
        self.G34.apply(circuit, qubits, params[s:s+self.G34.num_params], **gate_args)
        s += self.G34.num_params
        self.U5.apply(circuit, q0, params[s:s+self.U5.num_params], **gate_args)
        s += self.U5.num_params
        self.U6.apply(circuit, q1, params[s:s+self.U6.num_params], **gate_args)
        
class ThreeUThreeGGate(CompositeGate):

    def __init__(self, 
            U1: OneQubitGate,
            U2: OneQubitGate,     
            G12: TwoQubitGate,
            U3: OneQubitGate,
            U4: OneQubitGate,     
            G34: TwoQubitGate,
            U5: OneQubitGate,
            U6: OneQubitGate,     
            G56: TwoQubitGate, 
            distribution: str="uniform",
            dist_params: Tuple[int] = [0, 2*np.pi],
            to_backend: Optional[Callable]=None,
    ):
        super().__init__(distribution=distribution, dist_params=dist_params, to_backend=to_backend)

        self.U1 = U1
        U1.set_backend(self.to_backend)
        self.U2 = U2
        U2.set_backend(self.to_backend)
        self.G12 = G12
        G12.set_backend(self.to_backend)
        self.U3 = U3
        U3.set_backend(self.to_backend)
        self.U4 = U4
        U4.set_backend(self.to_backend)
        self.G34 = G34
        G34.set_backend(self.to_backend)
        self.U5 = U5
        U5.set_backend(self.to_backend)
        self.U6 = U6
        U6.set_backend(self.to_backend)
        self.G56 = G56
        G56.set_backend(self.to_backend)

        self._name = f"-{U1.name}-|{G12.name}|-{U3.name}-|{G34.name}|-{U5.name}-|{G56.name}|-\n-{U2.name}-|{G12.name}|-{U4.name}-|{G34.name}|-{U6.name}-|{G56.name}|-"

        self.set_num_params()
        
    def __call__(self, params: Iterable, inds=["a6", "b6", "a0", "b0"]):

        s = 0
        T1 = self.U1(params[s:s+self.U1.num_params], inds=["a1", inds[2]])
        s += self.U1.num_params
        T2 = self.U2(params[s:s+self.U2.num_params], inds=["b1", inds[3]])
        s += self.U2.num_params
        TG12 = self.G12(params[s:s+self.G12.num_params], inds=["a2", "b2", "a1", "b1"])
        s += self.G12.num_params
        T3 = self.U3(params[s:s+self.U3.num_params], inds=["a3", "a2"])
        s += self.U3.num_params
        T4 = self.U4(params[s:s+self.U4.num_params], inds=["b4", "b2"])
        s += self.U4.num_params
        TG34 = self.G34(params[s:s+self.G34.num_params], inds=["a4", "b4", "a3", "b3"])
        s += self.G34.num_params
        T5 = self.U5(params[s:s+self.U5.num_params], inds=["a5", "a4"])
        s += self.U5.num_params
        T6 = self.U6(params[s:s+self.U6.num_params], inds=["b5", "b4"])
        s += self.U6.num_params
        TG56 = self.G56(params[s:s+self.G56.num_params], inds=[inds[0], inds[1], "a5", "b5"])

        return tensor_contract(
            T1,
            T2,
            TG12,
            T3,
            T4,
            TG34,
            T5,
            T6,
            TG56,
            output_inds=inds,
            optimize="auto-hq",
        ).data
        
        
    def apply(self, circuit: qtn.Circuit, qubits: Tuple[int], params: Optional[Iterable]=None, **gate_args):

        q0, q1 = qubits
        
        s = 0
        self.U1.apply(circuit, q0, params[s:s+self.U1.num_params], **gate_args)
        s += self.U1.num_params
        self.U2.apply(circuit, q1, params[s:s+self.U2.num_params], **gate_args)
        s += self.U2.num_params
        self.G12.apply(circuit, qubits, params[s:s+self.G12.num_params], **gate_args)
        s += self.G12.num_params
        self.U3.apply(circuit, q0, params[s:s+self.U3.num_params], **gate_args)
        s += self.U3.num_params
        self.U4.apply(circuit, q1, params[s:s+self.U4.num_params], **gate_args)
        s += self.U4.num_params
        self.G34.apply(circuit, qubits, params[s:s+self.G34.num_params], **gate_args)
        s += self.G34.num_params
        self.U5.apply(circuit, q0, params[s:s+self.U5.num_params], **gate_args)
        s += self.U5.num_params
        self.U6.apply(circuit, q1, params[s:s+self.U6.num_params], **gate_args)
        s += self.U6.num_params
        self.G56.apply(circuit, qubits, params[s:s+self.G56.num_params], **gate_args)
        
class FourUThreeGGate(CompositeGate):

    def __init__(self, 
            U1: OneQubitGate,
            U2: OneQubitGate,     
            G12: TwoQubitGate,
            U3: OneQubitGate,
            U4: OneQubitGate,     
            G34: TwoQubitGate,
            U5: OneQubitGate,
            U6: OneQubitGate,     
            G56: TwoQubitGate,
            U7: OneQubitGate,
            U8: OneQubitGate,     
            distribution: str="uniform",
            dist_params: Tuple[int] = [0, 2*np.pi],
            to_backend: Optional[Callable]=None,
    ):
        super().__init__(distribution=distribution, dist_params=dist_params, to_backend=to_backend)

        self.U1 = U1
        U1.set_backend(self.to_backend)
        self.U2 = U2
        U2.set_backend(self.to_backend)
        self.G12 = G12
        G12.set_backend(self.to_backend)
        self.U3 = U3
        U3.set_backend(self.to_backend)
        self.U4 = U4
        U4.set_backend(self.to_backend)
        self.G34 = G34
        G34.set_backend(self.to_backend)
        self.U5 = U5
        U5.set_backend(self.to_backend)
        self.U6 = U6
        U6.set_backend(self.to_backend)
        self.G56 = G56
        G56.set_backend(self.to_backend)
        self.U7 = U7
        U7.set_backend(self.to_backend)
        self.U8 = U8
        U8.set_backend(self.to_backend)

        self._name = f"-{U1.name}-|{G12.name}|-{U3.name}-|{G34.name}|-{U5.name}-|{G56.name}|-{U7.name}-\n-{U2.name}-|{G12.name}|-{U4.name}-|{G34.name}|-{U6.name}-|{G56.name}|-{U8.name}-"
        
        self.set_num_params()

    def __call__(self, params: Iterable, inds=["a7", "b7", "a0", "b0"]):

        s = 0
        T1 = self.U1(params[s:s+self.U1.num_params], inds=["a1", inds[2]])
        s += self.U1.num_params
        T2 = self.U2(params[s:s+self.U2.num_params], inds=["b1", inds[3]])
        s += self.U2.num_params
        TG12 = self.G12(params[s:s+self.G12.num_params], inds=["a2", "b2", "a1", "b1"])
        s += self.G12.num_params
        T3 = self.U3(params[s:s+self.U3.num_params], inds=["a3", "a2"])
        s += self.U3.num_params
        T4 = self.U4(params[s:s+self.U4.num_params], inds=["b4", "b2"])
        s += self.U4.num_params
        TG34 = self.G34(params[s:s+self.G34.num_params], inds=["a4", "b4", "a3", "b3"])
        s += self.G34.num_params
        T5 = self.U5(params[s:s+self.U5.num_params], inds=["a5", "a4"])
        s += self.U5.num_params
        T6 = self.U6(params[s:s+self.U6.num_params], inds=["b5", "b4"])
        s += self.U6.num_params
        TG56 = self.G56(params[s:s+self.G56.num_params], inds=["a6", "b6", "a5", "b5"])
        s += self.G56.num_params
        T7 = self.U7(params[s:s+self.U7.num_params], inds=[inds[0], "a6"])
        s += self.U7.num_params
        T8 = self.U8(params[s:s+self.U8.num_params], inds=[inds[1], "b6"])

        return tensor_contract(
            T1,
            T2,
            TG12,
            T3,
            T4,
            TG34,
            T5,
            T6,
            TG56,
            T7,
            T8,
            output_inds=inds,
            optimize="auto-hq",
        ).data
        
    def apply(self, circuit: qtn.Circuit, qubits: Tuple[int], params: Optional[Iterable]=None, **gate_args):

        q0, q1 = qubits
        
        s = 0
        self.U1.apply(circuit, q0, params[s:s+self.U1.num_params], **gate_args)
        s += self.U1.num_params
        self.U2.apply(circuit, q1, params[s:s+self.U2.num_params], **gate_args)
        s += self.U2.num_params
        self.G12.apply(circuit, qubits, params[s:s+self.G12.num_params], **gate_args)
        s += self.G12.num_params
        self.U3.apply(circuit, q0, params[s:s+self.U3.num_params], **gate_args)
        s += self.U3.num_params
        self.U4.apply(circuit, q1, params[s:s+self.U4.num_params], **gate_args)
        s += self.U4.num_params
        self.G34.apply(circuit, qubits, params[s:s+self.G34.num_params], **gate_args)
        s += self.G34.num_params
        self.U5.apply(circuit, q0, params[s:s+self.U5.num_params], **gate_args)
        s += self.U5.num_params
        self.U6.apply(circuit, q1, params[s:s+self.U6.num_params], **gate_args)
        s += self.U6.num_params
        self.G56.apply(circuit, qubits, params[s:s+self.G56.num_params], **gate_args)
        s += self.G56.num_params
        self.U7.apply(circuit, q0, params[s:s+self.U7.num_params], **gate_args)
        s += self.U7.num_params
        self.U8.apply(circuit, q1, params[s:s+self.U8.num_params], **gate_args)
        s += self.U8.num_params