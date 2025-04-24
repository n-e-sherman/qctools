import warnings
import quimb.tensor as qtn

from quimb.tensor.optimize import TNOptimizer
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable, Dict, Any

from qctools.utils import reorder_gates, early_stopping_callback, EarlyStopException, TorchConverter, get_gate_tuples, TraceFidelityTNLoss


class DeformationSweeper(ABC):

    def __init__(self, attempts: int=10, to_backend: Callable=TorchConverter(), 
                 epochs: int=1000, rel_tol: float=1e-8, tol: float=1e-3, progbar: bool=True,
                 early_stopping: bool=True,
                 **opt_args
    ):

        self.attempts = 10
        self.to_backend = to_backend if to_backend else lambda x: x
        self.epochs = epochs
        self.rel_tol = rel_tol
        self.tol = tol
        self.progbar = progbar
        self.opt_args = opt_args
        self.early_stopping = early_stopping

    def sweep(self, qc: qtn.Circuit) -> qtn.Circuit:

        qc_full = self.initialize_qc(qc)

        # NOTE: introduce a "rounds" argument to do repeated sweeps?
        sweep_schedule = self.get_schedule(qc_full)
        for i, patch_args in enumerate(sweep_schedule):
            msg = f"patch: {i+1}/{len(sweep_schedule)}"
            qc_target = self.get_patch_target(qc_full, **patch_args.get('patch_target', {}))
            qc_pre, qc_post = self.get_pre_post_circuits(qc_target, qc_full)
            for attempt in range(self.attempts):
                qc_train = self.get_patch_train(qc_target, **patch_args.get('patch_train', {}))
                qc_opt = self.train_patch(qc_train, qc_target, msg=msg, **patch_args.get('train', {}))
                if qc_opt:
                    break
            if not qc_opt:
                raise RuntimeError(f"patch training did not converge at {msg}")
            qc_full = self.combine_qc(qc_pre, qc_opt, qc_post)

        return qc_full

    @abstractmethod
    def get_schedule(self, qc_full: qtn.Circuit, *args, **kwargs):
        """Derived classes must define the schedule for sweeping."""
        pass

    @abstractmethod
    def get_patch_target(self, qc: qtn.Circuit, *args, **kwargs) -> qtn.Circuit:
        """Derived classes must define how to get a patch target."""
        pass

    @abstractmethod
    def get_patch_train(self, qc_target: qtn.Circuit, *args, **kwargs) -> qtn.Circuit:
        """Derived classes must define how to get a patch train."""
        pass

    def train_patch(self, qc_train: qtn.Circuit, qc_target: qtn.Circuit, msg: str='', **kwargs):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings only
            loss = self.get_loss(qc_target)
            opt = TNOptimizer(
                qc_train,
                loss,
                autodiff_backend='torch',
                progbar=self.progbar,
                callback=lambda opt: self._update_progbar_callback(opt, msg, self.tol),
                **self.opt_args
            )

            try:
                qc_opt = opt.optimize(self.epochs, tol=self.rel_tol)
            except EarlyStopException:
                qc_opt = opt.get_tn_opt()  # Return the last optimized quantum circuit state
            except:
                raise Exception
            if opt.loss > self.tol:
                return False
        qc_opt.apply_to_arrays(self.to_backend)
        return qc_opt
    
    def get_loss(self, qc_target: qtn.Circuit) -> Callable:

        return TraceFidelityTNLoss(qc_target.get_uni())

    def initialize_qc(self, qc: qtn.Circuit) -> qtn.Circuit:

        _qc = qtn.Circuit(qc.N)
        for gate in qc.gates:
            _qc.apply_gate(gate.label, params=self.to_backend(gate.params), qubits=gate.qubits, parametrize=False)
        qc_full = reorder_gates(_qc)
        qc_full.apply_to_arrays(self.to_backend)
        return qc_full

    @classmethod
    def combine_qc(cls, qc_pre: qtn.Circuit, qc_opt: qtn.Circuit, qc_post: qtn.Circuit, parametrize: bool=False) -> qtn.Circuit:

        N = max(qc_pre.N, qc_opt.N, qc_post.N)
        qc_full = qtn.Circuit(N)
        for gate in qc_pre.gates + qc_opt.gates + qc_post.gates:
            qc_full.apply_gate(gate.label, params=gate.params, qubits=gate.qubits, parametrize=parametrize)
        return reorder_gates(qc_full)

    @classmethod
    def get_pre_post_circuits(cls, qc_patch: qtn.Circuit, qc_full: qtn.Circuit) -> Tuple[qtn.Circuit]:

        sub_gates = get_gate_tuples(qc_patch)
        qc_pre = qtn.Circuit(qc_full.N)
        qc_post = qtn.Circuit(qc_full.N)

        pre = True
        for gate in qc_full.gates:
            _gate = (*gate.qubits, gate.round)
            if _gate in sub_gates:
                pre = False
            else:
                if pre:
                    qc_pre.apply_gate(gate)
                else:
                    qc_post.apply_gate(gate)
        return qc_pre, qc_post
        
    # @classmethod
    # def get_gate_tuples(cls, qc: qtn.Circuit) -> List[Tuple[int]]:

    #     gates = []
    #     for gate in qc.gates:
    #         gates.append((*gate.qubits, gate.round))
    #     return gates

    def _update_progbar_callback(self, opt, postfix_str: str, loss_threshold: float):
        if hasattr(opt, '_pbar') and opt._pbar is not None:
            opt._pbar.set_postfix_str(postfix_str)  # Directly set the postfix string
        if self.early_stopping:
            early_stopping_callback(opt, loss_threshold)
        return False  # Returning False means "continue optimization"