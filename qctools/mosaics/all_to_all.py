import seaborn as sns
import colorsys
import copy
import inspect
import random
import numpy as np
import matplotlib.pyplot as plt

from typing import Set, Tuple, Dict, Iterable, List, Optional

from ._base import Mosaic

class AllToAllMosaic(Mosaic):

    def __init__(self, N, T, pbc=False, N_initial=1, min_patch_size=1, max_patch_size=4, unassigned_id: int=-1, missing_id: int=-2):
        
        super().__init__(N_initial=N_initial, min_patch_size=min_patch_size, max_patch_size=max_patch_size, unassigned_id=unassigned_id, missing_id=missing_id)
        

        self.N = N
        self.T = T
        self.pbc=pbc

        frame = inspect.currentframe()
        args, vargs, varkw, locals = inspect.getargvalues(frame)
        skip_kwargs = ['self', 'frame']
        self._kwargs = {key: locals[key] for key in locals if key not in skip_kwargs}

    
    def validate_brick_proposals(self, bricks: Iterable[Tuple[int]], patch_id: int) -> List[Tuple[int]]:

        # NOTE: This is specific to brickwall and IQC-RQC mosaic
        valid_bricks = []
        for brick in bricks:
            q1, q2, t = brick
            wires = copy.deepcopy(self.wires)
            wires[q1, t] = patch_id
            wires[q2, t] = patch_id
            wire_patch_seq1 = [x for x in np.unique(wires[q1]) if not x == self.missing_id]
            wire_patch_seq2 = [x for x in np.unique(wires[q2]) if not x == self.missing_id]

            # check geometric constraint on q1
            valid1 = True
            for pid in wire_patch_seq1:
                ts = np.argwhere(wires[q1] == pid).ravel()
                if np.any(np.diff(ts) > 1):
                    valid1 = False
                    break
            
            # check geometric constraint on q2
            valid2 = True
            for pid in wire_patch_seq2:
                ts = np.argwhere(wires[q2] == pid).ravel()
                if np.any(np.diff(ts) > 1):
                    valid2 = False
                    break

            if valid1 and valid2:
                valid_bricks.append(brick)

        return valid_bricks

    def _get_brick_proposals(self, patch: Dict[int, Tuple[int]]) -> Set[Tuple[int]]:

        
        brick_proposals = []
        for gate in patch:
            q0, q1, t = gate
            brick_proposals += [gate for gate in self.unassigned_bricks if gate[-1] == t]
            brick_proposals += [gate for gate in self.unassigned_bricks if (gate[0] == q0) and (gate[1] != q1) and (abs(gate[-1] - t) == 1)]
            brick_proposals += [gate for gate in self.unassigned_bricks if (gate[1] == q1) and (gate[0] != q0) and (abs(gate[-1] - t) == 1)]
        return set(brick_proposals)


    def _initialize_wires_and_bricks(self):

        self.all_bricks = []
        self.wires = np.zeros((self.N, self.T)) + self.missing_id
        for t in range(self.T):
            _qubits = list(range(self.N))
            np.random.shuffle(_qubits)
            np.random.shuffle(_qubits)
            for i in range(0, len(_qubits) - 1, 2):
                qubits = [_qubits[i], _qubits[i+1]]
                qubits.sort()
                self.wires[qubits[0], t] = self.unassigned_id
                self.wires[qubits[1], t] = self.unassigned_id
                self.all_bricks.append((*qubits, t))

        self.unassigned_bricks = set(copy.deepcopy(self.all_bricks))
        self.patches = {}
        self.patch_sizes = {}
    