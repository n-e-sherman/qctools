import seaborn as sns
import colorsys
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

from typing import Set, Tuple, Dict, Iterable, List, Optional
from .brickwall import BrickwallMosaic

class ConstrainedBrickwallMosaic(BrickwallMosaic):

    def __init__(self, N, T, max_width=None, max_depth=None, shift=0, pbc=False, N_initial=1, min_patch_size=1, max_patch_size=4, unassigned_id: int=-1, missing_id: int=-2):
        
        super().__init__(N, T, shift=shift, pbc=pbc, N_initial=N_initial, min_patch_size=min_patch_size, max_patch_size=max_patch_size, unassigned_id=unassigned_id, missing_id=missing_id)
        
        self.max_width=max_width
        self.max_depth=max_depth
    
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
            valid = True
            for pid in wire_patch_seq1:
                ts = np.argwhere(wires[q1] == pid).ravel()
                if np.any(np.diff(ts) > 1):
                    valid = False
                    break
            
            # check geometric constraint on q2
            for pid in wire_patch_seq2:
                ts = np.argwhere(wires[q2] == pid).ravel()
                if np.any(np.diff(ts) > 1):
                    valid = False
                    break
            
            # check width
            patch = self.patches[patch_id]
            bonds = [g[:2] for g in patch]
            bonds.append(brick[:2])
            qs = np.unique(np.array(bonds).ravel())
            if self.max_width and (len(qs) > self.max_width):
                valid = False
                break

            # check depth
            ts = np.unique([g[-1] for g in patch] + [brick[-1]])
            if self.max_depth and (len(ts) > self.max_width):
                valid = False
                break

            if valid:
                valid_bricks.append(brick)

        return valid_bricks

