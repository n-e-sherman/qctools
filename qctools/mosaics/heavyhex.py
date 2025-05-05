import copy
import inspect
import random
import numpy as np

from typing import Set, Tuple, Dict, Iterable, List, Optional

from ._base import Mosaic

class HeavyHexMosaic(Mosaic):

    _valid_coverings = ['2', '3']
    def __init__(self, G, T,
                 seed=None, 
                 covering='2', max_trials=1000,
                 max_width=None, max_depth=None,
                 N_initial=1, min_patch_size=10, max_patch_size=20, unassigned_id: int=-1, missing_id: int=-2):
        
        super().__init__(N_initial=N_initial, min_patch_size=min_patch_size, max_patch_size=max_patch_size, unassigned_id=unassigned_id, missing_id=missing_id)

        if covering == '2':
            self.generate_covering = lambda: self.generate_2layer_covering(G, seed=seed, max_trials=max_trials)
            self.T_coverings = T // 2
            self.T_final_covering = T % 2
        elif covering == '3':
            self.generate_covering = lambda: self.generate_3layer_covering(G, seed=seed, max_trials=max_trials)
            self.T_coverings = T // 3
            self.T_final_covering = T % 3
        else:
            raise RuntimeError(f"covering type: {covering} is not a valid choice.")
        
        self.G = G
        self.N = G.order()
        self.T = T # T is total number of layers, may need a partial covering to match

        self.max_width=max_width
        self.max_depth=max_depth

        frame = inspect.currentframe()
        args, vargs, varkw, locals = inspect.getargvalues(frame)
        skip_kwargs = ['self', 'frame', 'G']
        self._kwargs = {key: locals[key] for key in locals if key not in skip_kwargs}

    def _initialize_wires_and_bricks(self):

        self.all_bricks = []
        self.wires = np.zeros((self.N, self.T)) + self.missing_id
        self.coverings = []

        # All full coverings
        t = 0
        for t_covering in range(self.T_coverings):
            layers = self.generate_covering()
            self.coverings.append(layers)
            for layer in layers:
                for _qubits in layer:
                    qubits = list(_qubits)
                    qubits.sort()
                    self.wires[qubits[0], t] = self.unassigned_id
                    self.wires[qubits[1], t] = self.unassigned_id
                    self.all_bricks.append((*qubits, t))
                t += 1
        
        # final covering
        self.final_covering = None
        if not self.T_final_covering == 0:
            layers = self.generate_covering()
            self.final_covering = layers
            for _t in range(self.T_final_covering):
                layer = layers[_t]
                for _qubits in layer:
                    qubits = list(_qubits)
                    qubits.sort()
                    self.wires[qubits[0], t] = self.unassigned_id
                    self.wires[qubits[1], t] = self.unassigned_id
                    self.all_bricks.append((*qubits, t))
                t += 1

        self.unassigned_bricks = set(copy.deepcopy(self.all_bricks))
        self.patches = {}
        self.patch_sizes = {}

    def validate_brick_proposals(self, bricks: Iterable[Tuple[int]], patch_id: int) -> List[Tuple[int]]:

        # NOTE: This is specific to echo mosaic
        valid_bricks = []
        for brick in bricks:
            q1, q2, t = brick
            wires = copy.deepcopy(self.wires)
            wires[q1, t] = patch_id
            wires[q2, t] = patch_id
            # wire_patch_seq1 = [x for x in np.unique(wires[q1]) if not x == self.missing_id]
            # wire_patch_seq2 = [x for x in np.unique(wires[q2]) if not x == self.missing_id]
            wire_no_missing1 = [x for x in wires[q1] if not x == self.missing_id]
            wire_no_missing2 = [x for x in wires[q2] if not x == self.missing_id]

            # check geometric constraint on q1
            valid = True
            for pid in np.unique(wire_no_missing1):
                ts = np.argwhere(wire_no_missing1 == pid).ravel()
                if np.any(np.diff(ts) > 1):
                    valid = False
                    break
            
            # check geometric constraint on q2
            for pid in np.unique(wire_no_missing2):
                ts = np.argwhere(wire_no_missing2 == pid).ravel()
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

            # check depth
            ts = np.unique([g[-1] for g in patch] + [brick[-1]])
            if self.max_depth and (len(ts) > self.max_depth):
                valid = False

            if valid:
                valid_bricks.append(brick)

        return valid_bricks

    def _get_brick_proposals(self, patch: Dict[int, Tuple[int]]) -> Set[Tuple[int]]:

        brick_proposals = []
        for gate in patch:
            qubits = gate[:-1]
            t = gate[-1]

            # t' = t \pm 1 bricks
            brick_proposal = [gate for gate in self.unassigned_bricks if ((gate[0] in qubits) or (gate[1] in qubits)) and (abs(gate[-1] - t) == 1)] # t \pm 1 gates
            qs = self.get_unique_qs(brick_proposal)

            # t' = t bricks
            for q1 in qs:
                for q2 in self.G.neighbors(q1):
                    qubits = [q1, q2]
                    qubits.sort()
                    brick = (*qubits, t)
                    brick_proposal.append(brick)
            brick_proposals += brick_proposal

        return set(brick_proposals)
    
    def _select_starter_bricks(self, starter_bricks, N):
        
        # prefer brick starts with higher degree
        zs = list(set([len(list(self.G.neighbors(node))) for node in self.G.nodes]))
        for z in zs[::-1]:
            qs = self.get_unique_qs(starter_bricks)
            qs_z = [q for q in qs if len(list(self.G.neighbors(q))) == z]
            preferred_bricks = [brick for brick in starter_bricks if (brick[0] in qs_z) or (brick[1] in qs_z)]
            if len(preferred_bricks) > 0:
                # prefer lower t
                t_min = min([brick[-1] for brick in preferred_bricks])
                final_bricks = [brick for brick in preferred_bricks if brick[-1] == t_min]
                return random.sample(final_bricks, min(N, len(final_bricks)))

        return random.sample(starter_bricks, min(N, len(starter_bricks)))
    
    def get_unique_qs(self, bricks):

        qs = set()
        for brick in bricks:
            qs.update(brick[:-1])
        return qs

    @classmethod
    def generate_2layer_covering(cls, G, seed=None, max_trials=1000):
        if seed is not None:
            random.seed(seed)

        degrees = dict(G.degree)
        num_nodes = len(G.nodes)

        best_layer1, best_layer2 = set(), set()
        best_bond_count = 0

        for trial in range(max_trials):
            layer1, layer2 = set(), set()
            used1, used2 = set(), set()
            assigned_edges = set()

            # Step 1: randomly assign spoke bonds
            for node in G.nodes:
                if degrees[node] == 1:
                    neighbor = next(G.neighbors(node))
                    edge = tuple(sorted((node, neighbor)))
                    if edge in assigned_edges:
                        continue
                    if random.random() < 0.5:
                        layer1.add(edge)
                        used1.update(edge)
                    else:
                        layer2.add(edge)
                        used2.update(edge)
                    assigned_edges.add(edge)

            # Step 2: prioritize z=3 nodes with fewest available edges first
            remaining_edges = [e for e in G.edges if tuple(sorted(e)) not in assigned_edges]
            edge_priority = []

            for edge in remaining_edges:
                u, v = edge
                score = degrees[u] + degrees[v]  # Prefer lower degree nodes first
                edge_priority.append((score, edge))

            edge_priority.sort()  # Lowest score = highest priority
            remaining_edges = [e for _, e in edge_priority]
            random.shuffle(remaining_edges)  # Add some randomness still

            for u, v in remaining_edges:
                edge = tuple(sorted((u, v)))
                if edge in assigned_edges:
                    continue
                if u not in used1 and v not in used1:
                    layer1.add(edge)
                    used1.update(edge)
                    assigned_edges.add(edge)
                elif u not in used2 and v not in used2:
                    layer2.add(edge)
                    used2.update(edge)
                    assigned_edges.add(edge)

            # Evaluate total bonds placed
            total_bonds = len(layer1) + len(layer2)

            # Keep best covering found
            if total_bonds > best_bond_count:
                covered_nodes = used1.union(used2)
                if len(covered_nodes) == num_nodes:
                    best_layer1 = layer1.copy()
                    best_layer2 = layer2.copy()
                    best_bond_count = total_bonds

        if best_bond_count == 0:
            raise RuntimeError(f"Failed to cover all qubits after {max_trials} trials.")

        return best_layer1, best_layer2


    # @classmethod
    # def generate_2layer_covering(cls, G, seed=None, max_trials=1000):
    #     if seed is not None:
    #         random.seed(seed)

    #     degrees = dict(G.degree)
    #     num_nodes = len(G.nodes)

    #     for trial in range(max_trials):
    #         layer1, layer2 = set(), set()
    #         used1, used2 = set(), set()
    #         assigned_edges = set()

    #         # Step 1: randomly assign spoke bonds
    #         for node in G.nodes:
    #             if degrees[node] == 1:
    #                 neighbor = next(G.neighbors(node))
    #                 edge = tuple(sorted((node, neighbor)))
    #                 if edge in assigned_edges:
    #                     continue
    #                 if random.random() < 0.5:
    #                     layer1.add(edge)
    #                     used1.update(edge)
    #                 else:
    #                     layer2.add(edge)
    #                     used2.update(edge)
    #                 assigned_edges.add(edge)

    #         # Step 2: process remaining edges (including z=3) in random order
    #         remaining_edges = [e for e in G.edges if tuple(sorted(e)) not in assigned_edges]
    #         random.shuffle(remaining_edges)

    #         for u, v in remaining_edges:
    #             edge = tuple(sorted((u, v)))
    #             if edge in assigned_edges:
    #                 continue
    #             if u not in used1 and v not in used1:
    #                 layer1.add(edge)
    #                 used1.update(edge)
    #                 assigned_edges.add(edge)
    #             elif u not in used2 and v not in used2:
    #                 layer2.add(edge)
    #                 used2.update(edge)
    #                 assigned_edges.add(edge)
    #             # If neither is assignable, skip it

    #         covered_nodes = used1.union(used2)
    #         if len(covered_nodes) == num_nodes:
    #             return layer1, layer2

    #     raise RuntimeError(f"Failed to cover all qubits after {max_trials} trials.")
    
    @classmethod
    def generate_3layer_covering(cls, G, seed=None, max_trials=1000):
        if seed is not None:
            random.seed(seed)

        for trial in range(max_trials):
            layers = [set(), set(), set()]
            used_nodes = [set(), set(), set()]
            edges = list(G.edges)
            random.shuffle(edges)

            success = True
            for u, v in edges:
                edge = tuple(sorted((u, v)))
                options = [0, 1, 2]
                random.shuffle(options)
                placed = False

                for i in options:
                    if u not in used_nodes[i] and v not in used_nodes[i]:
                        layers[i].add(edge)
                        used_nodes[i].add(u)
                        used_nodes[i].add(v)
                        placed = True
                        break

                if not placed:
                    success = False
                    break

            if success:
                return layers

        raise RuntimeError(f"Failed to find full 3-layer covering after {max_trials} trials.")

