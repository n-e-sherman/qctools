from abc import ABC, abstractmethod
import inspect
import seaborn as sns
import colorsys
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

from typing import Set, Tuple, Dict, Iterable, List, Optional

class Mosaic(ABC):

    def __init__(self, N_initial=1, min_patch_size=1, max_patch_size=4, unassigned_id: int=-1, missing_id: int=-2):

        assert min_patch_size <= max_patch_size, f"min_patch_size: {min_patch_size} is not less than max_patch_size: {max_patch_size}"
        assert unassigned_id < 0, f"unassigned_id: {unassigned_id} is invalid, must be negative"
        assert missing_id < 0, f"unassigned_id: {missing_id} is invalid, must be negative"
        assert missing_id != unassigned_id, f"missing_id and unassigned_id must be different, but both were set as {unassigned_id}"

        self.N_initial=N_initial
        self.min_patch_size=min_patch_size
        self.max_patch_size=max_patch_size
        self.unassigned_id=unassigned_id
        self.missing_id=missing_id
        self.patches = {}

        frame = inspect.currentframe()
        args, vargs, varkw, locals = inspect.getargvalues(frame)
        skip_kwargs = ['self', 'frame']
        self._kwargs = {key: locals[key] for key in locals if key not in skip_kwargs}

    def get_log_params(self, tag: Optional[str]=None):
        
        if tag:
            tag += '-'
        else:
            tag = ''
        return {f"{tag}{k}": v for k,v in self._kwargs.items()}
        
    def build(self, N_patches: int=5, max_no_growth=5, max_steps: int=1000):

        # initialize
        self._initialize_wires_and_bricks()
        self.add_patches(self.N_initial)
        self.grow_patches()

        # grow patches 
        step = 0
        no_growth_steps = 0
        while len(self.unassigned_bricks) > 0:
            N_patches = self.add_patches(N_patches)

            # check if any new patches were made
            if N_patches == 0:
                print(f'no new starter bricks, stopping at step: {step}')
                break
            

            # check growth
            growth = self.grow_patches()
            if not growth:
                no_growth_steps += 1
            else:
                no_growth_steps = 0
            if no_growth_steps >= max_no_growth:
                print(f'no growth for {no_growth_steps} steps at step: {step}')
                break

            # check steps
            step += 1
            if step >= max_steps:
                print(f'max_steps reached at step: {step}')
                break


    def add_patches(self, N):

        patch_ids = [i+self.N_patches for i in range(N)]
        if np.any([patch_id in self.patches for patch_id in patch_ids]):
            raise RuntimeError("trying to add a patch where a prexisting patch already exists")
        
        starter_bricks = self._get_patch_starter_bricks(N)
            
        for patch_id, brick in zip(patch_ids, starter_bricks):
            self.patches[patch_id] = []
            self.patch_sizes[patch_id] = random.sample(range(self.min_patch_size, self.max_patch_size+1), 1)[0]
            self.add_brick_to_patch(brick, patch_id)

        return len(starter_bricks)

    def add_brick_to_patch(self, brick: Tuple[int], patch_id: int):

        self.patches[patch_id].append(brick)
        self.unassigned_bricks.remove(brick)

        q1, q2, t = brick
        self.wires[q1, t] = patch_id
        self.wires[q2, t] = patch_id

    def grow_patches(self, max_steps: int=1000) -> bool:
        
        step = 0
        any_growth = False
        while True:
            growth = self._grow_patches_step()
            if not any_growth and growth:
                any_growth = True
            step += 1
            if not growth:
                break
            if step >= max_steps:
                break
        return any_growth

    def _grow_patches_step(self) -> bool:

        growth = False
        for patch_id, patch in self.patches.items():
            if len(patch) >= self.patch_sizes[patch_id]: # patch is fully grown
                continue

            brick_proposals = self._get_brick_proposals(patch)
            
            available_proposed_bricks = brick_proposals.intersection(self.unassigned_bricks)
            if len(available_proposed_bricks) == 0: # patch can not grow further
                continue

            # check constraint on brick_options
            brick_options = self.validate_brick_proposals(available_proposed_bricks, patch_id)
            if len(brick_options) == 0:
                continue

            # can add a brick
            new_brick = random.sample(brick_options, 1)[0]
            self.add_brick_to_patch(new_brick, patch_id)
            growth = True
        
        return growth
    
    def get_patches(self):
        return self.patches

    @property
    def N_patches(self):
        return len(self.patches)
    
    @abstractmethod
    def validate_brick_proposals(self, bricks: Iterable[Tuple[int]], patch_id: int) -> List[Tuple[int]]:
        """Derived classes must define how to validate brick proposals based on currently assigned patches."""
        pass

    @abstractmethod
    def _get_brick_proposals(self, patch: Dict[int, Tuple[int]]) -> Set[Tuple[int]]:
        """Derived classes must define proposed bricks by possible bricks for a patch that are consistent with the full mosaic design."""
        pass

    @abstractmethod
    def _initialize_wires_and_bricks(self):
        """define the wires and the set of all bricks for the mosaic design."""
        pass

    def _get_patch_starter_bricks(self, N):

        all_starter_bricks = self._get_all_patch_starter_bricks()
        starter_bricks = self._select_starter_bricks(all_starter_bricks, N)
        return starter_bricks
    
    def _select_starter_bricks(self, starter_bricks, N):

        return random.sample(starter_bricks, min(N, len(starter_bricks)))
    
    def _get_all_patch_starter_bricks(self):

        t_boundaries = self._get_t_boundaries()
        starter_bricks = []
        for brick in self.unassigned_bricks:
            q1, q2, t = brick
            if (t_boundaries[q1] == t) and (t_boundaries[q2] == t):
                starter_bricks.append(brick)

        return starter_bricks
    
    def _get_t_boundaries(self):
        """Finds the earliest unassigned timestep for each qubit."""
        t_boundaries = np.zeros((self.N))
        
        for q in range(self.N):
            unassigned_times = np.argwhere(self.wires[q] == -1)
            
            if unassigned_times.size == 0:  # If no unassigned bricks remain
                t_boundaries[q] = -1  # Mark it as fully assigned
            else:
                t_boundaries[q] = np.min(unassigned_times)  # Get the earliest unassigned time

        return t_boundaries

    
    def draw(self, cmap_name="tab20", seaborn_cmap: bool=True, show_unassigned: bool=True, unassigned_color: str='w', show_reflection: bool=True, scale=1.0, x_scale=None, y_scale=None, return_fig=False):

        if x_scale is None:
            x_scale = self.N / 3
        if y_scale is None:
            y_scale = self.T
        fig, ax = plt.subplots(figsize=(x_scale * scale, y_scale * scale / 2))  # Adjust aspect ratio

        colors = self._generate_patch_colors(self.N_patches, cmap_name, seaborn_cmap)
        if show_unassigned:
            self._draw_patch(list(self.unassigned_bricks), ax, color=unassigned_color, show_reflection=show_reflection)
        for patch_id, patch in self.patches.items():
            self._draw_patch(patch, ax, color=colors[patch_id], show_reflection=show_reflection)

        if return_fig:
            return fig, ax
        
    def _generate_patch_colors(self, N_patches, cmap_name: str="tab20", seaborn_cmap: bool=True):

        cmap = plt.get_cmap(cmap_name, N_patches)  # Sample N_patches colors from the colormap
        return [cmap(i) for i in range(N_patches)]
    
    def _draw_patch(self, gates, ax, color='k', show_reflection=False):
        """
        Visualizes a compact and horizontally stretched brickwall quantum circuit
        where qubits are along the x-axis and time steps are along the y-axis.

        Parameters:
            gates (list of tuples): List of (q1, q2, t) tuples representing two-qubit gates.
            num_qubits (int): Total number of qubits.
            max_depth (int): Maximum depth (time steps).
            color (str): Color of the bricks.
        """

        num_qubits = self.N
        max_depth = self.T

        if not gates:
            print("No gates to display.")
            return

        # Draw vertical qubit lines (time evolution lines)
        ymin = -0.5
        if show_reflection:
            ymin = -0.5 - max_depth
        for q in range(num_qubits):
            ax.plot([q, q], [ymin, max_depth - 0.5], color="black", linewidth=1, linestyle="-", zorder=0)

        # Define brick dimensions
        brick_width = 0.8  # Wider along qubit axis
        brick_height = 0.8  # Narrower along time axis
        pbc_shift = 0.5
        pbc_alpha = 1.0

        # Draw gates as rectangles (bricks)
        for gate in gates:
            _q1, _q2, t = gate
            q1, q2 = np.sort([_q1, _q2])
            if abs(q1 - q2) == 1:
            # Rectangle coordinates: (x, y) = (qubit, time)
                rect = plt.Rectangle((q1 - brick_width / 2, t - brick_height / 2), 
                                    q2 - q1 + brick_width, brick_height, 
                                    alpha=1, facecolor=color, edgecolor="black", linewidth=1)
                ax.add_patch(rect)
            else: # PBC brick
                rect = plt.Rectangle((q1 - brick_width / 2-pbc_shift, t - brick_height / 2), 
                                    q1 + brick_width+pbc_shift, brick_height, 
                                    alpha=pbc_alpha, facecolor=color, edgecolor="black", linewidth=1, ls='-')
                ax.add_patch(rect)
                rect = plt.Rectangle((q2 - brick_width / 2, t - brick_height / 2), 
                                    q2 + brick_width+pbc_shift, brick_height, 
                                    alpha=pbc_alpha, facecolor=color, edgecolor="black", linewidth=1, ls='-')
                ax.add_patch(rect)
            
            if show_reflection:
                t = -t
                if abs(q1 - q2) == 1:
                # Rectangle coordinates: (x, y) = (qubit, time)
                    rect = plt.Rectangle((q1 - brick_width / 2, t - brick_height / 2 - 1), 
                                        q2 - q1 + brick_width, brick_height, 
                                        alpha=1, facecolor=color, edgecolor="w", linewidth=1)
                    ax.add_patch(rect)
                else: # PBC brick
                    rect = plt.Rectangle((q1 - brick_width / 2-pbc_shift, t - brick_height / 2 - 1), 
                                        q1 + brick_width+pbc_shift, brick_height, 
                                        alpha=pbc_alpha, facecolor=color, edgecolor="w", linewidth=1)
                    ax.add_patch(rect)
                    rect = plt.Rectangle((q2 - brick_width / 2, t - brick_height / 2 - 1), 
                                        q2 + brick_width+pbc_shift, brick_height, 
                                        alpha=pbc_alpha, facecolor=color, edgecolor="w", linewidth=1)
                    ax.add_patch(rect)

        # Formatting
        ax.set_xlim(-0.8, num_qubits - 0.2)
        ax.set_ylim(ymin, max_depth - 0.5)
        ax.set_xticks([2*_ for _ in range(num_qubits//2)])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlabel(r"$N$", fontsize=16)
        ax.set_ylabel(r"$T$", fontsize=16)

        # Remove unnecessary axis lines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
            
    
    # def _generate_patch_colors(self, N_patches, cmap_name: str="tab20", seaborn_cmap: bool=True):
    #     """
    #     Generates a list of visually distinct colors for N_patches.
        
    #     Parameters:
    #         N_patches (int): Number of patches (unique colors needed).
    #         cmap_name (str): Name of the Matplotlib colormap to use (default: 'tab10').

    #     Returns:
    #         List of color values (hex or RGB tuples).
    #     """
    #     try:
    #         cmap = plt.get_cmap(cmap_name, N_patches)  # Sample N_patches colors from the colormap
    #         return [cmap(i) for i in range(N_patches)]
    #     except:
    #         pass
    #     try:
    #         return self._generate_patch_colors_seaborn(N_patches, cmap_name=cmap_name)
    #     except:
    #         pass
    #     try:
    #         return self._generate_patch_colors_colorsys(N_patches)
    #     except Exception as e:
    #         raise e

    # def _generate_patch_colors_seaborn(self, N_patches, cmap_name: str='husl'):
    #     return sns.color_palette(cmap_name, N_patches)

    # def _generate_patch_colors_colorsys(self, N_patches):
    #     """
    #     Generates N visually distinct colors using the HSLuv color space.

    #     Parameters:
    #         N_patches (int): Number of patches.

    #     Returns:
    #         List of RGB tuples.
    #     """
    #     colors = [
    #         colorsys.hsv_to_rgb(i / N_patches, 0.7, 0.9)  # Hue spread evenly, with high saturation and brightness
    #         for i in range(N_patches)
    #     ]
    #     return colors



    