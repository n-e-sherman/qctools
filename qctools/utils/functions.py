import torch
import quimb.tensor as qtn

from typing import Tuple

def i_to_b(index, num_qubits, order="big"):
    """Convert an integer index to a bitstring of a given length, with optional ordering.

    Args:
        index (int): The integer to convert.
        num_qubits (int): The number of qubits (length of the bitstring).
        order (str): The bit ordering, either "little" or "big".
                     "little" gives the least significant bit first (default).
                     "big" gives the most significant bit first.

    Returns:
        str: The bitstring representation of the index.
    """
    if order not in ["little", "big"]:
        raise ValueError("Order must be either 'little' or 'big'")

    # Convert to a bitstring
    bitstring = format(index, f"0{num_qubits}b")

    # Reverse the bitstring for little-endian order
    if order == "little":
        bitstring = bitstring[::-1]

    return bitstring



def b_to_i(bitstring, order="big"):
    """Convert a bitstring to an integer index, with optional ordering.

    Args:
        bitstring (str): The bitstring to convert.
        order (str): The bit ordering, either "little" or "big".
                     "little" treats the bitstring as little-endian (default).
                     "big" treats the bitstring as big-endian.

    Returns:
        int: The integer index represented by the bitstring.
    """
    if order not in ["little", "big"]:
        raise ValueError("Order must be either 'little' or 'big'")

    # Reverse the bitstring for little-endian order
    if order == "little":
        bitstring = bitstring[::-1]

    # Convert the bitstring to an integer
    return int(bitstring, 2)




def gauge_tensor_loss(circ: qtn.Circuit, target: torch.Tensor, 
                      gauge: bool=True, 
                      gauge_inds: Tuple[int]=(0, 0), 
                      alpha_thresh: float=1e-7
):

        U = circ.get_uni().to_dense()
        if gauge:
            alpha = U[*gauge_inds]
            beta = target[*gauge_inds]
            if abs(alpha) > alpha_thresh:
                U = U * (beta / alpha)
        return (abs(U - target)**2).sum()



def bitstring_loss(circ: qtn.Circuit, target: str):

    return -torch.abs(circ.amplitude(target))**2