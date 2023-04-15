"""
CPEN 400Q work

Helper functions to assist in the analytic decomposition of an MPS with
arbitrary bond dimensions to a parameterized quantum circuit of layers of
two-qubit unitary gates

Author: @abhishekabhishek
"""
import tensornetwork as tn
import numpy as np

from MPScumulant import MPS_c


def truncate_mps(mps: tn.FiniteMPS, max_singular_values: int = 2):
    """
    Crude truncation of an MPS where we apply SVD to each core tensor
    individually, throw away the (extra singular values, extra rows from u, and
    extra columns from v), and construct a new MPS

    Args:
        mps (tn.FiniteMPS): a tn MPS object with core tensors with
            arbitrary bond dimensions
        max_singular_values (int): No. of maximum singular values to keep in
            the truncated MPS core tensors

    Returns:
        truncated_mps (tn.FinteMPS): a tn MPS object with core tensors which
            are truncated to a specified maximum bond dimension 
    """
    truncated_tensors = []
    mps_tensors = mps.tensors

    for _, tensor in enumerate(mps_tensors):
        # perform svd on the core tensor
        _u, _s, _vh, _ = mps.svd(tensor,
                                 max_singular_values=max_singular_values)

        # compute a truncated SVD approximation
        # TODO double check this works correctly for the order 3 tensor
        truncated_tensor = np.matmul(
            _u[:max_singular_values], np.matmul(
                np.diag(_s), _vh[:, :max_singular_values]
            )
        )
        truncated_tensors.append(truncated_tensor)

    return tn.FiniteMPS(truncated_tensors, canonicalize=False)


def tn_to_mps(mps: tn.FiniteMPS):
    """
    Manually convert a tensorNetwork.FiniteMPS object to MPScumulant.MPS_c
    object which is used by most of our MPS to PQC mapping functions

    Args:
        mps (tn.FiniteMPS): tn MPS object to be converted

    Returns:
        mps (MPScumulant.MPS_c): MPS object containing the core tensors and
            bond dimensions.
    """
    mps_tensors, mps_bonddims = mps.tensors, mps.bond_dimensions
    mps_c = MPS_c(len(mps.tensors))
    mps_c.matrices, mps_c.bond_dimension = mps_tensors, \
        np.array(mps_bonddims[1:], dtype=np.int16)
    return mps_c


def apply_conjugate_unitaries(mps_unitaries: list, mps: tn.FiniteMPS,
                              max_singular_value: int = 4):
    """
    Apply the disentangler (i.e. a conjugated mapped quantum circuit)
    corresponding to a chi = 2 MPS

    Note that this applies the disentangler in-place on the MPS

    Args:
        mps_unitaries (list): list of 2-qubit unitaries extradted from the
            chi=2 MPS
        mps (tn.FiniteMPS): tn MPS object to apply the disentangler to - this
            is the MPS with chi > 2 bond dimensions

    Returns:
        None
    """
    # construct the disentangler using the MPS unitaries
    disentangler = []
    for unitary in mps_unitaries:
        disentangler.append(unitary.conj().T)

    # apply the single site gate first
    mps.apply_one_site_gate(disentangler[0], site=0)

    # apply the two site gates (hopefully) in the correct order
    for idx in range(len(disentangler)-1):
        gate = disentangler[idx+1]

        # TODO not sure on the correctness of this reshaping but the two-site
        # operator needs to be rank 4 and this is the only valid way
        gate = gate.reshape(2, 2, 2, 2)

        # need to set the center position in order for SVD truncation to work
        mps.center_position = idx
        mps.position(idx)

        mps.apply_two_site_gate(gate, site1=idx, site2=idx+1,
                                max_singular_values=max_singular_value)
