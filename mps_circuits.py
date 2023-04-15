"""
CPEN 400Q work

Pennylane implementation of quantum circuits constructed using the classically
trained MPS

Author : @abhishekabhishek
"""
import pennylane as qml
import numpy as np
from mps_circuit_helpers import is_unitary


def mps_unitaries_to_circuit(mps_unitaries, shots: int = None,
                             return_state: bool = False):
    """
    Given a list of unitaries extracted from an MPS, initialize a quantum
    circuit with the trained parameters

    Args:
        mps_unitaries (list): List of multi-qubit unitaries constructed from
            the MPS using the helper functions in mps_circuit_helpers.py
        shots (int): No. of shots - this is used to either have the circuit
            return probabilities of the computational basis states (shots=None)
            or return actual samples from the QCBM (shots = 1024)
        return_state (bool): If true, the qnode returns the quantum state, not
            probabilities or samples

    Returns:
        qml.QNode : Initialized quantum circuit corresponding to the trained
            MPS
    """
    # check all matrices in the list are actually unitaries
    for idx, unitary in enumerate(mps_unitaries):
        assert is_unitary(unitary), \
            f"the matrix at idx {idx} in the list is not unitary"
        assert unitary.shape[0]%2 == 0, \
            f"the unitary size at idx {idx} is not a power-of-two : \
                {unitary.shape}"

    n_wires = len(mps_unitaries)
    dev = qml.device("default.qubit", wires=n_wires, shots=shots)

    @qml.qnode(dev)
    def circuit():
        # starting from wire 0, apply the multi-qubit unitaries in the list
        # in a staircase format
        for wire in range(n_wires-1, -1, -1):
            unitary = mps_unitaries[wire]
            n_qubits = int(np.log2(unitary.shape[0]))
            u_wires = [wire] + list(range(wire-1, wire-n_qubits, -1))
            u_wires.reverse()
            qml.QubitUnitary(unitary, wires=u_wires)

        # return bitstring samples if number of shots specified
        if return_state:
            return qml.state()

        if shots is not None:
            return qml.sample()

        # else return the probs of bitstrings
        return qml.probs(wires=range(n_wires))

    return circuit


def mps_unitaries_to_circuit_template(mps_unitaries):
    """
    This is almost exactly the same circuit/function as above but it returns
    a qfunc compared to a qnode and can therefore be used as a template in
    building circuits which have multiple two-qubit unitary layers

    Args:
        mps_unitaries (list): List of multi-qubit unitaries constructed from
            the MPS using the helper functions in mps_circuit_helpers.py

    Returns:
        qml.qfunc : Mapped quantum circuit subroutine
    """
    # check all matrices in the list are actually unitaries
    for idx, unitary in enumerate(mps_unitaries):
        assert is_unitary(unitary), \
            f"the matrix at idx {idx} in the list is not unitary"
        assert unitary.shape[0]%2 == 0, \
            f"the unitary size at idx {idx} is not a power-of-two : \
                {unitary.shape}"

    n_wires = len(mps_unitaries)

    def circuit_template():
        # starting from wire 0, apply the multi-qubit unitaries in the list
        # in a staircase format
        for wire in range(n_wires-1, -1, -1):
            unitary = mps_unitaries[wire]
            n_qubits = int(np.log2(unitary.shape[0]))
            u_wires = [wire] + list(range(wire-1, wire-n_qubits, -1))
            u_wires.reverse()
            qml.QubitUnitary(unitary, wires=u_wires)

    return circuit_template


def mps_layers_to_circuit(mps_layer_list: list, shots: int = None,
                          return_state: bool = False):
    """
    Given a list of 2-qubit unitary layers, initialize a decomposed MPS
    circuit

    Args:
        mps_layer_list (list): List of 2-qubit unitary layers i.e. list of
            lists where each item list consists of a layer of 2-qubit unitaries
        shots (int): No. of shots - this is used to either have the circuit
            return probabilities of the computational basis states (shots=None)
            or return actual samples from the QCBM (shots = 1024)
        return_state (bool): If true, the qnode returns the quantum state, not
            probabilities or samples

    Returns:
        qml.QNode : Initialized quantum circuit corresponding to the trained
            MPS
    """
    # according to the arXiV:2209.00595, we need to reverse the list ordering
    # such that the last extracted from the MPS get applied first
    layer_list = mps_layer_list.copy()
    layer_list.reverse()

    n_wires = len(layer_list[0])
    dev = qml.device("default.qubit", wires=n_wires, shots=shots)

    @qml.qnode(dev)
    def circuit():
        for mps_unitaries in layer_list:
            # use the single layer template above
            mps_unitaries_to_circuit_template(mps_unitaries)()

        # return bitstring samples if number of shots specified
        if return_state:
            return qml.state()

        if shots is not None:
            return qml.sample()

        # else return the probs of bitstrings
        return qml.probs(wires=range(n_wires))

    return circuit
