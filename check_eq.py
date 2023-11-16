import os
import json
import time
from glob import glob
from pathlib import Path

from pytket import Circuit
from pytket.qasm import circuit_from_qasm_str
from pytket.extensions.cutensornet import TensorNetwork
import cuquantum as cq
import numpy as np


def test_equivalence(circ1, circ2):
    n_qubits = circ1.n_qubits
    assert n_qubits == circ2.n_qubits

    bell_states = Circuit(2 * n_qubits)
    for q in range(n_qubits):
        bell_states.H(q)
    for q in range(n_qubits):
        bell_states.CX(q, q + n_qubits)

    ket_circ = bell_states.copy()
    ket_circ.add_circuit(circ1, qubits=[q for q in range(n_qubits)])
    bra_circ = bell_states
    bra_circ.add_circuit(circ2, qubits=[q for q in range(n_qubits)])

    # Create the TNs of the circuits
    ket_net = TensorNetwork(ket_circ)
    bra_net = TensorNetwork(bra_circ)
    # Concatenate one with the other, netB is the adjoint
    overlap_net = ket_net.vdot(bra_net)
    # Run the contraction
    overlap = cq.contract(*overlap_net)

    return np.isclose(overlap, 1)


def run(max_qubits, results):
    old_circs = "bef"
    new_circs = "aft"

    n_skipped = 0
    n_success = 0
    n_fail = 0

    for name in os.listdir(old_circs):
        old_circ_f = os.path.join(old_circs, name)
        name = Path(name).stem

        try:
            old_circ = load_circuit(old_circ_f)
        except ValueError:
            # Ignore this file
            # We only support tket1 json and qasm files
            continue

        json_new_circs = glob(os.path.join(new_circs, name + "*.json"))
        qasm_new_circs = glob(os.path.join(new_circs, name + "*.qasm"))

        for new_circ_f in json_new_circs + qasm_new_circs:
            new_circ = load_circuit(new_circ_f)

            if new_circ.n_qubits != old_circ.n_qubits:
                print(
                    f"{old_circ_f} and {new_circ_f} have different qubit counts ({old_circ.n_qubits} vs {new_circ.n_qubits})"
                )
                n_fail += 1
                continue

            print(
                f"Checking equivalence for {old_circ_f} and {new_circ_f} ({new_circ.n_qubits} qb, {old_circ.n_gates} -> {new_circ.n_gates} gates)"
            )
            if new_circ.n_qubits > max_qubits:
                print("Skip")
                n_skipped += 1
                continue
            start_time = time.time()
            is_eq = test_equivalence(new_circ, old_circ)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if is_eq:
                print(f"{name}: OK ({elapsed_time:.2f}s))")
                n_success += 1
            else:
                print(f"{name}: FAIL ({elapsed_time:.2f}s))")
                n_fail += 1

            results.append((name, is_eq, elapsed_time))

    print(f"Done. Success/Fail/Skipped ({n_success}/{n_fail}/{n_skipped}).")


def load_circuit(file):
    """Load a circuit from a tket1 json or qasm file."""
    file = Path(file)
    # Check the extension
    if file.suffix == ".json":
        with open(file, "r") as f:
            json_circ = json.load(f)
            # Ignore classical registers
            json_circ["bits"] = []
            circ = Circuit.from_dict(json_circ)
    elif file.suffix == ".qasm":
        with open(file, "r") as f:
            # Ignore classical registers
            qasm = "".join(line for line in f if "creg" not in line)
            circ = circuit_from_qasm_str(qasm)
    else:
        raise ValueError(f"Unknown file extension: {file.suffix}")

    return circ


if __name__ == "__main__":
    import csv

    results = []

    MAX_QUBITS = 40

    print("Starting")
    try:
        run(MAX_QUBITS, results)
    finally:
        # Save results to CSV file
        with open("results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Success", "Elapsed Time"])
            for row in results:
                writer.writerow(row)
