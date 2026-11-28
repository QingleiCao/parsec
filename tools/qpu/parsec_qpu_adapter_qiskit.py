#!/usr/bin/env python3
#
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#
"""An adapter between PaRSEC's QPU device and Qiskit.

PaRSEC's adapter backend (--mca device_qpu_backend adapter) is the client of
this program: it sends circuits as OpenQASM 3 and gets counts back, over a
Unix-domain socket, one line of JSON per message (tools/qpu/README.md has the
protocol). PaRSEC does not start this program and links to nothing of it.

This version runs the jobs on Qiskit Aer, a simulator: on this machine, with
no account and no network. It reaches no quantum device and has no code that
could. The option --backend exists so that a later version can add one, and
takes nothing but "aer" here.

    parsec_qpu_adapter_qiskit.py --socket /path/of/the.sock
    parsec_qpu_adapter_qiskit.py --socket s.sock --device generic:5:line

With --device the circuits are transpiled to the gates and the connectivity
of a device model, and run with its noise: Qiskit's GenericBackendV2, a made-up
device, not the model of an existing one.

What a job gives depends on --seed and on the key of each evaluation, and on
nothing else: not on the order of the jobs, not on what shares a job, not on
how often a job was tried.

It needs qiskit, qiskit-aer and qiskit-qasm3-import (the reader of OpenQASM 3
that qiskit.qasm3.loads uses).
"""

import argparse
import os
import re
import sys
import threading

# The protocol half is a module beside this file, however the program was started,
# and loading it writes nothing beside the sources.
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parsec_qpu_adapter_protocol import Adapter, Refused, add_arguments, check_arguments, mix64, run  # noqa: E402,F401


class Simulator:
    """Qiskit Aer, as the thing that runs the jobs."""

    def __init__(self, args):
        # The libraries below start a thread per core each unless told
        # otherwise, which is no help to circuits of a few qubits and a
        # burden to the application the adapter runs beside.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
            os.environ.setdefault(name, str(args.threads))
        # Imported here, so that --help works without Qiskit.
        import qiskit
        import qiskit_aer
        from qiskit import qasm3, transpile
        from qiskit_aer import AerSimulator

        self.qasm3 = qasm3
        self.transpile = transpile
        self.seed = args.seed
        self.level = args.optimization_level
        self.versions = "qiskit %s, qiskit-aer %s" % (qiskit.__version__, qiskit_aer.__version__)
        self.lock = threading.Lock()        # the reader and the transpiler are given one circuit at a time
        device = args.device
        if device in (None, "", "none"):
            self.backend = AerSimulator(method=args.method, max_parallel_threads=args.threads)
            self.name = "aer"
            self.qubits = args.qubits
        else:
            from qiskit.providers.fake_provider import GenericBackendV2

            parts = device.split(":")
            if parts[0] != "generic" or len(parts) not in (2, 3) or not parts[1].isdigit():
                raise SystemExit("--device takes none, or generic:<qubits>[:line|ring|full], not '%s'" % device)
            n = int(parts[1])
            shape = parts[2] if len(parts) == 3 else "line"
            if n < 2 or shape not in ("line", "ring", "full"):
                raise SystemExit("--device generic:<qubits>[:line|ring|full] with at least 2 qubits, not '%s'" % device)
            edges = None
            if shape != "full":
                edges = [[q, q + 1] for q in range(n - 1)]
                if shape == "ring" and n > 2:
                    edges.append([n - 1, 0])
                edges = edges + [[b, a] for a, b in edges]
            model = GenericBackendV2(n, coupling_map=edges, seed=args.seed, noise_info=not args.no_noise)
            if args.no_noise:
                self.backend = AerSimulator(method=args.method, max_parallel_threads=args.threads, coupling_map=model.coupling_map,
                                            basis_gates=[g for g in model.operation_names if g not in ("delay", "reset", "measure")])
            else:
                self.backend = AerSimulator.from_backend(model, method=args.method, max_parallel_threads=args.threads)
            self.name = "aer, as the device model generic:%d:%s%s" % (n, shape, "" if not args.no_noise else ", without its noise")
            self.qubits = min(n, args.qubits)

    def prepare(self, text):
        """A circuit from its text, transpiled once: (circuit, its parameters by slot)."""
        try:
            with self.lock:
                circuit = self.qasm3.loads(text)
        except Exception as error:          # whatever the reader raises: the circuit is not taken
            raise Refused("rejected", "the circuit is not read: %s" % str(error)[:200])
        if circuit.num_qubits > self.qubits:
            raise Refused("rejected", "a circuit of %d qubits, and this adapter takes %d" % (circuit.num_qubits, self.qubits))
        # The k-th input of the text is parameter slot k.
        names = re.findall(r"^\s*input\s+\w+(?:\s*\[\s*\d+\s*\])?\s+(\w+)\s*;", text, flags=re.MULTILINE)
        try:
            with self.lock:
                ready = self.transpile(circuit, self.backend, optimization_level=self.level, seed_transpiler=self.seed)
        except Exception as error:
            raise Refused("rejected", "the circuit is not transpiled: %s" % str(error)[:200])
        by_name = {p.name: p for p in ready.parameters}
        # An input the circuit declares and does not use has no parameter left.
        return ready, [by_name.get(name) for name in names]

    def run(self, ready, slots, values, shots, key):
        """Counts of one evaluation, as a list of [bitstring as "0x...", count]."""
        bound = ready.assign_parameters({p: v for p, v in zip(slots, values) if p is not None})
        seed = mix64(mix64(self.seed) ^ key) & 0x7FFFFFFF
        result = self.backend.run(bound, shots=shots, seed_simulator=seed).result()
        if not result.success:
            raise RuntimeError("the simulator did not run the circuit")
        counts = {}
        for bits, count in result.get_counts(0).items():
            value = int(bits.replace(" ", ""), 2)
            counts[value] = counts.get(value, 0) + int(count)
        return [["0x%x" % value, counts[value]] for value in sorted(counts)]


def main():
    parser = argparse.ArgumentParser(description="An adapter between PaRSEC's QPU device and Qiskit; this version runs "
                                     "the jobs on Qiskit Aer, a simulator, and reaches no quantum device.",
                                     allow_abbrev=False)       # an option is what it is written as, in full
    add_arguments(parser)
    parser.add_argument("--backend", default="aer", help="what runs the jobs: aer, the only one this version has")
    parser.add_argument("--method", default="automatic", help="the simulation method of Aer (default: automatic)")
    parser.add_argument("--device", default="none",
                        help="none (default): an ideal simulator; generic:<qubits>[:line|ring|full]: a made-up device "
                        "model, to whose gates and connectivity the circuits are transpiled, with its noise")
    parser.add_argument("--no-noise", action="store_true", help="with --device: its gates and connectivity, without its noise")
    parser.add_argument("--seed", type=int, default=1, help="seed of the samples, of the device model and of the transpiler (default 1)")
    parser.add_argument("--qubits", type=int, default=24, help="widest circuit taken (default 24, or the size of the device)")
    parser.add_argument("--workers", type=int, default=2, help="jobs run at the same time (default 2)")
    parser.add_argument("--threads", type=int, default=1, help="threads one simulation may use (default 1)")
    parser.add_argument("--optimization-level", type=int, default=1, choices=(0, 1, 2, 3), help="of the transpiler (default 1)")
    args = parser.parse_args()
    if args.backend != "aer":
        raise SystemExit("this version has one backend, aer: it reaches no quantum device, and '%s' is not built" % args.backend)
    check_arguments(args)
    return run(args, Simulator(args), args.workers)


if __name__ == "__main__":
    sys.exit(main())
