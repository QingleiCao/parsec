#!/usr/bin/env python3
#
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#
"""An adapter between PaRSEC's QPU device and CUDA-Q.

PaRSEC's adapter backend (--mca device_qpu_backend adapter) is the client of
this program: it sends circuits as OpenQASM 3 and gets counts back, over a
Unix-domain socket, one line of JSON per message (tools/qpu/README.md has the
protocol). PaRSEC does not start this program and links to nothing of it.

The jobs run on a simulator of CUDA-Q, on this machine, with no account and
no network: the CPU simulator (--target qpp-cpu, the default) or the GPU one
(--target nvidia). The program takes no other target, reaches no quantum
device and has no code that could.

    parsec_qpu_adapter_cudaq.py --socket /path/of/the.sock
    CUDA_VISIBLE_DEVICES=2 parsec_qpu_adapter_cudaq.py --socket s.sock --target nvidia

One adapter is one QPU of CUDA-Q, and runs one job at a time. For several
QPUs, start several adapters, each on a socket and, for the GPU simulator,
each with a GPU of its own; PaRSEC takes one entry of device_qpu_devices for
each. --qpu chooses a QPU of a target that has several (nvidia with --option
mqpu has one per visible GPU). With the CPU simulator the adapter sees no GPU:
it hides them from CUDA-Q, which would otherwise start on one.

What a job gives depends on --seed and on the key of each evaluation, and on
nothing else: not on the order of the jobs, not on what shares a job, not on
how often a job was tried. CUDA-Q has one seed for the process, and setting
it waits for the job that runs: so the seed of an evaluation is set when no
job runs, right before its own.

The circuits are read here, by a reader for the text PaRSEC writes
(parsec_qpu_qasm_write) and for no other OpenQASM 3: CUDA-Q has no reader for
a text with parameters. A text is taken only if writing the circuit that was
read from it gives the same text again; any other is refused when it is given.

It needs cudaq. With --check-reader <directory> in place of every other
option it needs nothing: it reads the texts tests/runtime/qpu/qpu_qasm_test -o
wrote there, and says whether it read each as the circuit it is.
"""

import argparse
import math
import os
import re
import sys

# The protocol half is a module beside this file, however the program was started,
# and loading it writes nothing beside the sources.
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parsec_qpu_adapter_protocol import Refused, add_arguments, check_arguments, mix64, run  # noqa: E402

# The targets this program sets, each with the options it takes: simulators, all of them.
TARGETS = {"qpp-cpu": ("",), "nvidia": ("", "mqpu")}

# ---- the reader: the text parsec_qpu_qasm_write() writes ----

ONE_QUBIT = ("h", "x", "y", "z")
ROTATIONS = ("rx", "ry", "rz")
TWO_QUBITS = ("cx", "cz")
TWO_QUBIT_ROTATIONS = ("rxx", "ryy", "rzz")
# rxx, ryy and rzz are not gates of the standard library: the text defines the
# ones it uses, in these words. Another definition is another gate, and is refused.
DEFINITIONS = {
    "rxx": "gate rxx(theta) a, b { h a; h b; cx a, b; rz(theta) b; cx a, b; h b; h a; }",
    "ryy": "gate ryy(theta) a, b { rx(pi/2) a; rx(pi/2) b; cx a, b; rz(theta) b; cx a, b; rx(-pi/2) a; rx(-pi/2) b; }",
    "rzz": "gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }",
}
MAX_QUBITS = 64

_NUMBER = r"-?[0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?"
_INDEX = r"(0|[1-9][0-9]*)"
_RE_INPUT = re.compile(r"input float\[64\] p%s;" % _INDEX)
_RE_QUBITS = re.compile(r"qubit\[%s\] q;" % _INDEX)
_RE_BITS = re.compile(r"bit\[%s\] c;" % _INDEX)
_RE_MEASURE = re.compile(r"c\[%s\] = measure q\[%s\];" % (_INDEX, _INDEX))
_RE_GATE = re.compile(r"([a-z]+)(?:\((%s)(?: \* p%s)?\))? q\[%s\](?:, q\[%s\])?;" % (_NUMBER, _INDEX, _INDEX, _INDEX))


class Circuit:
    """A circuit as its text gives it."""

    def __init__(self):
        self.qubits = 0
        self.params = 0
        self.ops = []           # (name, q0, q1 or None, parameter slot or None, angle or None); a measurement is ("measure", q, ...)
        self.scaled = []        # (slot, coefficient) of every angle that is a coefficient times a parameter
        self.measured = []      # the measured qubits, by increasing index: bit i of a result is qubit i
        self.kernel = None      # built by the thread that runs the jobs, at the first job that needs it


def write_number(value):
    """A double as parsec_qpu_qasm_write() writes it: 17 significant digits, and always a point or an exponent."""
    text = "%.17g" % value
    return text if "." in text or "e" in text else text + ".0"


def write_circuit(circuit):
    """The text parsec_qpu_qasm_write() writes for a circuit."""
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";']
    lines.extend("input float[64] p%d;" % k for k in range(circuit.params))
    used = set(op[0] for op in circuit.ops)
    lines.extend(DEFINITIONS[name] for name in TWO_QUBIT_ROTATIONS if name in used)
    lines.append("qubit[%d] q;" % circuit.qubits)
    lines.append("bit[%d] c;" % circuit.qubits)
    for name, q0, q1, slot, angle in circuit.ops:
        if name == "measure":
            lines.append("c[%d] = measure q[%d];" % (q0, q0))
            continue
        line = name
        if angle is not None:
            line += "(%s)" % write_number(angle) if slot is None else "(%s * p%d)" % (write_number(angle), slot)
        line += " q[%d];" % q0 if q1 is None else " q[%d], q[%d];" % (q0, q1)
        lines.append(line)
    return "\n".join(lines) + "\n"


def read_circuit(text):
    """The circuit of a text that parsec_qpu_qasm_write() wrote. Raises Refused, with the line, for any other."""
    circuit = Circuit()
    lines = text.split("\n")
    at = 0

    def refuse(why):
        raise Refused("rejected", "line %d: %s" % (at + 1, why))

    def expect(line):
        nonlocal at
        if at >= len(lines) or lines[at] != line:
            refuse("'%s' is expected here" % line)
        at += 1

    if not text.endswith("\n"):
        at = len(lines) - 1
        refuse("the text does not end with the end of a line")
    lines.pop()                 # what follows the last end of line: nothing
    expect("OPENQASM 3.0;")
    expect('include "stdgates.inc";')
    while at < len(lines) and lines[at].startswith("input "):
        match = _RE_INPUT.fullmatch(lines[at])
        if match is None or int(match.group(1)) != circuit.params:
            refuse("the inputs are 'input float[64] p0;', p1 and so on, in that order")
        circuit.params += 1
        at += 1
    defined = set()
    for name in TWO_QUBIT_ROTATIONS:
        if at < len(lines) and lines[at].startswith("gate %s(" % name):
            if lines[at] != DEFINITIONS[name]:
                refuse("a definition of %s that is not the one PaRSEC writes" % name)
            defined.add(name)
            at += 1
    match = _RE_QUBITS.fullmatch(lines[at]) if at < len(lines) else None
    if match is None:
        refuse("'qubit[n] q;' is expected here")
    circuit.qubits = int(match.group(1))
    if not 1 <= circuit.qubits <= MAX_QUBITS:
        refuse("a circuit has 1 to %d qubits" % MAX_QUBITS)
    at += 1
    expect("bit[%d] c;" % circuit.qubits)

    measured = set()

    def qubit(group):
        q = int(group)
        if q >= circuit.qubits:
            refuse("the circuit has no qubit %d" % q)
        if q in measured:
            refuse("an operation follows the measurement of its qubit")
        return q

    while at < len(lines):
        line = lines[at]
        match = _RE_MEASURE.fullmatch(line)
        if match is not None:
            if match.group(1) != match.group(2):
                refuse("qubit i is measured into bit i")
            q = qubit(match.group(2))
            measured.add(q)
            circuit.ops.append(("measure", q, None, None, None))
            at += 1
            continue
        match = _RE_GATE.fullmatch(line)
        if match is None:
            refuse("not an operation as PaRSEC writes one")
        name, number, slot, first, second = match.groups()
        rotation = name in ROTATIONS or name in TWO_QUBIT_ROTATIONS
        two = name in TWO_QUBITS or name in TWO_QUBIT_ROTATIONS
        if not (name in ONE_QUBIT or rotation or two):
            refuse("'%s' is not a gate of the circuit format" % name)
        if name in TWO_QUBIT_ROTATIONS and name not in defined:
            refuse("%s is used and not defined" % name)
        if rotation != (number is not None):
            refuse("%s takes %s" % (name, "one angle" if rotation else "no angle"))
        if two != (second is not None):
            refuse("%s is on %s" % (name, "two qubits" if two else "one qubit"))
        q0 = qubit(first)
        q1 = qubit(second) if two else None
        if q1 == q0:
            refuse("the two qubits of an operation are the same")
        angle = None
        if rotation:
            angle = float(number)
            if not math.isfinite(angle):
                refuse("an angle that is not a finite number")
        if slot is not None:
            slot = int(slot)
            if slot >= circuit.params:
                refuse("the text has no input p%d" % slot)
        circuit.ops.append((name, q0, q1, slot, angle))
        at += 1
    circuit.measured = sorted(measured)
    circuit.scaled = sorted(set((op[3], op[4]) for op in circuit.ops if op[3] is not None))
    if not measured:
        refuse("the circuit measures nothing")      # which is no circuit to PaRSEC: its writer writes none
    # What was read, written again, is the text: or the text is of another
    # form than PaRSEC's (a number written another way, a definition that is
    # not used), and what it means is not for this reader to say.
    if write_circuit(circuit) != text:
        at = 0
        refuse("a text that is not as PaRSEC writes this circuit")
    return circuit


def check_reader(directory):
    """The texts qpu_qasm_test -o wrote: c*.qasm with its circuit in c*.ops, which have to be read as that circuit;
    r*.qasm, which the reader of PaRSEC refuses, and o*.qasm, which it takes and its writer does not write:
    neither may be taken here."""
    names = sorted(os.listdir(directory))
    wrong = []
    nb_read = nb_refused = nb_other = 0
    for name in names:
        if not name.endswith(".qasm"):
            continue
        with open(os.path.join(directory, name), "rb") as f:
            text = f.read().decode("utf-8", "surrogateescape")
        if name[0] in "ro":
            try:
                read_circuit(text)
                wrong.append("%s: a text the reader has to refuse was taken" % name)
            except Refused:
                if name[0] == "r":
                    nb_refused += 1
                else:
                    nb_other += 1
            continue
        with open(os.path.join(directory, name[:-5] + ".ops")) as f:
            want = f.read().split("\n")
        try:
            circuit = read_circuit(text)
        except Refused as refused:
            wrong.append("%s: a text PaRSEC wrote is refused: %s" % (name, refused.why))
            continue
        # One line for the circuit, then one per operation, as qpu_qasm_test writes them:
        # the opcode, the qubits, the slot (-1 for none) and the angle, by its bits.
        opcodes = dict(h=1, x=2, y=3, z=4, rx=5, ry=6, rz=7, cx=8, cz=9, rxx=10, ryy=11, rzz=12, measure=13)
        have = ["%d %d %d" % (circuit.qubits, circuit.params, len(circuit.ops))]
        for op, q0, q1, slot, angle in circuit.ops:
            have.append("%d %d %d %d %s" % (opcodes[op], q0, 0 if q1 is None else q1, -1 if slot is None else slot,
                                            (0.0 if angle is None else angle).hex()))
        if want and want[-1] == "":
            want.pop()
        same = len(want) == len(have) and want[0] == have[0]
        for a, b in zip(want[1:], have[1:]) if same else ():
            fa, fb = a.split(" "), b.split(" ")
            # The angle is compared as a number and by its sign: two programs write the same double in two ways.
            x, y = float.fromhex(fa[4]), float.fromhex(fb[4])
            if fa[:4] != fb[:4] or x != y or math.copysign(1.0, x) != math.copysign(1.0, y):
                same = False
                break
        if not same:
            wrong.append("%s: read as another circuit than the one that was written" % name)
        else:
            nb_read += 1
    for line in wrong[:20]:
        sys.stderr.write("parsec_qpu_adapter_cudaq: %s\n" % line)
    if wrong or nb_read == 0 or nb_refused == 0 or nb_other == 0:
        sys.stderr.write("parsec_qpu_adapter_cudaq: the reader failed on %d texts (%d read as written; refused as they have "
                         "to be: %d that PaRSEC refuses, %d of other forms)\n" % (len(wrong), nb_read, nb_refused, nb_other))
        return 1
    print("parsec_qpu_adapter_cudaq: %d texts read as the circuits that were written; refused as they have to be: "
          "%d that PaRSEC refuses, %d of other forms" % (nb_read, nb_refused, nb_other))
    return 0


# ---- what runs the jobs ----

class Simulator:
    """One QPU of a simulator of CUDA-Q, as the thing that runs the jobs.

    Everything of CUDA-Q after the target is set is called by one thread, the
    one that runs the jobs (the protocol half is given one worker): a kernel
    is built there, at the first job of its circuit, the seed is set there,
    and the sample is taken there.
    """

    def __init__(self, args):
        # The CPU simulator takes every core unless told otherwise, which is a
        # burden to the application the adapter runs beside. --threads decides;
        # without it, what the environment says, and one thread if it says nothing.
        if args.threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(args.threads)
        else:
            os.environ.setdefault("OMP_NUM_THREADS", "1")
        # CUDA-Q takes its GPU simulator, and a GPU, as soon as it sees one,
        # before any target is set: an adapter for the CPU simulator sees none.
        if args.target == "qpp-cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Imported here, so that --help and --check-reader work without CUDA-Q.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # on what sample will be in a later release, and on a package in preview
            import cudaq

        self.cudaq = cudaq
        self.seed = args.seed
        self.qubits = args.qubits
        self.qpu = args.qpu
        try:
            if args.option:
                cudaq.set_target(args.target, option=args.option)
            else:
                cudaq.set_target(args.target)
            target = cudaq.get_target()
        except Exception as error:
            raise SystemExit("CUDA-Q does not set the target %s: %s" % (args.target, str(error)[:200]))
        if target.name != args.target:
            raise SystemExit("CUDA-Q was asked for the target %s and has %s" % (args.target, target.name))
        nb_qpus = target.num_qpus()
        if not 0 <= args.qpu < nb_qpus:
            raise SystemExit("--qpu %d, and the target %s has %d QPU(s) here" % (args.qpu, args.target, nb_qpus))
        version = cudaq.__version__.split()
        self.versions = "cudaq %s" % (version[2] if len(version) > 2 else cudaq.__version__)
        # Letters, digits, blanks and , . : _ - only: the client prints nothing else of a name.
        self.name = "cuda-q, target %s%s, QPU %d of %d" % (args.target, ", option %s" % args.option if args.option else "",
                                                          args.qpu, nb_qpus)

    def prepare(self, text):
        """A circuit from its text: (circuit, its parameter slots). The kernel is built later, by the thread of the jobs."""
        circuit = read_circuit(text)
        if circuit.qubits > self.qubits:
            raise Refused("rejected", "a circuit of %d qubits, and this adapter takes %d" % (circuit.qubits, self.qubits))
        return circuit, list(range(circuit.params))

    def build(self, circuit):
        """The kernel of a circuit. rxx, ryy and rzz as the text defines them: CUDA-Q's builder has none of the three."""
        if circuit.params:
            kernel, p = self.cudaq.make_kernel(list[float])
        else:
            kernel, p = self.cudaq.make_kernel(), None
        q = kernel.qalloc(circuit.qubits)
        half = math.pi / 2
        for name, q0, q1, slot, angle in circuit.ops:
            if name == "measure":
                continue
            if angle is not None and slot is not None:
                angle = angle * p[slot]
            if name in ONE_QUBIT:
                getattr(kernel, name)(q[q0])
            elif name in ROTATIONS:
                getattr(kernel, name)(angle, q[q0])
            elif name in TWO_QUBITS:
                getattr(kernel, name)(q[q0], q[q1])
            elif name == "rzz":
                kernel.cx(q[q0], q[q1]); kernel.rz(angle, q[q1]); kernel.cx(q[q0], q[q1])
            elif name == "rxx":
                kernel.h(q[q0]); kernel.h(q[q1])
                kernel.cx(q[q0], q[q1]); kernel.rz(angle, q[q1]); kernel.cx(q[q0], q[q1])
                kernel.h(q[q1]); kernel.h(q[q0])
            elif name == "ryy":
                kernel.rx(half, q[q0]); kernel.rx(half, q[q1])
                kernel.cx(q[q0], q[q1]); kernel.rz(angle, q[q1]); kernel.cx(q[q0], q[q1])
                kernel.rx(-half, q[q0]); kernel.rx(-half, q[q1])
            else:
                raise RuntimeError("an operation the reader took and the builder does not know")
        # Nothing follows a measurement on its qubit, so the measurements can
        # all stand at the end; in the order of the qubits, which is the order
        # of the characters of a result.
        for qubit in circuit.measured:
            kernel.mz(q[qubit])
        return kernel

    def run(self, circuit, slots, values, shots, key):
        """Counts of one evaluation, as a list of [bitstring as "0x...", count]."""
        # An angle that is no finite number (a coefficient times a value that
        # is too large, or no number at all) makes CUDA-Q end the process, and
        # every job of every client with it: such an evaluation fails here.
        for slot, coefficient in circuit.scaled:
            if not math.isfinite(coefficient * values[slot]):
                raise ValueError("an angle that is not a finite number")
        if circuit.kernel is None:
            circuit.kernel = self.build(circuit)
        # No job runs now: this thread is the only one that starts any, and it
        # waited for the last one. So the seed is set at once, and is the one
        # the next sample draws from.
        self.cudaq.set_random_seed((mix64(mix64(self.seed) ^ key) & 0x7FFFFFFF) or 1)
        if circuit.params:
            result = self.cudaq.sample_async(circuit.kernel, values, shots_count=shots, qpu_id=self.qpu).get()
        else:
            result = self.cudaq.sample_async(circuit.kernel, shots_count=shots, qpu_id=self.qpu).get()
        counts = {}
        total = 0
        for bits, count in result.items():
            if len(bits) != len(circuit.measured):
                raise RuntimeError("a result of another width than what was measured")
            value = 0
            for position, bit in enumerate(bits):       # character i is the i-th measured qubit
                if bit == "1":
                    value |= 1 << circuit.measured[position]
            counts[value] = counts.get(value, 0) + int(count)
            total += int(count)
        if total != shots:
            raise RuntimeError("a result with another number of shots than was asked for")
        return [["0x%x" % value, counts[value]] for value in sorted(counts)]


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--check-reader":
        return check_reader(sys.argv[2])
    parser = argparse.ArgumentParser(description="An adapter between PaRSEC's QPU device and CUDA-Q; it runs the jobs on a "
                                     "simulator of CUDA-Q and reaches no quantum device.",
                                     epilog="--check-reader <directory>, alone, checks the reader on the texts of qpu_qasm_test -o.",
                                     allow_abbrev=False)       # an option is what it is written as, in full
    # A circuit that is kept is a compiled kernel, about a megabyte of it.
    add_arguments(parser, max_prepared=256)
    parser.add_argument("--target", default="qpp-cpu", choices=sorted(TARGETS),
                        help="the simulator: qpp-cpu, on the CPU (default), or nvidia, on a GPU")
    parser.add_argument("--option", default="", help="an option of the target: mqpu with nvidia, a QPU for each visible GPU")
    parser.add_argument("--qpu", type=int, default=0, help="which QPU of the target runs the jobs (default 0)")
    parser.add_argument("--seed", type=int, default=1, help="seed of the samples (default 1)")
    parser.add_argument("--qubits", type=int, default=24, help="widest circuit taken (default 24)")
    parser.add_argument("--threads", type=int, default=None,
                        help="threads the CPU simulator may use (default: OMP_NUM_THREADS of the environment, or 1)")
    args = parser.parse_args()
    if args.threads is not None and args.threads < 1:
        raise SystemExit("--threads takes 1 or more")
    if args.option not in TARGETS[args.target]:
        raise SystemExit("the target %s takes %s here, not the option '%s'"
                         % (args.target, " or ".join(o for o in TARGETS[args.target] if o) or "no option", args.option))
    if not 1 <= args.qubits <= MAX_QUBITS:
        raise SystemExit("--qubits takes 1 to %d" % MAX_QUBITS)
    check_arguments(args)
    # One worker: one job at a time, and one thread for everything of CUDA-Q.
    return run(args, Simulator(args), 1)


if __name__ == "__main__":
    sys.exit(main())
