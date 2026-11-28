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
import json
import os
import re
import select
import signal
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

PROTOCOL = 1
MAX_LINE = 256 << 20


def log(message):
    """What happens, never what a message holds."""
    if ARGS.verbose:
        sys.stderr.write("parsec_qpu_adapter: %s\n" % message)
        sys.stderr.flush()


def mix64(x):
    """splitmix64, as PaRSEC's parsec_qpu_mix64()."""
    mask = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return x ^ (x >> 31)


class Refused(Exception):
    """A request that is not taken; the client is told why."""

    def __init__(self, error, why):
        super().__init__(why)
        self.error = error
        self.why = why


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


class Job:
    def __init__(self, number, tag, entries):
        self.number = number
        self.tag = tag
        self.entries = entries          # (ready circuit, slots, values, shots, key)
        self.state = "running"          # then done, failed or cancelled
        self.reported = False
        self.cancel = False
        self.results = None
        self.t_start = time.monotonic_ns()
        self.t_end = None


class Session:
    """What one client has here. It outlives the client's connection."""

    def __init__(self, name):
        self.name = name
        self.jobs = {}                  # by number
        self.tags = {}                  # tag -> number
        self.rfd, self.wfd = os.pipe()  # a byte when a job ends
        os.set_blocking(self.rfd, False)
        os.set_blocking(self.wfd, False)
        self.users = 0                  # connections that use it now
        self.t_left = None              # since when none does
        self.closed = False

    def close(self):
        """With the adapter's lock held: nobody writes to the pipe after this."""
        self.closed = True
        for job in self.jobs.values():
            job.cancel = True
        self.jobs.clear()
        self.tags.clear()
        for fd in (self.rfd, self.wfd):
            try:
                os.close(fd)
            except OSError:
                pass


class Adapter:
    def __init__(self, args):
        self.args = args
        self.simulator = Simulator(args)
        self.lock = threading.Lock()
        self.sessions = {}
        self.circuits = {}              # name -> (text, ready circuit, slots)
        self.next_job = 1
        self.pool = ThreadPoolExecutor(max_workers=args.workers)
        self.counters = dict(hellos=0, resumes=0, prepares=0, submits=0, jobs=0, tag_hits=0, waits=0, wakes=0,
                             results=0, cancels=0, abandons=0, byes=0, failed_jobs=0)

    # ---- the jobs ----

    def work(self, session, job):
        results = []
        state = "done"
        try:
            for ready, slots, values, shots, key in job.entries:
                if job.cancel:
                    state = "cancelled"
                    break
                results.append(self.simulator.run(ready, slots, values, shots, key))
        except Exception as error:      # a job that fails is a job that failed, whatever the reason
            log("a job failed: %s" % type(error).__name__)
            state = "failed"
        with self.lock:
            job.t_end = time.monotonic_ns()
            job.results = results if state == "done" else None
            if job.state == "running":
                job.state = state
                if state == "failed":
                    self.counters["failed_jobs"] += 1
            # Under the lock: a session that was closed has given its pipe
            # back, and the number may be another file's by now.
            if not session.closed:
                try:
                    os.write(session.wfd, b"j")
                except OSError:
                    pass                # a full pipe wakes all the same

    # ---- the requests ----

    def hello(self, request):
        name = request.get("session")
        if request.get("protocol") != PROTOCOL:
            raise Refused("protocol", "this adapter speaks version %d of the protocol" % PROTOCOL)
        if not isinstance(name, str):
            raise Refused("bad_request", "hello takes protocol and session")
        with self.lock:
            self.counters["hellos"] += 1
            session = self.sessions.get(name)
            resumed = session is not None
            if resumed:
                self.counters["resumes"] += 1
                for job in session.jobs.values():
                    job.reported = False        # told again: the client may not have heard
            else:
                session = self.sessions[name] = Session(name)
            session.users += 1
            session.t_left = None
        return session, dict(protocol=PROTOCOL, backend=self.simulator.name, qubits=self.simulator.qubits,
                             max_shots=self.args.max_shots, max_jobs=0, max_circuits=self.args.max_circuits,
                             physical=False, resumed=resumed)

    def prepare(self, request):
        name, text = request.get("circuit"), request.get("qasm")
        if not isinstance(name, str) or not isinstance(text, str):
            raise Refused("bad_request", "prepare takes circuit and qasm")
        with self.lock:
            self.counters["prepares"] += 1
            known = self.circuits.get(name)
        if known is not None:
            if known[0] != text:
                raise Refused("rejected", "another circuit has this name")
            return {}
        ready, slots = self.simulator.prepare(text)
        with self.lock:
            if len(self.circuits) >= self.args.max_prepared:
                self.circuits.pop(next(iter(self.circuits)))    # the oldest; it is given again if needed
            self.circuits[name] = (text, ready, slots)
        return {}

    def submit(self, session, request):
        tag, entries = request.get("tag"), request.get("entries")
        if not isinstance(tag, str) or not isinstance(entries, list):
            raise Refused("bad_request", "submit takes tag and entries")
        with self.lock:
            self.counters["submits"] += 1
            if tag in session.tags:
                self.counters["tag_hits"] += 1
                return dict(job="0x%x" % session.tags[tag])
            if not 1 <= len(entries) <= self.args.max_circuits:
                raise Refused("rejected", "a job of %d circuits, and this adapter takes 1 to %d" % (len(entries), self.args.max_circuits))
            taken = []
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("circuit"), str):
                    raise Refused("rejected", "an entry takes circuit, shots, key, attempt and params")
                known = self.circuits.get(entry["circuit"])
                if known is None:
                    raise Refused("unknown_circuit", "prepare it first")
                values, shots, key = entry.get("params"), entry.get("shots"), entry.get("key")
                if (not isinstance(values, list) or len(values) != len(known[2]) or
                        not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values) or
                        not isinstance(shots, int) or isinstance(shots, bool) or shots < 1 or
                        (self.args.max_shots and shots > self.args.max_shots) or not isinstance(key, str)):
                    raise Refused("rejected", "an entry takes circuit, shots, key, attempt, and as many params as the circuit has inputs")
                try:
                    key = int(key, 16)
                except ValueError:
                    raise Refused("rejected", "a key is written 0x and hexadecimal digits")
                taken.append((known[1], known[2], [float(v) for v in values], shots, key))
            job = Job(self.next_job, tag, taken)
            self.next_job += 1
            session.jobs[job.number] = job
            session.tags[tag] = job.number
            self.counters["jobs"] += 1
        self.pool.submit(self.work, session, job)
        return dict(job="0x%x" % job.number)

    def news(self, session, most):
        """The jobs of a session that ended and were not told yet."""
        told = []
        with self.lock:
            for job in session.jobs.values():
                if job.state != "running" and not job.reported and len(told) < most:
                    job.reported = True
                    told.append(dict(job="0x%x" % job.number, state=job.state))
        return told

    def find(self, session, request):
        try:
            return session.jobs.get(int(request.get("job"), 16))
        except (TypeError, ValueError):
            return None

    def results(self, session, request):
        now = time.monotonic_ns()
        with self.lock:
            job = self.find(session, request)
            if job is None or job.state != "done":
                raise Refused("unknown_job", "no job of that name has ended well here")
            del session.jobs[job.number]
            session.tags.pop(job.tag, None)
            self.counters["results"] += len(job.results)
        return dict(entries=[dict(since_start_ns=now - job.t_start, since_end_ns=now - job.t_end, counts=counts)
                             for counts in job.results])

    def cancel(self, session, request):
        with self.lock:
            self.counters["cancels"] += 1
            job = self.find(session, request)
            if job is None:
                raise Refused("unknown_job", "there is no job of that name here")
            job.cancel = True
            del session.jobs[job.number]
            session.tags.pop(job.tag, None)
        return {}

    def abandon(self, session, request):
        tag = request.get("tag")
        if not isinstance(tag, str):
            raise Refused("bad_request", "abandon takes tag")
        with self.lock:
            self.counters["abandons"] += 1
            number = session.tags.pop(tag, None)
            job = session.jobs.pop(number, None) if number is not None else None
            if job is not None:
                job.cancel = True
        # Which job went with the tag: the client may hold an end of it that is nobody's now.
        return dict(found=True, job="0x%x" % job.number) if job is not None else dict(found=False)

    def bye(self, session):
        with self.lock:
            self.counters["byes"] += 1
            self.sessions.pop(session.name, None)
            session.close()
        return {}

    def leave(self, session):
        """A connection that used a session is over."""
        with self.lock:
            session.users -= 1
            if session.users <= 0:
                session.t_left = time.monotonic()

    def drop_idle(self):
        """Sessions whose client went away without a word, long ago, and did not come back."""
        now = time.monotonic()
        with self.lock:
            for name in [n for n, s in self.sessions.items()
                         if s.users <= 0 and s.t_left is not None and now - s.t_left > self.args.idle_drop]:
                log("a session without a client is dropped")
                self.sessions.pop(name).close()

    # ---- one client ----

    def serve(self, conn):
        session = None
        buffer = b""
        waiting = None                  # (seq, deadline, most) while a wait has no answer yet

        def send(seq, ok, fields):
            message = dict(seq=seq, ok=ok)
            message.update(fields)
            conn.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode())

        try:
            while True:
                # A wait is answered when there is news, when its time is up, or when the client says wake.
                if session is not None and session.closed:
                    break               # another connection said goodbye for it
                if waiting is not None:
                    told = self.news(session, waiting[2])
                    left = waiting[1] - time.monotonic()
                    if told or left <= 0:
                        send(waiting[0], True, dict(jobs=told))
                        waiting = None
                        continue
                if b"\n" not in buffer:
                    watched = [conn] + ([session.rfd] if waiting is not None else [])
                    readable, _, _ = select.select(watched, [], [], left if waiting is not None else None)
                    if waiting is not None and session.rfd in readable:
                        try:
                            os.read(session.rfd, 4096)
                        except OSError:
                            pass
                    if conn in readable:
                        data = conn.recv(1 << 16)
                        if not data:
                            break
                        buffer += data
                        if len(buffer) > MAX_LINE:
                            break
                    continue
                line, buffer = buffer.split(b"\n", 1)
                try:
                    request = json.loads(line)
                    op, seq = request["op"], request.get("seq")
                except (ValueError, KeyError, TypeError):
                    send(None, False, dict(error="bad_request", why="a line that is not a request"))
                    continue
                if op == "wake":
                    with self.lock:
                        self.counters["wakes"] += 1
                    if waiting is not None:
                        send(waiting[0], True, dict(jobs=self.news(session, waiting[2])))
                        waiting = None
                    continue
                try:
                    if waiting is not None:
                        waiting = None
                        raise Refused("bad_request", "a request while another one waits for its answer")
                    if op == "hello":
                        if session is not None:
                            self.leave(session)
                            session = None
                        session, fields = self.hello(request)
                    elif op == "stats":
                        with self.lock:
                            fields = dict(self.counters)
                            fields.update(sessions=len(self.sessions), circuits=len(self.circuits),
                                          live_jobs=sum(len(s.jobs) for s in self.sessions.values()))
                    elif session is None:
                        raise Refused("bad_request", "say hello first")
                    elif op == "prepare":
                        fields = self.prepare(request)
                    elif op == "submit":
                        fields = self.submit(session, request)
                    elif op == "wait":
                        timeout, most = request.get("timeout_ms"), request.get("max")
                        if not isinstance(timeout, int) or not isinstance(most, int) or timeout < 0 or most < 1:
                            raise Refused("bad_request", "wait takes timeout_ms and max")
                        with self.lock:
                            self.counters["waits"] += 1
                        waiting = (seq, time.monotonic() + timeout / 1000.0, most)
                        continue
                    elif op == "results":
                        fields = self.results(session, request)
                    elif op == "cancel":
                        fields = self.cancel(session, request)
                    elif op == "abandon":
                        fields = self.abandon(session, request)
                    elif op == "bye":
                        fields = self.bye(session)
                        session = None
                        send(seq, True, fields)
                        break
                    else:
                        raise Refused("bad_request", "no such request")
                    send(seq, True, fields)
                except Refused as refused:
                    send(seq, False, dict(error=refused.error, why=refused.why))
                except OSError:
                    raise
                except Exception as error:      # a request of a shape nothing above expected
                    log("a request was not understood: %s" % type(error).__name__)
                    send(seq, False, dict(error="bad_request", why="the request is not of the protocol"))
        except OSError:
            pass                        # the client is gone
        finally:
            conn.close()
            if session is not None:
                self.leave(session)


def main():
    global ARGS
    parser = argparse.ArgumentParser(description="An adapter between PaRSEC's QPU device and Qiskit; this version runs "
                                     "the jobs on Qiskit Aer, a simulator, and reaches no quantum device.")
    parser.add_argument("--socket", required=True, help="path of the Unix-domain socket to listen on")
    parser.add_argument("--backend", default="aer", help="what runs the jobs: aer, the only one this version has")
    parser.add_argument("--method", default="automatic", help="the simulation method of Aer (default: automatic)")
    parser.add_argument("--device", default="none",
                        help="none (default): an ideal simulator; generic:<qubits>[:line|ring|full]: a made-up device "
                        "model, to whose gates and connectivity the circuits are transpiled, with its noise")
    parser.add_argument("--no-noise", action="store_true", help="with --device: its gates and connectivity, without its noise")
    parser.add_argument("--seed", type=int, default=1, help="seed of the samples, of the device model and of the transpiler (default 1)")
    parser.add_argument("--qubits", type=int, default=24, help="widest circuit taken (default 24, or the size of the device)")
    parser.add_argument("--max-circuits", type=int, default=64, help="most circuits in one job (default 64)")
    parser.add_argument("--max-shots", type=int, default=0, help="most shots of one circuit (default 0: no limit)")
    parser.add_argument("--max-prepared", type=int, default=4096, help="most circuits kept transpiled (default 4096)")
    parser.add_argument("--workers", type=int, default=2, help="jobs run at the same time (default 2)")
    parser.add_argument("--threads", type=int, default=1, help="threads one simulation may use (default 1)")
    parser.add_argument("--optimization-level", type=int, default=1, choices=(0, 1, 2, 3), help="of the transpiler (default 1)")
    parser.add_argument("--idle-drop", type=float, default=600.0,
                        help="seconds after which what a client left behind without a goodbye is dropped (default 600)")
    parser.add_argument("-v", "--verbose", action="store_true", help="say on stderr what happens (never what a message holds)")
    ARGS = parser.parse_args()
    if ARGS.backend != "aer":
        raise SystemExit("this version has one backend, aer: it reaches no quantum device, and '%s' is not built" % ARGS.backend)
    if len(os.fsencode(ARGS.socket)) > 100:
        raise SystemExit("the path of the socket is too long for a Unix-domain socket")

    adapter = Adapter(ARGS)

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    # What an earlier run left at that path is removed, if it is a socket: nothing else is.
    try:
        import stat
        if stat.S_ISSOCK(os.lstat(ARGS.socket).st_mode):
            os.unlink(ARGS.socket)
        else:
            raise SystemExit("there is a file at %s that is not a socket; it is left alone" % ARGS.socket)
    except FileNotFoundError:
        pass
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    mask = os.umask(0o077)              # the socket is for the user who started the adapter, and nobody else
    try:
        listener.bind(ARGS.socket)
    finally:
        os.umask(mask)
    listener.listen(16)
    log("listening on %s; jobs run on %s (%s)" % (ARGS.socket, adapter.simulator.name, adapter.simulator.versions))
    try:
        while True:
            readable, _, _ = select.select([listener], [], [], 30.0)
            adapter.drop_idle()
            if readable:
                conn, _ = listener.accept()
                threading.Thread(target=adapter.serve, args=(conn,), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()
        try:
            os.unlink(ARGS.socket)
        except FileNotFoundError:
            pass
        adapter.pool.shutdown(wait=False, cancel_futures=True)
    log("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
