#
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#
"""The half of an adapter that is the protocol, for the adapter programs here.

PaRSEC's adapter backend (--mca device_qpu_backend adapter) is the client of
an adapter: it sends circuits as OpenQASM 3 and gets counts back, over a
Unix-domain socket, one line of JSON per message (tools/qpu/README.md has the
protocol). This module is the side of the adapter in that exchange: the
sessions, the jobs, the requests and the socket. What runs a job is not here.
A program gives it a simulator, an object with

    name, versions   what it is, for hello and for the log
    qubits           the widest circuit it takes
    prepare(text)    a circuit from its OpenQASM 3 text, ready to run, and one
                     entry per input of the text: (ready, slots). Raises
                     Refused for a text it does not take.
    run(ready, slots, values, shots, key)
                     the counts of one evaluation, as a list of
                     [bitstring as "0x...", count]

and the number of jobs that may run at the same time. Every adapter built on
this says in hello that it drives no quantum device: the module has no other
answer, and no program here has code that reaches one.

This is not a program: parsec_qpu_adapter_qiskit.py and
parsec_qpu_adapter_cudaq.py are.
"""

import json
import os
import re
import select
import signal
import socket
import stat
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

PROTOCOL = 1
MAX_LINE = 256 << 20
KEY = re.compile(r"0x[0-9a-fA-F]{1,16}")       # 64 bits, as the protocol writes them


VERBOSE = False


def log(message):
    """What happens, never what a message holds."""
    if VERBOSE:
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
    def __init__(self, args, simulator, workers):
        self.args = args
        self.simulator = simulator
        self.lock = threading.Lock()
        self.sessions = {}
        self.circuits = {}              # name -> (text, ready circuit, slots)
        self.next_job = 1
        self.pool = ThreadPoolExecutor(max_workers=workers)
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
                if KEY.fullmatch(key) is None:
                    raise Refused("rejected", "a key is written 0x and one to sixteen hexadecimal digits")
                key = int(key, 16)
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
                        # The wait gets its answer first, as at a wake: every request gets one.
                        send(waiting[0], True, dict(jobs=self.news(session, waiting[2])))
                        waiting = None
                        raise Refused("bad_request", "a request while another one waited for its answer")
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


def add_arguments(parser, max_prepared=4096):
    """The options every adapter program has."""
    parser.add_argument("--socket", required=True, help="path of the Unix-domain socket to listen on")
    parser.add_argument("--max-circuits", type=int, default=64, help="most circuits in one job (default 64)")
    parser.add_argument("--max-shots", type=int, default=0, help="most shots of one circuit (default 0: no limit)")
    parser.add_argument("--max-prepared", type=int, default=max_prepared,
                        help="most circuits kept ready to run; the oldest goes, and is asked for again when a job needs it "
                        "(default %d)" % max_prepared)
    parser.add_argument("--idle-drop", type=float, default=600.0,
                        help="seconds after which what a client left behind without a goodbye is dropped (default 600)")
    parser.add_argument("-v", "--verbose", action="store_true", help="say on stderr what happens (never what a message holds)")


def check_arguments(args):
    """Before a simulator is made, which may take seconds."""
    global VERBOSE
    VERBOSE = args.verbose
    if len(os.fsencode(args.socket)) > 100:
        raise SystemExit("the path of the socket is too long for a Unix-domain socket")


def run(args, simulator, workers):
    """Listen on the socket and serve the clients until the program is told to stop."""
    adapter = Adapter(args, simulator, workers)

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    # What an earlier run left at that path is removed, if it is a socket that
    # nobody listens on: nothing else is.
    try:
        if not stat.S_ISSOCK(os.lstat(args.socket).st_mode):
            raise SystemExit("there is a file at %s that is not a socket; it is left alone" % args.socket)
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(2.0)
        try:
            probe.connect(args.socket)
        except OSError:
            os.unlink(args.socket)      # left behind
        else:
            raise SystemExit("an adapter listens at %s already; it is left alone" % args.socket)
        finally:
            probe.close()
    except FileNotFoundError:
        pass
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    mask = os.umask(0o077)              # the socket is for the user who started the adapter, and nobody else
    try:
        listener.bind(args.socket)
    finally:
        os.umask(mask)
    listener.listen(16)
    mine = os.lstat(args.socket)        # at the end, the path is removed only if it is still this socket
    log("listening on %s; jobs run on %s (%s)" % (args.socket, adapter.simulator.name, adapter.simulator.versions))
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
            there = os.lstat(args.socket)
            if (there.st_dev, there.st_ino) == (mine.st_dev, mine.st_ino):
                os.unlink(args.socket)
        except FileNotFoundError:
            pass
        # No job starts any more, and the one that runs ends with the evaluation it is in.
        with adapter.lock:
            for session in adapter.sessions.values():
                for job in session.jobs.values():
                    job.cancel = True
        adapter.pool.shutdown(wait=False, cancel_futures=True)
    log("stopped")
    return 0
