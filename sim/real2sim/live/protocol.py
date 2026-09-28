"""The wire format between the live sim server and whoever drives it.

The server runs natively in an engine's own interpreter (the mjlab uv env for MuJoCo,
the Isaac conda env for Isaac); the lerobot side runs in the Docker image. They share
a Unix socket in the bind-mounted repo (sim/outputs/real2sim/live/sim.sock), which works
across the container boundary with no port publishing and no network.

A message is a JSON header plus raw array buffers:

    u32 header length | header JSON | buffer 0 | buffer 1 | ...

Arrays in the object are replaced by {"__nd__": i, "dtype": ..., "shape": [...]} and
their bytes follow the header in order. Not pickle: the two ends run different Pythons
and different numpy majors (1.26 in the Isaac env, 2.x elsewhere), and numpy 2 pickles
reference numpy._core, which numpy 1.26 cannot import.
"""

from __future__ import annotations

import json
import socket
import struct

import numpy as np

_LEN = struct.Struct("!I")
_BUF = struct.Struct("!Q")


def _pack(obj, bufs: list):
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        bufs.append(a)
        return {"__nd__": len(bufs) - 1, "dtype": a.dtype.str, "shape": list(a.shape)}
    if isinstance(obj, dict):
        return {str(k): _pack(v, bufs) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_pack(v, bufs) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _unpack(obj, bufs: list):
    if isinstance(obj, dict):
        if "__nd__" in obj:
            return np.frombuffer(bufs[obj["__nd__"]], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
        return {k: _unpack(v, bufs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unpack(v, bufs) for v in obj]
    return obj


def send(sock: socket.socket, obj) -> None:
    bufs: list = []
    head = json.dumps(_pack(obj, bufs)).encode()
    sizes = b"".join(_BUF.pack(b.nbytes) for b in bufs)
    sock.sendall(_LEN.pack(len(head)) + head + _LEN.pack(len(bufs)) + sizes)
    for b in bufs:  # array bytes straight from their memory, no joined copy
        sock.sendall(memoryview(b).cast("B"))


def _read(sock: socket.socket, n: int) -> bytearray:
    out = bytearray(n)
    mv, got = memoryview(out), 0
    while got < n:
        k = sock.recv_into(mv[got:], n - got)
        if not k:
            raise ConnectionError("peer closed the connection")
        got += k
    return out


def recv(sock: socket.socket):
    (hl,) = _LEN.unpack(_read(sock, _LEN.size))
    head = json.loads(_read(sock, hl))
    (nb,) = _LEN.unpack(_read(sock, _LEN.size))
    raw = _read(sock, _BUF.size * nb) if nb else b""
    sizes = [_BUF.unpack_from(raw, i * _BUF.size)[0] for i in range(nb)]
    bufs = [_read(sock, n) for n in sizes]  # bytearrays: np.frombuffer views them writable, no copy
    return _unpack(head, bufs)


def call(sock: socket.socket, op: str, **kw):
    """One request/response. Server-side errors come back as {"error": str} and raise here."""
    send(sock, {"op": op, **kw})
    r = recv(sock)
    if isinstance(r, dict) and "error" in r:
        raise RuntimeError(f"sim server: {op}: {r['error']}")
    return r


def connect(path: str, timeout_s: float = 10.0) -> socket.socket:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    s.connect(path)
    return s
