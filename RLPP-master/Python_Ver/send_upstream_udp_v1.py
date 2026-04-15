#!/usr/bin/env python3
"""
Send upstream_frame_v1 UDP datagrams to drive RLPP_realtime_udp_infer.

One datagram = one time bin:
  magic/version/seq/tbin/Nx + Nx float64 values (little-endian).

Example:
  python Python_Ver/send_upstream_udp_v1.py --nx 32 --bins 2000 --port 46000 --hz 200
"""

from __future__ import annotations

import argparse
import socket
import struct
import time
from typing import List

MAGIC = 0x31565055  # 'UPV1' LE
VER = 1


def pack_frame(seq: int, tbin: int, x: List[float]) -> bytes:
    nx = len(x)
    hdr = struct.pack("<IHHQii", MAGIC, VER, 0, seq, tbin, nx)
    body = struct.pack("<" + "d" * nx, *x)
    return hdr + body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=46000)
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--bins", type=int, default=2000)
    ap.add_argument("--hz", type=float, default=200.0, help="bins per second (200Hz => 5ms)")
    ap.add_argument("--mode", choices=("zeros", "impulse", "rand01"), default="zeros")
    args = ap.parse_args()

    if args.nx <= 0:
        raise SystemExit("nx must be positive")
    if args.hz <= 0:
        raise SystemExit("hz must be positive")

    period = 1.0 / float(args.hz)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.host, int(args.port))

    t0 = time.perf_counter()
    next_t = t0
    for i in range(args.bins):
        tbin = i + 1
        seq = i
        if args.mode == "zeros":
            x = [0.0] * args.nx
        elif args.mode == "impulse":
            x = [0.0] * args.nx
            x[i % args.nx] = 1.0
        else:
            # simple deterministic pseudo-random without numpy dependency
            x = [((1103515245 * (i + 1) + 12345 + j) & 0xFFFF) / 65535.0 for j in range(args.nx)]

        pkt = pack_frame(seq, tbin, x)
        sock.sendto(pkt, dest)

        next_t += period
        now = time.perf_counter()
        dt = next_t - now
        if dt > 0:
            time.sleep(dt)

    sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

