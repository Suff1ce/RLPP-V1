#!/usr/bin/env python3
"""
UDP loopback test (same role as RLPP_hardware_udp_loopback): one datagram per v1 frame to 127.0.0.1.
Requires a free UDP port; matches hardware_frame_v1 / verify_hardware_trace_v1 layout.
"""

from __future__ import annotations

import argparse
import socket
import struct
import threading
import time

MAGIC_LE = 0x31504C52


def unpack_frame(data: bytes, off: int) -> tuple[int, tuple] | None:
    """Return (consumed, (seq, tbin, valid, lab, logits)) or None."""
    if off + 32 > len(data):
        return None
    magic, ver, flags, seq, tbin, valid, lab, k = struct.unpack_from("<IHHQiiii", data, off)
    if magic != MAGIC_LE or ver != 1 or k <= 0:
        return None
    need = 32 + 8 * k
    if off + need > len(data):
        return None
    logits = struct.unpack_from("<" + "d" * k, data, off + 32)
    return need, (seq, tbin, valid, lab, k, logits)


def count_frames(blob: bytes) -> int:
    off = 0
    n = 0
    while off < len(blob):
        r = unpack_frame(blob, off)
        if r is None:
            break
        consumed, _ = r
        off += consumed
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, help="hardware_trace_v1.bin")
    ap.add_argument("--port", type=int, default=45123)
    ap.add_argument("--first-timeout", type=float, default=30.0)
    ap.add_argument("--per-frame-timeout", type=float, default=5.0)
    args = ap.parse_args()

    with open(args.trace, "rb") as f:
        trace = f.read()

    nframes = count_frames(trace)
    if nframes == 0 and len(trace) > 0:
        print("No valid frames")
        return 1

    print(f"frames={nframes} bytes={len(trace)} port={args.port}")

    recv_count = [0]

    def receiver() -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", args.port))
        s.settimeout(args.first_timeout)
        got = 0
        try:
            while got < nframes:
                if got > 0:
                    s.settimeout(args.per_frame_timeout)
                data, _addr = s.recvfrom(65536)
                off = 0
                r = unpack_frame(data, off)
                if r is None:
                    print("recv unpack failed")
                    return
                consumed, _ = r
                if consumed != len(data):
                    print("extra bytes in datagram")
                    return
                got += 1
        except OSError as e:
            print("recv error:", e)
            return
        finally:
            s.close()
        recv_count[0] = got

    t = threading.Thread(target=receiver, daemon=False)
    t.start()
    time.sleep(0.2)

    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = ("127.0.0.1", args.port)
    off = 0
    while off < len(trace):
        r = unpack_frame(trace, off)
        if r is None:
            print("send: bad frame at", off)
            send_sock.close()
            t.join()
            return 1
        consumed, _ = r
        chunk = trace[off : off + consumed]
        send_sock.sendto(chunk, dest)
        off += consumed
    send_sock.close()

    t.join()

    if recv_count[0] != nframes:
        print(f"UDP_LOOPBACK FAIL: got {recv_count[0]} expected {nframes}")
        return 1
    print("UDP_LOOPBACK OK (Python)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
