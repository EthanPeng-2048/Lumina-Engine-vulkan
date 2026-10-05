#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Lumina-Engine lossless GGUF unpacker (container v1 + v2).
#
# Reverses tools/lumina_gguf_pack.py and reproduces the original .gguf
# byte-for-byte.  Used for round-trip verification and as the specification
# for the in-engine C++ decoder (src/lumina-gguf.cpp).
# ---------------------------------------------------------------------------
import hashlib
import os
import struct
import sys
import zlib

import numpy as np

MAGIC = b"LGGC"
GGML_TYPE_F32 = 0
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14
TYPE_SIZE = {GGML_TYPE_F32: 4, GGML_TYPE_Q4_K: 144, GGML_TYPE_Q6_K: 210}
BLOCK_SIZE = {GGML_TYPE_F32: 1, GGML_TYPE_Q4_K: 256, GGML_TYPE_Q6_K: 256}


class Inflater:
    """Streaming zlib decompressor that can yield exactly n bytes at a time."""
    def __init__(self, f):
        self.f = f
        self.decomp = zlib.decompressobj(15)

    def read_exact(self, n):
        out = bytearray()
        while len(out) < n:
            data = self.decomp.unconsumed_tail
            if not data:
                data = self.f.read(8 * 1024 * 1024)
                if not data:
                    raise IOError("unexpected end of deflate stream")
            out += self.decomp.decompress(data, n - len(out))
        return bytes(out)


def _merge_nibbles(lo, hi, nb, n):
    lo = np.frombuffer(lo, dtype=np.uint8).reshape(nb, n // 2)
    hi = np.frombuffer(hi, dtype=np.uint8).reshape(nb, n // 2)
    even = (lo & 0x0F) | ((hi & 0x0F) << 4)
    odd = (lo >> 4) | ((hi >> 4) << 4)
    qs = np.empty((nb, n), dtype=np.uint8)
    qs[:, 0::2] = even
    qs[:, 1::2] = odd
    return qs


# --------------------------- v1 -------------------------------------------
def merge_q4k_v1(streams, nb):
    d, dmin, sc, lo, hi = streams
    out = np.empty((nb, 144), dtype=np.uint8)
    out[:, 0:2] = np.frombuffer(d, dtype=np.uint8).reshape(nb, 2)
    out[:, 2:4] = np.frombuffer(dmin, dtype=np.uint8).reshape(nb, 2)
    out[:, 4:16] = np.frombuffer(sc, dtype=np.uint8).reshape(nb, 12)
    out[:, 16:144] = _merge_nibbles(lo, hi, nb, 128)
    return out.tobytes()


def merge_q6k_v1(streams, nb):
    d, sc, lo, hi, qh = streams
    out = np.empty((nb, 210), dtype=np.uint8)
    out[:, 0:128] = _merge_nibbles(lo, hi, nb, 128)
    out[:, 128:192] = np.frombuffer(qh, dtype=np.uint8).reshape(nb, 64)
    out[:, 192:208] = np.frombuffer(sc, dtype=np.uint8).reshape(nb, 16)
    out[:, 208:210] = np.frombuffer(d, dtype=np.uint8).reshape(nb, 2)
    return out.tobytes()


# --------------------------- v2 -------------------------------------------
def merge_q4k_v2(streams, nb):
    dlo, dhi, mlo, mhi, scb, mnb, nib = streams
    sc = np.frombuffer(scb, dtype=np.uint8).reshape(nb, 8)
    mn = np.frombuffer(mnb, dtype=np.uint8).reshape(nb, 8)
    order = np.argsort(((sc.astype(np.int64) << 6) | mn).reshape(-1), kind="stable")

    v = np.frombuffer(nib, dtype=np.uint8).reshape(-1)
    seq = np.empty(nb * 8 * 32, dtype=np.uint8)
    seq[0::2] = v & 0x0F
    seq[1::2] = v >> 4
    vals = np.empty((nb * 8, 32), dtype=np.uint8)
    vals[order] = seq.reshape(nb * 8, 32)
    vals = vals.reshape(nb, 8, 32)

    out = np.empty((nb, 144), dtype=np.uint8)
    out[:, 0] = np.frombuffer(dlo, dtype=np.uint8)
    out[:, 1] = np.frombuffer(dhi, dtype=np.uint8)
    out[:, 2] = np.frombuffer(mlo, dtype=np.uint8)
    out[:, 3] = np.frombuffer(mhi, dtype=np.uint8)

    for j in range(4):
        out[:, 4 + j] = (sc[:, j] & 63) | (((sc[:, j + 4] >> 4) & 3) << 6)
        out[:, 8 + j] = (mn[:, j] & 63) | (((mn[:, j + 4] >> 4) & 3) << 6)
        out[:, 12 + j] = (sc[:, j + 4] & 0x0F) | ((mn[:, j + 4] & 0x0F) << 4)

    for g in range(4):
        out[:, 16 + g * 32: 16 + (g + 1) * 32] = vals[:, 2 * g, :] | (vals[:, 2 * g + 1, :] << 4)
    return out.tobytes()


def merge_q6k_v2(streams, nb):
    dlo, dhi, sce, sco, lo, hi, qh = streams
    out = np.empty((nb, 210), dtype=np.uint8)
    out[:, 208] = np.frombuffer(dlo, dtype=np.uint8)
    out[:, 209] = np.frombuffer(dhi, dtype=np.uint8)
    sc = np.empty((nb, 16), dtype=np.uint8)
    sc[:, 0::2] = np.frombuffer(sce, dtype=np.uint8).reshape(nb, 8)
    sc[:, 1::2] = np.frombuffer(sco, dtype=np.uint8).reshape(nb, 8)
    out[:, 192:208] = sc
    out[:, 128:192] = np.frombuffer(qh, dtype=np.uint8).reshape(nb, 64)
    out[:, 0:128] = _merge_nibbles(lo, hi, nb, 128)
    return out.tobytes()


def _tensor_lens(version, tt, nbytes):
    if tt == GGML_TYPE_Q4_K:
        nb = nbytes // 144
        if version >= 2:
            return [nb, nb, nb, nb, nb * 8, nb * 8, nb * 128], merge_q4k_v2, nb
        return [nb * 2, nb * 2, nb * 12, nb * 64, nb * 64], merge_q4k_v1, nb
    if tt == GGML_TYPE_Q6_K:
        nb = nbytes // 210
        if version >= 2:
            return [nb, nb, nb * 8, nb * 8, nb * 64, nb * 64, nb * 64], merge_q6k_v2, nb
        return [nb * 2, nb * 16, nb * 64, nb * 64, nb * 64], merge_q6k_v1, nb
    return [nbytes], None, nbytes


def _read_container(f):
    if f.read(4) != MAGIC:
        raise ValueError("not a Lumina compressed GGUF")
    version, flags, orig_size, data_offset, n_tensors, _ = struct.unpack("<IIQQII", f.read(32))
    if version not in (1, 2):
        raise ValueError("unsupported container version %d" % version)
    tensors = []
    for _ in range(n_tensors):
        tt, _, nbytes = struct.unpack("<IIQ", f.read(16))
        tensors.append((tt, nbytes))
    return version, orig_size, data_offset, tensors


def _read_tensor_streams(inflater, version, tt, nbytes):
    lens, merge, nb = _tensor_lens(version, tt, nbytes)
    streams = [inflater.read_exact(n) for n in lens]
    return merge(streams, nb) if merge else streams[0]


def unpack(src, dst):
    with open(src, "rb") as f:
        version, orig_size, data_offset, tensors = _read_container(f)
        inflater = Inflater(f)
        with open(dst, "wb") as out:
            remaining = data_offset
            while remaining > 0:
                n = min(remaining, 16 * 1024 * 1024)
                buf = inflater.read_exact(n)
                out.write(buf)
                remaining -= len(buf)
            for tt, nbytes in tensors:
                if nbytes == 0:
                    continue
                out.write(_read_tensor_streams(inflater, version, tt, nbytes))
    return os.path.getsize(dst), orig_size


def verify(packed, original):
    """Decompress `packed` and compare byte-for-byte against `original`."""
    with open(packed, "rb") as f:
        version, orig_size, data_offset, tensors = _read_container(f)
        inflater = Inflater(f)

        h_in = hashlib.sha256()
        h_out = hashlib.sha256()

        with open(original, "rb") as g:
            remaining = data_offset
            while remaining > 0:
                n = min(remaining, 16 * 1024 * 1024)
                buf = inflater.read_exact(n)
                h_in.update(g.read(len(buf)))
                h_out.update(buf)
                remaining -= len(buf)
            for tt, nbytes in tensors:
                if nbytes == 0:
                    continue
                buf = _read_tensor_streams(inflater, version, tt, nbytes)
                h_in.update(g.read(len(buf)))
                h_out.update(buf)
            extra = g.read(1)

    ok = (h_in.digest() == h_out.digest()) and not extra and orig_size == os.path.getsize(original)
    return ok, h_in.hexdigest(), h_out.hexdigest()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        got, want = unpack(sys.argv[1], sys.argv[2])
        print("unpacked to %s: %d bytes (expected %d)" % (sys.argv[2], got, want))
    elif len(sys.argv) == 4 and sys.argv[1] == "verify":
        ok, a, b = verify(sys.argv[2], sys.argv[3])
        print("verify:", "OK" if ok else "FAIL", a, b)
        sys.exit(0 if ok else 1)
    else:
        print("usage: %s <packed.gguf.lumina> <output.gguf>" % sys.argv[0])
        print("       %s verify <packed.gguf.lumina> <original.gguf>" % sys.argv[0])
        sys.exit(1)
