#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Lumina-Engine lossless GGUF packer (container v2)
#
#   Shrinks a .gguf file WITHOUT quantization and WITHOUT pruning.
#
# By default the packer writes `<input>.lumina`, i.e. `A.gguf` ->
# `A.gguf.lumina`, so the original model stays untouched and the two are easy
# to tell apart.  Lumina-Engine detects the container by its `LGGC` magic and
# loads either extension transparently.
#
# It never changes a single weight value: every tensor is byte-for-byte
# reconstructed at load time.  The pipeline is a reversible, k-quant aware
# de-interleaving followed by DEFLATE:
#
#   * Q4_K: split each 144-byte block into fp16 byte-planes (d, dmin), the
#     8+8 six-bit scales/mins, and the 4-bit quants.  The quants are grouped
#     by sub-block and stably sorted by the context key (scale << 6 | min),
#     which moves each sub-block next to others with the same distribution.
#     This alone recovers ~1.2% of the quant stream over plain field split.
#   * Q6_K: fp16 byte-plane for d, even/odd split for the int8 scales, and the
#     usual low/high nibble + qh split.
#   * F32: stored whole.
#
# All streams are concatenated in a single zlib/DEFLATE stream, so the engine
# can keep using its bundled miniz decoder.
#
# Container layout (little endian):
#
#   char   magic[4]      = "LGGC"
#   u32    version       = 2
#   u32    flags         = 0
#   u64    orig_size     total size of the original .gguf
#   u64    data_offset   start of the tensor data section in the .gguf
#   u32    n_tensors
#   u32    reserved
#   { u32 type; u32 reserved; u64 nbytes } * n_tensors
#   <zlib stream: original header || split(tensor_0) || split(tensor_1) ... >
#
# The original header (everything before data_offset), including the GGUF
# metadata and tensor table, is stored verbatim so the normal GGUF parser
# accepts the reconstructed file unchanged.
# ---------------------------------------------------------------------------
import os
import struct
import sys
import zlib

import numpy as np

MAGIC = b"LGGC"
VERSION = 2

GGML_TYPE_F32 = 0
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14

TYPE_SIZE = {GGML_TYPE_F32: 4, GGML_TYPE_Q4_K: 144, GGML_TYPE_Q6_K: 210}
BLOCK_SIZE = {GGML_TYPE_F32: 1, GGML_TYPE_Q4_K: 256, GGML_TYPE_Q6_K: 256}


def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def skip_value(f, t):
    if t in (0, 1, 7):   f.read(1)
    elif t == 2:         f.read(2)
    elif t == 3:         f.read(2)
    elif t == 4:         f.read(4)
    elif t == 5:         f.read(4)
    elif t == 6:         f.read(4)
    elif t == 8:         read_string(f)
    elif t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            skip_value(f, et)
    elif t == 10:        f.read(8)
    elif t == 11:        f.read(8)
    elif t == 12:        f.read(8)
    else:
        raise ValueError("unknown gguf type %d" % t)


def parse_gguf(f):
    """Read the header/metadata/tensor table, return (data_offset, tensors)."""
    f.seek(0)
    magic = f.read(4)
    if magic == MAGIC:
        raise ValueError("input is already a Lumina compressed GGUF")
    if magic != b"GGUF":
        raise ValueError("not a GGUF file (magic=%r)" % magic)
    ver = struct.unpack("<I", f.read(4))[0]
    if ver != 3:
        raise ValueError("unsupported GGUF version %d" % ver)
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    alignment = 32
    for _ in range(n_kv):
        key = read_string(f)
        t = struct.unpack("<I", f.read(4))[0]
        if key == "general.alignment" and t == 4:
            alignment = struct.unpack("<I", f.read(4))[0]
        else:
            skip_value(f, t)
    tensors = []
    for _ in range(n_tensors):
        name = read_string(f)
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        tt = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        n = 1
        for d in dims:
            n *= d
        if tt not in TYPE_SIZE:
            raise ValueError("unsupported tensor type %d for '%s'" % (tt, name))
        nbytes = (n // BLOCK_SIZE[tt]) * TYPE_SIZE[tt]
        tensors.append((name, tt, off, nbytes))
    pos = f.tell()
    if n_tensors > 0:
        pos = (pos + alignment - 1) // alignment * alignment
    return pos, tensors


def _split_nibbles(qs):
    """qs: (nb, N) uint8 -> packed (lo, hi) streams, each nb*N/2 bytes."""
    even = qs[:, 0::2]
    odd = qs[:, 1::2]
    lo = ((even & 0x0F) | ((odd & 0x0F) << 4)).astype(np.uint8)
    hi = ((even >> 4) | ((odd >> 4) << 4)).astype(np.uint8)
    return lo.tobytes(), hi.tobytes()


def _q4_scales(a):
    """a: (nb,144) uint8 -> sc (nb,8), mn (nb,8), each a 6-bit field."""
    qq = a[:, 4:16]
    nb = a.shape[0]
    sc = np.empty((nb, 8), np.uint8)
    mn = np.empty((nb, 8), np.uint8)
    for j in range(8):
        if j < 4:
            sc[:, j] = qq[:, j] & 63
            mn[:, j] = qq[:, j + 4] & 63
        else:
            sc[:, j] = (qq[:, j + 4] & 0x0F) | (((qq[:, j - 4] >> 6) & 3) << 4)
            mn[:, j] = (qq[:, j + 4] >> 4) | (((qq[:, j] >> 6) & 3) << 4)
    return sc, mn


def split_q4k(data):
    """144-byte blocks -> scales, byte-planes and context-sorted nibbles."""
    nb = len(data) // 144
    a = np.frombuffer(data, dtype=np.uint8).reshape(nb, 144)
    sc, mn = _q4_scales(a)

    vals = np.empty((nb, 8, 32), dtype=np.uint8)
    for s in range(8):
        grp = a[:, 16 + (s // 2) * 32: 16 + (s // 2) * 32 + 32]
        vals[:, s, :] = (grp & 0x0F) if s % 2 == 0 else (grp >> 4)

    key = (sc.astype(np.int64) << 6) | mn
    order = np.argsort(key.reshape(-1), kind="stable")
    v = vals.reshape(nb * 8, 32)[order].reshape(-1)
    nib = ((v[0::2]) | (v[1::2] << 4)).astype(np.uint8).tobytes()

    return [a[:, 0].tobytes(), a[:, 1].tobytes(),
            a[:, 2].tobytes(), a[:, 3].tobytes(),
            sc.tobytes(), mn.tobytes(), nib]


def split_q6k(data):
    """210-byte blocks -> fp16 byte-plane, even/odd scales, ql/ql/qh split."""
    nb = len(data) // 210
    a = np.frombuffer(data, dtype=np.uint8).reshape(nb, 210)
    lo, hi = _split_nibbles(a[:, 0:128])
    sc = a[:, 192:208]
    return [a[:, 208].tobytes(), a[:, 209].tobytes(),
            sc[:, 0::2].tobytes(), sc[:, 1::2].tobytes(),
            lo, hi, a[:, 128:192].tobytes()]


def split_tensor(tt, data):
    if tt == GGML_TYPE_Q4_K:
        return split_q4k(data)
    if tt == GGML_TYPE_Q6_K:
        return split_q6k(data)
    return [data]


def pack(src, dst, level=9):
    with open(src, "rb") as f:
        data_offset, tensors = parse_gguf(f)
        orig_size = os.path.getsize(src)

        co = zlib.compressobj(level, zlib.DEFLATED, 15)  # zlib format

        with open(dst, "wb") as out:
            out.write(MAGIC)
            out.write(struct.pack("<IIQQII", VERSION, 0, orig_size, data_offset,
                                  len(tensors), 0))
            for _, tt, _, nbytes in tensors:
                out.write(struct.pack("<IIQ", tt, 0, nbytes))

            # header region (metadata + tensor table)
            f.seek(0)
            remaining = data_offset
            while remaining > 0:
                chunk = f.read(min(remaining, 8 * 1024 * 1024))
                if not chunk:
                    raise IOError("unexpected EOF while reading header")
                out.write(co.compress(chunk))
                remaining -= len(chunk)

            # tensor data, de-interleaved per tensor
            for name, tt, off, nbytes in tensors:
                f.seek(data_offset + off)
                data = f.read(nbytes)
                if len(data) != nbytes:
                    raise IOError("unexpected EOF in tensor '%s'" % name)
                for stream in split_tensor(tt, data):
                    out.write(co.compress(stream))
            out.write(co.flush())
    return orig_size


def default_output(src):
    """A.gguf -> A.gguf.lumina (keeps the original file intact)."""
    return src + ".lumina"


def main():
    argv = sys.argv[1:]
    if not argv or len(argv) > 3:
        print("usage: %s <input.gguf> [output.gguf.lumina] [level]" % sys.argv[0])
        print("       default output: <input.gguf>.lumina")
        return 1
    src = argv[0]
    dst = argv[1] if len(argv) >= 2 else default_output(src)
    level = int(argv[2]) if len(argv) >= 3 else 9
    orig = pack(src, dst, level)
    new = os.path.getsize(dst)
    print("packed %s" % src)
    print("  original : %d bytes (%.2f GiB)" % (orig, orig / 1024 ** 3))
    print("  packed   : %d bytes (%.2f GiB)" % (new, new / 1024 ** 3))
    print("  ratio    : %.2f%%  (saved %.1f MiB)" % (new / orig * 100.0, (orig - new) / 1024 ** 2))
    print("  output   : %s" % dst)
    print("  original left untouched; load either file directly with Lumina-Engine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
