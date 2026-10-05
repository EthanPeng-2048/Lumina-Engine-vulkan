#include "lumina-gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "miniz/miniz.h"

// keep this translation unit self-contained (it only needs a log sink)
#ifndef LUMINA_LOG_INFO
#define LUMINA_LOG_INFO(...) do { if (std::getenv("LUMINA_VERBOSE") != nullptr) { std::fprintf(stderr, __VA_ARGS__); } } while (0)
#endif

#define LUMINA_MAGIC "LGGC"
#define LUMINA_VERSION_MIN 1u
#define LUMINA_VERSION_MAX 2u

#define GGML_TYPE_Q4_K 12u
#define GGML_TYPE_Q6_K 14u

namespace {

struct lumina_tensor {
    uint32_t type;
    uint64_t nbytes;
};

static bool read_exact(FILE * f, void * dst, size_t n) {
    return n == 0 || fread(dst, 1, n, f) == n;
}

// Streaming zlib inflater fed from a FILE.  Can produce an exact number of
// bytes, which lets us rebuild the file tensor by tensor without ever holding
// the whole (multi-GiB) model in memory.
struct lumina_inflater {
    FILE * f = nullptr;
    mz_stream strm{};
    std::vector<unsigned char> in;
    bool eof = false;
    bool ok  = false;

    bool init(FILE * file) {
        f = file;
        in.resize(4 * 1024 * 1024);
        std::memset(&strm, 0, sizeof(strm));
        if (mz_inflateInit(&strm) != MZ_OK) {
            return false;
        }
        ok = true;
        return true;
    }

    bool read(void * dst, size_t n) {
        unsigned char * out = static_cast<unsigned char *>(dst);
        size_t got = 0;
        while (got < n) {
            if (strm.avail_in == 0 && !eof) {
                const size_t r = fread(in.data(), 1, in.size(), f);
                if (r == 0) {
                    eof = true;
                }
                strm.next_in  = in.data();
                strm.avail_in = static_cast<unsigned int>(r);
            }

            const size_t want = std::min<size_t>(n - got, static_cast<size_t>(UINT32_MAX));
            strm.next_out  = out + got;
            strm.avail_out = static_cast<unsigned int>(want);

            const int ret = mz_inflate(&strm, MZ_NO_FLUSH);
            const size_t produced = want - strm.avail_out;
            got += produced;

            if (ret == MZ_STREAM_END) {
                return got == n;
            }
            if (ret != MZ_OK) {
                // no forward progress and no more input => truncated stream
                if (ret == MZ_BUF_ERROR && produced == 0 && eof) {
                    return false;
                }
                if (ret != MZ_BUF_ERROR) {
                    return false;
                }
            }
            if (produced == 0 && eof && strm.avail_in == 0) {
                return false;
            }
        }
        return true;
    }

    ~lumina_inflater() {
        if (ok) {
            mz_inflateEnd(&strm);
        }
    }
};

static uint64_t fnv1a(const std::string & s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

// ---------------------------------------------------------------------------
// v1 decoding
// ---------------------------------------------------------------------------

static void unpack_q4_k_v1(const unsigned char * buf, unsigned char * out, uint64_t nbytes) {
    const uint64_t nb = nbytes / 144;
    const unsigned char * sd   = buf;
    const unsigned char * sdmn = buf + nb * 2;
    const unsigned char * ssc  = buf + nb * 4;
    const unsigned char * slo  = buf + nb * 16;
    const unsigned char * shi  = buf + nb * 80;
    for (uint64_t i = 0; i < nb; ++i) {
        unsigned char * o = out + i * 144;
        o[0] = sd[2 * i];
        o[1] = sd[2 * i + 1];
        o[2] = sdmn[2 * i];
        o[3] = sdmn[2 * i + 1];
        std::memcpy(o + 4, ssc + i * 12, 12);
        for (int j = 0; j < 64; ++j) {
            const unsigned char a = slo[i * 64 + j];
            const unsigned char b = shi[i * 64 + j];
            o[16 + 2 * j]     = (a & 0x0F) | ((b & 0x0F) << 4);
            o[16 + 2 * j + 1] = (a >> 4)   | ((b >> 4) << 4);
        }
    }
}

static void unpack_q6_k_v1(const unsigned char * buf, unsigned char * out, uint64_t nbytes) {
    const uint64_t nb = nbytes / 210;
    const unsigned char * sd  = buf;
    const unsigned char * ssc = buf + nb * 2;
    const unsigned char * slo = buf + nb * 18;
    const unsigned char * shi = buf + nb * 82;
    const unsigned char * sqh = buf + nb * 146;
    for (uint64_t i = 0; i < nb; ++i) {
        unsigned char * o = out + i * 210;
        for (int j = 0; j < 64; ++j) {
            const unsigned char a = slo[i * 64 + j];
            const unsigned char b = shi[i * 64 + j];
            o[2 * j]     = (a & 0x0F) | ((b & 0x0F) << 4);
            o[2 * j + 1] = (a >> 4)   | ((b >> 4) << 4);
        }
        std::memcpy(o + 128, sqh + i * 64, 64);
        std::memcpy(o + 192, ssc + i * 16, 16);
        o[208] = sd[2 * i];
        o[209] = sd[2 * i + 1];
    }
}

// ---------------------------------------------------------------------------
// v2 decoding
//
// Q4_K per-tensor stream layout (bytes, nb = nbytes/144):
//   [ d_lo:nb ][ d_hi:nb ][ dmin_lo:nb ][ dmin_hi:nb ]
//   [ scale:nb*8 ][ min:nb*8 ][ nibbles:nb*128 ]
// The nibbles are the 4-bit quant values grouped by sub-block and stably
// sorted by the context key (scale << 6 | min); the 32 values of a sub-block
// are packed two per byte, low nibble first.
//
// Q6_K per-tensor stream layout (nb = nbytes/210):
//   [ d_lo:nb ][ d_hi:nb ][ scales_even:nb*8 ][ scales_odd:nb*8 ]
//   [ ql_lo:nb*64 ][ ql_hi:nb*64 ][ qh:nb*64 ]
// ---------------------------------------------------------------------------

static void stable_order_sc_min(const unsigned char * sc, const unsigned char * mn,
                                uint64_t n, std::vector<uint32_t> & order) {
    uint32_t counts[4096];
    uint32_t pos[4096];
    std::memset(counts, 0, sizeof(counts));
    for (uint64_t p = 0; p < n; ++p) {
        const uint32_t k = (static_cast<uint32_t>(sc[p]) << 6) | mn[p];
        ++counts[k];
    }
    uint32_t acc = 0;
    for (int i = 0; i < 4096; ++i) {
        pos[i] = acc;
        acc += counts[i];
    }
    order.resize(n);
    for (uint64_t p = 0; p < n; ++p) {
        const uint32_t k = (static_cast<uint32_t>(sc[p]) << 6) | mn[p];
        order[pos[k]++] = static_cast<uint32_t>(p);
    }
}

static void unpack_q4_k_v2(const unsigned char * buf, unsigned char * out, uint64_t nbytes) {
    const uint64_t nb = nbytes / 144;
    const unsigned char * sdlo = buf;
    const unsigned char * sdhi = buf + nb;
    const unsigned char * smlo = buf + nb * 2;
    const unsigned char * smhi = buf + nb * 3;
    const unsigned char * ssc  = buf + nb * 4;
    const unsigned char * smn  = buf + nb * 12;
    const unsigned char * snib = buf + nb * 20;

    std::vector<uint32_t> order;
    stable_order_sc_min(ssc, smn, nb * 8, order);

    // scales + fp16 byte-plane, zero the qs region so nibbles can be OR-ed in
    for (uint64_t i = 0; i < nb; ++i) {
        unsigned char * o = out + i * 144;
        o[0] = sdlo[i];
        o[1] = sdhi[i];
        o[2] = smlo[i];
        o[3] = smhi[i];
        const unsigned char * s8 = ssc + i * 8;
        const unsigned char * m8 = smn + i * 8;
        for (int j = 0; j < 4; ++j) {
            o[4 + j]  = (s8[j] & 63) | ((s8[j + 4] >> 4) << 6);
            o[8 + j]  = (m8[j] & 63) | ((m8[j + 4] >> 4) << 6);
            o[12 + j] = (s8[j + 4] & 0x0F) | ((m8[j + 4] & 0x0F) << 4);
        }
        std::memset(o + 16, 0, 128);
    }

    const uint64_t nsub = nb * 8;
    for (uint64_t q = 0; q < nsub; ++q) {
        const uint32_t p = order[q];
        const uint64_t blk = p >> 3;
        const int s = static_cast<int>(p & 7);
        const uint64_t base = 16 + (s >> 1) * 32;
        const unsigned char * src = snib + q * 16;
        unsigned char * o = out + blk * 144 + base;
        if (s & 1) {
            for (int i = 0; i < 16; ++i) {
                const unsigned char b = src[i];
                o[2 * i]     |= static_cast<unsigned char>((b & 0x0F) << 4);
                o[2 * i + 1] |= static_cast<unsigned char>((b >> 4) << 4);
            }
        } else {
            for (int i = 0; i < 16; ++i) {
                const unsigned char b = src[i];
                o[2 * i]     |= static_cast<unsigned char>(b & 0x0F);
                o[2 * i + 1] |= static_cast<unsigned char>(b >> 4);
            }
        }
    }
}

static void unpack_q6_k_v2(const unsigned char * buf, unsigned char * out, uint64_t nbytes) {
    const uint64_t nb = nbytes / 210;
    const unsigned char * sdlo = buf;
    const unsigned char * sdhi = buf + nb;
    const unsigned char * sce  = buf + nb * 2;
    const unsigned char * sco  = buf + nb * 10;
    const unsigned char * slo  = buf + nb * 18;
    const unsigned char * shi  = buf + nb * 82;
    const unsigned char * sqh  = buf + nb * 146;
    for (uint64_t i = 0; i < nb; ++i) {
        unsigned char * o = out + i * 210;
        o[208] = sdlo[i];
        o[209] = sdhi[i];
        const unsigned char * se = sce + i * 8;
        const unsigned char * so = sco + i * 8;
        for (int j = 0; j < 8; ++j) {
            o[192 + 2 * j]     = se[j];
            o[192 + 2 * j + 1] = so[j];
        }
        std::memcpy(o + 128, sqh + i * 64, 64);
        const unsigned char * a = slo + i * 64;
        const unsigned char * b = shi + i * 64;
        for (int j = 0; j < 64; ++j) {
            o[2 * j]     = (a[j] & 0x0F) | ((b[j] & 0x0F) << 4);
            o[2 * j + 1] = (a[j] >> 4)   | ((b[j] >> 4) << 4);
        }
    }
}

static std::string lumina_temp_path(const char * path, uint64_t orig_size) {
    std::string name = "lumina-" + std::to_string(fnv1a(std::string(path) + ":" + std::to_string(orig_size))) + ".gguf";
    try {
        return (std::filesystem::temp_directory_path() / name).string();
    } catch (...) {
        // fall back to a sibling file
        std::filesystem::path p(path);
        return (p.parent_path() / (p.filename().string() + ".lumina-tmp")).string();
    }
}

} // namespace

bool lumina_gguf_is_compressed(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (f == nullptr) {
        return false;
    }
    char magic[4] = { 0 };
    const bool ok = read_exact(f, magic, sizeof(magic)) && std::memcmp(magic, LUMINA_MAGIC, 4) == 0;
    std::fclose(f);
    return ok;
}

std::string lumina_gguf_resolve_path(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (f == nullptr) {
        return path;
    }

    char magic[4] = { 0 };
    if (!read_exact(f, magic, sizeof(magic)) || std::memcmp(magic, LUMINA_MAGIC, 4) != 0) {
        std::fclose(f);
        return path;
    }

    uint32_t version = 0;
    uint32_t flags   = 0;
    uint64_t orig_size   = 0;
    uint64_t data_offset = 0;
    uint32_t n_tensors   = 0;
    uint32_t reserved    = 0;

    if (!read_exact(f, &version, 4) || !read_exact(f, &flags, 4) ||
        !read_exact(f, &orig_size, 8) || !read_exact(f, &data_offset, 8) ||
        !read_exact(f, &n_tensors, 4) || !read_exact(f, &reserved, 4)) {
        std::fclose(f);
        throw std::runtime_error("lumina: truncated compressed GGUF header");
    }

    if (version < LUMINA_VERSION_MIN || version > LUMINA_VERSION_MAX) {
        std::fclose(f);
        throw std::runtime_error("lumina: unsupported compressed GGUF version " + std::to_string(version));
    }

    std::vector<lumina_tensor> tensors(n_tensors);
    for (uint32_t i = 0; i < n_tensors; ++i) {
        uint32_t tt = 0;
        uint32_t pad = 0;
        uint64_t nb = 0;
        if (!read_exact(f, &tt, 4) || !read_exact(f, &pad, 4) || !read_exact(f, &nb, 8)) {
            std::fclose(f);
            throw std::runtime_error("lumina: truncated compressed GGUF tensor table");
        }
        tensors[i] = { tt, nb };
    }

    const std::string tmp = lumina_temp_path(path, orig_size);
    std::error_code ec;
    if (std::filesystem::exists(tmp, ec) && !ec && std::filesystem::file_size(tmp, ec) == orig_size && !ec) {
        LUMINA_LOG_INFO("%s: reusing decompressed model %s\n", __func__, tmp.c_str());
        std::fclose(f);
        return tmp;
    }

    LUMINA_LOG_INFO("%s: decompressing lossless container (v%u) %s -> %s\n",
                    __func__, version, path, tmp.c_str());

    FILE * out = std::fopen(tmp.c_str(), "wb");
    if (out == nullptr) {
        std::fclose(f);
        throw std::runtime_error("lumina: cannot open temporary file " + tmp);
    }

    lumina_inflater inf;
    bool ok = inf.init(f);

    uint64_t written = 0;
    std::vector<unsigned char> stream;
    std::vector<unsigned char> block;

    // 1) header region (GGUF magic, metadata, tensor table + padding), verbatim
    {
        const size_t chunk = 4 * 1024 * 1024;
        std::vector<unsigned char> buf(chunk);
        uint64_t remaining = data_offset;
        while (ok && remaining > 0) {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(remaining, chunk));
            ok = inf.read(buf.data(), n) && std::fwrite(buf.data(), 1, n, out) == n;
            remaining -= n;
            written += n;
        }
    }

    // 2) one tensor at a time: de-interleave, then write the original bytes
    for (size_t i = 0; ok && i < tensors.size(); ++i) {
        const lumina_tensor & t = tensors[i];
        if (t.nbytes == 0) {
            continue;
        }
        // v2 enlarges only the Q4_K stream (scale/min are unpacked to bytes)
        uint64_t stream_size = t.nbytes;
        if (version >= 2 && t.type == GGML_TYPE_Q4_K) {
            if (t.nbytes % 144 != 0) { ok = false; break; }
            stream_size = (t.nbytes / 144) * 148;
        }
        stream.resize(static_cast<size_t>(stream_size));
        block.resize(static_cast<size_t>(t.nbytes));
        ok = inf.read(stream.data(), stream.size());
        if (!ok) {
            break;
        }
        switch (t.type) {
            case GGML_TYPE_Q4_K:
                if (t.nbytes % 144 != 0) { ok = false; break; }
                if (version >= 2) {
                    unpack_q4_k_v2(stream.data(), block.data(), t.nbytes);
                } else {
                    unpack_q4_k_v1(stream.data(), block.data(), t.nbytes);
                }
                break;
            case GGML_TYPE_Q6_K:
                if (t.nbytes % 210 != 0) { ok = false; break; }
                if (version >= 2) {
                    unpack_q6_k_v2(stream.data(), block.data(), t.nbytes);
                } else {
                    unpack_q6_k_v1(stream.data(), block.data(), t.nbytes);
                }
                break;
            default:
                std::memcpy(block.data(), stream.data(), t.nbytes);
                break;
        }
        if (!ok) {
            break;
        }
        ok = std::fwrite(block.data(), 1, t.nbytes, out) == t.nbytes;
        written += t.nbytes;
    }

    std::fclose(f);
    if (std::fclose(out) != 0) {
        ok = false;
    }

    if (!ok || written != orig_size) {
        std::filesystem::remove(tmp, ec);
        throw std::runtime_error("lumina: failed to decompress model (wrote " + std::to_string(written) +
                                 " of " + std::to_string(orig_size) + " bytes)");
    }

    LUMINA_LOG_INFO("%s: decompressed %.2f GiB\n", __func__, written / 1024.0 / 1024.0 / 1024.0);
    return tmp;
}
