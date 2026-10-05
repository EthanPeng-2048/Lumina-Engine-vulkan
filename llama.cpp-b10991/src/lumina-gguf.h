#pragma once

// ---------------------------------------------------------------------------
// Lumina-Engine lossless GGUF support.
//
// A "Lumina compressed GGUF" is a lossless container produced by
// tools/lumina_gguf_pack.py.  By convention it is named `A.gguf.lumina`
// (the original `A.gguf` is left untouched), but detection is purely by the
// `LGGC` file magic, so the extension does not matter.  It stores the exact
// same weights as the original model - no quantization, no pruning - but
// de-interleaves the k-quant blocks (and, for v2, context-sorts the 4-bit
// streams) and deflates the result.  `lumina_gguf_resolve_path()` turns such
// a path into a temporary, standard .gguf file that the rest of llama.cpp can
// load transparently.
//
// Container versions:
//   v1 : field split + DEFLATE.
//   v2 : field split + fp16 byte-plane split + (scale,min)-sorted 4-bit
//        streams + DEFLATE.  Roughly 1% smaller than v1, byte-identical on
//        decode.
// ---------------------------------------------------------------------------

#include <string>

// true if the file begins with the Lumina container magic.
bool lumina_gguf_is_compressed(const char * path);

// If `path` is a Lumina compressed GGUF, decompress it to a temporary .gguf
// file and return that path.  Otherwise return `path` unchanged.  Throws
// std::runtime_error on a malformed container.
std::string lumina_gguf_resolve_path(const char * path);
