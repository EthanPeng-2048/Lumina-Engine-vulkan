#pragma once

#include "ggml-cpu-impl.h"
#include "ggml.h"

// GGML internal header

// Batched mul_mat path specialised for the 1-bit Q1_0 format used by the Bonsai models.
//
// The plain vec_dot path re-expands the 1-bit signs of every src0 row for each src1 column.
// This path expands a small row group once per K-block and keeps the resulting sign vectors in
// vector registers while it sweeps a tile of src1 columns, so the expansion is amortised over
// the tile without any extra memory traffic.
//
// A dedicated single-column path (token generation) uses the same per-128 activation format for
// one src1 column at a time, which cuts the float conversions of a Q1_0 block from four (one per
// Q8_0 sub-block) to one. The number of src0 rows processed per group defaults to 1 and can be
// tuned with GGML_Q1_GEMV_ROWS.

#ifdef __cplusplus
extern "C" {
#endif

// whether the Q1_0 register-tiled gemm should be used for this mul_mat
bool ggml_cpu_q1t_supports_mul_mat(const struct ggml_tensor * dst);

// shared scratch bytes for the per-128 activation of src1
size_t ggml_cpu_q1t_scratch_size(const struct ggml_tensor * dst);

// must be called after src1 has been converted to q8_0 into params->wdata and the threads
// have synchronized on it
void ggml_compute_forward_mul_mat_q1t(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
