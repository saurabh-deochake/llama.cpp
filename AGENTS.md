# MTP SYCL Performance Fix — Implementation Summary

## Status: Option A (GPU-side embd batch) — fully implemented

## Core Changes (12 files, ~400 lines)

### Architecture
1. **`include/llama.h`** — Added `const void * device_embd` to `llama_batch`
2. **`src/llama-batch.h`** — Added `const void * device_embd` to `llama_ubatch`
3. **`src/llama-batch.cpp`** — `ubatch_add()` skips CPU memcpy when `device_embd` is set, passes pointer through
4. **`src/llama-graph.cpp`** — `set_input()` calls `ggml_backend_sycl_set_embd_device()` for D2D copy instead of CPU→GPU upload
5. **`common/common.cpp`** — `common_batch_clear()` resets `device_embd = nullptr`

### Device Pointer API
6. **`src/llama-context.h`** — Added `embd_pre_norm_device` field + `get_embeddings_pre_norm_device()` + `alloc_device_buffer/free_device_buffer/device_memcpy`
7. **`src/llama-context.cpp`** — Capture `t_h_pre_norm->data` after encode/decode; reset in `output_reserve()`; implement device memory helpers
8. **`src/llama-ext.h`** — C API wrappers: `llama_get_embeddings_pre_norm_device`, `llama_alloc_device_buffer`, etc.

### SYCL Backend (ggml-sycl)
9. **`ggml/include/ggml-sycl.h`** — Declared `ggml_backend_sycl_set_embd_device`, `_device_malloc`, `_device_free`, `_device_memcpy`
10. **`ggml/src/ggml-sycl/ggml-sycl.cpp`** — Implemented D2D memcpy helper + USM device memory alloc/free/memcpy

### MTP Speculative Decoding
11. **`common/speculative.cpp`** — `common_speculative_state_mtp`:
    - `process()`: D2D shift copy + CPU→GPU pending_h upload via USM workspace, sets `device_embd`
    - `draft()`: Uses device pointers from `ctx_dft` output instead of CPU `llama_get_embeddings_pre_norm_ith`
    - Pending_h capture: async GPU→CPU copy via device pointer offsets (avoids `output_reorder()` sync)

## Data Flow (after fix)

```
process():
  dev_h_tgt = get_embeddings_pre_norm_device(ctx_tgt)  // no sync
  D2D: dev_h_tgt[0..n-2] → workspace[1..n-1]          // shift right
  CPU→GPU: pending_h[seq] → workspace[0]                // async per-seq
  batch.device_embd = workspace
  llama_decode(ctx_dft, batch)                          // set_input does D2D

draft():
  batch.device_embd = workspace
  llama_decode(ctx_dft, batch)
  while drafting:
    dev_row = ctx_dft->embd_pre_norm_device[i_batch]   // no sync
    D2D: dev_row → workspace[pos]
    llama_decode(ctx_dft, batch)
```
