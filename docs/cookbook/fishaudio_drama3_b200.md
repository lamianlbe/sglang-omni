# Fish Audio Drama 3 on B200

This recipe targets the official block-FP8 Drama 3 checkpoint on a single NVIDIA B200 (SM100). It combines the existing asynchronous decode, CUDA graph, fused tower, sampler, and MPS paths with tuned expert GEMMs and a quantizer upsampling optimization. Measurements use SGLang 0.5.23, Triton 3.8.0, PyTorch 2.14.1+cu130, CUDA 13, and driver 580.173.02. They do not establish performance on SM120 or other software versions.

## Start the measured configuration

Install the Fish Audio dependencies using the [Drama 3 cookbook](fishaudio_drama3.md). From the repository root, set the two local checkpoint paths and launch:

```bash
export DRAMA3_CHECKPOINT=/path/to/drama-3
export DRAMA3_CODEC=/path/to/codec
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export SGLANG_MOE_CONFIG_DIR="$PWD/examples/configs/drama3_b200_moe"

sgl-omni serve \
  --model-path "$DRAMA3_CHECKPOINT" \
  --host 127.0.0.1 \
  --port 8000 \
  --vocoder.factory.codec_path "$DRAMA3_CODEC" \
  --tts_engine.factory.max_running_requests 256 \
  --tts_engine.factory.enable_async_decode true \
  --tts_engine.factory.tower_fused_layers true \
  --tts_engine.factory.triton_sampling true \
  --tts_engine.engine.moe_runner_backend triton \
  --mps on
```

The command uses the final high-concurrency capacity of 256. Use 128 to reproduce the fixed-capacity comparison below; client concurrency and engine capacity are separate settings.

The optional expert configuration directory is versioned for Triton 3.8.0 and NVIDIA B200. It contains separate up and down projection configurations; both files are required to reproduce the TMA experiment. Unset `SGLANG_MOE_CONFIG_DIR` to use SGLang's normal configuration selection. Warm the full serving path before measuring throughput; loading weights and capturing graphs are not steady inference.

The quantizer optimization is selected automatically for FP16 weights on compute capability (10, 0), only for ungrouped, undilated, unpadded transposed convolutions with kernel size equal to stride. Other shapes, dtypes, and devices retain their existing implementation. It does not change checkpoint precision, codebook count, sample rate, or the FP32 GDN recurrent state.

## End-to-end serving results

The following comparison uses the previous optimized b200 commit (1f7e33177c17fb243262a3296b69147fd1450c60) as its baseline. Both configurations keep engine capacity 128, MPS, asynchronous decode, sampler, and fused tower enabled. The candidate adds the quantizer projection and the separate TMA configuration files. The baseline uses the earlier non-TMA tuning preserved under examples/configs/drama3_b200_moe/baseline; the code commit alone does not identify that tuning. Point SGLANG_MOE_CONFIG_DIR at this baseline subdirectory when running the previous commit.

Three repetitions alternate deployment order and warm 256 requests after each restart. Each repetition measures 256 requests at concurrency 32 and 1024 requests at concurrency 128 and 256. The client streams a fixed 256-text English SeedTTS corpus with reference audio, temperature 1.0, top-p 0.9, and a 2048-token generation budget. Audio throughput is generated audio seconds per wall second; request throughput is also reported because sampling changes audio lengths. Profiling is disabled for these measurements.

| Client concurrency | Baseline audio seconds/s | Candidate audio seconds/s | Gain | Baseline requests/s | Candidate requests/s |
| --- | --- | --- | --- | --- | --- |
| 32 | 84.18 ± 1.01 | 88.78 ± 0.82 | 5.5% | 19.59 | 20.74 |
| 128 | 110.82 ± 0.28 | 137.28 ± 0.69 | 23.9% | 25.85 | 32.05 |
| 256 | 114.62 ± 0.12 | 141.86 ± 0.74 | 23.8% | 26.73 | 33.16 |

Values are mean ± sample standard deviation across three runs, not confidence intervals. All 13,824 measured requests across the two configurations completed without request errors or generation-budget truncation. The corpus SHA-256 is 8e789cc186a2f51c6529ad31f81316c39a993b838fbb01adad67a61869aba281. These are local closed-loop results, not an internet latency or service-level capacity guarantee.

At concurrency 128, mean p95 time to first audio decreased from 3.240 to 2.186 seconds. A separate 4096-request run sustained 136.86 audio seconds per second without errors or truncation. Immediate-playback stall incidence increased from 78.9% to 84.2%, while stall duration per request decreased from 0.478 to 0.380 seconds; throughput and playback continuity need separate service objectives.

The subsequent capacity-256 experiment measured 143.90 ± 0.27 audio seconds per second at client concurrency 256 (three runs of 1024 requests), about 1.44% above capacity 128. Mean p95 time to first audio was 6.058 seconds. A fresh 4096-request run at capacity/concurrency 256 sustained 138.73 audio seconds per second with zero errors and truncations; its p95 time to first audio was 6.357 seconds. This later capacity sweep was not interleaved with the earlier runs, so its smaller gain should be rechecked for other workloads. Capacity 256 is the final high-concurrency preset; the repeated code-change comparison above remains fixed at capacity 128.

## Why the quantizer change helps

The production codec has two non-overlapping stride-2 quantizer upsampling layers. At a 96-frame streaming window, Nsight Compute measured the previous cuDNN direct kernels at 1.339 and 2.019 milliseconds, with less than 0.1% of peak DRAM throughput and no Tensor Core activity. The replacement expresses each layer as a phase-ordered dense projection, enabling Tensor Core GEMMs. The complete codec, including the remaining layers and layout conversions, decreased from 7.701 to 4.375 milliseconds at batch one.

Phase ordering preserves the transposed convolution's mathematical output and sample positions. Floating-point accumulation order changes: fixed-codebook comparisons over the actual 4/14/26/58/96-frame streaming windows retain output shape and finite samples, but FP16 output is not bitwise identical. CPU double-precision tests check the phase mapping, bias handling, and noncontiguous inputs against `ConvTranspose1d`; GPU codec tests retain the existing tolerance.

## What the profile establishes

Two-second captures taken during sustained load separate complete decode CUDA graphs from prefill and first-frame graphs. The graph signature is 80 expert GEMM nodes and 30 GDN recurrent nodes. At client concurrency 128, the previous optimized service spent 32.81% of its complete decode kernel duration in expert GEMMs, 12.21% in GDN state updates, 22.52% in dense GEMMs and split-K reductions, and 4.03% in full-attention decode. The remaining time includes normalization, activation, sampling, routing, quantization, copies, and other kernels.

After the combined optimization, a separate concurrency-128 capture attributed 30.92% of complete decode kernel duration to expert GEMMs, 15.08% to GDN state updates, and 20.12% to dense GEMMs and split-K reductions. Its padded batches ranged from 64 to 120; the shift in batch distribution means these shares are not fixed-work kernel speedup measurements.

These percentages use summed GPU kernel durations within complete decode graphs, not wall-clock time or total device utilization. Kernels on concurrent streams and processes can overlap. Client concurrency also differs from the padded decode graph batch: the concurrency-128 capture used batches between 32 and 112. Shared dense kernel names do not uniquely identify the backbone or codebook tower.

Real-input expert replay at batch 128 reached 70.08% of sustained peak DRAM throughput for layer-zero up projection, while the layer-39 down projection reached 36.22%. Routing changes the workload: the four captured layers activated 213, 170, 134, and 77 of 256 experts. Uniform random routing is insufficient for selecting a production configuration. TMA reduced complete expert-call latency by 6.6% to 9.6% on these four captures; all 48 layer/batch-boundary comparisons matched the previous configuration exactly.

A separate, model-shaped GDN probe reached 65.61% of sustained peak DRAM throughput and 11.69% achieved occupancy. This supports investigating state traffic and register pressure while retaining FP32 state. Small dense GEMMs and elementwise kernels did not saturate Tensor Core throughput; low bandwidth utilization alone does not make a kernel compute-bound. Counter runs use cold-cache replay, while latency measurements use separate warmed CUDA graphs. Neither replaces end-to-end serving measurements.

## Validation limits

The changed vocoder and convolution suites passed 103 tests, and the repository pre-commit checks passed. The preceding broader validation still has two strict CUDA graph/eager parity failures, also reproducible with the earlier sampler and fused-tower additions disabled. Those failures remain unresolved; the complete test suite is not claimed to pass.

On a paired 64-text English ASR screen, corpus WER was 0.28% for the baseline and 0.43% for the candidate. Sampling was not fixed per request, so this small difference is not a causal estimate of codec quality.

Fixed-codebook numerical checks and a small ASR regression screen do not replace blind listening, speaker-similarity evaluation, or long-generation testing. High aggregate audio throughput also does not establish acceptable time to first audio or uninterrupted playback. Measure those service objectives alongside throughput before choosing production admission limits.

For two GPUs, compare two independent replicas against AR/codec stage separation before drawing conclusions about disaggregation. Prefill/decode separation needs a distinct workload and measurement of KV plus GDN-state transfer; it is not part of this single-card recipe.
