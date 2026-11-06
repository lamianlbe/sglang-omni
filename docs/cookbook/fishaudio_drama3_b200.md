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

## Follow-up module attribution and kernel experiments

The 2026-11-07 follow-up starts from b82e2ad974fefa068e65583ba537e5f59062b4b7 and retains engine capacity 256, checkpoint precision, MPS, and asynchronous decode. The preceding approximately 24% improvement belongs to the earlier fixed-capacity comparison; the following experiments measure additional opportunities.

GPU begin/end markers in a separate diagnostic source tree distinguish backbone, shared tower, and codebook tower kernels, including GEMMs with identical names. A complete decode replay establishes the graph-node mapping; only graphs containing 80 expert GEMMs, 30 GDN updates, four shared-layer marker pairs, and nine codebook-layer pairs contribute. Marker durations are excluded. These are shares of summed decode kernel duration, not wall time, and exclude the independent vocoder process. Diagnostic throughput is not used as a performance result.

| Client concurrency | Backbone | Shared tower | Codebook tower | Outside marked modules | Observed padded batches |
| --- | --- | --- | --- | --- | --- |
| 128 | 79.89% | 4.45% | 10.03% | 5.64% | 64, 72, 80, 88, 96, 104 |
| 256 | 77.96% | 4.74% | 10.76% | 6.54% | 32, 40, 48, 56, 64, 72, 80 |

Dense GEMMs account for 14.96% of total decode kernel duration in the backbone, 1.73% in the shared tower, and 3.87% in the codebook tower in the concurrency-128 capture. At concurrency 256 those shares are 17.16%, 1.97%, and 4.45%. Most measured dense work therefore belongs to the backbone. Two-second samples have different batch distributions and do not establish universal module shares.

### GDN launch geometry

A 120-configuration search covers batch sizes 1, 8, 32, 64, 128, and 256; value tiles 4, 8, 16, 32, and 64; and one, two, four, or eight warps. Each configuration compares 32 recurrent steps, shuffled state slots, and negative padding indices against the packed reference with FP32 state and BF16 inputs/output. The service candidate uses value tile 4 and one warp at batch one and batches at least 32; it keeps the original geometry otherwise. Selected geometry comparisons are bitwise identical in this probe.

At batch 128, paired cold-cache Nsight Compute runs measured 95.328 microseconds for tile 32 and 81.120 microseconds for tile 4. Registers per thread fell from 201 to 53, achieved occupancy rose from 11.64% to 43.34%, and measured DRAM throughput rose from 5.048 to 5.931 TB/s, or 65.82% to 77.33% of sustained peak. Eligible warps per scheduler increased from 0.33 to 1.62. These counters identify register pressure and state-memory traffic as constraints; they do not establish a Tensor Core compute bottleneck.

The full-service comparison runs baseline then candidate, followed by candidate then baseline, warming 256 requests after each restart and measuring 2048 requests at each concurrency. All 16,384 measured requests complete without errors or generation-budget truncation.

| Client concurrency | Baseline audio seconds/s | GDN candidate audio seconds/s | Change | Baseline requests/s | Candidate requests/s |
| --- | --- | --- | --- | --- | --- |
| 128 | 136.352 ± 0.116 | 137.184 ± 0.022 | +0.61% | 31.835 | 32.037 |
| 256 | 140.762 ± 0.139 | 140.671 ± 0.403 | -0.06% | 32.822 | 32.818 |

Values are means and sample standard deviations of two runs per configuration. The highest-concurrency throughput does not improve, so this candidate is retained as an experiment rather than enabled in the recipe. A faster isolated kernel need not accelerate a pipeline sharing device resources with other stages.

### Dense projections and expert routing

A BF16 gate/up GEMM plus SiLU fusion search uses a real codebook-layer checkpoint and 128 configurations across batches 32, 64, 128, and 256. The best valid fused kernels take 10.89, 17.02, 23.14, and 44.44 microseconds, compared with 8.41, 9.21, 10.09, and 14.26 microseconds for cuBLAS plus the existing activation. These are warm-cache microbenchmarks, not whole-tower timings. None is adopted.

A separate cuBLAS/cuBLASLt comparison cycles all 13 real tower-layer weight sets, with a working set larger than L2. Summing the mean layer time for each of five projection types yields approximately 43.68/43.67, 43.25/43.04, 44.92/44.92, and 53.84/53.86 microseconds for cuBLAS/cuBLASLt at batches 32/64/128/256. Selecting cuBLASLt alone does not provide a material improvement.

The additional expert search tests 75 configurations using captured activations, routes, FP8 weights, and scales from layers 0, 7, 20, and 39. It varies row tiles 8/16/32, column tiles 64/128, row grouping 1/4, and TMA. At batch 32, the best four-layer sum falls from 337.19 to 328.99 microseconds; at batches 64 and 128, the existing configurations remain fastest at 377.99 and 422.47 microseconds. The selected results match the reference numerically. Four captured layers do not justify changing the configuration for all 40 layers.

### Buffered recurrent state

Existing ReplaySSM with FP32 state and ring buffers still converts reconstruction operands to BF16. In the 32-step probe, cache length 16 reduces batch-128 latency from 98.21 to 76.90 microseconds and batch-256 latency from 190.23 to 143.10 microseconds, but flushed-state errors reach approximately 0.0043 and 0.0048. It fails the FP32 state comparison tolerance of relative 0.0001 and absolute 0.00001. FlashInfer also fails that state tolerance; CuTe DSL fails this probe's numerical comparisons. These outcomes apply to the tested shapes, padding, and software versions rather than every use of those backends.

An FP32-vector/TF32x3 reconstruction prototype lowers 256-step state error to approximately 0.0000022 or less, but takes 158.80 microseconds at batch 128 and 307.99 microseconds at batch 256. It is slower than the packed reference.

A second prototype computes the buffered low-rank updates directly against q and k, avoiding a full state write on ordinary steps and replaying FP32 updates sequentially at flush. All 81 configurations complete their 256-step microbenchmark state checks. The best observed times are 22.40, 84.88, and 160.45 microseconds at batches 32, 128, and 256, with maximum state error around 0.0000002. A stricter CUDA-graph lifecycle check, including long memory, strided inputs, forced flush, request reordering, and state-slot reuse, fails the BF16 output tolerance at one element with absolute difference 0.0001220703125. This failure is retained; no production backend switch follows from the microbenchmark.

The separate FP32-output lifecycle diagnostic retains the same workload and compares arithmetic before BF16 output rounding. Both batch 32 and 128 complete 1024 steps, including 25 reorderings and eight state-slot reuses. Maximum FP32 output error is below 0.00000019; maximum flushed-state error is 0.00000430. Approximately 0.051% and 0.054% of outputs round to a different BF16 value. This supports a small arithmetic difference crossing BF16 rounding boundaries, but does not make the original stricter BF16-output test pass or establish quality equivalence.

For a four-step ring, a simple traffic model changes full-state accesses from one read plus one write per step to one read plus one-quarter write per step, a 37.5% reduction in full-state bytes before ring traffic and cache effects. The arithmetic also changes: buffered vector products replace ordinary-step state reconstruction, while flush steps replay updates. This is a traffic model, not a measured 37.5% service improvement. The prototype's benefit must be evaluated with the actual mix of small and large decode batches and the added ring/cursor operations.

The follow-up archive retains replay_readout.py, both lifecycle tests, prepare_readout_source.py, and src-readout.patch. Its isolated service enables linear ReplaySSM with cache length four and substitutes the experimental readout callable. Enabling `tts_engine.engine.enable_linear_replayssm` alone in this repository selects the installed SGLang implementation, not this FP32 prototype. The recipe's default backend remains unchanged. A maintained integration should attach a model-scoped decode kernel through the GDN dispatcher before graph capture, rather than reuse the experiment's process-wide callable replacement.

The prototype's full-service comparison uses candidate then baseline, followed by baseline then candidate, with a fresh restart and 256-request warmup per deployment. The first candidate repetition is its preceding service screen with identical settings. Each deployment measures 2048 requests at concurrency 128 and another 2048 at concurrency 256. All 16,384 formal requests complete without errors or generation-budget truncation.

| Client concurrency | Baseline audio seconds/s | FP32 readout audio seconds/s | Change | Baseline requests/s | FP32 readout requests/s |
| --- | --- | --- | --- | --- | --- |
| 128 | 136.942 ± 0.315 | 140.464 ± 0.480 | +2.57% | 32.003 | 32.784 |
| 256 | 140.860 ± 0.235 | 144.432 ± 0.898 | +2.54% | 32.887 | 33.677 |

Values are means and sample standard deviations of two runs, not confidence intervals. Mean p95 first-audio latency changes from 2.278 to 2.119 seconds at concurrency 128 and from 6.306 to 6.043 seconds at concurrency 256. Playback-stall incidence after 0.25 seconds of buffering increases from 38.55% to 45.09% and from 48.78% to 51.42%, respectively. The approximately 2.5% throughput gain is an experimental result with an accompanying streaming tradeoff, not an unqualified production improvement.

A subsequent 8192-request prototype run at concurrency 256 sustains 141.27 audio seconds per second without errors or generation-budget truncation. Its lower rate than the shorter comparison runs is retained as a separate observation; the repeated 2048-request mean is not a sustained-throughput guarantee.

The final fresh-deployment 8192-request baseline run sustains 137.52 audio seconds per second, also with zero errors and truncation. The prototype is 2.73% faster in these two longer observations, but the long runs are sequential rather than repeated alternating trials. The follow-up records 69,760 non-warmup HTTP requests across comparisons, scheduling, audio collection, and both long runs, excluding profiler loads; none has a request error or generation-budget truncation.

A separate one-second CUPTI capture under load records 1530 calls to replay_readout_kernel and no calls to the packed reference, confirming that the service exercises the experimental implementation. After stopping the service, Whisper turbo transcribes the paired 64-text English samples: both configurations have corpus WER 0.43%, with no sample above 50% WER. Sampling is not fixed per request, and this small ASR screen does not establish perceptual or speaker-similarity equivalence.

### Scheduling and codec batching

Ten independent deployments each warm 256 requests and measure 2048 requests at concurrency 256. The baseline uses prefill/decode interval 4, a prefill token budget of 16384, and codec batch 8 with a 2 ms wait. Every candidate changes only the listed setting. Baselines before and after the scan measure 141.40 and 140.60 audio seconds/s; all 20,480 formal requests complete without errors or generation-budget truncation.

| Setting | Audio seconds/s | Requests/s | First-audio p95, seconds | Requests with playback stalls after 0.25 s buffering |
| --- | --- | --- | --- | --- |
| Baseline before | 141.40 | 33.043 | 6.214 | 50.4% |
| Prefill/decode interval 2 | 141.20 | 33.012 | 5.899 | 69.9% |
| Prefill/decode interval 8 | 137.54 | 32.071 | 6.975 | 32.3% |
| Prefill/decode interval 16 | 135.57 | 31.640 | 7.190 | 21.6% |
| Prefill token budget 2048 | 138.17 | 32.273 | 6.493 | 35.9% |
| Prefill token budget 4096 | 140.42 | 32.814 | 6.343 | 47.0% |
| Prefill token budget 8192 | 139.81 | 32.653 | 6.425 | 46.4% |
| Codec batch 1, wait 0 ms | 141.20 | 32.948 | 6.293 | 44.8% |
| Codec batch 4, wait 0 ms | 140.22 | 32.737 | 6.342 | 47.5% |
| Baseline after | 140.60 | 32.770 | 6.341 | 46.8% |

No candidate exceeds the initial baseline. Larger prefill intervals trade throughput and first-audio latency for fewer playback stalls; interval 2 has the opposite latency/stall tradeoff. The throughput recipe therefore retains the original settings. This single screening pass identifies workload-specific tradeoffs, not statistically established optimal latency settings.

### Remaining opportunities and two-GPU experiments

Further expert work should target real-routing weight reuse, padding, dispatch, and a specialized SM100 implementation; another small launch-parameter grid has limited evidence of benefit. The current expert path is already near a substantial fraction of sustained HBM bandwidth on some layers. Any replacement must preserve the checkpoint's scales, gate/up layout, activation semantics, and output checks before service testing.

Next, investigate backbone dense/elementwise fusion and GDN state traffic with the actual serving batch distribution. The buffered-state experiments show why preserving an FP32 allocation is not sufficient to preserve arithmetic precision. Shared/codebook tower work remains useful, but its smaller measured share limits the potential return. If a component occupies 10% of serial decode duration, making it twice as fast would improve that duration by only about 5.3%; this is an illustrative Amdahl estimate, not a prediction for the overlapping serving pipeline.

For the next two-GPU machine, first measure two independent replicas at the same total hardware budget. Then compare AR/codec separation and prefill/decode separation, recording aggregate audio throughput, requests/s, first-audio latency, playback stalls, transfer bandwidth, and per-GPU utilization. GDN state alone occupies 60 MiB per request across 30 layers, before KV state: 30 layers times 32 heads times 128 times 128 FP32 values. Measure the actual state-transfer path and concurrent admission behavior; this single-GPU study does not establish disaggregated performance or ReplaySSM compatibility with disaggregation.

## Validation limits

The changed vocoder and convolution suites passed 103 tests, and the repository pre-commit checks passed. The preceding broader validation still has two strict CUDA graph/eager parity failures, also reproducible with the earlier sampler and fused-tower additions disabled. Those failures remain unresolved; the complete test suite is not claimed to pass.

On a paired 64-text English ASR screen, corpus WER was 0.28% for the baseline and 0.43% for the candidate. Sampling was not fixed per request, so this small difference is not a causal estimate of codec quality.

Fixed-codebook numerical checks and a small ASR regression screen do not replace blind listening, speaker-similarity evaluation, or long-generation testing. High aggregate audio throughput also does not establish acceptable time to first audio or uninterrupted playback. Measure those service objectives alongside throughput before choosing production admission limits.

For two GPUs, compare two independent replicas against AR/codec stage separation before drawing conclusions about disaggregation. Prefill/decode separation needs a distinct workload and measurement of KV plus GDN-state transfer; it is not part of this single-card recipe.
