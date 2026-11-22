---
title: "Roadmap"
description: "What I'm writing about next: hands-on experiments on AI inference infrastructure, with code you can run."
---

This is the plan for what I'll write, in order. Every post follows the same rule: **one question, one experiment you can run, one thing you change, one chart that shows what happened.** The code for each post lives in a companion repo, `inference-labs`, with exact commands and versions.

I'm not trying to out-write the excellent blogs already on these topics. I'm trying to make mine practical and reproducible, and to learn by changing things and watching what breaks.

## How the experiments run

- **Local machine:** a GTX 1080 (Pascal, compute capability 6.1, 8 GB). It runs llama.cpp and Ollama well, and PyTorch only on CUDA 12.6 builds of version 2.14 or older.
- **Cloud GPU (free Colab or Kaggle T4, compute capability 7.5):** for vLLM, Triton and FlashAttention-style work, which need compute capability 7.0 or higher.
- **Every post says which machine produced its numbers.**

## Series 0: Build it first (nanoGPT)

Start by building a tiny GPT myself, then add the serving tricks by hand. Seeing the cost go down when I add a KV cache is a better introduction than reading about it.

- [ ] **0.1 A tiny GPT on a 2016 GPU.** Follow Karpathy's nanoGPT on the character-level Shakespeare set. Run it with `compile=False` and fp16 or fp32 (Pascal has no bf16 and no Triton).
- [ ] **0.2 Adding a KV cache.** Write generation with and without the cache. Check that the logits match, then plot tokens/s against sequence length.
- [ ] **0.3 Batching and padding.** Generate for several prompts at once and see what padding costs.
- [ ] **0.4 Where the memory goes.** Derive the KV-cache size formula and check it against measured GPU memory.
- [ ] **0.5 A paged KV cache, toy version.** Replace one big buffer with fixed-size blocks and a block table, and measure the waste you removed.
- [ ] **0.6 Speculative decoding in 100 lines.** Use a small draft model and plot acceptance rate against speedup.
- [ ] **0.7 Quantizing the weights.** Convert to int8, then compare speed, memory and loss.

## Series 1: Measure before you optimize

- [ ] **1.1 Anatomy of one request.** Tokenize, prefill, decode. Time to first token versus per-token latency, measured on llama.cpp and Ollama.
- [ ] **1.2 Benchmarking honestly.** A load generator with open-loop and closed-loop traffic, reporting p50/p95/p99. Reused by every later post.
- [ ] **1.3 Why decoding is memory-bound.** Tokens/s against batch size and quantization, compared with the memory-bandwidth estimate.
- [ ] **1.4 Sampling from scratch.** Temperature, top-k, top-p and min-p, with an interactive slider.
- [ ] **1.5 Running inference on a 2016 GPU.** Why fp16 is slow on Pascal, what int8 and 4-bit change, and how much model fits in 8 GB.
- [ ] **1.6 Why temperature 0 isn't deterministic.** The same prompt under different batch sizes drifts.

## Series 2: Structured output and tool calls

This is my core topic, and all of it runs locally.

- [ ] **2.1 Constrained decoding from scratch.** Mask the logits with a regex or a finite-state machine, and measure the overhead.
- [ ] **2.2 JSON schema to grammar.** llama.cpp GBNF, and cases where the output is valid but wrong.
- [ ] **2.3 How tool calling really works.** Chat templates, and the raw model output for different model families.
- [ ] **2.4 Writing a tool-call parser.** Parallel calls, malformed JSON, and streaming partial output.
- [ ] **2.5 Streaming JSON parsing.** Incremental parsing of tool-call arguments.
- [ ] **2.6 The conformance suite.** The same test cases run against Ollama and llama.cpp (and vLLM on a cloud GPU), published as a pass/fail matrix. This is the flagship project.

## Series 3: Inside a serving engine

Local where possible; the vLLM posts run on a cloud T4.

- [ ] **3.1 Continuous batching.** A toy scheduler that compares static and continuous batching, then the same experiment with vLLM's `max_num_seqs`.
- [ ] **3.2 Prefix caching.** Time to first token with a shared system prompt, with and without `--enable-prefix-caching`.
- [ ] **3.3 Chunked prefill and head-of-line blocking.**
- [ ] **3.4 Quantization compared.** GGUF Q4/Q8 on llama.cpp locally, AWQ and GPTQ on a T4. Speed, memory and perplexity.
- [ ] **3.5 Speculative decoding in a real engine.** llama.cpp draft models and vLLM.
- [ ] **3.6 Reading the vLLM source.** Trace one request from the API to the scheduler, with logging patched in.
- [ ] **3.7 Guided decoding in vLLM.** Outlines and xgrammar overhead compared with my own implementation from 2.1.

## Series 4: Adapters and scale

- [ ] **4.1 LoRA from scratch.** The math, the effect of rank, and merging. Fine-tune a tiny model.
- [ ] **4.2 Serving LoRA adapters.** Loading adapters at runtime in llama.cpp and Ollama, then multi-LoRA serving in vLLM.
- [ ] **4.3 FlashAttention intuition.** Naive versus tiled attention, on a cloud GPU.
- [ ] **4.4 Prefill/decode disaggregation and tensor parallelism.** Concepts plus a small cost model, since I don't have a multi-GPU setup.

## What else to write

- **Paper notes.** Short notes on PagedAttention, Orca (continuous batching), FlashAttention and speculative decoding, each paired with the experiment that tests its claim.
- **Mistakes and dead ends.** What I tried that didn't work and why. These are often the most useful posts.
- **Setup guides.** Reproducible environments for Pascal, Ollama, llama.cpp and a cloud T4.
- **Tool comparisons.** The same workload on Ollama, llama.cpp and vLLM, with caveats about what each measures.
- **A running benchmark post.** A single page that I update as I learn more.

## Interactive pieces I want to build

- A sampling-distribution slider (1.4).
- A continuous-batching visualizer where requests enter and leave a batch (3.1).
- A KV-cache size calculator (0.4).
- A "reproduce this" box at the end of every post with commands and versions.

## Suggested order

0.1 → 0.2 → 1.2 → 1.4 → 0.4 → 2.1 → 2.3 → 2.4 → 2.6, then the rest as I go. That gets the tool-call work out early while the first posts build the vocabulary.
