"""Benchmark offline embedding throughput using LLM.embed().

Unlike benchmark_serving.py (which goes through the HTTP API), this script
drives the vLLM engine in-process. Benefits:
  - No jnp.split recompilation: all batches use the same padded token bucket,
    because warmup pre-compiles every shape before measurement.
  - No HTTP/async overhead in the critical path.
  - Deterministic batch composition (the scheduler sees all requests at once).

Usage:
    # Random prompts, input length 4, 256 prompts
    python benchmark_offline_embedding.py \\
        --model Qwen/Qwen3-Embedding-8B \\
        --random-input-len 4 \\
        --num-prompts 256

    # With merged sequence path
    MERGE_MIXED_SEQS=1 python benchmark_offline_embedding.py \\
        --model Qwen/Qwen3-Embedding-8B \\
        --random-input-len 4 \\
        --num-prompts 256
"""

import gc
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_core import SampleRequest
from benchmark_dataset import RandomDataset

from vllm import LLM
from vllm.engine.arg_utils import EngineArgs

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser


@dataclass
class EmbeddingBenchmarkMetrics:
    num_requests: int
    total_input_tokens: int
    duration_s: float
    request_throughput: float    # req/s
    token_throughput: float      # tok/s
    mean_latency_ms: float
    median_latency_ms: float
    std_latency_ms: float
    p99_latency_ms: float
    p999_latency_ms: float


def sample_requests(
    tokenizer,
    num_prompts: int,
    input_len: int,
    range_ratio: float,
    seed: int,
) -> list[SampleRequest]:
    np.random.seed(seed)
    random.seed(seed)
    dataset = RandomDataset(random_seed=seed)
    return dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_prompts,
        input_len=input_len,
        output_len=1,           # unused for embeddings
        range_ratio=range_ratio,
    )


def run_embed(llm: LLM, requests: list[SampleRequest],
              use_tqdm: bool = False) -> float:
    """Run llm.embed() on requests and return wall-clock seconds."""
    prompts = [{"prompt_token_ids": r.prompt} if isinstance(r.prompt, list)
               else r.prompt for r in requests]
    t0 = time.perf_counter()
    llm.embed(prompts, use_tqdm=use_tqdm)
    return time.perf_counter() - t0


def benchmark(
    llm: LLM,
    requests: list[SampleRequest],
    num_iters_warmup: int,
    num_iters: int,
    batch_size: int,
    tokenizer,
) -> EmbeddingBenchmarkMetrics:
    # Split into batches
    batches = [requests[i:i + batch_size]
               for i in range(0, len(requests), batch_size)]

    # Warmup — pre-compiles all bucket sizes the measurement will hit
    print(f"Warming up ({num_iters_warmup} iterations over "
          f"{len(batches)} batch(es) of up to {batch_size} requests)...")
    for _ in tqdm(range(num_iters_warmup), desc="Warmup"):
        for batch in batches:
            run_embed(llm, batch)
    print("Warmup done.")

    # Measurement
    print(f"Benchmarking ({num_iters} iterations)...")
    per_iter_latencies: list[float] = []
    for _ in tqdm(range(num_iters), desc="Benchmark"):
        t_iter_start = time.perf_counter()
        for batch in batches:
            run_embed(llm, batch)
        per_iter_latencies.append(time.perf_counter() - t_iter_start)

    # prompt_len counts content tokens only (RandomDataset strips special tokens
    # before storing it). Add back the special tokens the model will actually see
    # (e.g. BOS) so the reported token count matches what the TPU processes.
    num_special = tokenizer.num_special_tokens_to_add()
    total_input_tokens = sum(r.prompt_len + num_special for r in requests)
    n = len(requests)

    # Per-request latency from the fastest iteration (most stable measurement)
    best_iter_s = min(per_iter_latencies)
    per_req_ms = [best_iter_s / n * 1000] * n  # uniform per-request estimate

    # Report over the mean of all iterations for throughput
    mean_iter_s = float(np.mean(per_iter_latencies))

    return EmbeddingBenchmarkMetrics(
        num_requests=n,
        total_input_tokens=total_input_tokens,
        duration_s=mean_iter_s,
        request_throughput=n / mean_iter_s,
        token_throughput=total_input_tokens / mean_iter_s,
        mean_latency_ms=float(np.mean(per_iter_latencies)) / n * 1000,
        median_latency_ms=float(np.median(per_iter_latencies)) / n * 1000,
        std_latency_ms=float(np.std(per_iter_latencies)) / n * 1000,
        p99_latency_ms=float(np.percentile(per_iter_latencies, 99)) / n * 1000,
        p999_latency_ms=float(
            np.percentile(per_iter_latencies, 99.9)) / n * 1000,
    )


def print_metrics(m: EmbeddingBenchmarkMetrics) -> None:
    w = 50
    print("=" * w)
    print(f"{'Offline Embedding Benchmark Results':^{w}}")
    print("=" * w)
    print(f"{'Requests processed:':<35} {m.num_requests}")
    print(f"{'Total input tokens:':<35} {m.total_input_tokens}")
    print(f"{'Avg input tokens/request:':<35} {m.total_input_tokens/m.num_requests:.1f}")
    print(f"{'Duration (s):':<35} {m.duration_s:.3f}")
    print("-" * w)
    print(f"{'Request throughput (req/s):':<35} {m.request_throughput:.2f}")
    print(f"{'Token throughput (tok/s):':<35} {m.token_throughput:.2f}")
    print("-" * w)
    print(f"{'Mean latency/req (ms):':<35} {m.mean_latency_ms:.2f}")
    print(f"{'Median latency/req (ms):':<35} {m.median_latency_ms:.2f}")
    print(f"{'Std latency/req (ms):':<35} {m.std_latency_ms:.2f}")
    print(f"{'P99 latency/req (ms):':<35} {m.p99_latency_ms:.2f}")
    print(f"{'P99.9 latency/req (ms):':<35} {m.p999_latency_ms:.2f}")
    print("=" * w)


def main(args):
    seed = getattr(args, "seed", 0)
    random.seed(seed)
    np.random.seed(seed)

    tokenizer_id = args.tokenizer if args.tokenizer else args.model
    tokenizer = get_tokenizer(
        tokenizer_id,
        tokenizer_mode=getattr(args, "tokenizer_mode", "auto"),
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )

    print(f"Sampling {args.num_prompts} requests "
          f"(input_len={args.random_input_len}, "
          f"range_ratio={args.random_range_ratio})...")
    requests = sample_requests(
        tokenizer=tokenizer,
        num_prompts=args.num_prompts,
        input_len=args.random_input_len,
        range_ratio=args.random_range_ratio,
        seed=seed,
    )

    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM.from_engine_args(engine_args)

    gc.collect()
    gc.freeze()

    metrics = benchmark(
        llm=llm,
        requests=requests,
        num_iters_warmup=args.num_iters_warmup,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
    )
    print_metrics(metrics)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark offline embedding throughput via LLM.embed().")

    parser.add_argument("--num-prompts", type=int, default=256,
                        help="Total number of prompts to embed.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Requests per llm.embed() call. "
                             "Use a value <= max_num_seqs.")
    parser.add_argument("--num-iters-warmup", type=int, default=5,
                        help="Warmup iterations (pre-compiles all JIT shapes).")
    parser.add_argument("--num-iters", type=int, default=10,
                        help="Measurement iterations.")

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument("--random-input-len", type=int, default=4,
                               help="Input token length per request.")
    random_group.add_argument("--random-range-ratio", type=float, default=0.0,
                               help="Length variation ± this fraction of "
                                    "random-input-len. 0 = all requests "
                                    "have exactly the same length.")

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
