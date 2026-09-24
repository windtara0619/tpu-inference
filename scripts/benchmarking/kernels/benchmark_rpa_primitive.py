# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evidence-gated TPU microbenchmark for an RPA LLO ``vadd.f32`` candidate.

This program is a target-runtime adapter for KernelAgents Phase 6.  Each
invocation performs exactly one compile, warmup, or measurement phase and emits
one JSON object on stdout.  Compiler dumps and stderr are retained by the
generic Phase 6 executor.

The repeated variant contains ``repetitions + 1`` serial FP32 vector additions;
the control contains one.  Therefore ``(repeated - control) / repetitions`` is
the calibrated dependent-add estimate.  Timing is rejected unless a generated
final LLO bundle proves that the compiler retained the requested opcode count.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VECTOR_SHAPE = (8, 128)
DTYPE = "float32"
VARIANTS = ("repeated", "control")
PHASES = ("compile", "warmup", "measurement")


@dataclass(frozen=True)
class LloProof:
    artifact_path: str
    artifact_digest: str
    opcode: str
    opcode_count: int


def _replace_flags(original: str, replacements: Sequence[str]) -> str:
    """Replace named ``--flag`` values while retaining unrelated arguments."""

    replacement_names = {item.split("=", 1)[0] for item in replacements}
    retained = [
        item
        for item in shlex.split(original)
        if item.split("=", 1)[0] not in replacement_names
    ]
    return shlex.join([*retained, *replacements])


def configure_compiler_dumps(dump_directory: Path) -> None:
    """Configure fresh compiler dumps before JAX or libtpu is imported."""

    dump_directory = dump_directory.resolve()
    xla_directory = dump_directory / "xla"
    mosaic_directory = dump_directory / "mosaic"
    llo_directory = dump_directory / "llo"
    cache_directory = dump_directory / "jax_cache"
    for directory in (
        xla_directory,
        mosaic_directory,
        llo_directory,
        cache_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["XLA_FLAGS"] = _replace_flags(
        os.environ.get("XLA_FLAGS", ""),
        (
            f"--xla_dump_to={xla_directory}",
            "--xla_dump_hlo_as_text",
        ),
    )
    os.environ["LIBTPU_INIT_ARGS"] = _replace_flags(
        os.environ.get("LIBTPU_INIT_ARGS", ""),
        (
            f"--xla_mosaic_dump_to={mosaic_directory}",
            "--xla_mosaic_enable_dump_debug_info=true",
            "--xla_mosaic_enable_llo_source_annotations=true",
            f"--xla_jf_dump_to={llo_directory}",
            "--xla_jf_dump_llo_text=true",
            "--xla_jf_dump_llo_static_gaps=true",
            "--xla_jf_emit_annotations=true",
        ),
    )
    # A per-phase empty cache directory prevents an older persistent entry from
    # suppressing the LLO dump required by this evidence gate.
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_directory)
    os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"


def _read_maybe_gzip(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            return stream.read()
    return path.read_bytes()


def find_llo_proof(
    dump_directory: Path,
    *,
    expected_opcode: str,
    minimum_opcode_count: int,
) -> LloProof:
    """Select the final LLO bundle with the most exact opcode occurrences."""

    if not expected_opcode or minimum_opcode_count <= 0:
        raise ValueError("opcode proof requires a name and a positive count")
    opcode_pattern = re.compile(
        rb"(?<![A-Za-z0-9_.])"
        + re.escape(expected_opcode.encode("ascii"))
        + rb"(?![A-Za-z0-9_.])"
    )
    matches: list[tuple[int, str, Path, bytes]] = []
    for path in sorted(dump_directory.rglob("*final_bundles*.txt*")):
        if not path.is_file():
            continue
        payload = _read_maybe_gzip(path)
        count = len(opcode_pattern.findall(payload))
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        matches.append((count, digest, path, payload))
    if not matches:
        raise RuntimeError(
            f"no final LLO bundle was generated beneath {dump_directory}"
        )
    count, digest, path, _ = max(
        matches, key=lambda item: (item[0], item[2].as_posix())
    )
    if count < minimum_opcode_count:
        observed = ", ".join(
            f"{candidate.relative_to(dump_directory)}={candidate_count}"
            for candidate_count, _, candidate, _ in matches
        )
        raise RuntimeError(
            f"compiler retained {count} {expected_opcode} instruction(s), "
            f"but {minimum_opcode_count} are required; all bundles: {observed}"
        )
    return LloProof(
        artifact_path=str(path.resolve()),
        artifact_digest=digest,
        opcode=expected_opcode,
        opcode_count=count,
    )


def _build_program(*, operation_count: int, name: str):
    """Build a one-program native-vector Pallas kernel lazily."""

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if operation_count <= 0:
        raise ValueError("operation_count must be positive")

    def serial_add_kernel(x_ref, delta_ref, output_ref):
        value = x_ref[...]
        delta = delta_ref[...]
        for _ in range(operation_count):
            value = value + delta
        output_ref[...] = value

    vector_spec = pl.BlockSpec(VECTOR_SHAPE, lambda _program_id: (0, 0))
    call = pl.pallas_call(
        serial_add_kernel,
        out_shape=jax.ShapeDtypeStruct(VECTOR_SHAPE, jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(vector_spec, vector_spec),
            out_specs=vector_spec,
            grid=(1,),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
        ),
        name=name,
    )
    return jax.jit(call)


def _compile_program(*, variant: str, repetitions: int):
    import jax
    import jax.numpy as jnp

    operation_count = repetitions + 1 if variant == "repeated" else 1
    program = _build_program(
        operation_count=operation_count,
        name=f"rpa_vadd_f32_{variant}_{operation_count}",
    )
    x = jnp.full(VECTOR_SHAPE, 1.25, dtype=jnp.float32)
    delta = jnp.full(VECTOR_SHAPE, 1.0 / 1024.0, dtype=jnp.float32)
    compile_start_ns = time.perf_counter_ns()
    executable = program.lower(x, delta).compile()
    compile_duration_us = (time.perf_counter_ns() - compile_start_ns) / 1_000.0
    return jax, jnp, executable, x, delta, operation_count, compile_duration_us


def _check_result(
    jax_module: Any,
    jnp_module: Any,
    output: Any,
    *,
    operation_count: int,
) -> None:
    jax_module.block_until_ready(output)
    expected = jnp_module.full(
        VECTOR_SHAPE,
        1.25 + operation_count / 1024.0,
        dtype=jnp_module.float32,
    )
    if not bool(jnp_module.all(output == expected)):
        maximum_error = float(jnp_module.max(jnp_module.abs(output - expected)))
        raise RuntimeError(
            f"serial vadd correctness check failed; maximum error={maximum_error}"
        )


def _proof_payload(proof: LloProof) -> dict[str, object]:
    return {
        "executable_digest": proof.artifact_digest,
        "llo_artifact_path": proof.artifact_path,
        "llo_artifact_digest": proof.artifact_digest,
        "llo_opcode": proof.opcode,
        "llo_opcode_count": proof.opcode_count,
    }


def run_phase(args: argparse.Namespace) -> dict[str, object]:
    configure_compiler_dumps(args.dump_directory)
    (
        jax_module,
        jnp_module,
        executable,
        x,
        delta,
        operation_count,
        compile_duration_us,
    ) = _compile_program(variant=args.variant, repetitions=args.repetitions)
    proof = find_llo_proof(
        args.dump_directory,
        expected_opcode=args.expected_opcode,
        minimum_opcode_count=args.minimum_opcode_count,
    )
    common: dict[str, object] = {
        "phase": args.phase,
        "trial_index": args.trial_index,
        "variant": args.variant,
        "operation_count": operation_count,
        "shape": list(VECTOR_SHAPE),
        "dtype": DTYPE,
        **_proof_payload(proof),
    }
    if args.phase == "compile":
        return {
            **common,
            "cache_behavior": "cold_miss",
            "compile_duration_us": compile_duration_us,
        }

    for _ in range(args.warmup_iterations):
        _check_result(
            jax_module,
            jnp_module,
            executable(x, delta),
            operation_count=operation_count,
        )
    if args.phase == "warmup":
        return {
            **common,
            "cache_behavior": "warm_hit",
            "warmup_iterations": args.warmup_iterations,
        }

    samples_us = []
    for _ in range(args.samples):
        start_ns = time.perf_counter_ns()
        output = executable(x, delta)
        jax_module.block_until_ready(output)
        samples_us.append((time.perf_counter_ns() - start_ns) / 1_000.0)
    _check_result(
        jax_module,
        jnp_module,
        output,
        operation_count=operation_count,
    )
    return {
        **common,
        "cache_behavior": "warm_hit",
        "synchronization_method": "jax_block_until_ready",
        "synchronization_detail": (
            "time.perf_counter_ns around one compiled call followed by "
            "jax.block_until_ready"
        ),
        "measured_samples_us": samples_us,
    }


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--trial-index", type=int, required=True)
    parser.add_argument("--repetitions", type=_positive_integer, required=True)
    parser.add_argument("--samples", type=_positive_integer, required=True)
    parser.add_argument("--warmup-iterations", type=_positive_integer, required=True)
    parser.add_argument("--dump-directory", type=Path, required=True)
    parser.add_argument("--expected-opcode", required=True)
    parser.add_argument("--minimum-opcode-count", type=_positive_integer, required=True)
    args = parser.parse_args(argv)
    if args.trial_index < 0:
        parser.error("--trial-index must be nonnegative")
    if args.expected_opcode != "vadd.f32":
        parser.error("this adapter supports only --expected-opcode=vadd.f32")
    expected_minimum = args.repetitions if args.variant == "repeated" else 1
    if args.minimum_opcode_count != expected_minimum:
        parser.error(
            "--minimum-opcode-count must equal repetitions for repeated and 1 "
            "for control"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    payload = run_phase(parse_args(argv))
    json.dump(payload, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
