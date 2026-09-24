from __future__ import annotations

import gzip
import importlib.util
import os
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[4]
    / "scripts"
    / "benchmarking"
    / "kernels"
    / "benchmark_rpa_primitive.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_rpa_primitive", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_find_llo_proof_selects_highest_count_and_digests_plaintext(tmp_path: Path):
    lower = tmp_path / "a-final_bundles.txt"
    lower.write_text("%v0 = vadd.f32 %v1, %v2\n", encoding="utf-8")
    selected = tmp_path / "nested" / "b-final_bundles.txt.gz"
    selected.parent.mkdir()
    payload = b"%v0 = vadd.f32 %v1, %v2 ;; %v3 = vadd.f32 %v0, %v2\n"
    with gzip.open(selected, "wb") as stream:
        stream.write(payload)

    proof = MODULE.find_llo_proof(
        tmp_path,
        expected_opcode="vadd.f32",
        minimum_opcode_count=2,
    )

    assert proof.artifact_path == str(selected.resolve())
    assert proof.opcode_count == 2
    assert proof.artifact_digest.startswith("sha256:")


def test_find_llo_proof_rejects_folded_chain(tmp_path: Path):
    (tmp_path / "only-final_bundles.txt").write_text(
        "%v0 = vadd.f32 %v1, %v2\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="but 16 are required"):
        MODULE.find_llo_proof(
            tmp_path,
            expected_opcode="vadd.f32",
            minimum_opcode_count=16,
        )


def test_configure_compiler_dumps_replaces_destinations(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "XLA_FLAGS", "--unrelated=true --xla_dump_to=/old --xla_dump_hlo_as_text"
    )
    monkeypatch.setenv(
        "LIBTPU_INIT_ARGS",
        "--another=true --xla_mosaic_dump_to=/old-mosaic --xla_jf_dump_to=/old-llo",
    )

    MODULE.configure_compiler_dumps(tmp_path)

    assert "--unrelated=true" in os.environ["XLA_FLAGS"]
    assert "--xla_dump_to=/old" not in os.environ["XLA_FLAGS"]
    assert f"--xla_dump_to={tmp_path / 'xla'}" in os.environ["XLA_FLAGS"]
    assert "--another=true" in os.environ["LIBTPU_INIT_ARGS"]
    assert "/old-mosaic" not in os.environ["LIBTPU_INIT_ARGS"]
    assert f"--xla_jf_dump_to={tmp_path / 'llo'}" in os.environ["LIBTPU_INIT_ARGS"]


def test_cli_contract_rejects_inconsistent_opcode_gate(tmp_path: Path):
    argv = [
        "--phase",
        "compile",
        "--variant",
        "repeated",
        "--trial-index",
        "0",
        "--repetitions",
        "16",
        "--samples",
        "3",
        "--warmup-iterations",
        "2",
        "--dump-directory",
        str(tmp_path),
        "--expected-opcode",
        "vadd.f32",
        "--minimum-opcode-count",
        "1",
    ]

    with pytest.raises(SystemExit):
        MODULE.parse_args(argv)
