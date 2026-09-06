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

from unittest.mock import MagicMock, patch

import pytest

# Adjust the import below to match the actual location of your function
# For example: from my_module.moe_selector import select_moe_backend_from_fused_moe_config
from tpu_inference.layers.common.moe import FusedMoEMethodBase, MoEBackend
from tpu_inference.layers.common.sharding import ShardingAxisName
# We assume the function is in a file that can be imported.
# If testing locally without the full repo, you might need to adjust imports.
from tpu_inference.layers.vllm.interface.moe import \
    select_moe_backend_from_fused_moe_config


# Mock the FusedMoEConfig so we don't need vllm installed to test the logic
@pytest.fixture
def mock_moe_config():
    """Creates a mock object mimicking FusedMoEConfig."""
    config = MagicMock()
    return config


@pytest.mark.parametrize(
    "env_use_ep_kernel, config_use_ep, expected_backend",
    [
        # Case 1: EP Kernel Env=True, Config EP=True -> FUSED_MOE
        (True, True, MoEBackend.FUSED_MOE),

        # Case 2: EP Kernel Env=False, Config EP=True -> GMM_EP
        (False, True, MoEBackend.GMM_EP),

        # Case 3: EP Kernel Env=False, Config EP=False -> GMM_TP
        (False, False, MoEBackend.GMM_TP),

        # Case 4: EP Kernel Env=True, Config EP=False -> Fallback to GMM_TP
        (True, False, MoEBackend.GMM_TP),
    ])
def test_select_moe_backend_logic(monkeypatch, mock_moe_config,
                                  env_use_ep_kernel, config_use_ep,
                                  expected_backend):
    """
    Tests the main logic paths for backend selection based on environment variables
    and the MoE configuration.
    """
    # Use patch as a context manager to limit the scope of the change to this block.
    # This avoids using monkeypatch and ensures the mock is cleaned up immediately.
    with patch("tpu_inference.envs.USE_MOE_EP_KERNEL", env_use_ep_kernel):
        mock_moe_config.use_ep = config_use_ep

        result = select_moe_backend_from_fused_moe_config(mock_moe_config)

    assert result == expected_backend


@pytest.mark.parametrize(
    "moe_backend, expert_axis, expect_ep_axis_name_kwarg",
    [
        # Under DP-attention (NEW_MODEL_DESIGN), ShardingAxisName.EXPERT spans
        # more than one physical mesh axis (e.g. attn_dp *and* model).
        # FUSED_MOE must pick that up instead of the single "model" axis it
        # would otherwise default to.
        (MoEBackend.FUSED_MOE, ("attn_dp", "model"), ("attn_dp", "model")),
        # Plain tensor-parallel-only configs collapse to a single axis.
        (MoEBackend.FUSED_MOE, "model", "model"),
        # Non-fused backends don't need an ep_axis_name at all.
        (MoEBackend.GMM_EP, ("attn_dp", "model"), None),
        (MoEBackend.GMM_TP, ("attn_dp", "model"), None),
    ])
def test_fused_moe_method_base_uses_sharding_axis_name_expert(
        moe_backend, expert_axis, expect_ep_axis_name_kwarg):
    """FUSED_MOE's ep_axis_name must track ShardingAxisName.EXPERT (which may
    span multiple mesh axes under DP-attention), not the caller-supplied
    single-axis default."""
    ShardingAxisName.override(EXPERT=expert_axis)
    try:
        # The positional ep_axis_name arg is the legacy single-axis default
        # every call site passes; it must be ignored in favor of EXPERT.
        method = FusedMoEMethodBase(moe_backend, ep_axis_name="model")
    finally:
        ShardingAxisName.reset()

    if expect_ep_axis_name_kwarg is None:
        assert "ep_axis_name" not in method.extra_backend_kwargs
    else:
        assert method.extra_backend_kwargs[
            "ep_axis_name"] == expect_ep_axis_name_kwarg
