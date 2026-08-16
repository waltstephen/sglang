"""Launch paths read the configured parallel sizes, not the live ones.

`get_parallel().pp_size` and its four siblings are read-through properties over
the process groups, so they answer only after distributed init. The launcher
decides how many processes to spawn *before* that, and a live read there raises
`Distributed environment is not initialized` -- a startup crash no unit test
reaches, because nothing short of booting a server runs the launcher.
"""

import ast
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PACKAGE_ROOT = pathlib.Path(sglang.__file__).resolve().parent

# The five sizes ParallelContext shadows with a live property; each has a
# `configured_*` accessor that answers from the config bag instead.
_LIVE_SHADOWED = {
    "tp_size": "configured_tp_size()",
    "pp_size": "configured_pp_size()",
    "moe_dp_size": "configured_moe_dp_size()",
    "attn_cp_size": "configured_attn_cp_size()",
    "dcp_size": "configured_dcp_size()",
}

# Modules that run before this process joins its process groups.
_PRE_DIST = (
    "srt/entrypoints/engine.py",
    "srt/entrypoints/http_server.py",
    "srt/entrypoints/sidecar.py",
    "srt/ray/engine.py",
    "srt/ray/data_parallel_controller.py",
    "srt/managers/data_parallel_controller.py",
)


class TestLaunchPathsReadConfiguredSizes(CustomTestCase):
    def test_no_live_topology_read_before_distributed_init(self):
        offenders = []
        for rel in _PRE_DIST:
            path = _PACKAGE_ROOT / rel
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                if node.attr not in _LIVE_SHADOWED:
                    continue
                base = node.value
                if (
                    isinstance(base, ast.Call)
                    and isinstance(base.func, ast.Name)
                    and base.func.id == "get_parallel"
                ):
                    offenders.append(
                        f"{rel}:{node.lineno} reads the live {node.attr}; "
                        f"use {_LIVE_SHADOWED[node.attr]}"
                    )
        self.assertEqual(
            offenders,
            [],
            "launch paths run before distributed init:\n  " + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
