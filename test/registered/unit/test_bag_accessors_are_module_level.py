"""The config-bag accessors are module functions, never object members.

`get_disagg()` and its siblings answer for the whole process. Calling one as
`something.get_disagg()` reads as "this object's disaggregation config" and
raises at runtime, on whichever path first reaches it -- a sweep that rewrites
`x.server_args.field` into `x.get_disagg().field` produces exactly that, and
only a served request notices.
"""

import ast
import pathlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sglang

_PACKAGE_ROOT = pathlib.Path(sglang.__file__).resolve().parent

_BAG_ACCESSORS = frozenset(
    {
        "get_exec",
        "get_memory",
        "get_schedule",
        "get_model",
        "get_spec",
        "get_serving",
        "get_observability",
        "get_disagg",
        "get_lora",
        "get_mm",
        "get_device",
        "get_parallel",
    }
)


def _runtime_context_names(tree):
    """Names this module imported from runtime_context.

    Scoping the check to them keeps it away from unrelated APIs that happen to
    share a name (``multiprocessing.get_context``, a communicator's
    ``get_device``).
    """
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith(
            "runtime_context"
        ):
            for a in node.names:
                name = a.asname or a.name
                if name in _BAG_ACCESSORS:
                    names.add(name)
    return names


class TestBagAccessorsAreCalledAsFunctions(CustomTestCase):
    def test_no_accessor_is_called_on_an_object(self):
        offenders = []
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            imported = _runtime_context_names(tree)
            if not imported:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute) or func.attr not in imported:
                    continue
                if isinstance(func.value, ast.Name) and func.value.id == "self":
                    # A class may own a same-named member; the sweep's mistake
                    # is calling the accessor on a *config-carrying* object.
                    continue
                offenders.append(
                    f"{path.relative_to(_PACKAGE_ROOT.parent)}:{node.lineno} "
                    f"calls .{func.attr}() on an object"
                )
        self.assertEqual(
            offenders,
            [],
            "config-bag accessors answer for the process, not for an object:\n  "
            + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
