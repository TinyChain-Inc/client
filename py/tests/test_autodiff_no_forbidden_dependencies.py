"""Static architecture check: the autodiff package imports no ILC-specific module.

The extension seam's whole design point (structured dependency analysis,
extensible program lowering, traced optimizer updates) is that `client` owns
generic graph analysis and traversal while a downstream consumer -- of which
ILC is the concrete motivating example -- owns its own target representation,
supported-operator mapping, fusion policy, and runtime. If a later change ever
made `tinychain.autodiff` import something ILC-specific, that would silently
reverse the boundary this whole task exists to establish. Reviewing for that
by hand does not scale; this mechanically walks the import graph instead,
following the same AST-over-source-tree style as the existing symbolic
instantiation guards.
"""

from __future__ import annotations

import ast
import pathlib


_AUTODIFF_ROOT = pathlib.Path(__file__).resolve().parents[1] / "tinychain" / "autodiff"


def _imported_module_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _has_ilc_segment(module_name: str) -> bool:
    """Report whether any dotted path segment of *module_name* names ILC.

    Matches an exact ``ilc`` segment or an ``ilc_``/``ilc-api``-style prefixed
    segment, so both ``import ilc`` and ``from ilc_api.target import Foo``
    are caught, while an unrelated identifier that merely contains the
    substring ``ilc`` (there are none in this codebase today) is not.
    """
    for segment in module_name.split("."):
        lowered = segment.lower()
        if lowered == "ilc" or lowered.startswith("ilc_") or lowered.startswith("ilc-"):
            return True
    return False


def test_autodiff_package_imports_no_ilc_specific_module() -> None:
    violations: list[str] = []
    for path in sorted(_AUTODIFF_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for module_name in _imported_module_names(tree):
            if _has_ilc_segment(module_name):
                violations.append(f"{path.relative_to(_AUTODIFF_ROOT)}: imports {module_name!r}")

    assert not violations, "forbidden ILC-specific import found:\n" + "\n".join(violations)
