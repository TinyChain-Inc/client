from __future__ import annotations

import ast
import pathlib


_PY_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _PY_ROOT / "tinychain"

_ALLOWLIST_SEGMENTS: dict[str, set[str]] = {
    "state/scalar/refs.py": {"TCRef(IdRef(name))"},
}


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_symbolic_instantiation_style_guards() -> None:
    violations: list[str] = []

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative_path = path.relative_to(_SRC_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            name = _call_name(node.func)
            line = getattr(node, "lineno", 0)
            segment = ast.get_source_segment(source, node) or ""

            if name == "State":
                violations.append(f"{relative_path}:{line}: {segment}")
                continue

            if name != "TCRef" or not node.args:
                continue

            first_arg = node.args[0]
            if not isinstance(first_arg, ast.Call):
                continue

            if _call_name(first_arg.func) != "IdRef":
                continue

            allowed = segment in _ALLOWLIST_SEGMENTS.get(relative_path, set())
            if not allowed:
                violations.append(f"{relative_path}:{line}: {segment}")

    assert not violations, "forbidden symbolic construction forms found:\n" + "\n".join(violations)
