from __future__ import annotations

import ast
import pathlib


_PY_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _PY_ROOT / "tinychain"

_ALLOWLIST_SEGMENTS: dict[str, set[str]] = {
    "state/scalar/refs.py": {"TCRef(IdRef(name))"},
}

_CANONICAL_PATH_PREFIXES: tuple[str, ...] = (
    "/state",
    "/service",
    "/lib",
    "/class",
    "/host",
    "/healthz",
)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _collect_subclasses_by_base() -> dict[str, set[str]]:
    direct_children: dict[str, set[str]] = {}
    all_classes: set[str] = set()

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            class_name = node.name
            all_classes.add(class_name)
            for base in node.bases:
                base_name = _call_name(base)
                if base_name is None:
                    continue
                direct_children.setdefault(base_name, set()).add(class_name)

    subclasses_by_base: dict[str, set[str]] = {name: set() for name in all_classes}

    for base in all_classes:
        visited: set[str] = set()
        stack = list(direct_children.get(base, ()))
        while stack:
            child = stack.pop()
            if child in visited:
                continue
            visited.add(child)
            stack.extend(direct_children.get(child, ()))
        subclasses_by_base[base] = visited

    return subclasses_by_base


def _build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parent_by_child: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_by_child[child] = parent
    return parent_by_child


def _is_docstring_literal(node: ast.Constant, parent_by_child: dict[ast.AST, ast.AST]) -> bool:
    parent = parent_by_child.get(node)
    if not isinstance(parent, ast.Expr) or parent.value is not node:
        return False

    owner = parent_by_child.get(parent)
    if not isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return False

    return bool(owner.body) and owner.body[0] is parent


def test_symbolic_instantiation_style_guards() -> None:
    violations: list[str] = []
    subclasses_by_base = _collect_subclasses_by_base()

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

            if name in subclasses_by_base and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Call):
                    inner_name = _call_name(first_arg.func)
                    if inner_name in subclasses_by_base[name]:
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


def test_runtime_paths_use_uri_helpers() -> None:
    violations: list[str] = []

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative_path = path.relative_to(_SRC_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        parent_by_child = _build_parent_map(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if not node.value.startswith(_CANONICAL_PATH_PREFIXES):
                continue
            if _is_docstring_literal(node, parent_by_child):
                continue

            line = getattr(node, "lineno", 0)
            violations.append(f"{relative_path}:{line}: {node.value!r}")

    assert not violations, "forbidden canonical path literals found:\n" + "\n".join(violations)
