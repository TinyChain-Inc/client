from __future__ import annotations

import ast
import pathlib


_PY_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _PY_ROOT / "tinychain"

_CANONICAL_PATH_PREFIXES: tuple[str, ...] = (
    "/state",
    "/service",
    "/lib",
    "/class",
    "/host",
    "/healthz",
)

_ALLOWED_TRY_IMPORT_ROOTS: tuple[str, ...] = ("tensorflow", "torch", "jax")
_TYPE_LADDER_ALLOWED_FUNCTIONS: frozenset[str] = frozenset(
    {
        # Explicit encode/decode boundaries.
        "autobox",
        "_json_of",
        "from_json",
        "_value_runtime",
        "_scalar_like",
        # Explicit route/result normalization boundaries.
        "_compile_route_rtype",
        "_compile_opdef_route",
        "_to_opref",
    }
)


def _is_state_root_uri_call(node: ast.Call) -> bool:
    name = _call_name(node.func)
    if name not in {"uri", "path"} or len(node.args) != 1:
        return False
    arg = node.args[0]
    return isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value == "state"


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

            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "id"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "TCRef"
            ):
                violations.append(f"{relative_path}:{line}: {segment}")
                continue

            if name in subclasses_by_base and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Call):
                    inner_name = _call_name(first_arg.func)
                    if inner_name in subclasses_by_base[name]:
                        violations.append(f"{relative_path}:{line}: {segment}")
                        continue

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


def test_native_state_paths_use_type_uri_subjects() -> None:
    violations: list[str] = []

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative_path = path.relative_to(_SRC_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            name = _call_name(node.func)
            if name not in {"path", "uri"} or not node.args:
                continue

            first = node.args[0]
            if not (isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value == "state"):
                continue

            if relative_path == "state/base.py" and _is_state_root_uri_call(node):
                continue

            line = getattr(node, "lineno", 0)
            segment = ast.get_source_segment(source, node) or ""
            violations.append(f"{relative_path}:{line}: {segment}")

    assert not violations, "forbidden direct state-root URI/path construction found:\n" + "\n".join(violations)


def _import_root_from_node(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        if not node.names:
            return None
        return node.names[0].name.split(".", 1)[0]
    if isinstance(node, ast.ImportFrom):
        if node.module is None:
            return None
        return node.module.split(".", 1)[0]
    return None


def test_try_imports_are_banned_except_large_dependency_type_converters() -> None:
    violations: list[str] = []

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative_path = path.relative_to(_SRC_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue

            imported_roots = [
                root
                for stmt in node.body
                for root in [_import_root_from_node(stmt)]
                if root is not None
            ]
            if not imported_roots:
                continue

            disallowed = [root for root in imported_roots if root not in _ALLOWED_TRY_IMPORT_ROOTS]
            if not disallowed:
                continue

            line = getattr(node, "lineno", 0)
            segment = ast.get_source_segment(source, node) or ""
            violations.append(f"{relative_path}:{line}: {segment}")

    assert not violations, "forbidden try-import patterns found:\n" + "\n".join(violations)


def test_symbolic_cmp_key_helpers_are_banned() -> None:
    violations: list[str] = []

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative_path = path.relative_to(_SRC_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "_cmp_key":
                continue

            line = getattr(node, "lineno", 0)
            violations.append(f"{relative_path}:{line}: def _cmp_key(...)")

    assert not violations, "forbidden _cmp_key helpers found:\n" + "\n".join(violations)


def test_coerce_naming_is_banned_in_symbolic_runtime() -> None:
    violations: list[str] = []

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative_path = path.relative_to(_SRC_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and "coerce" in node.name:
                line = getattr(node, "lineno", 0)
                violations.append(f"{relative_path}:{line}: def {node.name}(...)")
            elif isinstance(node, ast.Attribute) and isinstance(node.attr, str) and "coerce" in node.attr:
                line = getattr(node, "lineno", 0)
                segment = ast.get_source_segment(source, node) or node.attr
                violations.append(f"{relative_path}:{line}: {segment}")

    assert not violations, "forbidden coerce naming found:\n" + "\n".join(violations)


def test_runtime_uri_composition_bans_local_uri_constructors_and_constant_tables() -> None:
    violations: list[str] = []
    target_files = {
        "state/scalar/__init__.py",
        "state/scalar/refs.py",
        "state/scalar/opdef.py",
        "autodiff/reflection.py",
        "autodiff/routes.py",
        "codec.py",
    }

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative_path = path.relative_to(_SRC_ROOT).as_posix()
        if relative_path not in target_files:
            continue

        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for stmt in tree.body:
            targets: list[ast.expr] = []
            if isinstance(stmt, ast.Assign):
                targets = list(stmt.targets)
            elif isinstance(stmt, ast.AnnAssign):
                targets = [stmt.target]

            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id.endswith("_URI") or target.id.endswith("_PATH"):
                    line = getattr(target, "lineno", 0)
                    violations.append(f"{relative_path}:{line}: {target.id}")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.endswith("_tag") or node.name.endswith("_path") or node.name.endswith("_uri"):
                    line = getattr(node, "lineno", 0)
                    violations.append(f"{relative_path}:{line}: def {node.name}(...)")
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.endswith("_TAG"):
                        line = getattr(node, "lineno", 0)
                        violations.append(f"{relative_path}:{line}: {target.id}")

    assert not violations, "forbidden runtime URI constructor/table patterns found:\n" + "\n".join(violations)


def test_runtime_bans_uri_constructor_path_extraction() -> None:
    violations: list[str] = []

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative_path = path.relative_to(_SRC_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if node.attr != "path":
                continue
            if not isinstance(node.value, ast.Call):
                continue
            if _call_name(node.value.func) != "URI":
                continue

            line = getattr(node, "lineno", 0)
            segment = ast.get_source_segment(source, node) or "URI(...).path"
            violations.append(f"{relative_path}:{line}: {segment}")

    assert not violations, "forbidden URI(...).path extraction found; use URI values in domain logic and str(URI(...)) at boundaries:\n" + "\n".join(violations)


def test_runtime_bans_redundant_str_uri_wraps() -> None:
    violations: list[str] = []

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative_path = path.relative_to(_SRC_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node.func) != "str" or len(node.args) != 1:
                continue

            inner = node.args[0]
            if not isinstance(inner, ast.Call) or _call_name(inner.func) != "URI":
                continue
            if len(inner.args) != 1 or inner.keywords:
                continue

            subject = inner.args[0]
            if isinstance(subject, ast.Name) and subject.id.endswith("_URI"):
                line = getattr(node, "lineno", 0)
                segment = ast.get_source_segment(source, node) or "str(URI(...))"
                violations.append(f"{relative_path}:{line}: {segment}")
            elif isinstance(subject, ast.Attribute) and subject.attr.endswith("_URI"):
                line = getattr(node, "lineno", 0)
                segment = ast.get_source_segment(source, node) or "str(URI(...))"
                violations.append(f"{relative_path}:{line}: {segment}")

    assert not violations, "forbidden redundant str(URI(<..._URI>)) wrappers found; use str(<..._URI>) directly:\n" + "\n".join(violations)


def test_runtime_bans_uri_parse_str_wraps() -> None:
    violations: list[str] = []

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative_path = path.relative_to(_SRC_ROOT).as_posix()
        if relative_path == "uri.py":
            continue

        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "parse":
                continue
            if _call_name(node.func.value) != "URI":
                continue
            if len(node.args) != 1:
                continue

            arg = node.args[0]
            if isinstance(arg, ast.Call) and _call_name(arg.func) == "str":
                line = getattr(node, "lineno", 0)
                segment = ast.get_source_segment(source, node) or "URI.parse(str(...))"
                violations.append(f"{relative_path}:{line}: {segment}")

    assert not violations, "forbidden URI.parse(str(...)) wrappers found; pass URI values directly (URI(...)) or parse raw strings at boundaries:\n" + "\n".join(violations)


def _is_isinstance_call(test: ast.AST) -> bool:
    if not isinstance(test, ast.Call):
        return False

    name = _call_name(test.func)
    return name == "isinstance" and len(test.args) >= 1


def _ladder_subject_key(test: ast.AST) -> str | None:
    if not _is_isinstance_call(test):
        return None

    assert isinstance(test, ast.Call)
    subject = test.args[0]
    if not isinstance(subject, (ast.Name, ast.Attribute)):
        return None

    return ast.dump(subject)


def _is_elif(node: ast.If, parent_by_child: dict[ast.AST, ast.AST]) -> bool:
    parent = parent_by_child.get(node)
    if not isinstance(parent, ast.If):
        return False
    return bool(parent.orelse) and parent.orelse[0] is node


def _iter_if_ladder(node: ast.If) -> list[ast.If]:
    ladder = [node]
    cursor = node
    while len(cursor.orelse) == 1 and isinstance(cursor.orelse[0], ast.If):
        cursor = cursor.orelse[0]
        ladder.append(cursor)
    return ladder


def test_type_ladder_dispatch_is_banned_outside_explicit_dispatch_boundaries() -> None:
    violations: list[str] = []

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative_path = path.relative_to(_SRC_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        parent_by_child = _build_parent_map(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if _is_elif(node, parent_by_child):
                continue

            ladder = _iter_if_ladder(node)
            if len(ladder) < 3:
                continue

            keys = [_ladder_subject_key(branch.test) for branch in ladder]
            if any(key is None for key in keys):
                continue
            if len(set(keys)) != 1:
                continue

            owner = parent_by_child.get(node)
            while owner is not None and not isinstance(owner, ast.FunctionDef):
                owner = parent_by_child.get(owner)

            owner_name = owner.name if isinstance(owner, ast.FunctionDef) else None
            if owner_name in _TYPE_LADDER_ALLOWED_FUNCTIONS:
                continue

            line = getattr(node, "lineno", 0)
            segment = ast.get_source_segment(source, node) or "if ..."
            violations.append(f"{relative_path}:{line}: {segment}")

    assert not violations, (
        "forbidden shared-helper type-dispatch ladders found; move behavior to type-specific implementations:\n"
        + "\n".join(violations)
    )
