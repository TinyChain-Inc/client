from __future__ import annotations

import copy
import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Iterable


class AutographError(ValueError):
    pass


class AutographSyntaxError(AutographError):
    pass


class AutographNameError(AutographError):
    pass


class AutographControlFlowError(AutographError):
    pass


class AutographAssignmentError(AutographError):
    pass


class AutographMixedContextError(AutographError):
    pass


_RESERVED_NAMES = {
    "self",
    "cxt",
    "ctx",
    "txn",
    "result",
    "bind",
    "bind_auto",
    "form",
    "value",
    "_tc_autograph",
    "_tc_cxt",
}

_ALLOWED_GLOBALS = {
    "slice",
    "tc",
}


@dataclass
class _IfAssignments:
    map_expr: ast.expr
    names: list[str]


def transform(form, *, source: str | None = None):
    sig = inspect.signature(form)
    params = list(sig.parameters.values())
    if not params or params[0].name != "self":
        raise AutographSyntaxError("autograph requires a method whose first parameter is `self`")

    injected = {"cxt", "ctx", "txn"}
    if any(param.name in injected for param in params[1:]):
        raise AutographMixedContextError("explicit cxt/ctx/txn parameters are not supported in autograph mode")

    src = source if source is not None else inspect.getsource(form)
    src = textwrap.dedent(src)
    mod = ast.parse(src)
    fn = _find_function(mod, form.__name__)
    if fn is None:
        raise AutographSyntaxError(f"autograph could not locate function {form.__name__}")

    arg_names = {param.name for param in params[1:]}
    transformer = _AutographTransformer(arg_names)
    new_fn = transformer.transform(fn)
    new_mod = ast.Module(body=[new_fn], type_ignores=[])
    ast.fix_missing_locations(new_mod)

    globals_dict = dict(form.__globals__)
    if form.__closure__:
        for name, cell in zip(form.__code__.co_freevars, form.__closure__, strict=True):
            globals_dict[name] = cell.cell_contents

    filename = inspect.getsourcefile(form) or "<autograph>"
    exec(compile(new_mod, filename, "exec"), globals_dict)
    transformed = globals_dict[form.__name__]
    transformed.__defaults__ = form.__defaults__
    transformed.__kwdefaults__ = form.__kwdefaults__
    transformed.__annotations__ = getattr(form, "__annotations__", {}).copy()
    transformed.__doc__ = form.__doc__
    transformed.__module__ = form.__module__
    return transformed


def _find_function(mod: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


class _AutographTransformer(ast.NodeTransformer):
    def __init__(self, params: set[str]) -> None:
        self._params = params
        self._locals: set[str] = set()
        self._temp_counter = 0

    def transform(self, fn: ast.FunctionDef) -> ast.FunctionDef:
        fn = ast.FunctionDef(
            name=fn.name,
            args=fn.args,
            body=self._transform_body(fn.body),
            decorator_list=[],
            returns=fn.returns,
            type_comment=fn.type_comment,
        )
        return fn

    def _transform_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
            out.append(body[0])
            body = body[1:]

        out.extend(self._autograph_preamble())
        i = 0
        while i < len(body):
            stmt = body[i]
            # Ergonomic sugar: `if cond: return x; return y` lowers to one Cond return.
            if (
                isinstance(stmt, ast.If)
                and not stmt.orelse
                and _is_return_branch(stmt.body)
                and i + 1 < len(body)
                and isinstance(body[i + 1], ast.Return)
            ):
                stmt = ast.If(test=stmt.test, body=stmt.body, orelse=[body[i + 1]])
                out.extend(self._lower_stmt(stmt))
                i += 2
                continue

            out.extend(self._lower_stmt(stmt))
            i += 1
        return out

    def _autograph_preamble(self) -> list[ast.stmt]:
        import_stmt = ast.Import(names=[ast.alias(name="tinychain", asname="_tc_autograph")])
        ctx_stmt = ast.Assign(
            targets=[ast.Name(id="cxt", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(value=ast.Name(id="_tc_autograph", ctx=ast.Load()), attr="state", ctx=ast.Load()),
                    attr="context",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            ),
        )
        alias_stmt = ast.Assign(
            targets=[ast.Name(id="_tc_cxt", ctx=ast.Store())],
            value=ast.Name(id="cxt", ctx=ast.Load()),
        )
        return [import_stmt, ctx_stmt, alias_stmt]

    def _lower_stmt(self, stmt: ast.stmt) -> list[ast.stmt]:
        if isinstance(stmt, ast.Assign):
            return self._lower_assign(stmt)
        if isinstance(stmt, ast.AugAssign):
            return self._lower_augassign(stmt)
        if isinstance(stmt, ast.Return):
            return [self._lower_return(stmt)]
        if isinstance(stmt, ast.If):
            return self._lower_if(stmt)
        if isinstance(stmt, ast.While):
            return self._lower_while(stmt)
        if isinstance(stmt, ast.For):
            return self._lower_for(stmt)
        if isinstance(stmt, ast.Pass):
            return [stmt]
        if isinstance(stmt, ast.AsyncFor):
            raise AutographControlFlowError("async for is not supported in autograph mode")
        if isinstance(stmt, (ast.With, ast.AsyncWith, ast.Try, ast.Raise)):
            raise AutographControlFlowError(f"unsupported control flow: {type(stmt).__name__}")
        if isinstance(stmt, (ast.Global, ast.Nonlocal)):
            raise AutographControlFlowError(f"unsupported statement: {type(stmt).__name__}")
        if isinstance(stmt, ast.Expr):
            raise AutographSyntaxError("expression statements are not supported in autograph mode")
        raise AutographSyntaxError(f"unsupported statement: {type(stmt).__name__}")

    def _lower_assign(self, stmt: ast.Assign) -> list[ast.stmt]:
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            raise AutographAssignmentError("only single-target assignments are supported")
        name = stmt.targets[0].id
        self._check_name_binding(name)
        value = self._lower_expr(stmt.value)
        self._locals.add(name)
        return [ast.Assign(targets=[self._cxt_attr(name, ast.Store())], value=value)]

    def _lower_augassign(self, stmt: ast.AugAssign) -> list[ast.stmt]:
        if not isinstance(stmt.target, ast.Name):
            raise AutographAssignmentError("only single-target assignments are supported")
        name = stmt.target.id
        if name not in self._locals:
            self._check_name_binding(name)
        left = self._name_load(name)
        right = self._lower_expr(stmt.value)
        value = ast.BinOp(left=left, op=stmt.op, right=right)
        self._locals.add(name)
        return [ast.Assign(targets=[self._cxt_attr(name, ast.Store())], value=value)]

    def _lower_return(self, stmt: ast.Return) -> ast.Return:
        value = self._lower_expr(stmt.value) if stmt.value is not None else ast.Constant(value=None)
        return ast.Return(
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id="cxt", ctx=ast.Load()), attr="result", ctx=ast.Load()),
                args=[value],
                keywords=[],
            )
        )

    def _lower_if(self, stmt: ast.If) -> list[ast.stmt]:
        const = _eval_const_bool(stmt.test)
        if const is not None:
            branch = stmt.body if const else stmt.orelse
            out: list[ast.stmt] = []
            for inner in branch:
                out.extend(self._lower_stmt(inner))
            return out

        if _is_return_branch(stmt.body) and (
            _is_return_branch(stmt.orelse)
            or (len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If))
        ):
            return [self._lower_if_return(stmt)]

        then_assigns = self._lower_if_assignments(stmt)
        temp_name = self._temp_name("_if_map_")
        out: list[ast.stmt] = [
            ast.Assign(
                targets=[ast.Name(id=temp_name, ctx=ast.Store())],
                value=then_assigns.map_expr,
            )
        ]
        for name in then_assigns.names:
            if name not in self._locals:
                self._check_name_binding(name)
                self._locals.add(name)
            out.append(
                ast.Assign(
                    targets=[self._cxt_attr(name, ast.Store())],
                    value=ast.Subscript(
                        value=ast.Name(id=temp_name, ctx=ast.Load()),
                        slice=ast.Constant(value=name),
                        ctx=ast.Load(),
                    ),
                )
            )
        return out

    def _lower_if_return(self, stmt: ast.If) -> ast.Return:
        allowed = {*self._params, *self._locals}
        cond = _replace_names(stmt.test, allowed, "_tc_autograph")
        then_expr = self._lower_if_return_expr(stmt.body[0], allowed)
        else_expr = self._lower_if_return_orelse(stmt.orelse, allowed)
        return ast.Return(value=self._tc_cond(cond, then_expr, else_expr))

    def _lower_if_return_expr(self, stmt: ast.stmt, allowed: set[str]) -> ast.expr:
        if not isinstance(stmt, ast.Return):
            raise AutographSyntaxError("if return branch must contain a single return statement")
        return _replace_names(stmt.value, allowed, "_tc_autograph")

    def _lower_if_return_orelse(self, orelse: list[ast.stmt], allowed: set[str]) -> ast.expr:
        if len(orelse) == 1 and isinstance(orelse[0], ast.If):
            return self._lower_if_return(orelse[0]).value
        if len(orelse) == 1 and isinstance(orelse[0], ast.Return):
            return _replace_names(orelse[0].value, allowed, "_tc_autograph")
        raise AutographSyntaxError("if return branch must contain a single return statement")

    def _lower_if_assignments(self, stmt: ast.If) -> _IfAssignments:
        if _contains_disallowed_control(stmt.body):
            raise AutographControlFlowError("nested control flow is not supported in if branches")
        if not (len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If)) and _contains_disallowed_control(
            stmt.orelse
        ):
            raise AutographControlFlowError("nested control flow is not supported in if branches")

        allowed = {*self._params, *self._locals}
        map_expr, names = self._lower_if_map(stmt, allowed)
        return _IfAssignments(map_expr=map_expr, names=names)

    def _lower_if_map(self, stmt: ast.If, allowed: set[str]) -> tuple[ast.expr, list[str]]:
        if _contains_disallowed_control(stmt.body):
            raise AutographControlFlowError("nested control flow is not supported in if branches")
        if not (len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If)) and _contains_disallowed_control(
            stmt.orelse
        ):
            raise AutographControlFlowError("nested control flow is not supported in if branches")

        cond_expr = _replace_names(stmt.test, allowed, "_tc_autograph")
        then_form, then_names = self._collect_if_branch_form(stmt.body, allowed)

        if not stmt.orelse:
            # No-else assignment sugar keeps existing bindings unchanged.
            missing = [name for name in then_names if name not in allowed]
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise AutographAssignmentError(
                    f"if without else may only assign previously bound names: {missing_str}"
                )
            else_form = [(name, _id_call(name)) for name in then_names]
            else_names = list(then_names)
        elif len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
            else_map_expr, else_names = self._lower_if_map(stmt.orelse[0], allowed)
            else_form = self._if_map_to_form(else_map_expr, else_names, allowed)
        else:
            else_form, else_names = self._collect_if_branch_form(stmt.orelse, allowed)

        if set(then_names) != set(else_names):
            raise AutographAssignmentError("if branches must assign the same set of names")

        then_op = _opdef_post(then_form + [("result", _dict_expr(_id_map(then_names)))])
        else_op = _opdef_post(else_form + [("result", _dict_expr(_id_map(then_names)))])
        map_expr = self._tc_cond_op(cond_expr, then_op, else_op)
        return map_expr, then_names

    def _collect_if_branch_form(
        self, stmts: list[ast.stmt], allowed: set[str]
    ) -> tuple[list[tuple[str, ast.expr]], list[str]]:
        if not stmts:
            raise AutographAssignmentError("if branches must assign at least one name")
        form: list[tuple[str, ast.expr]] = []
        names: list[str] = []
        bound = set(allowed)

        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1:
                    raise AutographAssignmentError("if assignment branches must contain only simple assignments")
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    name = target.id
                    if name in names:
                        raise AutographAssignmentError(f"duplicate assignment to {name} in branch")
                    expr = _replace_names(stmt.value, bound, "_tc_autograph")
                    form.append((name, expr))
                    names.append(name)
                    bound.add(name)
                    continue
                if isinstance(target, ast.Tuple) and all(isinstance(elt, ast.Name) for elt in target.elts):
                    temp_name = self._unique_temp_name("_if_tuple_", bound, names)
                    value_expr = _replace_names(stmt.value, bound, "_tc_autograph")
                    form.append((temp_name, value_expr))
                    bound.add(temp_name)
                    for index, elt in enumerate(target.elts):
                        name = elt.id
                        if name in names:
                            raise AutographAssignmentError(f"duplicate assignment to {name} in branch")
                        form.append(
                            (
                                name,
                                ast.Subscript(
                                    value=_id_call(temp_name),
                                    slice=ast.Constant(value=index),
                                    ctx=ast.Load(),
                                ),
                            )
                        )
                        names.append(name)
                        bound.add(name)
                    continue
                raise AutographAssignmentError("if assignment branches must contain only simple assignments")
            if isinstance(stmt, ast.If):
                map_expr, nested_names = self._lower_if_map(stmt, bound)
                form.extend(self._if_map_to_form(map_expr, nested_names, bound, names))
                continue
            raise AutographAssignmentError("if assignment branches must contain only assignments or if statements")

        return form, names

    def _if_map_to_form(
        self,
        map_expr: ast.expr,
        names: list[str],
        bound: set[str],
        assigned: list[str] | None = None,
    ) -> list[tuple[str, ast.expr]]:
        if assigned is None:
            assigned = []
        form: list[tuple[str, ast.expr]] = []
        temp_name = self._unique_temp_name("_if_map_", bound, assigned)
        form.append((temp_name, map_expr))
        bound.add(temp_name)
        for name in names:
            if name in assigned:
                raise AutographAssignmentError(f"duplicate assignment to {name} in branch")
            form.append(
                (
                    name,
                    ast.Subscript(
                        value=_id_call(temp_name),
                        slice=ast.Constant(value=name),
                        ctx=ast.Load(),
                    ),
                )
            )
            assigned.append(name)
            bound.add(name)
        return form

    def _unique_temp_name(self, prefix: str, bound: set[str], assigned: list[str]) -> str:
        name = self._temp_name(prefix)
        while name in bound or name in assigned:
            name = self._temp_name(prefix)
        return name

    def _lower_while(self, stmt: ast.While) -> list[ast.stmt]:
        if stmt.orelse:
            raise AutographControlFlowError("while-else is not supported in autograph mode")
        if not stmt.body:
            raise AutographControlFlowError("while body must not be empty")

        const = _eval_const_bool(stmt.test)
        if const is True:
            raise AutographControlFlowError("while condition is always true")
        if const is False:
            return []

        assignments: list[ast.Assign] = []
        body_stmts: list[ast.stmt] = []
        for item in stmt.body:
            if isinstance(item, ast.Assign):
                if len(item.targets) != 1 or not isinstance(item.targets[0], ast.Name):
                    raise AutographAssignmentError("while body assignment must target a single name")
                assignments.append(item)
                body_stmts.append(item)
                continue
            if isinstance(item, ast.If):
                body_stmts.append(item)
                continue
            raise AutographControlFlowError("while body must contain only assignments or if statements")

        state_name = _find_while_state(assignments, self._locals)
        if state_name is None:
            raise AutographAssignmentError("while body must update a previously bound state variable")
        if not isinstance(body_stmts[-1], ast.Assign) or body_stmts[-1].targets[0].id != state_name:
            raise AutographAssignmentError("while state update must be the final statement in the loop body")

        next_state_name = self._temp_name(f"{state_name}_next_")

        cond_expr = _replace_names(stmt.test, {state_name}, "_tc_autograph")
        cond_op = _opdef_post([("result", cond_expr)])

        temp_names: set[str] = set()
        step_form: list[tuple[str, ast.expr]] = []
        for item in body_stmts:
            if isinstance(item, ast.Assign):
                name = item.targets[0].id
                if name != state_name:
                    if name in temp_names:
                        raise AutographAssignmentError(f"duplicate assignment to {name} in while body")
                    if name in _RESERVED_NAMES:
                        raise AutographNameError(f"namespace collision: {name}")
                    expr = _replace_names(item.value, {state_name, *temp_names}, "_tc_autograph")
                    temp_names.add(name)
                    step_form.append((name, expr))
                    continue

                expr = _replace_names(item.value, {state_name, *temp_names}, "_tc_autograph")
                step_form.append((next_state_name, expr))
                temp_names.add(next_state_name)
                continue

            if isinstance(item, ast.If):
                assigns = self._lower_while_if_assignments(item, {state_name, *temp_names})
                for name, expr in assigns:
                    if name == state_name:
                        raise AutographAssignmentError("while body may not assign state inside a conditional")
                    if name in temp_names:
                        raise AutographAssignmentError(f"duplicate assignment to {name} in while body")
                    if name in _RESERVED_NAMES:
                        raise AutographNameError(f"namespace collision: {name}")
                    temp_names.add(name)
                    step_form.append((name, expr))

        step_form.append(("result", _id_call(next_state_name)))
        step_op = _opdef_post(step_form)
        while_call = ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(value=ast.Name(id="_tc_autograph", ctx=ast.Load()), attr="state", ctx=ast.Load()),
                attr="while_loop",
                ctx=ast.Load(),
            ),
            args=[cond_op, step_op, self._name_load(state_name)],
            keywords=[],
        )
        return [ast.Assign(targets=[self._cxt_attr(state_name, ast.Store())], value=while_call)]

    def _lower_for(self, stmt: ast.For) -> list[ast.stmt]:
        if stmt.orelse:
            raise AutographControlFlowError("for-else is not supported in autograph mode")
        if not isinstance(stmt.target, ast.Name):
            raise AutographAssignmentError("for loop target must be a single name")
        item_name = stmt.target.id
        self._check_name_binding(item_name)

        items_expr = self._lower_expr(stmt.iter)

        form_items, last_name = self._lower_for_body(stmt.body, item_name)
        if last_name is None:
            raise AutographAssignmentError("for loop body must assign at least one name")

        form_items.append(("result", _id_call(last_name)))
        op_def = _opdef_post(form_items)

        for_each_call = ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id="_tc_autograph", ctx=ast.Load()),
                    attr="state",
                    ctx=ast.Load(),
                ),
                attr="for_each",
                ctx=ast.Load(),
            ),
            args=[items_expr],
            keywords=[
                ast.keyword(arg="item_name", value=ast.Constant(value=item_name)),
                ast.keyword(arg="op", value=op_def),
            ],
        )

        tmp_name = self._temp_name("_tmp_for_each")
        return [ast.Assign(targets=[self._cxt_attr(tmp_name, ast.Store())], value=for_each_call)]

    def _check_name_binding(self, name: str) -> None:
        if name in _RESERVED_NAMES:
            raise AutographNameError(f"name {name} is reserved in autograph mode")
        if name in self._params:
            raise AutographNameError(f"namespace collision: {name}")
        if name in self._locals:
            raise AutographNameError(f"duplicate assignment to {name}")

    def _lower_expr(self, expr: ast.expr) -> ast.expr:
        if isinstance(expr, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            raise AutographSyntaxError("comprehensions are not supported in autograph mode")
        if isinstance(expr, (ast.Await, ast.Yield, ast.YieldFrom)):
            raise AutographSyntaxError("async/yield expressions are not supported in autograph mode")
        expr = _rewrite_len_calls(expr)
        allowed = {*self._params, *self._locals, "self", *_ALLOWED_GLOBALS}
        _validate_load_names(expr, allowed, context="autograph expression")
        return self._lower_names(expr)

    def _lower_names(self, expr: ast.expr) -> ast.expr:
        class _NameRewriter(ast.NodeTransformer):
            def __init__(self, outer: _AutographTransformer) -> None:
                self._outer = outer

            def visit_Name(self, node: ast.Name) -> ast.AST:
                if isinstance(node.ctx, ast.Load) and node.id in self._outer._locals:
                    return self._outer._cxt_attr(node.id, ast.Load())
                return node

        return _NameRewriter(self).visit(expr)

    def _name_load(self, name: str) -> ast.expr:
        if name in self._locals:
            return self._cxt_attr(name, ast.Load())
        return ast.Name(id=name, ctx=ast.Load())

    def _cxt_attr(self, name: str, ctx: ast.expr_context) -> ast.Attribute:
        return ast.Attribute(value=ast.Name(id="cxt", ctx=ast.Load()), attr=name, ctx=ctx)

    def _tc_cond(self, cond: ast.expr, then_expr: ast.expr, else_expr: ast.expr) -> ast.expr:
        then_op = _opdef_post([("result", then_expr)])
        else_op = _opdef_post([("result", else_expr)])
        return self._tc_cond_op(cond, then_op, else_op)

    def _tc_cond_op(self, cond: ast.expr, then_op: ast.expr, else_op: ast.expr) -> ast.expr:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(value=ast.Name(id="_tc_autograph", ctx=ast.Load()), attr="state", ctx=ast.Load()),
                attr="cond",
                ctx=ast.Load(),
            ),
            args=[cond, then_op, else_op],
            keywords=[],
        )

    def _lower_for_body(
        self, body: list[ast.stmt], item_name: str
    ) -> tuple[list[tuple[str, ast.expr]], str | None]:
        if not body:
            raise AutographAssignmentError("for loop body must not be empty")

        temp_names: list[str] = []
        form_items: list[tuple[str, ast.expr]] = []
        last_name: str | None = None
        allowed = {item_name, *self._locals}

        for stmt in body:
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    raise AutographAssignmentError("for loop body assignment must target a single name")
                name = stmt.targets[0].id
                if name == item_name:
                    raise AutographAssignmentError("for loop body must not assign to the loop item")
                if name in temp_names:
                    raise AutographAssignmentError(f"duplicate assignment to {name} in for body")
                if name in self._locals or name in self._params or name in _RESERVED_NAMES:
                    raise AutographNameError(f"namespace collision: {name}")
                expr = _replace_names(stmt.value, allowed | set(temp_names), "_tc_autograph")
                temp_names.append(name)
                form_items.append((name, expr))
                last_name = name
                continue

            if isinstance(stmt, ast.If):
                assigns = self._lower_while_if_assignments(stmt, allowed | set(temp_names))
                for name, expr in assigns:
                    if name == item_name:
                        raise AutographAssignmentError("for loop body must not assign to the loop item")
                    if name in temp_names:
                        raise AutographAssignmentError(f"duplicate assignment to {name} in for body")
                    if name in self._locals or name in self._params or name in _RESERVED_NAMES:
                        raise AutographNameError(f"namespace collision: {name}")
                    temp_names.append(name)
                    form_items.append((name, expr))
                    last_name = name
                continue

            raise AutographControlFlowError("for loop body must contain only assignments or if statements")

        return form_items, last_name

    def _temp_name(self, prefix: str) -> str:
        name = f"{prefix}{self._temp_counter}"
        self._temp_counter += 1
        if name in self._locals or name in self._params or name in _RESERVED_NAMES:
            return self._temp_name(prefix)
        return name

    def _lower_while_if_assignments(
        self, stmt: ast.If, allowed: set[str]
    ) -> list[tuple[str, ast.expr]]:
        map_expr, names = self._lower_if_map(stmt, allowed)
        bound = set(allowed)
        return self._if_map_to_form(map_expr, names, bound)


def _eval_const_bool(expr: ast.expr) -> bool | None:
    try:
        value = ast.literal_eval(expr)
    except (TypeError, ValueError, SyntaxError):
        return None
    return value if isinstance(value, bool) else None


def _is_return_branch(stmts: Iterable[ast.stmt]) -> bool:
    items = list(stmts)
    return len(items) == 1 and isinstance(items[0], ast.Return)


def _contains_disallowed_control(stmts: list[ast.stmt]) -> bool:
    for stmt in stmts:
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
            return True
        if isinstance(stmt, ast.If):
            if _contains_disallowed_control(stmt.body) or _contains_disallowed_control(stmt.orelse):
                return True
    return False


def _validate_load_names(expr: ast.expr, allowed: set[str], *, context: str) -> None:
    class _LoadNameValidator(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load) and node.id not in allowed:
                raise AutographNameError(f"unsupported name {node.id} in {context}")
            self.generic_visit(node)

    _LoadNameValidator().visit(expr)


def _replace_names(expr: ast.expr, allowed: set[str], tc_name: str) -> ast.expr:
    expr = _rewrite_len_calls(expr)

    class _NameRewriter(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load) and node.id in allowed:
                return _id_call(node.id)
            if isinstance(node.ctx, ast.Load):
                if node.id in _ALLOWED_GLOBALS:
                    return node
                raise AutographNameError(f"unsupported name {node.id} in while expression")
            return node

    return _NameRewriter().visit(expr)


def _rewrite_len_calls(expr: ast.expr) -> ast.expr:
    class _LenRewriter(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "len"
                and len(node.args) == 1
                and not node.keywords
            ):
                arg = self.visit(node.args[0])
                return ast.Call(
                    func=ast.Attribute(value=arg, attr="len", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                )
            return self.generic_visit(node)

    return _LenRewriter().visit(expr)


def _dict_expr(items: dict[str, ast.expr]) -> ast.Dict:
    keys: list[ast.expr] = []
    values: list[ast.expr] = []
    for name in sorted(items.keys()):
        keys.append(ast.Constant(value=name))
        values.append(items[name])
    return ast.Dict(keys=keys, values=values)


def _id_map(names: list[str]) -> dict[str, ast.expr]:
    return {name: _id_call(name) for name in names}


def _id_call(name: str) -> ast.Call:
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(value=ast.Name(id="_tc_autograph", ctx=ast.Load()), attr="state", ctx=ast.Load()),
            attr="id",
            ctx=ast.Load(),
        ),
        args=[ast.Constant(value=name)],
        keywords=[],
    )


def _opdef_post(items: list[tuple[str, ast.expr]]) -> ast.Call:
    item_lambdas: list[ast.expr] = []
    for name, expr in items:
        item_lambdas.append(
            ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg="_tc_cxt")],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=ast.Tuple(
                    elts=[ast.Constant(value=name), expr],
                    ctx=ast.Load(),
                ),
            )
        )
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(value=ast.Name(id="_tc_autograph", ctx=ast.Load()), attr="_autograph", ctx=ast.Load()),
            attr="_autograph_opdef_post",
            ctx=ast.Load(),
        ),
        args=[ast.List(elts=item_lambdas, ctx=ast.Load())],
        keywords=[],
    )


def _autograph_opdef_post(item_fns):
    from .state import OpDef, scoped_context

    with scoped_context() as cxt:
        form: list[tuple[str, object]] = []
        cursor = 0
        for item_fn in item_fns:
            name, value = item_fn(cxt)
            ctx_form = list(cxt.form())
            if len(ctx_form) > cursor:
                form.extend(ctx_form[cursor:])
                cursor = len(ctx_form)
            form.append((name, value))
        from .state import PostOpDef

        return PostOpDef(form)


def _find_while_state(assignments: list[ast.Assign], locals_before: set[str]) -> str | None:
    candidates = []
    for assign in assignments:
        name = assign.targets[0].id
        if name in locals_before:
            candidates.append(name)
    if len(candidates) != 1:
        return None
    return candidates[0]
