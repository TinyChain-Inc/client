"""Composition helper for tracing parameter updates as ordinary Tensor code.

An optimizer update is, structurally, no different from an application loss:
it is a typed Tensor expression over declared inputs. This module supplies the
one piece that is otherwise missing from the public tracing surface -- typed
parameter/gradient/optimizer inputs and a single selected updated-parameter
output -- so an update can be authored and traced entirely with ordinary
`Tensor` operations (spec invariant 6) and then handed to the same structured
dependency analysis and extensible lowering seam as any other traced graph.

`Optimizer` below is the contract that binds an update expression to the names
of the inputs it reads, and `SGD` is its one concrete implementation:
`parameter - learning_rate * gradient`, written with the `-`/`*` operators
`Tensor` already supports -- nothing here constructs a `TensorNodeRecord` or a
concrete `TensorOperator` directly. `sgd_update` is the compatibility path for
callers that already import the reference expression as a function.

This module still defines no optimizer catalog, no state lifecycle, and no
consumer policy. An optimizer owns its expression, its declared input names,
and its own configuration; graph construction, dependency analysis, lowering,
provider execution, encrypted state lifecycle, and the training loop are all
outside it.

Named invariants and where each is enforced (spec-driven, each in one place):

* **Update-callable well-formedness.** The signature that will be invoked --
  a plain callable itself, or an `Optimizer`'s `update` method -- must accept
  exactly the declared keyword inputs. Checked once, before the builder is
  entered, in :func:`_validate_update_signature` -- this is what makes
  "invalid update callables fail before consumer execution" a structural
  property rather than an accident of statement order: the callable's body
  cannot run before its signature has been validated.
* **Declared inputs match what an optimizer reads.** For an `Optimizer`, the
  invariant above is enforced twice over, in the same place and at the same
  moment, because two different mistakes are possible. The signature check is
  applied to the `update` *method* -- not to the instance, whose
  `Optimizer.__call__` accepts arbitrary keywords and would therefore accept
  any declaration at all -- and catches an implementation whose parameters do
  not match the declared inputs. :func:`_validate_declared_optimizer_inputs`
  then catches a declaration naming inputs the expression never reads.
  Supplementing the signature check rather than replacing it is what keeps
  this path from being weaker than the plain-callable path; the declared names
  themselves are read through
  :func:`_required_optimizer_input_names`, so a malformed declaration is
  categorized rather than escaping as a raw `TypeError`.
* **Traced output validity.** The callable must return a `Tensor`. Checked
  once, immediately after invoking the callable and before `builder.build`.
* **Typed input spec well-formedness.** Each declared spec must be a mapping
  providing a ``dtype`` and a ``shape``. Checked once, in
  :func:`_typed_input_spec`, before the builder is entered. This is *not*
  delegated to `TensorGraphBuilder.input`, and cannot be: the specs used to be
  unpacked into it as ``**dict(spec)``, so a spec that was not a mapping, or
  that named its keys wrongly, failed in the unpack -- before the builder was
  ever reached -- and a keyword-argument `TypeError` carries no category. The
  spec is therefore read by key instead, exactly as the structured dependency
  analysis reads a type spec, and an unrecognized extra key is ignored for the
  same reason it is ignored there: one spec must not be accepted by the
  analysis and rejected by the tracer.
* **Declared optimizer inputs.** ``optimizer_inputs`` declares *which*
  keyword inputs exist, so a container that is not a mapping leaves the
  declared input set unestablished and the update callable's signature
  uncheckable. Checked once, in :func:`_resolve_optimizer_inputs`.
* **Typed input dtype and shape validity.** The dtype *value* is delegated
  entirely to `TensorGraphBuilder.input`'s existing
  `check_differentiable_dtype`. The shape's rank and dimensions are checked
  against `shape.parse_shape`, the same shared parser the dependency analysis
  uses, so the builder's own normalization can no longer be reached with a
  value it would reject with an uncategorized `ValueError`.
* **Shape/dtype compatibility of the traced expression** (for example a
  gradient shape incompatible with the parameter shape). Delegated entirely to
  the existing typed-tracing Sub/Mul shape inference; this module does not
  re-check operand shapes.
* **Inactive-tracing precondition.** Tracing must start with no builder
  already active. Delegated entirely to `TensorGraphBuilder`'s existing
  nested-context guard.
"""

from __future__ import annotations

import inspect
import keyword
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Optional

from .graph import TensorGraph, TensorGraphBuilder
from .protocol import AutodiffError
from .shape import parse_shape


_RESERVED_INPUT_NAMES: tuple[str, ...] = ("parameter", "gradient")


class Optimizer(ABC):
    """The contract an optimizer implements, and nothing wider than that.

    An update expression and the names of the inputs that expression reads are
    two facts that must agree. Without a contract binding them, a caller
    imports an update function by name and separately remembers to declare the
    inputs it needs, and nothing checks the pair. This class is that binding,
    and that is the whole of its justification: it is not a catalog, and it
    designs no second optimizer family.

    An optimizer owns exactly three things:

    * **The update expression**, authored in ordinary `Tensor` operations --
      :meth:`update`.
    * **The names of the optimizer inputs that expression reads** --
      :attr:`required_optimizer_inputs`.
    * **Validation of its own configuration**, where it has any. There is no
      member for this, deliberately: an implementation with configuration
      validates it in its own constructor, so the contract *admits*
      configuration without *mandating* it, and an implementation with none --
      like :class:`SGD` -- carries no empty hook.

    It owns nothing else. Graph construction, dependency analysis, lowering,
    provider execution, encrypted state lifecycle, and the training loop are
    all outside it. It holds no state and no persistence, and it knows nothing
    about dtype or shape: a caller still declares those, because they are
    properties of the values being trained rather than of the algorithm.

    An instance is callable, so it is usable wherever a plain update callable
    is. :func:`trace_parameter_update` checks an optimizer *both* ways, and an
    implementor should expect both. It binds :meth:`update` against the
    declared inputs, so an implementation whose parameters do not match what
    it declares is rejected; and it compares
    :attr:`required_optimizer_inputs` against the caller's declaration, so a
    declaration naming inputs the expression never reads is rejected too. The
    binding is applied to :meth:`update` rather than to the instance because
    the call path -- :meth:`__call__` -- accepts arbitrary keywords, so
    binding *that* would accept any declaration at all.
    """

    @property
    @abstractmethod
    def required_optimizer_inputs(self) -> tuple[str, ...]:
        """The names of the optimizer inputs :meth:`update` reads.

        These are the *optimizer* inputs only: ``parameter`` and ``gradient``
        are declared by every traced update and are never named here. The
        names must match the keys of the ``optimizer_inputs`` a caller
        declares to :func:`trace_parameter_update`, and must also match what
        :meth:`update` accepts -- the two are checked separately, because an
        implementation can get either one wrong on its own.

        An implementation may answer per instance -- configuration is free to
        decide which inputs the expression reads.
        """

    @abstractmethod
    def update(
        self, *, parameter: object, gradient: object, **optimizer_inputs: object
    ) -> object:
        """Return the updated parameter, as an ordinary `Tensor` expression.

        Called once per trace, by keyword, with a `Tensor` for ``parameter``,
        a `Tensor` for ``gradient``, and a `Tensor` for each name in
        :attr:`required_optimizer_inputs`. It must return the single updated
        parameter `Tensor`, and must build it with ordinary `Tensor`
        operations -- never by constructing a graph, a node record, or a
        concrete operator.

        This method's own signature *is* checked: it is bound against the
        declared inputs before any tracing begins, so parameters that do not
        match what :attr:`required_optimizer_inputs` declares are reported
        rather than reaching the expression.
        """

    def __call__(self, **optimizer_inputs: object) -> object:
        """Make an optimizer usable wherever a plain update callable is.

        This accepts arbitrary keywords deliberately, so that every optimizer
        is callable through one signature regardless of what it declares. The
        cost is that binding *this* signature proves nothing, which is why
        :func:`trace_parameter_update` binds :meth:`update` instead.
        """
        return self.update(**optimizer_inputs)


class SGD(Optimizer):
    """Stochastic gradient descent: ``parameter - learning_rate * gradient``.

    The first and, today, the only concrete :class:`Optimizer`. It declares
    one required optimizer input, ``learning_rate``, and carries no
    configuration, so it has nothing to validate in its constructor.

    The expression is written with the `Tensor` ``-``/``*`` operators; no
    `TensorNodeRecord` and no concrete `TensorOperator` is constructed here.
    """

    required_optimizer_inputs: tuple[str, ...] = ("learning_rate",)

    def update(
        self, *, parameter: object, gradient: object, learning_rate: object
    ) -> object:
        return parameter - learning_rate * gradient


@dataclass(frozen=True)
class TracedUpdate:
    """The finalized typed graph produced by tracing a parameter update.

    ``input_value_ids`` maps each declared input name (``"parameter"``,
    ``"gradient"``, and every ``optimizer_inputs`` key) to its stable value id
    in ``graph``, so a consumer binds runtime values by name instead of
    scanning ``graph.inputs`` or inferring order.
    """

    graph: TensorGraph
    updated_parameter_id: str
    input_value_ids: Mapping[str, str]


def trace_parameter_update(
    update: "Optimizer | Callable[..., object]",
    *,
    parameter: Mapping[str, object],
    gradient: Mapping[str, object],
    optimizer_inputs: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> TracedUpdate:
    """Trace an ordinary Tensor *update* into a finalized typed graph.

    *update* is either a plain callable or an :class:`Optimizer`. The two are
    traced through the same path and differ only in how the declared inputs
    are validated: a callable has its own signature bound against them, while
    an optimizer has :meth:`Optimizer.update` bound against them *and* its
    :attr:`Optimizer.required_optimizer_inputs` compared with them.

    ``parameter``, ``gradient``, and each ``optimizer_inputs`` value are typed
    input specs (``{"dtype": ..., "shape": ...}``), read by key and passed on
    to :meth:`TensorGraphBuilder.input`; an unrecognized extra key is ignored.
    ``update`` is called once, by keyword, with a ``Tensor`` for
    ``parameter``, a ``Tensor`` for ``gradient``, and a ``Tensor`` for each
    declared optimizer input; it must return the single updated-parameter
    ``Tensor``, expressed with ordinary Tensor operations.

    Every declared spec is validated before the builder is entered, so a
    malformed declaration never reaches the consumer's callable body.
    """
    resolved_optimizer_inputs = _resolve_optimizer_inputs(optimizer_inputs)
    if isinstance(update, Optimizer):
        # Two checks, because they catch two different mistakes. The names
        # catch a declaration naming inputs the expression never reads;
        # binding catches an implementation whose `update` parameters do not
        # match the declared inputs. Binding is applied to `update` -- the
        # method that actually runs -- and not to the instance: an optimizer
        # is invoked through `Optimizer.__call__`, which accepts arbitrary
        # keywords, so binding *that* signature would accept any declaration
        # at all. Dropping binding entirely would leave this path weaker than
        # the plain-callable path below for the very mistakes an optimizer
        # exists to catch.
        _validate_declared_optimizer_inputs(update, resolved_optimizer_inputs)
        _validate_update_signature(
            update.update,
            parameter=parameter,
            gradient=gradient,
            optimizer_inputs=resolved_optimizer_inputs,
            label=f"optimizer {type(update).__name__!r} update method",
        )
    else:
        _validate_update_signature(
            update,
            parameter=parameter,
            gradient=gradient,
            optimizer_inputs=resolved_optimizer_inputs,
        )
    # Ahead of the builder, like the signature check above and for the same
    # reason: failing before any trace begins is then a structural property
    # rather than an accident of statement order inside the trace.
    declared_specs = {
        "parameter": _typed_input_spec(parameter, label="parameter"),
        "gradient": _typed_input_spec(gradient, label="gradient"),
    }
    for name, spec in resolved_optimizer_inputs.items():
        declared_specs[name] = _typed_input_spec(spec, label=name)

    # Deferred import: importing collection.tensor at module scope would
    # initialize Tensor, whose recorder imports this package for concrete
    # operator identities (same rationale as TensorGraphBuilder.input).
    from ..collection.tensor import Tensor

    with TensorGraphBuilder() as builder:
        parameter_tensor = builder.input("parameter", **declared_specs["parameter"])
        gradient_tensor = builder.input("gradient", **declared_specs["gradient"])
        optimizer_tensors = {
            name: builder.input(name, **declared_specs[name])
            for name in resolved_optimizer_inputs
        }
        updated_parameter = update(
            parameter=parameter_tensor,
            gradient=gradient_tensor,
            **optimizer_tensors,
        )

    if not isinstance(updated_parameter, Tensor):
        raise AutodiffError(
            "invalid_update_output",
            "update callable must return a Tensor, got "
            f"{type(updated_parameter).__name__!r}",
        )

    graph = builder.build(outputs=[updated_parameter])

    input_value_ids = {
        "parameter": builder.value_id(parameter_tensor),
        "gradient": builder.value_id(gradient_tensor),
    }
    for name, tensor in optimizer_tensors.items():
        input_value_ids[name] = builder.value_id(tensor)

    return TracedUpdate(
        graph=graph,
        updated_parameter_id=builder.value_id(updated_parameter),
        input_value_ids=input_value_ids,
    )


def _resolve_optimizer_inputs(optimizer_inputs: object) -> Mapping[str, object]:
    """Normalize the declared optimizer inputs, rejecting a malformed container.

    ``optimizer_inputs`` declares *which* keyword inputs exist, so a container
    that is not a mapping leaves the declared input set unestablished and the
    update callable's signature uncheckable -- which is the failure
    ``invalid_update_signature`` owns. The category name points a reader at
    the callable, so the message says plainly that the argument is at fault
    and the callable is not.
    """
    if optimizer_inputs is None:
        return {}
    if not isinstance(optimizer_inputs, Mapping):
        raise AutodiffError(
            "invalid_update_signature",
            "the optimizer_inputs argument is at fault here, not the update "
            "callable: optimizer_inputs must be a mapping of input name to "
            f"typed input spec, got {type(optimizer_inputs).__name__!r}; the "
            "declared input set cannot be established, so the callable's "
            "signature cannot be checked against it",
        )
    for name in optimizer_inputs:
        _validate_optimizer_input_name(name)
    return optimizer_inputs


def _validate_optimizer_input_name(name: object) -> None:
    """Require one declared optimizer input name to be usable as an input.

    Whether a bad name is otherwise noticed depends entirely on the update
    callable, which is why it cannot be left to the builder. A callable
    declaring exact parameters cannot bind a name like ``"a b"``, so
    :func:`_validate_update_signature` rejects it first and the builder is
    never reached; a callable taking ``**kwargs`` binds any name at all, so
    the same declaration reaches ``TensorGraphBuilder.input`` and fails there
    with a raw ``ValueError``/``TypeError``. A ``**kwargs`` optimizer update
    is an ordinary shape to write, so the name set is validated in its own
    right rather than incidentally.

    The reserved names are checked here for a second reason: a collision is
    already reported for an exact-parameter callable, but for the wrong
    reason. Deduplicating the declared names leaves that callable an argument
    short, so the caller is told their *function* is missing a parameter when
    what is actually wrong is the name they gave an optimizer input.
    """
    if not isinstance(name, str) or not name:
        raise AutodiffError(
            "invalid_update_signature",
            "each optimizer_inputs key declares an input name, so it must be "
            f"a non-empty string; got {name!r}",
        )
    if name in _RESERVED_INPUT_NAMES:
        raise AutodiffError(
            "invalid_update_signature",
            f"optimizer_inputs declares an input named {name!r}, which is "
            "already declared by the argument of that name; the fault is the "
            "declared name, not the update callable -- rename the optimizer "
            f"input, as {_RESERVED_INPUT_NAMES!r} are reserved",
        )
    if not name.isidentifier() or keyword.iskeyword(name):
        raise AutodiffError(
            "invalid_update_signature",
            f"optimizer_inputs declares an input named {name!r}, which cannot "
            "be a keyword argument: an input name must be a valid, "
            "non-keyword Python identifier",
        )


def _typed_input_spec(spec: object, *, label: str) -> dict[str, object]:
    """Read one declared typed input spec, failing closed with a category.

    Mirrors ``dependencies._complete_typespec``: the spec is read by key
    rather than unpacked, an absent or unreadable ``dtype`` is
    ``missing_dtype_metadata``, an absent or malformed ``shape`` is
    ``missing_shape_metadata``, and an extra key is ignored. A consumer then
    meets one category per *structural* mistake -- a spec that is not a
    mapping, or that is missing or cannot produce one of the two keys --
    wherever they hit it.

    That parity is structural only, and deliberately does not extend to the
    values. A dtype that is present but not a dtype is ``dtype_not_
    differentiable`` here and ``missing_dtype_metadata`` in the analysis,
    because the value is judged by the builder rather than re-judged here.

    The guard is placed where the helper places it. In both, reading the
    **dtype** happens outside the guard, so a container that raises while its
    dtype is looked up escapes raw from either. Reading the **shape** happens
    inside, so ``IndexError``, ``TypeError`` and ``ValueError`` from a shape
    lookup become ``missing_shape_metadata`` in both.

    One limit is therefore shared rather than fixed: a shape lookup raising
    anything outside those three types still escapes raw, from this function
    and from the analysis helper alike.

    The dtype *value* is deliberately not judged here. `TensorGraphBuilder`'s
    ``check_differentiable_dtype`` already categorizes it, and re-checking
    would give one mistake two decision sites. The shape *is* checked here,
    because the builder's normalization rejects a malformed dimension with an
    uncategorized `ValueError`; ``parse_shape`` is the shared parser the
    dependency analysis uses and accepts exactly what the builder accepts.
    """
    if not isinstance(spec, Mapping):
        raise AutodiffError(
            "missing_dtype_metadata",
            f"typed input {label!r} must be declared as a mapping of 'dtype' "
            f"and 'shape', got {type(spec).__name__!r}",
        )
    if "dtype" not in spec:
        raise AutodiffError(
            "missing_dtype_metadata",
            f"typed input {label!r} declares no 'dtype'; it has "
            f"{sorted(str(key) for key in spec)}",
        )
    # Everything that touches the container for its shape goes inside the
    # guard, matching where the analysis helper reads its own -- its
    # `.get("shape")` is inside its `try`, and only its dtype read sits
    # outside. The membership test belongs inside for the same reason as the
    # read: `Mapping.__contains__` is defined in terms of `__getitem__`, so
    # `"shape" not in spec` runs the container's own code and can raise
    # exactly where the read can.
    try:
        if "shape" not in spec:
            raise AutodiffError(
                "missing_shape_metadata",
                f"typed input {label!r} declares no 'shape'; it has "
                f"{sorted(str(key) for key in spec)}",
            )
        declared_shape = spec["shape"]
        parse_shape(declared_shape, label=f"typed input {label!r} shape")
    except AutodiffError:
        raise
    except (IndexError, TypeError, ValueError) as exc:
        # The same three types the analysis helper normalizes, for the same
        # reason: a shape that raises while being read is an unreadable shape,
        # not a raw container exception for a consumer to catch.
        raise AutodiffError(
            "missing_shape_metadata",
            f"typed input {label!r} has no readable ranked shape",
        ) from exc
    return {"dtype": spec["dtype"], "shape": declared_shape}


def _validate_update_signature(
    update: Callable[..., object],
    *,
    parameter: Mapping[str, object],
    gradient: Mapping[str, object],
    optimizer_inputs: Mapping[str, Mapping[str, object]],
    label: str = "update callable",
) -> None:
    """Require *update* to accept exactly the declared typed inputs by keyword.

    This is the single point where *signature* well-formedness is enforced --
    the declared *names* an optimizer reads are a separate mistake, checked in
    :func:`_validate_declared_optimizer_inputs`. It runs before any builder is
    entered or any typed input is declared, so a rejected callable never has
    its body invoked.

    It serves both entry paths. For a plain callable, *update* is the callable
    itself. For an `Optimizer`, it is the bound `update` method rather than
    the instance -- binding the instance would bind `Optimizer.__call__`,
    which accepts arbitrary keywords and therefore accepts anything. *label*
    names whichever of the two is at fault.
    """
    required_names = ("parameter", "gradient", *optimizer_inputs.keys())
    try:
        signature = inspect.signature(update)
        signature.bind(**dict.fromkeys(required_names))
    except TypeError as exc:
        raise AutodiffError(
            "invalid_update_signature",
            f"{label} must accept exactly the declared typed inputs "
            f"{required_names!r} by keyword: {exc}",
        ) from exc


def _required_optimizer_input_names(optimizer: Optimizer) -> tuple[str, ...]:
    """Read one optimizer's declared input names, failing closed with a category.

    ``required_optimizer_inputs`` is written by an implementation, so a
    malformed one is that implementation's bug -- but it arrives through this
    public entry point, and nothing here may leave through a raw `TypeError`.
    A declaration that cannot be read as a collection of names is therefore
    reported as what it is: a declaration fault, in the same category as every
    other one on this path.

    A bare string is rejected rather than iterated. It is the likely mistake
    for an optimizer with a single input, and iterating it would silently
    declare one input per character.
    """
    declared = optimizer.required_optimizer_inputs
    if isinstance(declared, str):
        raise AutodiffError(
            "invalid_update_signature",
            f"optimizer {type(optimizer).__name__!r} declares its required "
            f"optimizer inputs as the string {declared!r}; a single name must "
            "still be declared as a collection, for example "
            f"({declared!r},) -- otherwise it declares one input per character",
        )
    try:
        names = tuple(declared)
    except TypeError as exc:
        raise AutodiffError(
            "invalid_update_signature",
            f"optimizer {type(optimizer).__name__!r} must declare its "
            "required optimizer inputs as a collection of names, got "
            f"{type(declared).__name__!r}",
        ) from exc
    for name in names:
        if not isinstance(name, str) or not name:
            raise AutodiffError(
                "invalid_update_signature",
                f"optimizer {type(optimizer).__name__!r} declares a required "
                f"optimizer input named {name!r}; each declared name must be "
                "a non-empty string",
            )
    return names


def _validate_declared_optimizer_inputs(
    optimizer: Optimizer, optimizer_inputs: Mapping[str, Mapping[str, object]]
) -> None:
    """Require the declared optimizer inputs to be exactly what *optimizer* reads.

    This is the optimizer half of update-callable well-formedness. It runs in
    the same place and at the same moment as the signature check does for a
    plain callable -- before the builder is entered and before any typed input
    spec is read -- so a rejected declaration never reaches the optimizer's
    expression.

    It supplements binding rather than replacing it. Binding the *instance*
    is vacuous -- `Optimizer.__call__` accepts arbitrary keywords, so
    `inspect.signature(optimizer).bind(...)` succeeds for any declared input
    set -- but binding the `update` method is not, and the caller does both.
    The two catch different mistakes: binding catches an implementation whose
    parameters do not match the declared inputs, and this catches a
    declaration naming inputs the expression never reads.

    The comparison is by name set: the inputs are passed by keyword, so their
    order carries no meaning.
    """
    required_names = _required_optimizer_input_names(optimizer)
    declared_names = tuple(optimizer_inputs)
    missing = tuple(name for name in required_names if name not in optimizer_inputs)
    unexpected = tuple(name for name in declared_names if name not in required_names)
    if missing or unexpected:
        raise AutodiffError(
            "invalid_update_signature",
            f"optimizer {type(optimizer).__name__!r} reads the optimizer "
            f"inputs {required_names!r}, but the declared optimizer_inputs "
            f"are {declared_names!r}: missing {missing!r}, unexpected "
            f"{unexpected!r}",
        )


_SGD_COMPATIBILITY_REFERENCE = SGD()


def sgd_update(*, parameter: object, gradient: object, learning_rate: object) -> object:
    """Compatibility path for the reference SGD update.

    This is the function that existed before :class:`Optimizer`, kept working
    for callers that already import it. It authors nothing of its own: the
    expression ``parameter - learning_rate * gradient`` now lives in
    :class:`SGD`, and this delegates to a shared instance of it, so the two
    can never drift apart. `SGD` is stateless, so one shared instance is safe.

    New callers should pass ``SGD()`` to :func:`trace_parameter_update`
    instead, which additionally checks the declared optimizer inputs against
    what the expression actually reads -- a check a plain callable cannot get.
    """
    return _SGD_COMPATIBILITY_REFERENCE.update(
        parameter=parameter, gradient=gradient, learning_rate=learning_rate
    )
