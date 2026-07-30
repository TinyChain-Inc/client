from __future__ import annotations

import cmath
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .base import State
from .scalar import Scalar
from ..uri import URI


class Value(Scalar):
    __slots__ = ("_value",)

    __uri__: URI = URI(State, "scalar", "value")

    def __init__(self, value: None | bool | int | float | complex | str | dict[str, "Value"] | list["Value"] = None):
        super().__init__(value)
        self._value = value

    def to_json(self) -> object:
        raise TypeError(f"{type(self).__name__}.to_json must be implemented by a concrete Value subclass")

    @staticmethod
    def from_json(obj: Any) -> "Value":
        if obj is None:
            return Null()
        if isinstance(obj, bool):
            return Number(obj)
        if isinstance(obj, (int, float)):
            return Number(obj)
        if isinstance(obj, str):
            return String(obj)
        if isinstance(obj, list):
            return Tuple(obj)

        if isinstance(obj, dict) and len(obj) == 1:
            (key, value), = obj.items()
            if isinstance(key, str):
                value_type = _value_class_for_uri(key)
                if value_type is not None:
                    return value_type._from_json(value)

            if isinstance(key, str) and (key.startswith("/") or "://" in key):
                if value != []:
                    raise TypeError("expected link value to be an empty list")
                return Link(key)

        if isinstance(obj, dict):
            return Map(obj)

        raise TypeError(f"cannot decode Value from {type(obj).__name__}")

    @classmethod
    def _from_json(cls, obj: Any) -> "Value":
        raise TypeError("Value.from_json is not implemented for the base class")


def _iter_value_subclasses(base: type[Value]) -> Iterator[type[Value]]:
    for subclass in base.__subclasses__():
        yield subclass
        yield from _iter_value_subclasses(subclass)


def _value_class_for_uri(uri_path: str) -> type[Value] | None:
    if uri_path == str(URI(Value)):
        return Value

    for value_type in _iter_value_subclasses(Value):
        if str(URI(value_type)) == uri_path:
            return value_type

    return None


def _decode_value_item(obj: object) -> Value:
    if isinstance(obj, Value):
        return obj

    return Value.from_json(obj)


def _encode_value_item(value: Value) -> object:
    return value.to_json()


def form_of(value: "Value | object") -> object:
    if isinstance(value, Value):
        return value._value
    return value


class Null(Value):
    __slots__ = ()

    __uri__: URI = URI(Value, "none")

    def __init__(self):
        super().__init__(None)

    def to_json(self) -> object:
        return None

    @classmethod
    def _from_json(cls, obj: Any) -> "Null":
        return cls()


class Link(Value):
    __slots__ = ()

    __uri__: URI = URI(Value, "link")

    def __init__(self, value: URI | str):
        if not isinstance(value, (URI, str)):
            raise TypeError("expected link value")

        super().__init__(str(value))

    def to_json(self) -> object:
        return {str(URI(Link)): str(form_of(self))}

    @classmethod
    def _from_json(cls, obj: Any) -> "Link":
        if not isinstance(obj, str):
            raise TypeError("expected link value")
        return cls(obj)


class String(Value):
    __slots__ = ("op",)

    __uri__: URI = URI(Value, "string")

    def __init__(self, value: str | object):
        from ..opref import OpRef as RuntimeOpRef
        from .scalar import OpRef as StateOpRef, TCRef, tcref_form_of

        if isinstance(value, TCRef):
            value = tcref_form_of(value)

        if isinstance(value, (RuntimeOpRef, StateOpRef)):
            super().__init__(None)
            object.__setattr__(self, "op", value)
        else:
            if not isinstance(value, str):
                raise TypeError("expected string value")
            super().__init__(value)
            object.__setattr__(self, "op", None)

    def render(self, params: dict[str, object] | None = None, **kwargs: object) -> "String":
        if params is not None and kwargs:
            raise ValueError("String.render accepts a dict or kwargs, not both")

        from .scalar import OpRef as StateOpRef, TCRef, autobox, form_of, tcref_form_of

        render_params = kwargs if params is None else params
        if self.op is None and all(_is_literal_render_value(value) for value in render_params.values()):
            rendered = str(form_of(self))
            for key, value in render_params.items():
                rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
            return String(rendered)

        rendered_scalar = autobox(self)
        render_fn = getattr(rendered_scalar, "_string_render", None)
        if not callable(render_fn):
            raise TypeError("expected string render to produce an op ref")

        rendered_form = form_of(render_fn(render_params))
        if not isinstance(rendered_form, TCRef):
            raise TypeError("expected string render to produce an op ref")

        ref_form = tcref_form_of(rendered_form)
        if not isinstance(ref_form, StateOpRef):
            raise TypeError("expected string render to produce an op ref")

        return String(ref_form)

    def to_json(self) -> object:
        return form_of(self)

    @classmethod
    def _from_json(cls, obj: Any) -> "String":
        if not isinstance(obj, str):
            raise TypeError("expected string value")
        return cls(obj)


def _is_literal_render_value(value: object) -> bool:
    return isinstance(value, (str, bool, int, float, URI))


def _subject_of_scalar(value: "Value") -> str:
    from ._ops import subject_of

    return subject_of(value._scalar())


class Number(Value):
    __slots__ = ("op",)

    __uri__: URI = URI(Value, "number")

    def __init__(self, value: bool | int | float | object):
        op = _as_opref(value)
        if op is not None:
            super().__init__(None)
            object.__setattr__(self, "op", op)
            return

        if not isinstance(value, (bool, int, float)):
            raise TypeError("expected number value")

        super().__init__(value)
        object.__setattr__(self, "op", None)

    def to_json(self) -> object:
        return form_of(self)

    @classmethod
    def _from_json(cls, obj: Any) -> "Number":
        return cls(obj)

    def _scalar(self):
        from .scalar import autobox

        return autobox(self)

    def _binary(self, op_name: str, other: object, literal_op) -> "Number":
        from .scalar import (
            PostOpRef,
            autobox,
        )

        if self.op is None and isinstance(other, Number) and other.op is None:
            return Number(literal_op(form_of(self), form_of(other)))

        if self.op is None and isinstance(other, (int, float)) and not isinstance(other, bool):
            return Number(literal_op(form_of(self), other))

        subject = _subject_of_scalar(self)
        opref = PostOpRef(f"{subject}/{op_name}", {"r": autobox(other)})
        return Number(opref)

    @staticmethod
    def _number_operand(value: object) -> "Number":
        if isinstance(value, Number):
            return value
        return Number(value)

    def add(self, other: object) -> "Number":
        return self._binary("add", other, lambda l, r: l + r)

    def sub(self, other: object) -> "Number":
        return self._binary("sub", other, lambda l, r: l - r)

    def mul(self, other: object) -> "Number":
        return self._binary("mul", other, lambda l, r: l * r)

    def div(self, other: object) -> "Number":
        return self._binary("div", other, lambda l, r: l / r)

    def __add__(self, other: object) -> "Number":
        return self.add(other)

    def __radd__(self, other: object) -> "Number":
        return Number._number_operand(other).add(self)

    def __sub__(self, other: object) -> "Number":
        return self.sub(other)

    def __rsub__(self, other: object) -> "Number":
        return Number._number_operand(other).sub(self)

    def __mul__(self, other: object) -> "Number":
        return self.mul(other)

    def __rmul__(self, other: object) -> "Number":
        return Number._number_operand(other).mul(self)

    def __truediv__(self, other: object) -> "Number":
        return self.div(other)

    def __rtruediv__(self, other: object) -> "Number":
        return Number._number_operand(other).div(self)


class Integer(Number):
    __slots__ = ()

    __uri__: URI = URI(Number, "integer")

    def __init__(self, value: int | object):
        op = _as_opref(value)
        if op is not None:
            super().__init__(op)
            return

        if isinstance(value, bool):
            raise TypeError("bool is not an integer")
        if not isinstance(value, int):
            raise TypeError("expected integer value")

        super().__init__(value)


class Float(Number):
    __slots__ = ()

    __uri__: URI = URI(Number, "float")

    def __init__(self, value: int | float | object):
        op = _as_opref(value)
        if op is not None:
            super().__init__(op)
            return

        if isinstance(value, bool):
            raise TypeError("bool is not a float")
        if not isinstance(value, (int, float)):
            raise TypeError("expected float value")

        super().__init__(float(value))


class Complex(Number):
    __slots__ = ()

    __uri__: URI = URI(Number, "complex")

    def __init__(self, value: complex | object):
        op = _as_opref(value)
        if op is not None:
            super().__init__(op)
            return

        if not isinstance(value, complex):
            raise TypeError("expected complex value")

        Value.__init__(self, value)
        object.__setattr__(self, "op", None)

    @staticmethod
    def _complex_operand(value: object) -> "Complex":
        if isinstance(value, Complex):
            return value

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return Complex(complex(value))

        return Complex(value)

    def _binary_complex(self, op_name: str, other: object, literal_op) -> "Complex":
        from .scalar import PostOpRef, autobox

        if self.op is None:
            rhs = Complex._complex_operand(other)
            if rhs.op is None:
                left_form = form_of(self)
                right_form = form_of(rhs)
                assert isinstance(left_form, complex)
                assert isinstance(right_form, complex)
                return Complex(literal_op(left_form, right_form))

        subject = _subject_of_scalar(self)
        opref = PostOpRef(f"{subject}/{op_name}", {"r": autobox(other)})
        return Complex(opref)

    def add(self, other: object) -> "Complex":
        return self._binary_complex("add", other, lambda l, r: l + r)

    def sub(self, other: object) -> "Complex":
        return self._binary_complex("sub", other, lambda l, r: l - r)

    def mul(self, other: object) -> "Complex":
        return self._binary_complex("mul", other, lambda l, r: l * r)

    def div(self, other: object) -> "Complex":
        return self._binary_complex("div", other, lambda l, r: l / r)

    def __add__(self, other: object) -> "Complex":
        return self.add(other)

    def __radd__(self, other: object) -> "Complex":
        return Complex._complex_operand(other).add(self)

    def __sub__(self, other: object) -> "Complex":
        return self.sub(other)

    def __rsub__(self, other: object) -> "Complex":
        return Complex._complex_operand(other).sub(self)

    def __mul__(self, other: object) -> "Complex":
        return self.mul(other)

    def __rmul__(self, other: object) -> "Complex":
        return Complex._complex_operand(other).mul(self)

    def __truediv__(self, other: object) -> "Complex":
        return self.div(other)

    def __rtruediv__(self, other: object) -> "Complex":
        return Complex._complex_operand(other).div(self)

    def conjugate(self) -> "Complex":
        from .scalar import PostOpRef

        if self.op is None:
            value_form = form_of(self)
            assert isinstance(value_form, complex)
            return Complex(value_form.conjugate())

        subject = _subject_of_scalar(self)
        return Complex(PostOpRef(f"{subject}/conjugate", {}))

    def exp(self) -> "Complex":
        from .scalar import PostOpRef

        if self.op is None:
            value_form = form_of(self)
            assert isinstance(value_form, complex)
            return Complex(cmath.exp(value_form))

        subject = _subject_of_scalar(self)
        return Complex(PostOpRef(f"{subject}/exp", {}))

    def log(self) -> "Complex":
        from .scalar import PostOpRef

        if self.op is None:
            value_form = form_of(self)
            assert isinstance(value_form, complex)
            return Complex(cmath.log(value_form))

        subject = _subject_of_scalar(self)
        return Complex(PostOpRef(f"{subject}/log", {}))


class I64(Integer):
    __slots__ = ()

    __uri__: URI = URI(Integer, "i64")


class U64(Integer):
    __slots__ = ()

    __uri__: URI = URI(Integer, "u64")

    def __init__(self, value: int | object):
        super().__init__(value)

        if self.op is None:
            value_form = form_of(self)
            assert isinstance(value_form, int)
            if value_form < 0:
                raise ValueError("u64 cannot be negative")


class F32(Float):
    __slots__ = ()

    __uri__: URI = URI(Float, "32")


class F64(Float):
    __slots__ = ()

    __uri__: URI = URI(Float, "64")


class C64(Complex):
    __slots__ = ()

    __uri__: URI = URI(Complex, "64")


class C128(Complex):
    __slots__ = ()

    __uri__: URI = URI(Complex, "128")


# Backward-compatible alias: bool literals are represented as Number values.
Bool = Number


class Map(Value):
    __slots__ = ("op",)

    __uri__: URI = URI(Value, "map")

    def __init__(self, value: Mapping[str, object] | object):
        op = _as_opref(value)
        if op is not None:
            super().__init__(None)
            object.__setattr__(self, "op", op)
            return

        if not isinstance(value, Mapping):
            raise TypeError("expected map value")

        decoded: dict[str, Value] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("expected map key to be a string")
            decoded[key] = _decode_value_item(item)

        super().__init__(decoded)
        object.__setattr__(self, "op", None)

    def __iter__(self) -> Iterator[str]:
        value_form = form_of(self)
        if value_form is None:
            raise TypeError("cannot iterate a deferred Map without execution")
        assert isinstance(value_form, dict)
        return iter(value_form)

    def keys(self) -> Iterable[str]:
        value_form = form_of(self)
        if value_form is None:
            raise TypeError("cannot read keys from a deferred Map without execution")
        assert isinstance(value_form, dict)
        return value_form.keys()

    def values(self) -> Iterable[Value]:
        value_form = form_of(self)
        if value_form is None:
            raise TypeError("cannot read values from a deferred Map without execution")
        assert isinstance(value_form, dict)
        return value_form.values()

    def items(self) -> Iterable[tuple[str, Value]]:
        value_form = form_of(self)
        if value_form is None:
            raise TypeError("cannot read items from a deferred Map without execution")
        assert isinstance(value_form, dict)
        return value_form.items()

    def to_json(self) -> object:
        value_form = form_of(self)
        assert isinstance(value_form, dict)
        return {key: _encode_value_item(item) for key, item in value_form.items()}

    @classmethod
    def _from_json(cls, obj: Any) -> "Map":
        if not isinstance(obj, dict):
            raise TypeError("expected map value")
        return cls(obj)


class Tuple(Value):
    __slots__ = ("op",)

    __uri__: URI = URI(Value, "tuple")

    def __init__(self, value: Sequence[object] | object):
        op = _as_opref(value)
        if op is not None:
            super().__init__(None)
            object.__setattr__(self, "op", op)
            return

        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise TypeError("expected tuple value")

        super().__init__([_decode_value_item(item) for item in value])
        object.__setattr__(self, "op", None)

    def __iter__(self) -> Iterator[Value]:
        value_form = form_of(self)
        if value_form is None:
            raise TypeError("cannot iterate a deferred Tuple without execution")
        assert isinstance(value_form, list)
        return iter(value_form)

    def __len__(self) -> int:
        value_form = form_of(self)
        if value_form is None:
            raise TypeError("cannot get len of a deferred Tuple without execution")
        assert isinstance(value_form, list)
        return len(value_form)

    def __getitem__(self, index: int) -> Value:
        value_form = form_of(self)
        if value_form is None:
            raise TypeError("cannot index a deferred Tuple without execution")
        assert isinstance(value_form, list)
        return value_form[index]

    def to_json(self) -> object:
        value_form = form_of(self)
        assert isinstance(value_form, list)
        return [_encode_value_item(item) for item in value_form]

    @classmethod
    def _from_json(cls, obj: Any) -> "Tuple":
        if not isinstance(obj, list):
            raise TypeError("expected tuple value")
        return cls(obj)


def _as_opref(value: object) -> object | None:
    from ..opref import OpRef as RuntimeOpRef
    from .scalar import OpRef as StateOpRef

    if isinstance(value, (RuntimeOpRef, StateOpRef)):
        return value

    return None
