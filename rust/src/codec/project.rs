//! Project native semantic variants into the existing Python authoring API.
use number_general::{Number, NumberInstance};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyComplex, PyDict, PyList};
use tc_ir::{OpDef, OpRef, Scalar, TCRef};
use tc_value::Value;

use super::Budget;

fn construct<'py>(
    py: Python<'py>,
    module: &str,
    name: &str,
    args: impl pyo3::call::PyCallArgs<'py>,
) -> PyResult<Py<PyAny>> {
    Ok(py.import(module)?.getattr(name)?.call1(args)?.unbind())
}

pub(crate) fn value(py: Python<'_>, value: &Value, budget: &Budget) -> PyResult<Py<PyAny>> {
    budget.charge(std::mem::size_of::<Value>())?;
    let module = "tinychain.state.value";
    match value {
        Value::None => construct(py, module, "Null", ()),
        Value::String(text) => {
            budget.charge(text.len())?;
            construct(py, module, "String", (text,))
        }
        Value::Link(link) => construct(py, module, "Link", (link.to_string(),)),
        Value::Bytes(bytes) => {
            budget.charge(bytes.len())?;
            construct(py, module, "Bytes", (PyBytes::new(py, bytes),))
        }
        Value::Number(number) => match number {
            Number::Bool(n) => construct(py, module, "Bool", (bool::from(*n),)),
            Number::Int(n) => construct(py, module, "Integer", (i64::from(*n),)),
            Number::UInt(n) => construct(py, module, "U64", (u64::from(*n),)),
            Number::Float(n) => construct(
                py,
                module,
                if n.class() == number_general::FloatType::F32 {
                    "F32"
                } else {
                    "F64"
                },
                (f64::from(*n),),
            ),
            Number::Complex(n) => {
                let (re, im, name) = match n {
                    number_general::Complex::C32(n) => (f64::from(n.re), f64::from(n.im), "C64"),
                    number_general::Complex::C64(n) => (n.re, n.im, "C128"),
                };

                construct(py, module, name, (PyComplex::from_doubles(py, re, im),))
            }
        },
        Value::Tuple(values) => {
            let items = PyList::empty(py);
            for item in values {
                items.append(self::value(py, item, budget)?)?;
            }

            construct(py, module, "Tuple", (items,))
        }
    }
}

pub(crate) fn scalar(py: Python<'_>, scalar: &Scalar, budget: &Budget) -> PyResult<Py<PyAny>> {
    budget.charge(std::mem::size_of::<Scalar>())?;
    match scalar {
        Scalar::Value(v) => construct(
            py,
            "tinychain.state.scalar",
            "autobox",
            (value(py, v, budget)?,),
        ),
        Scalar::Map(values) => {
            let map = PyDict::new(py);
            for (key, value) in values.iter() {
                map.set_item(key.as_str(), self::scalar(py, value, budget)?)?;
            }

            construct(py, "tinychain.state.scalar", "map_of", (map,))
        }
        Scalar::Tuple(values) => {
            let list = PyList::empty(py);
            for value in values {
                list.append(self::scalar(py, value, budget)?)?;
            }

            construct(py, "tinychain.state.scalar", "tuple_of", (list,))
        }
        Scalar::Ref(value) => construct(
            py,
            "tinychain.state.scalar",
            "autobox",
            (reference(py, value, budget)?,),
        ),
        Scalar::Op(value) => construct(
            py,
            "tinychain.state.scalar",
            "Scalar",
            (opdef(py, value, budget)?,),
        ),
    }
}

// Python's Value facade also includes Map/Tuple authoring containers. Their
// grammar remains Scalar's; this projection introduces no native Value variant.
pub(crate) fn value_form(py: Python<'_>, scalar: &Scalar, budget: &Budget) -> PyResult<Py<PyAny>> {
    budget.charge(std::mem::size_of::<Scalar>())?;
    match scalar {
        Scalar::Value(v) => value(py, v, budget),
        Scalar::Tuple(values) => {
            let list = PyList::empty(py);
            for item in values {
                list.append(value_form(py, item, budget)?)?;
            }

            construct(py, "tinychain.state.value", "Tuple", (list,))
        }
        Scalar::Map(values) => {
            let map = PyDict::new(py);
            for (key, item) in values.iter() {
                map.set_item(key.as_str(), value_form(py, item, budget)?)?;
            }

            construct(py, "tinychain.state.value", "Map", (map,))
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a literal Value",
        )),
    }
}

pub(crate) fn reference(py: Python<'_>, value: &TCRef, budget: &Budget) -> PyResult<Py<PyAny>> {
    budget.charge(std::mem::size_of::<TCRef>())?;
    let module = "tinychain.state.scalar.refs";
    match value {
        TCRef::Id(id) => construct(
            py,
            module,
            "IdRef",
            (id.to_string().trim_start_matches('$'),),
        ),
        TCRef::Op(op) => match op {
            OpRef::Get((subject, key)) => construct(
                py,
                module,
                "GetOpRef",
                (subject.to_string(), scalar(py, key, budget)?),
            ),
            OpRef::Put((subject, key, value)) => construct(
                py,
                module,
                "PutOpRef",
                (
                    subject.to_string(),
                    scalar(py, key, budget)?,
                    scalar(py, value, budget)?,
                ),
            ),
            OpRef::Delete((subject, key)) => construct(
                py,
                module,
                "DeleteOpRef",
                (subject.to_string(), scalar(py, key, budget)?),
            ),
            OpRef::Post((subject, values)) => {
                let params = PyDict::new(py);
                for (key, value) in values.iter() {
                    params.set_item(key.as_str(), scalar(py, value, budget)?)?;
                }

                construct(py, module, "PostOpRef", (subject.to_string(), params))
            }
        },
        TCRef::Cond(value) => construct(
            py,
            module,
            "Cond",
            (
                reference(py, &value.cond, budget)?,
                scalar(py, &value.then, budget)?,
                scalar(py, &value.or_else, budget)?,
            ),
        ),
        TCRef::After(value) => construct(
            py,
            module,
            "After",
            (
                scalar(py, &value.when, budget)?,
                scalar(py, &value.then, budget)?,
            ),
        ),
        TCRef::While(value) => construct(
            py,
            module,
            "While",
            (
                scalar(py, &value.cond, budget)?,
                scalar(py, &value.closure, budget)?,
                scalar(py, &value.state, budget)?,
            ),
        ),
        TCRef::ForEach(value) => construct(
            py,
            module,
            "ForEach",
            (
                scalar(py, &value.items, budget)?,
                scalar(py, &value.op, budget)?,
                value.item_name.as_str(),
            ),
        ),
    }
}

pub(crate) fn opdef(py: Python<'_>, op: &OpDef, budget: &Budget) -> PyResult<Py<PyAny>> {
    let form = PyList::empty(py);
    for (name, value) in op.form() {
        form.append((name.as_str(), scalar(py, value, budget)?))?;
    }

    let module = "tinychain.state.scalar.opdef";
    match op {
        OpDef::Get((key, _)) => construct(py, module, "GetOpDef", (key.as_str(), form)),
        OpDef::Put((key, value, _)) => {
            construct(py, module, "PutOpDef", (key.as_str(), value.as_str(), form))
        }
        OpDef::Post(_) => construct(py, module, "PostOpDef", (form,)),
        OpDef::Delete((key, _)) => construct(py, module, "DeleteOpDef", (key.as_str(), form)),
    }
}

pub(crate) fn materialized_value(
    py: Python<'_>,
    native: &Value,
    budget: &Budget,
) -> PyResult<Py<PyAny>> {
    budget.charge(std::mem::size_of::<Value>())?;
    match native {
        Value::None => Ok(py.None()),
        Value::Tuple(values) => {
            let out = PyList::empty(py);
            for item in values {
                out.append(materialized_value(py, item, budget)?)?;
            }

            Ok(out.into_any().unbind())
        }
        // The Value projection already preserves native numerical width and
        // Link normalization. Unwrap its authoring form at this explicit
        // foreign materialization boundary, without encoding it again.
        _ => Ok(value(py, native, budget)?
            .bind(py)
            .getattr("_form")?
            .unbind()),
    }
}

pub(crate) fn materialized_scalar(
    py: Python<'_>,
    native: &Scalar,
    budget: &Budget,
) -> PyResult<Py<PyAny>> {
    match native {
        Scalar::Value(value) => materialized_value(py, value, budget),
        Scalar::Tuple(values) => {
            let out = PyList::empty(py);
            for item in values {
                out.append(materialized_scalar(py, item, budget)?)?;
            }

            Ok(out.into_any().unbind())
        }
        Scalar::Map(values) => {
            let out = PyDict::new(py);
            for (key, item) in values.iter() {
                out.set_item(key.as_str(), materialized_scalar(py, item, budget)?)?;
            }

            Ok(out.into_any().unbind())
        }
        Scalar::Ref(value) => reference(py, value, budget),
        Scalar::Op(value) => opdef(py, value, budget),
    }
}
