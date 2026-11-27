use std::sync::Arc;

use destream::FromStream;
use futures::TryStreamExt;
use get_size::GetSize;
use number_general::{Number, NumberInstance};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyComplex, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use tc_ir::{After, Cond, ForEach, Map, OpDef, OpRef, Scalar, Subject, TCRef, While};
use tc_value::class::NativeClass;
use tc_value::{Value, ValueType};

use super::{Budget, DEFAULT_LIMIT, Decoder, Error, en, project};

#[pyclass(frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct ScalarHandle {
    pub(crate) inner: Scalar,
}

impl ScalarHandle {
    fn checked(inner: Scalar) -> PyResult<Self> {
        let valid = match &inner {
            Scalar::Value(value) => value.within_depth(tc_value::MAX_VALUE_DEPTH),
            other => other.within_depth(tc_ir::MAX_SCALAR_DEPTH),
        };

        if !valid {
            return Err(bad("native composition exceeds its structural depth"));
        }

        if inner.get_size() > DEFAULT_LIMIT {
            return Err(bad("native composition exceeds byte budget"));
        }

        Ok(Self { inner })
    }
}

fn bad(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn native_map(values: Vec<(String, ScalarHandle)>) -> PyResult<Map<Scalar>> {
    let mut map = Map::new();
    for (key, value) in values {
        if map.insert(key.parse().map_err(bad)?, value.inner).is_some() {
            return Err(bad("duplicate map key"));
        }
    }

    Ok(map)
}

fn value(handle: ScalarHandle) -> PyResult<Value> {
    match handle.inner {
        Scalar::Value(value) => Ok(value),
        _ => Err(bad("expected a native Value")),
    }
}

fn reference(handle: ScalarHandle) -> PyResult<TCRef> {
    match handle.inner {
        Scalar::Ref(value) => Ok(*value),
        _ => Err(bad("expected a native reference")),
    }
}

#[pymethods]
impl ScalarHandle {
    #[staticmethod]
    fn literal(raw: &Bound<'_, PyAny>) -> PyResult<Self> {
        let value = if raw.is_none() {
            Value::None
        } else if raw.is_instance_of::<PyBool>() {
            Value::from(Number::from(raw.extract::<bool>()?))
        } else if raw.is_instance_of::<PyInt>() {
            if let Ok(n) = raw.extract::<i64>() {
                Number::from(n).into()
            } else {
                Number::from(raw.extract::<u64>()?).into()
            }
        } else if raw.is_instance_of::<PyFloat>() {
            Number::from(raw.extract::<f64>()?).into()
        } else if let Ok(n) = raw.cast::<PyComplex>() {
            Number::from([n.real(), n.imag()]).into()
        } else if let Ok(text) = raw.cast::<PyString>() {
            let text = text.to_str()?;
            if text.len() > DEFAULT_LIMIT {
                return Err(bad("literal exceeds materialization budget"));
            }

            Value::from(text.to_owned())
        } else if let Ok(bytes) = raw.cast::<PyBytes>() {
            if bytes.as_bytes().len() > DEFAULT_LIMIT {
                return Err(bad("literal exceeds materialization budget"));
            }

            Value::from(Arc::<[u8]>::from(bytes.as_bytes()))
        } else {
            return Err(bad("expected a primitive Python literal"));
        };

        Self::checked(value.into())
    }

    #[staticmethod]
    fn link(text: &str) -> PyResult<Self> {
        Self::checked(Value::from(text.parse::<pathlink::Link>().map_err(bad)?).into())
    }

    #[staticmethod]
    fn number(raw: &Bound<'_, PyAny>, dtype: &str) -> PyResult<Self> {
        let Scalar::Value(Value::Number(number)) = Self::literal(raw)?.inner else {
            return Err(bad("expected a number"));
        };

        let path = dtype.parse::<pathlink::PathBuf>().map_err(bad)?;
        let Some(ValueType::Number(dtype)) = ValueType::from_path(&path) else {
            return Err(bad("expected numerical type path"));
        };

        Self::checked(Value::from(number.try_into_type(dtype).map_err(bad)?).into())
    }

    #[staticmethod]
    fn tuple(values: Vec<Self>) -> PyResult<Self> {
        Self::checked(Scalar::Tuple(values.into_iter().map(|v| v.inner).collect()))
    }

    #[staticmethod]
    fn value_tuple(values: Vec<Self>) -> PyResult<Self> {
        Self::checked(Value::Tuple(values.into_iter().map(value).collect::<PyResult<_>>()?).into())
    }

    #[staticmethod]
    fn map(values: Vec<(String, Self)>) -> PyResult<Self> {
        Self::checked(Scalar::Map(native_map(values)?))
    }

    #[staticmethod]
    fn idref(text: &str) -> PyResult<Self> {
        Self::checked(TCRef::Id(text.parse().map_err(bad)?).into())
    }

    #[staticmethod]
    fn get(subject: &str, key: Self) -> PyResult<Self> {
        Self::checked(
            TCRef::Op(OpRef::Get((
                subject.parse::<Subject>().map_err(bad)?,
                key.inner,
            )))
            .into(),
        )
    }

    #[staticmethod]
    fn put(subject: &str, key: Self, value: Self) -> PyResult<Self> {
        Self::checked(
            TCRef::Op(OpRef::Put((
                subject.parse().map_err(bad)?,
                key.inner,
                value.inner,
            )))
            .into(),
        )
    }

    #[staticmethod]
    fn post(subject: &str, params: Vec<(String, Self)>) -> PyResult<Self> {
        Self::checked(
            TCRef::Op(OpRef::Post((
                subject.parse().map_err(bad)?,
                native_map(params)?,
            )))
            .into(),
        )
    }

    #[staticmethod]
    fn delete(subject: &str, key: Self) -> PyResult<Self> {
        Self::checked(TCRef::Op(OpRef::Delete((subject.parse().map_err(bad)?, key.inner))).into())
    }

    #[staticmethod]
    fn opdef(method: &str, parameters: Vec<String>, form: Vec<(String, Self)>) -> PyResult<Self> {
        let parameters = parameters
            .into_iter()
            .map(|p| p.parse::<tc_ir::Id>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(bad)?;

        let form = form
            .into_iter()
            .map(|(name, value)| Ok((name.parse().map_err(bad)?, value.inner)))
            .collect::<PyResult<_>>()?;

        let op = match (method, &parameters[..]) {
            ("GET", [key]) => OpDef::Get((key.clone(), form)),
            ("PUT", [key, value]) => OpDef::Put((key.clone(), value.clone(), form)),
            ("POST", []) => OpDef::Post(form),
            ("DELETE", [key]) => OpDef::Delete((key.clone(), form)),
            _ => return Err(bad("invalid operation definition parameters")),
        };

        op.validate().map_err(bad)?;
        Self::checked(Scalar::Op(op))
    }

    #[staticmethod]
    fn cond(condition: Self, then: Self, or_else: Self) -> PyResult<Self> {
        Self::checked(
            TCRef::Cond(Box::new(Cond::new(
                reference(condition)?,
                then.inner,
                or_else.inner,
            )))
            .into(),
        )
    }

    #[staticmethod]
    fn after(when: Self, then: Self) -> PyResult<Self> {
        Self::checked(TCRef::After(Box::new(After::new(when.inner, then.inner))).into())
    }

    #[staticmethod]
    fn while_loop(condition: Self, op: Self, state: Self) -> PyResult<Self> {
        Self::checked(
            TCRef::While(Box::new(While::new(condition.inner, op.inner, state.inner))).into(),
        )
    }

    #[staticmethod]
    fn for_each(items: Self, op: Self, name: &str) -> PyResult<Self> {
        Self::checked(
            TCRef::ForEach(Box::new(ForEach::new(
                items.inner,
                op.inner,
                name.parse().map_err(bad)?,
            )))
            .into(),
        )
    }

    fn link_text(&self) -> PyResult<String> {
        if let Scalar::Value(Value::Link(link)) = &self.inner {
            Ok(link.to_string())
        } else {
            Err(bad("expected a Link"))
        }
    }
}

#[pyclass]
pub(crate) struct Codec {
    runtime: Arc<tokio::runtime::Runtime>,
    limit: usize,
}

impl Codec {
    pub(super) fn scalar_wire(&self, scalar: &Scalar) -> PyResult<Py<PyAny>> {
        if !scalar.within_depth(tc_ir::MAX_SCALAR_DEPTH) {
            return Err(bad("Scalar nesting exceeds limit"));
        }

        self.wait(en::encode(scalar, Budget::new(self.limit)))?
            .map_err(Into::into)
    }

    pub(crate) fn wait<F: std::future::Future + Send>(&self, future: F) -> PyResult<F::Output>
    where
        F::Output: Send,
    {
        super::wait(&self.runtime, future)
    }

    fn decode<T: FromStream<Context = ()> + Send>(
        &self,
        raw: Py<PyAny>,
        budget: Budget,
    ) -> PyResult<T> {
        self.wait(async {
            let mut decoder = Decoder::new(raw, budget);
            let value = T::from_stream((), &mut decoder).await?;
            decoder.finish()?;
            Ok::<_, Error>(value)
        })?
        .map_err(Into::into)
    }

    async fn project_response(
        raw: Py<PyAny>,
        budget: Budget,
        remaining: usize,
    ) -> Result<Py<PyAny>, Error> {
        if remaining == 0 {
            return Err(Error::Message("response nesting exceeds limit".into()));
        }

        budget.charge(std::mem::size_of::<Py<PyAny>>())?;
        enum Node {
            Map(Vec<(String, Py<PyAny>)>),
            Seq(Vec<Py<PyAny>>),
            Collection(Py<PyAny>),
            Scalar(Py<PyAny>),
        }

        let node = Python::attach(|py| -> Result<Node, Error> {
            let value = raw.bind(py);
            if let Ok(map) = value.cast::<PyDict>() {
                let collection = py
                    .import("tinychain.state.collection")?
                    .getattr("Collection")?
                    .call_method1("decode", (value,))?;

                if !collection.is_none() {
                    return Ok(Node::Collection(collection.unbind()));
                }

                let ordinary = match map.iter().next() {
                    None => true,
                    Some((key, _)) => key.extract::<String>()?.parse::<tc_ir::Id>().is_ok(),
                };

                if ordinary {
                    budget.charge(
                        map.len()
                            .saturating_mul(std::mem::size_of::<(String, Py<PyAny>)>()),
                    )?;

                    let mut entries = Vec::with_capacity(map.len());
                    for (key, value) in map.iter() {
                        let key = key.extract::<String>()?;
                        key.parse::<tc_ir::Id>()
                            .map_err(|e| Error::Message(e.to_string()))?;

                        budget.charge(key.len())?;
                        entries.push((key, value.unbind()));
                    }

                    return Ok(Node::Map(entries));
                }
            }

            if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
                budget.charge(
                    value
                        .len()?
                        .saturating_mul(std::mem::size_of::<Py<PyAny>>()),
                )?;

                let values = if let Ok(list) = value.cast::<PyList>() {
                    list.iter().map(Bound::unbind).collect()
                } else {
                    value.cast::<PyTuple>()?.iter().map(Bound::unbind).collect()
                };

                return Ok(Node::Seq(values));
            }

            Ok(Node::Scalar(raw))
        })?;

        match node {
            Node::Collection(value) => Ok(value),
            Node::Map(entries) => {
                let map = Python::attach(|py| PyDict::new(py).unbind());
                for (key, raw) in entries {
                    let value =
                        Box::pin(Self::project_response(raw, budget.clone(), remaining - 1))
                            .await?;

                    Python::attach(|py| map.bind(py).set_item(key, value))?;
                }

                Ok(map.into_any())
            }
            Node::Seq(entries) => {
                let seq = Python::attach(|py| PyList::empty(py).unbind());
                for raw in entries {
                    let value =
                        Box::pin(Self::project_response(raw, budget.clone(), remaining - 1))
                            .await?;

                    Python::attach(|py| seq.bind(py).append(value))?;
                }

                Ok(seq.into_any())
            }
            Node::Scalar(raw) => {
                let mut decoder = Decoder::new(raw, budget.clone());
                let value = tc_ir::decode_scalar(&mut decoder, remaining).await?;
                decoder.finish()?;
                Python::attach(|py| project::materialized_scalar(py, &value, &budget))
                    .map_err(Error::from)
            }
        }
    }
}

#[pymethods]
impl Codec {
    #[new]
    #[pyo3(signature = (max_bytes = DEFAULT_LIMIT))]
    pub(super) fn new(max_bytes: usize) -> PyResult<Self> {
        if max_bytes == 0 {
            return Err(bad("codec byte budget must be positive"));
        }

        Ok(Self {
            runtime: Arc::new(
                tokio::runtime::Builder::new_current_thread()
                    .build()
                    .map_err(bad)?,
            ),
            limit: max_bytes,
        })
    }

    fn value_from_wire(&self, py: Python<'_>, raw: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let budget = Budget::new(self.limit);
        let value: Scalar = self.decode(raw, budget.clone())?;
        project::value_form(py, &value, &budget)
    }

    fn scalar_from_wire(&self, py: Python<'_>, raw: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let budget = Budget::new(self.limit);
        let value: Scalar = self.decode(raw, budget.clone())?;
        project::scalar(py, &value, &budget)
    }

    fn materialize(&self, raw: Py<PyAny>) -> PyResult<Py<PyAny>> {
        self.wait(Self::project_response(
            raw,
            Budget::new(self.limit),
            tc_ir::MAX_SCALAR_DEPTH + 1,
        ))?
        .map_err(Into::into)
    }

    fn ref_from_wire(&self, py: Python<'_>, raw: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let budget = Budget::new(self.limit);
        let value: TCRef = self.decode(raw, budget.clone())?;
        project::reference(py, &value, &budget)
    }

    fn opdef_from_wire(&self, py: Python<'_>, raw: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let budget = Budget::new(self.limit);
        let value: OpDef = self.decode(raw, budget.clone())?;
        project::opdef(py, &value, &budget)
    }

    fn to_wire(&self, value: PyRef<'_, ScalarHandle>) -> PyResult<Py<PyAny>> {
        self.scalar_wire(&value.inner)
    }

    fn to_value_wire(&self, value: PyRef<'_, ScalarHandle>) -> PyResult<Py<PyAny>> {
        if let Scalar::Value(native) = &value.inner {
            self.wait(en::encode(native, Budget::new(self.limit)))?
                .map_err(Into::into)
        } else {
            self.to_wire(value)
        }
    }

    fn wire_object(&self, raw: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let budget = Budget::new(self.limit);
        self.wait(async {
            let object = super::wire::Object {
                raw,
                budget: budget.clone(),
            };

            en::encode(object, budget).await
        })?
        .map_err(Into::into)
    }

    fn encode_wire(&self, raw: Py<PyAny>) -> PyResult<Vec<u8>> {
        self.wait(async {
            let budget = Budget::new(self.limit);
            let object = super::wire::Object {
                raw,
                budget: budget.clone(),
            };

            destream_json::encode(object)
                .map_err(|e| Error::Message(e.to_string()))?
                .try_fold(Vec::new(), |mut out, chunk| {
                    let budget = &budget;
                    async move {
                        budget.charge(chunk.len()).map_err(|e| {
                            <destream_json::en::Error as destream::en::Error>::custom(e)
                        })?;

                        out.extend_from_slice(&chunk);
                        Ok(out)
                    }
                })
                .await
                .map_err(|e| Error::Message(e.to_string()))
        })?
        .map_err(Into::into)
    }
}
