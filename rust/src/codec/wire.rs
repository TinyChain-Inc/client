//! Traverse explicit foreign wire objects, delegating native semantic leaves.
use std::collections::VecDeque;

use destream::en::{self, Error as _, IntoStream};
use get_size::GetSize;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use super::{Budget, Error, ScalarHandle};

pub(crate) struct Object {
    pub(crate) raw: Py<PyAny>,
    pub(crate) budget: Budget,
}

impl<'en> IntoStream<'en> for Object {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let events = Events {
            current: Some(self.raw),
            frames: Vec::new(),
            budget: self.budget,
        };

        encoder.encode_events(futures::stream::iter(
            events.map(|event| event.map_err(en::Error::custom)),
        ))
    }
}

struct Leaf(Py<PyAny>);

impl<'en> IntoStream<'en> for Leaf {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        Python::attach(|py| {
            let raw = self.0.bind(py);
            if let Ok(handle) = raw.extract::<PyRef<'_, ScalarHandle>>() {
                if !handle.inner.within_depth(tc_ir::MAX_SCALAR_DEPTH) {
                    return Err(en::Error::custom("Scalar nesting exceeds limit"));
                }

                // The output stream may independently retain this leaf after
                // the foreign object's borrow ends.
                return handle.inner.clone().into_stream(encoder);
            }

            if raw.is_none() {
                encoder.encode_unit()
            } else if raw.is_instance_of::<PyBool>() {
                encoder.encode_bool(raw.extract().map_err(en::Error::custom)?)
            } else if raw.is_instance_of::<PyInt>() {
                if let Ok(n) = raw.extract::<i64>() {
                    encoder.encode_i64(n)
                } else {
                    encoder.encode_u64(raw.extract().map_err(en::Error::custom)?)
                }
            } else if raw.is_instance_of::<PyFloat>() {
                encoder.encode_f64(raw.extract().map_err(en::Error::custom)?)
            } else {
                encoder.encode_str(raw.extract::<String>().map_err(en::Error::custom)?)
            }
        })
    }
}

struct Events {
    current: Option<Py<PyAny>>,
    frames: Vec<VecDeque<Py<PyAny>>>,
    budget: Budget,
}

impl Iterator for Events {
    type Item = Result<en::Event<Leaf>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut raw = match self.current.take() {
            Some(value) => value,
            None => match self.frames.last_mut()?.pop_front() {
                Some(value) => value,
                None => {
                    self.frames.pop();
                    return Some(Ok(en::Event::End));
                }
            },
        };

        Some(Python::attach(|py| {
            self.budget.charge(std::mem::size_of::<Py<PyAny>>())?;
            let value = raw.bind(py);
            let foreign_leaf = !value.is_none()
                && !value.is_instance_of::<PyBool>()
                && !value.is_instance_of::<PyInt>()
                && !value.is_instance_of::<PyFloat>()
                && !value.is_instance_of::<PyString>()
                && !value.is_instance_of::<PyDict>()
                && !value.is_instance_of::<PyList>()
                && !value.is_instance_of::<PyTuple>()
                && !value.is_instance_of::<ScalarHandle>();

            if foreign_leaf {
                raw = py
                    .import("tinychain._native")?
                    .call_method1("wire_leaf", (value,))?
                    .unbind();
            }

            let value = raw.bind(py);
            let map = value.cast::<PyDict>().ok();
            let seq = value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>();
            if map.is_some() || seq {
                if self.frames.len() >= tc_value::MAX_VALUE_DEPTH {
                    return Err(Error::custom("wire object nesting exceeds limit"));
                }

                let len = value.len()?;
                let slots = len
                    .checked_mul(if map.is_some() { 2 } else { 1 })
                    .and_then(|n| n.checked_mul(std::mem::size_of::<Py<PyAny>>()))
                    .ok_or_else(|| Error::custom("wire object size overflow"))?;

                self.budget.charge(slots)?;
                let mut children = VecDeque::new();
                if let Some(map) = map {
                    for (key, value) in map.iter() {
                        if !key.is_instance_of::<PyString>() {
                            return Err(Error::custom("wire map key must be a string"));
                        }

                        children.push_back(key.unbind());
                        children.push_back(value.unbind());
                    }

                    self.frames.push(children);
                    Ok(en::Event::MapStart(Some(len)))
                } else {
                    if let Ok(list) = value.cast::<PyList>() {
                        children.extend(list.iter().map(Bound::unbind));
                    } else {
                        children.extend(value.cast::<PyTuple>()?.iter().map(Bound::unbind));
                    }

                    self.frames.push(children);
                    Ok(en::Event::SeqStart(Some(len)))
                }
            } else {
                if let Ok(handle) = value.extract::<PyRef<'_, ScalarHandle>>() {
                    self.budget.charge(handle.inner.get_size())?;
                }

                if let Ok(text) = value.cast::<PyString>() {
                    self.budget.charge(text.to_str()?.len())?;
                }

                Ok(en::Event::Value(Leaf(raw)))
            }
        }))
    }
}
