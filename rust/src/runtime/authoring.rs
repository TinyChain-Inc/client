//! Normalize Python State composition directly; collection leaves delegate their
//! foreign schema/payload to the owning State/Collection codec with its capability.
use get_size::GetSize;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use tc_error::{TCError, TCResult};

use crate::codec::{Budget, Decoder, ScalarHandle};
use crate::{BoundRequest, State};
use super::types::PyStateHandle;

fn input(error: impl std::fmt::Display) -> TCError {
    TCError::bad_request(error.to_string())
}

pub(super) async fn into_state(
    raw: Py<PyAny>,
    request: &mut BoundRequest,
    budget: &Budget,
    remaining: usize,
) -> TCResult<State> {
    budget.charge(std::mem::size_of::<State>()).map_err(input)?;
    enum Node {
        Scalar(tc_ir::Scalar),
        Native(Py<PyAny>),
        Map(Vec<(String, Py<PyAny>)>),
        Tuple(Vec<Py<PyAny>>),
        Collection(Py<PyAny>),
    }

    let node = Python::attach(|py| -> PyResult<Node> {
        let value = raw.bind(py);
        if let Ok(handle) = value.extract::<PyRef<'_, ScalarHandle>>() {
            budget.charge(handle.inner.get_size())?;
            return Ok(Node::Scalar(handle.inner.clone()));
        }

        if value.is_instance_of::<PyStateHandle>() {
            return Ok(Node::Native(raw));
        }

        if let Ok(map) = value.cast::<PyDict>() {
            budget.charge(
                map.len()
                    .saturating_mul(std::mem::size_of::<(String, Py<PyAny>)>()),
            )?;

            let mut entries = Vec::with_capacity(map.len());
            for (key, value) in map.iter() {
                let key = key.cast::<PyString>()?.to_str()?;
                budget.charge(key.len())?;
                entries.push((key.to_owned(), value.unbind()));
            }

            return Ok(Node::Map(entries));
        }

        if let Ok(list) = value.cast::<PyList>() {
            budget.charge(list.len().saturating_mul(std::mem::size_of::<Py<PyAny>>()))?;
            return Ok(Node::Tuple(list.iter().map(Bound::unbind).collect()));
        }

        if let Ok(list) = value.cast::<PyTuple>() {
            budget.charge(list.len().saturating_mul(std::mem::size_of::<Py<PyAny>>()))?;
            return Ok(Node::Tuple(list.iter().map(Bound::unbind).collect()));
        }

        let collection = py
            .import("tinychain.state.collection")?
            .getattr("Collection")?;

        if value.is_instance(&collection)? {
            return Ok(Node::Collection(value.getattr("_form")?.unbind()));
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected normalized State authoring data",
        ))
    })
    .map_err(input)?;

    Ok(match node {
        Node::Native(value) => {
            let state = Python::attach(|py| {
                value
                    .bind(py)
                    .extract::<PyRef<'_, PyStateHandle>>()?
                    .retain_state(request)
            })
            .map_err(input)?
            .ok_or_else(|| input("expected a retained native State"))?;

            if !state.within_depth(remaining) {
                return Err(input("State composition exceeds nesting limit"));
            }

            state
        }
        Node::Scalar(value) => {
            if !value.within_depth(remaining) {
                return Err(input("Scalar composition exceeds nesting limit"));
            }

            State::from(value)
        }
        Node::Map(children) => {
            let child_depth = remaining
                .checked_sub(1)
                .ok_or_else(|| input("State composition exceeds nesting limit"))?;

            let mut map = tc_ir::Map::new();
            for (key, value) in children {
                let key = key.parse().map_err(input)?;
                let value = Box::pin(into_state(value, request, budget, child_depth)).await?;
                map.insert(key, value);
            }

            State::from(map)
        }
        Node::Tuple(children) => {
            let child_depth = remaining
                .checked_sub(1)
                .ok_or_else(|| input("State composition exceeds nesting limit"))?;

            let mut values = Vec::with_capacity(children.len());
            for value in children {
                values.push(Box::pin(into_state(value, request, budget, child_depth)).await?);
            }

            State::from(values)
        }
        Node::Collection(payload) => {
            let mut decoder = Decoder::new(payload, budget.clone());
            let state = tc_state::decode_state(request.txn().clone(), &mut decoder, remaining)
                .await
                .map_err(input)?;

            decoder.finish().map_err(input)?;
            state
        }
    })
}
