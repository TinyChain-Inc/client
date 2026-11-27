use futures::stream;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString, PyTuple};
use tokio::sync::OwnedSemaphorePermit;

use crate::BodyContract;
use super::types::PyStateHandle;

pub(super) fn decode_body(
    body: Option<PyStateHandle>,
    content_type: Option<&str>,
    request: &mut crate::BoundRequest,
    runtime: &tokio::runtime::Runtime,
) -> PyResult<(Option<crate::State>, Option<OwnedSemaphorePermit>)> {
    let deadline = request.deadline();
    deadline.check().map_err(super::tc_error)?;

    let contract = request
        .body_contract(content_type)
        .map_err(super::tc_error)?;

    let (limit, application) = match contract {
        BodyContract::Native(limit) => (limit, false),
        BodyContract::Definition(limit) | BodyContract::Wasm(limit) => (limit, true),
    };

    let wasm = matches!(contract, BodyContract::Wasm(_));
    if !wasm
        && let Some(handle) = &body
        && let Some(state) = handle.take_native_state(request, runtime)?
    {
        request.txn().check_active().map_err(super::tc_error)?;
        request.deadline().check().map_err(super::tc_error)?;
        let memory = if application {
            let bytes = handle.application_bytes(limit)?;
            Some(
                super::wait(
                    runtime,
                    deadline.run(request.admit_application_memory(bytes)),
                )?
                .map_err(super::tc_error)?,
            )
        } else {
            None
        };

        return Ok((Some(state), memory));
    }

    if body.is_none() && !application {
        return Ok((None, None));
    }

    if !wasm && let Some(handle) = &body {
        let raw = handle.value()?;
        let object = Python::attach(|py| {
            let raw = raw.bind(py);
            raw.is_instance_of::<PyDict>()
                || raw.is_instance_of::<PyList>()
                || raw.is_instance_of::<PyTuple>()
        });

        if object {
            return super::wait(
                runtime,
                deadline.run(async {
                    let memory = if application {
                        Some(request.admit_application_memory(limit).await?)
                    } else {
                        None
                    };

                    let mut decoder =
                        crate::codec::Decoder::new(raw, crate::codec::Budget::new(limit));

                    let state = request.decode_object(&mut decoder).await?;

                    decoder
                        .finish()
                        .map_err(|error| tc_error::TCError::bad_request(error.to_string()))?;

                    Ok((Some(state), memory))
                }),
            )?
            .map_err(super::tc_error);
        }
    }

    let bytes = match body {
        None => Vec::new(),
        Some(handle) => Python::attach(|py| {
            let value = handle.value()?;
            let any = value.bind(py);
            let bytes = if let Ok(bytes) = any.cast::<PyBytes>() {
                bytes.as_bytes()
            } else if let Ok(string) = any.cast::<PyString>() {
                string.to_str()?.as_bytes()
            } else {
                return Err(PyValueError::new_err("expected bytes or string body"));
            };

            if bytes.len() > limit {
                return Err(super::tc_error(tc_error::TCError::payload_too_large(
                    "request body exceeds its request bound",
                    tc_error::Pressure::new(
                        "/host/resource/request/body",
                        tc_error::PressureReason::QuotaExceeded,
                    ),
                )));
            }

            Ok(bytes.to_vec())
        })?,
    };

    deadline.check().map_err(super::tc_error)?;

    super::wait(
        runtime,
        deadline.run(async {
            let permit = if application {
                Some(request.admit_application_memory(bytes.len()).await?)
            } else {
                None
            };

            let state = if wasm {
                crate::wasm_installation(bytes.into())?
            } else {
                let input = stream::iter([Ok::<_, std::io::Error>(bytes.into())]);
                request.decode_json(input).await?
            };

            Ok((Some(state), permit))
        }),
    )?
    .map_err(super::tc_error)
}
