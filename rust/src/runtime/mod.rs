use pyo3::prelude::PyAnyMethods;

mod authoring;
mod kernel;
mod state_handle_conversions;
mod types;
mod wire;

pub(super) use crate::codec::wait;

pub(super) fn tc_error(err: tc_error::TCError) -> pyo3::PyErr {
    let exception = pyo3::exceptions::PyRuntimeError::new_err(err.message().to_string());
    pyo3::Python::attach(|py| {
        let value = exception.value(py);
        let _ = value.setattr("code", err.code().to_string());
        if let Some(pressure) = err.pressure() {
            let _ = value.setattr("reason", pressure.reason().to_string());
            let _ = value.setattr("resource", pressure.resource());
            let _ = value.setattr("retry_after_ms", pressure.retry_after_ms());
            let _ = value.setattr("reliability", pressure.reliability().to_string());
        }
    });
    exception
}

pub use kernel::register_python_api;

fn runtime() -> std::io::Result<std::sync::Arc<tokio::runtime::Runtime>> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .map(std::sync::Arc::new)
}

#[cfg(test)]
#[path = "../../tests/unit/runtime.rs"]
mod tests;
