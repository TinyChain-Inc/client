//! Foreign-object adapters delegate all TinyChain wire grammar to native codecs.

use std::fmt;
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

mod de;
mod en;
mod handle;
mod project;
mod wire;

pub(crate) use de::Decoder;
pub(crate) use handle::ScalarHandle;

pub(crate) fn wait<F>(runtime: &tokio::runtime::Runtime, future: F) -> PyResult<F::Output>
where
    F: std::future::Future + Send,
    F::Output: Send,
{
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "synchronous Python calls cannot run inside an active Tokio runtime",
        ));
    }

    Python::attach(|py| py.detach(|| Ok(runtime.block_on(future))))
}

#[cfg(feature = "local-host")]
pub(crate) fn project_scalar(value: &tc_ir::Scalar) -> PyResult<Py<PyAny>> {
    handle::Codec::new(DEFAULT_LIMIT)?.scalar_wire(value)
}

#[cfg(feature = "local-host")]
pub(crate) async fn into_wire<'en>(
    value: impl destream::IntoStream<'en>,
) -> Result<Py<PyAny>, Error> {
    en::encode(value, Budget::new(DEFAULT_LIMIT)).await
}

const DEFAULT_LIMIT: usize = 64 * 1024 * 1024;

#[derive(Debug)]
pub(crate) enum Error {
    Message(String),
    Python(PyErr),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) => f.write_str(message),
            Self::Python(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::convert::Infallible> for Error {
    fn from(value: std::convert::Infallible) -> Self {
        match value {}
    }
}

impl destream::de::Error for Error {
    fn custom<T: fmt::Display>(value: T) -> Self {
        Self::Message(value.to_string())
    }
}

impl destream::en::Error for Error {
    fn custom<T: fmt::Display>(value: T) -> Self {
        Self::Message(value.to_string())
    }
}

impl From<PyErr> for Error {
    fn from(value: PyErr) -> Self {
        Self::Python(value)
    }
}

impl From<pyo3::CastError<'_, '_>> for Error {
    fn from(value: pyo3::CastError<'_, '_>) -> Self {
        Self::Message(value.to_string())
    }
}

impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        match value {
            Error::Message(message) => pyo3::exceptions::PyValueError::new_err(message),
            Error::Python(error) => error,
        }
    }
}

// Independently retained child streams share one finite materialization budget.
#[derive(Clone)]
pub(crate) struct Budget(Arc<Mutex<usize>>);

impl Budget {
    pub(crate) fn new(limit: usize) -> Self {
        Self(Arc::new(Mutex::new(limit)))
    }

    pub(crate) fn charge(&self, bytes: usize) -> Result<(), Error> {
        let mut remaining = self.0.lock().expect("codec budget poisoned");
        *remaining = remaining
            .checked_sub(bytes)
            .ok_or_else(|| Error::Message("Python materialization exceeds byte budget".into()))?;

        Ok(())
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<handle::Codec>()?;
    module.add_class::<ScalarHandle>()?;
    Ok(())
}
