use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use get_size::GetSize;
use pyo3::prelude::*;

use crate::Method;
use super::wire::parse_method;

#[derive(Clone)]
pub struct PyKernelConfig {
    pub data_dir: Option<PathBuf>,
    pub workspace: Option<PathBuf>,
    pub host_id: String,
    pub limits: crate::HostLimits,
}

impl Default for PyKernelConfig {
    fn default() -> Self {
        Self {
            data_dir: None,
            workspace: None,
            host_id: "tc-py-host".to_string(),
            limits: crate::HostLimits::default(),
        }
    }
}

pub(super) fn apply_config_overrides(
    mut config: PyKernelConfig,
    request_ttl_secs: Option<u64>,
    max_request_bytes_unauth: Option<usize>,
) -> PyKernelConfig {
    if let Some(secs) = request_ttl_secs.filter(|value| *value > 0) {
        config.limits.transaction_ttl = std::time::Duration::from_secs(secs);
    }
    if let Some(max_bytes) = max_request_bytes_unauth.filter(|value| *value > 0) {
        config.limits.ingress.request_body_bytes = max_bytes;
    }
    config
}

#[pyclass(name = "StateHandle", from_py_object)]
#[derive(Clone)]
pub struct PyStateHandle {
    inner: StateHandle,
}

enum StateHandle {
    Python(Py<PyAny>),
    Authoring(Py<PyAny>),
    Scalar(tc_ir::Scalar),
    Definition(crate::Definition),
    Native(Box<NativeState>),
}

impl Clone for StateHandle {
    fn clone(&self) -> Self {
        match self {
            Self::Python(value) => Python::attach(|py| Self::Python(value.clone_ref(py))),
            Self::Authoring(value) => Python::attach(|py| Self::Authoring(value.clone_ref(py))),
            Self::Scalar(value) => Self::Scalar(value.clone()),
            Self::Definition(value) => Self::Definition(value.clone()),
            Self::Native(native) => Self::Native(native.clone()),
        }
    }
}

#[derive(Clone)]
struct NativeState {
    state: crate::State,
    txn: crate::TxnHandle,
    runtime: Arc<tokio::runtime::Runtime>,
    pending: Option<Arc<PendingRequest>>,
}

struct PendingRequest {
    pending: Mutex<Option<crate::BoundRequest>>,
}

impl PendingRequest {
    fn ensure_active(&self) -> PyResult<()> {
        let mut pending = self.pending.lock().expect("request lock poisoned");
        let result = match pending.as_ref() {
            Some(request) => request.deadline().check(),
            None => Ok(()),
        };

        if result.is_err() {
            pending.take();
        }

        result.map_err(super::tc_error)
    }

    fn discard(&self) {
        self.pending.lock().expect("request lock poisoned").take();
    }
}

#[pymethods]
impl PyStateHandle {
    #[staticmethod]
    pub fn from_authoring(value: Py<PyAny>) -> Self {
        Self {
            inner: StateHandle::Authoring(value),
        }
    }

    #[staticmethod]
    pub(crate) fn from_definition(
        identity: &str,
        body: crate::codec::ScalarHandle,
    ) -> PyResult<Self> {
        let identity = identity.parse().map_err(|error: pathlink::ParseError| {
            pyo3::exceptions::PyValueError::new_err(error.to_string())
        })?;

        Ok(Self {
            inner: StateHandle::Definition(crate::Definition(identity, body.inner)),
        })
    }

    #[new]
    pub fn new(value: Py<PyAny>) -> Self {
        let inner = Python::attach(|py| {
            if let Ok(handle) = value
                .bind(py)
                .extract::<PyRef<'_, crate::codec::ScalarHandle>>()
            {
                StateHandle::Scalar(handle.inner.clone())
            } else {
                StateHandle::Python(value)
            }
        });

        Self { inner }
    }

    pub fn clone_handle(&self) -> Self {
        self.clone()
    }

    pub fn value(&self) -> PyResult<Py<PyAny>> {
        let native = match &self.inner {
            StateHandle::Python(value) | StateHandle::Authoring(value) => {
                return Python::attach(|py| Ok(value.clone_ref(py)));
            }
            StateHandle::Scalar(value) => {
                return crate::codec::project_scalar(value);
            }
            StateHandle::Definition(value) => {
                return crate::codec::project_scalar(&tc_ir::Scalar::from(value.clone()));
            }
            StateHandle::Native(native) => native.as_ref(),
        };

        let NativeState {
            state,
            txn,
            runtime,
            pending,
        } = native;

        if let Err(error) = txn.check_active() {
            if let Some(pending) = pending {
                pending.discard();
            }

            return Err(super::tc_error(error));
        }

        if let Some(pending) = pending {
            pending.ensure_active()?;
        }

        let deadline = if let Some(pending) = pending {
            let retained = pending.pending.lock().expect("request lock poisoned");
            retained
                .as_ref()
                .map(crate::BoundRequest::deadline)
                .unwrap_or_else(|| txn.expiry_deadline())
        } else {
            txn.expiry_deadline()
        };

        let projected = (|| {
            let state = state.clone();
            let context = txn.clone();
            let project = async move {
                let view = context.view(state, deadline).await?;
                crate::codec::into_wire(view)
                    .await
                    .map_err(|err| tc_error::TCError::bad_request(err.to_string()))
            };

            let value = super::wait(runtime, deadline.run(project))?.map_err(super::tc_error)?;
            txn.check_active().map_err(super::tc_error)?;
            if let Some(pending) = pending {
                pending.ensure_active()?;
            }

            Ok(value)
        })();

        if let Some(pending) = pending {
            pending.discard();
        }

        projected
    }
}

impl PyStateHandle {
    pub(crate) fn from_terminal_state(
        state: crate::State,
        request: crate::BoundRequest,
        runtime: Arc<tokio::runtime::Runtime>,
    ) -> Self {
        let txn = request.txn().clone();
        Self {
            inner: StateHandle::Native(Box::new(NativeState {
                state,
                txn,
                runtime,
                pending: Some(Arc::new(PendingRequest {
                    pending: Mutex::new(Some(request)),
                })),
            })),
        }
    }

    pub(crate) fn application_bytes(&self, bound: usize) -> PyResult<usize> {
        let bytes = match &self.inner {
            StateHandle::Scalar(value) => value.get_size(),
            StateHandle::Definition(value) => value.0.get_size() + value.1.get_size(),
            // Guarded native State and deferred foreign composition already
            // have their own owners; conservatively reserve the admitted bound.
            _ => bound,
        };

        if bytes > bound {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "native application body exceeds its bound",
            ));
        }

        Ok(bytes)
    }

    pub(crate) fn take_native_state(
        &self,
        request: &mut crate::BoundRequest,
        runtime: &tokio::runtime::Runtime,
    ) -> PyResult<Option<crate::State>> {
        if let StateHandle::Authoring(value) = &self.inner {
            let value = Python::attach(|py| value.clone_ref(py));
            let deadline = request.deadline();
            let budget = crate::codec::Budget::new(request.native_body_limit());
            let state = super::wait(
                runtime,
                deadline.run(super::authoring::into_state(
                    value,
                    request,
                    &budget,
                    tc_ir::MAX_SCALAR_DEPTH,
                )),
            )?
            .map_err(super::tc_error)?;

            return Ok(Some(state));
        }

        self.retain_state(request)
    }

    pub(super) fn retain_state(
        &self,
        request: &mut crate::BoundRequest,
    ) -> PyResult<Option<crate::State>> {
        if let StateHandle::Scalar(value) = &self.inner {
            if !value.within_depth(tc_ir::MAX_SCALAR_DEPTH) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Scalar nesting exceeds limit",
                ));
            }

            return Ok(Some(crate::State::from(value.clone())));
        }

        if let StateHandle::Definition(value) = &self.inner {
            return Ok(Some(crate::State::from(value.clone())));
        }

        let StateHandle::Native(native) = &self.inner else {
            return Ok(None);
        };
        if let Some(pending) = &native.pending {
            pending.ensure_active()?;
        }

        request
            .retain_source(&native.txn)
            .map_err(super::tc_error)?;

        let state = native.state.clone();
        if let Some(pending) = &native.pending {
            pending.ensure_active()?;
            pending.discard();
        }

        Ok(Some(state))
    }
}

#[pyclass(name = "KernelRequest", from_py_object)]
#[derive(Clone)]
pub struct PyKernelRequest {
    pub(super) method: Method,
    pub(super) path: String,
    pub(super) headers: Vec<(String, String)>,
    pub(super) body: Option<PyStateHandle>,
}

impl fmt::Debug for PyKernelRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelRequest")
            .field("method", &self.method.as_str())
            .field("path", &self.path)
            .field("headers", &self.headers)
            .finish()
    }
}

#[pymethods]
impl PyKernelRequest {
    #[new]
    pub fn new(
        method: &str,
        path: &str,
        headers: Option<Vec<(String, String)>>,
        body: Option<PyStateHandle>,
    ) -> PyResult<Self> {
        Ok(Self {
            method: parse_method(method)?,
            path: path.to_string(),
            headers: headers.unwrap_or_default(),
            body,
        })
    }

    #[getter]
    pub fn method(&self) -> &'static str {
        self.method.as_str()
    }

    #[getter]
    pub fn path(&self) -> &str {
        &self.path
    }

    #[getter]
    pub fn headers(&self) -> Vec<(String, String)> {
        self.headers.clone()
    }

    #[getter]
    pub fn body(&self) -> Option<PyStateHandle> {
        self.body.clone()
    }
}

#[pyclass(name = "Response", from_py_object)]
#[derive(Clone)]
pub struct PyResponse {
    status: u16,
    pub(super) headers: Vec<(String, String)>,
    body: Option<PyStateHandle>,
}

#[pymethods]
impl PyResponse {
    #[new]
    pub fn new(
        status: u16,
        headers: Option<Vec<(String, String)>>,
        body: Option<PyStateHandle>,
    ) -> Self {
        Self {
            status,
            headers: headers.unwrap_or_default(),
            body,
        }
    }

    #[getter]
    pub fn status(&self) -> u16 {
        self.status
    }

    #[getter]
    pub fn headers(&self) -> Vec<(String, String)> {
        self.headers.clone()
    }

    #[getter]
    pub fn body(&self) -> Option<PyStateHandle> {
        self.body.clone()
    }
}
