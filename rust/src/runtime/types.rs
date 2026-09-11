use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use futures::TryStreamExt;
use pyo3::prelude::*;
use pyo3::types::PyString;
use tc_ir::IntoView;

use super::wire::parse_method;
use crate::Method;
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
    Native(Box<NativeState>),
}

impl Clone for StateHandle {
    fn clone(&self) -> Self {
        match self {
            Self::Python(value) => Python::attach(|py| Self::Python(value.clone_ref(py))),
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
    pending: Mutex<Option<crate::KernelRequestGuard>>,
    deadline: crate::Deadline,
}

impl PendingRequest {
    fn ensure_active(&self) -> PyResult<()> {
        if !self.deadline.is_expired() {
            return Ok(());
        }

        let mut pending = self.pending.lock().expect("request lock");
        if pending.is_none() {
            return Ok(());
        }
        let err = self.deadline.exceeded();
        pending.take();
        Err(super::tc_error(err))
    }

    fn discard(&self) {
        self.pending.lock().expect("request lock").take();
    }

    fn take(&self) -> Option<crate::KernelRequestGuard> {
        self.pending.lock().expect("request lock").take()
    }
}

#[pymethods]
impl PyStateHandle {
    #[new]
    pub fn new(value: Py<PyAny>) -> Self {
        Self {
            inner: StateHandle::Python(value),
        }
    }

    pub fn clone_handle(&self) -> Self {
        self.clone()
    }

    pub fn value(&self) -> PyResult<Py<PyAny>> {
        let native = match &self.inner {
            StateHandle::Python(value) => {
                return Python::attach(|py| Ok(value.clone_ref(py)));
            }
            StateHandle::Native(native) => native.as_ref(),
        };
        let NativeState {
            state,
            txn,
            runtime,
            pending,
        } = native;

        if let Some(pending) = pending {
            pending.ensure_active()?;
        }
        let encoded = super::wait(runtime, {
            let state = state.clone();
            let txn = txn.clone();
            async move {
                let view = state.into_view(txn).await.map_err(|err| err.to_string())?;
                let stream = destream_json::encode(view).map_err(|err| err.to_string())?;
                stream
                    .try_fold(Vec::new(), |mut bytes, chunk| async move {
                        bytes.extend_from_slice(&chunk);
                        Ok(bytes)
                    })
                    .await
                    .map_err(|err| err.to_string())
            }
        });
        let bytes = match encoded {
            Ok(bytes) => bytes,
            Err(error) => {
                if let Some(pending) = pending {
                    pending.discard();
                }
                return Err(pyo3::exceptions::PyValueError::new_err(error.to_string()));
            }
        };
        if let Some(pending) = pending {
            if let Some(guard) = pending.take() {
                super::wait(runtime, guard.finish_success()).map_err(super::tc_error)?;
            }
        }
        Python::attach(|py| {
            Ok(PyString::new(py, &String::from_utf8_lossy(&bytes))
                .into_any()
                .unbind())
        })
    }
}

impl PyStateHandle {
    pub(crate) fn from_terminal_state(
        state: crate::State,
        guard: crate::KernelRequestGuard,
        runtime: Arc<tokio::runtime::Runtime>,
    ) -> Self {
        let deadline = guard.deadline();
        let txn = guard.txn().clone();
        Self {
            inner: StateHandle::Native(Box::new(NativeState {
                state,
                txn,
                runtime: Arc::clone(&runtime),
                pending: Some(Arc::new(PendingRequest {
                    pending: Mutex::new(Some(guard)),
                    deadline,
                })),
            })),
        }
    }

    pub(crate) fn is_native(&self) -> bool {
        matches!(&self.inner, StateHandle::Native(_))
    }

    pub(crate) fn take_native_state(
        &self,
        runtime: &tokio::runtime::Runtime,
    ) -> PyResult<Option<crate::State>> {
        let StateHandle::Native(native) = &self.inner else {
            return Ok(None);
        };
        if let Some(pending) = &native.pending {
            pending.ensure_active()?;
            if let Some(guard) = pending.take() {
                super::wait(runtime, guard.finish_success()).map_err(super::tc_error)?;
            }
        }
        Ok(Some(native.state.clone()))
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

impl PyKernelRequest {
    pub(crate) fn method_enum(&self) -> Method {
        self.method
    }

    pub(crate) fn path_owned(&self) -> String {
        self.path.clone()
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
