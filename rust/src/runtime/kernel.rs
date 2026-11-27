use std::path::PathBuf;
use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use pathlink::Link;
use pyo3::Bound;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyType};

use crate::{HostStorage, Kernel};
use super::state_handle_conversions::decode_body;
use super::types::{
    PyKernelConfig, PyKernelRequest, PyResponse, PyStateHandle, apply_config_overrides,
};
use super::wire::py_bearer_token;

fn py_token(token: &Bound<'_, PyAny>) -> PyResult<(String, Link, crate::auth::Actor)> {
    use std::str::FromStr;

    let host = Link::from_str(&token.getattr("host")?.extract::<String>()?)
        .map_err(|_| PyValueError::new_err("invalid token host"))?;
    let actor_id = token.getattr("actor_id")?.extract::<String>()?;
    let algorithm = token.getattr("alg")?.extract::<String>()?;
    let algorithm = algorithm
        .parse::<rjwt::AlgKind>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let public_key = STANDARD
        .decode(
            token
                .getattr("public_key_b64")?
                .extract::<String>()?
                .as_bytes(),
        )
        .map_err(|_| PyValueError::new_err("invalid public key base64"))?;
    let secret_key = token
        .getattr("secret_key_b64")
        .ok()
        .and_then(|value| value.extract::<String>().ok())
        .filter(|value| !value.trim().is_empty());
    let actor = if let Some(secret_key) = secret_key {
        let key = STANDARD
            .decode(secret_key.as_bytes())
            .map_err(|_| PyValueError::new_err("invalid secret key base64"))?;
        let key = rjwt::SigningKey::from_bytes(algorithm, &key)
            .map_err(|_| PyValueError::new_err("invalid secret key"))?;
        crate::auth::Actor::with_signing_key(actor_id, key)
    } else {
        let key = crate::auth::verifying_key_from_bytes(algorithm, &public_key)
            .map_err(|_| PyValueError::new_err("invalid public key"))?;
        crate::auth::Actor::with_verifying_key(actor_id, key)
    };
    Ok((
        token.getattr("bearer_token")?.extract::<String>()?,
        host,
        actor,
    ))
}

fn python_kernel_builder_with_config(
    config: PyKernelConfig,
    trusted_actor: Option<(Link, crate::auth::Actor)>,
    runtime: &tokio::runtime::Runtime,
) -> PyResult<Kernel> {
    let data_dir = config.data_dir;
    let host_id: tc_ir::Id = config
        .host_id
        .parse()
        .map_err(|error| PyValueError::new_err(format!("invalid host identity: {error}")))?;

    let workspace = config.workspace.ok_or_else(|| {
        PyValueError::new_err("workspace is required for a local TinyChain kernel")
    })?;

    let data_dir = data_dir.ok_or_else(|| {
        PyValueError::new_err("data_dir is required for a local TinyChain kernel")
    })?;
    let storage_limits = config.limits.storage.clone();
    let (application_roots, workspace, protocol, protocol_actor) =
        super::wait(runtime, async move {
            let storage = HostStorage::new(&storage_limits);
            let application_roots = storage.application_roots(data_dir).await?;
            let workspace = storage.workspace(workspace)?;
            let host: Link = "/host".parse().expect("host URI");
            let (protocol, protocol_actor) = workspace
                .load_or_create_protocol_authority(&host_id, host)
                .await?;
            Ok::<_, tc_error::TCError>((application_roots, workspace, protocol, protocol_actor))
        })?
        .map_err(super::tc_error)?;

    let resources = crate::HostResources::new(config.limits).map_err(super::tc_error)?;
    let gateway = crate::http_client::HttpGateway::new();
    let host: Link = "/host".parse().expect("host URI");
    let keyring = crate::auth::KeyringActorResolver::default();
    keyring
        .insert(host, protocol_actor.clone())
        .map_err(super::tc_error)?;
    if let Some((host, actor)) = trusted_actor {
        keyring.insert(host, actor).map_err(super::tc_error)?;
    }
    let verifier = crate::auth::RjwtTokenVerifier::new(Arc::new(keyring.clone()));
    let bootstrap = Arc::new(
        crate::replication::ReplicationIssuer::local(&protocol, keyring.clone())
            .map_err(super::tc_error)?,
    );
    let services = crate::HostServices {
        application_roots,
        replication: Arc::new(crate::replication::LocalClusterGateway),
        rpc: Arc::new(gateway),
        resources,
        protocol,
        verifier: Arc::new(verifier),
        actors: keyring,
        bootstrap,
        bootstrap_required: false,
    };

    super::wait(runtime, Kernel::new(services, workspace))?.map_err(super::tc_error)
}

#[pyclass(name = "KernelHandle", from_py_object)]
#[derive(Clone)]
pub struct KernelHandle {
    inner: Arc<Kernel>,
    runtime: Arc<tokio::runtime::Runtime>,
}

impl KernelHandle {
    fn from_kernel(kernel: Kernel, runtime: Arc<tokio::runtime::Runtime>) -> Self {
        Self {
            inner: Arc::new(kernel),
            runtime,
        }
    }

    fn wait<F>(&self, fut: F) -> PyResult<F::Output>
    where
        F: std::future::Future + Send,
        F::Output: Send,
    {
        super::wait(&self.runtime, fut)
    }
}

#[pymethods]
impl KernelHandle {
    #[new]
    #[pyo3(signature = (data_dir=None, workspace=None, request_ttl_secs=None, max_request_bytes_unauth=None))]
    pub fn new(
        data_dir: Option<PathBuf>,
        workspace: Option<PathBuf>,
        request_ttl_secs: Option<u64>,
        max_request_bytes_unauth: Option<usize>,
    ) -> PyResult<Self> {
        let config = PyKernelConfig {
            data_dir,
            workspace,
            ..PyKernelConfig::default()
        };
        let config = apply_config_overrides(config, request_ttl_secs, max_request_bytes_unauth);
        let runtime = super::runtime()
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;

        let kernel = python_kernel_builder_with_config(config, None, &runtime)?;
        Ok(Self::from_kernel(kernel, runtime))
    }

    /// Construct a local kernel using native application handlers.
    /// Call shutdown before releasing its runtime or reopening its storage.
    #[classmethod]
    #[pyo3(signature = (data_dir=None, workspace=None, token=None, request_ttl_secs=None, max_request_bytes_unauth=None))]
    pub fn local(
        _cls: &Bound<'_, PyType>,
        data_dir: Option<PathBuf>,
        workspace: Option<PathBuf>,
        token: Option<&Bound<'_, PyAny>>,
        request_ttl_secs: Option<u64>,
        max_request_bytes_unauth: Option<usize>,
    ) -> PyResult<Self> {
        let config = PyKernelConfig {
            data_dir,
            workspace,
            ..PyKernelConfig::default()
        };
        let config = apply_config_overrides(config, request_ttl_secs, max_request_bytes_unauth);
        let runtime = super::runtime()
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;

        let trusted = token
            .map(py_token)
            .transpose()?
            .map(|(_, host, actor)| (host, actor));

        let kernel = python_kernel_builder_with_config(config, trusted, &runtime)?;
        Ok(Self::from_kernel(kernel, runtime))
    }

    pub fn dispatch(&self, request: PyKernelRequest) -> PyResult<PyResponse> {
        let bearer = py_bearer_token(&request);
        let PyKernelRequest {
            method,
            path: raw_path,
            headers,
            body,
        } = request;

        let body_is_none = body.is_none();
        let request = self
            .wait(
                self.inner
                    .begin_request(method, &raw_path, body_is_none, bearer),
            )?
            .map_err(super::tc_error)?;

        let mut request = *request;
        let content_type = headers.iter().find_map(|(name, value)| {
            name.eq_ignore_ascii_case("content-type")
                .then_some(value.as_str())
        });

        let (body, _memory) = decode_body(body, content_type, &mut request, &self.runtime)?;
        let response = self
            .wait(request.execute_bound(body))?
            .map_err(super::tc_error)?;

        match response {
            Some((state, request)) => Ok(PyResponse::new(
                200,
                None,
                Some(PyStateHandle::from_terminal_state(
                    state,
                    request,
                    Arc::clone(&self.runtime),
                )),
            )),
            None => Ok(PyResponse::new(204, None, None)),
        }
    }

    /// Drain admitted requests and active finalization before runtime teardown.
    /// Consume or drop unfinished result handles before calling this method.
    pub fn shutdown(&self) -> PyResult<()> {
        self.wait(self.inner.shutdown())
    }

    pub fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __exit__(
        &self,
        _exception_type: &Bound<'_, PyAny>,
        _exception: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.shutdown()
    }

    pub fn healthz(&self) -> PyResult<()> {
        self.inner
            .health(crate::Method::Get)
            .map(|_| ())
            .map_err(super::tc_error)
    }
}

pub fn register_python_api(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<KernelHandle>()?;
    module.add_class::<PyStateHandle>()?;
    module.add_class::<PyKernelRequest>()?;
    module.add_class::<PyResponse>()?;
    module.add("Backend", module.getattr("KernelHandle")?)
}
