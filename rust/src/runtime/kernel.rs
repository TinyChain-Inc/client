use std::path::PathBuf;
use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use pathlink::Link;
use pyo3::Bound;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyType};

use super::state_handle_conversions::{request_body_bytes, state_from_handle};
use super::types::{
    PyKernelConfig, PyKernelRequest, PyResponse, PyStateHandle, apply_config_overrides,
};
use super::wire::py_bearer_token;
use crate::{HostStorage, Kernel};

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
    let data_dir = config.data_dir.clone();
    let host_id: tc_ir::Id = config
        .host_id
        .parse()
        .map_err(|error| PyValueError::new_err(format!("invalid host identity: {error}")))?;
    let workspace = config.workspace.clone().ok_or_else(|| {
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
        })
        .map_err(super::tc_error)?;

    let resources = crate::HostResources::new(config.limits.clone()).map_err(super::tc_error)?;
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
    super::wait(
        runtime,
        Kernel::new(services, workspace, config.limits.transaction_ttl),
    )
    .map_err(super::tc_error)
}

#[pyclass(name = "KernelHandle", from_py_object)]
pub struct KernelHandle {
    inner: Arc<Kernel>,
    runtime: Arc<tokio::runtime::Runtime>,
    config: PyKernelConfig,
}

impl Clone for KernelHandle {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            runtime: Arc::clone(&self.runtime),
            config: self.config.clone(),
        }
    }
}

impl KernelHandle {
    fn from_kernel(
        kernel: Kernel,
        config: PyKernelConfig,
        runtime: Arc<tokio::runtime::Runtime>,
    ) -> Self {
        Self {
            inner: Arc::new(kernel),
            runtime,
            config,
        }
    }

    fn wait<F>(&self, fut: F) -> F::Output
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
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
        let runtime = super::runtime().expect("tokio runtime");
        let kernel = python_kernel_builder_with_config(config.clone(), None, &runtime)?;
        Ok(Self::from_kernel(kernel, config, runtime))
    }

    /// Construct a local kernel handle with no Python service handlers
    /// installed.
    ///
    /// This is intended for tooling/tests which only need the Rust `/lib` and
    /// `/healthz` handlers (e.g. WASM installs into a local `data_dir`)
    /// without providing Python callbacks.
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
        let runtime = super::runtime().expect("tokio runtime");
        let trusted = token
            .map(py_token)
            .transpose()?
            .map(|(_, host, actor)| (host, actor));
        let kernel = python_kernel_builder_with_config(config.clone(), trusted, &runtime)?;
        Ok(Self::from_kernel(kernel, config, runtime))
    }

    pub fn dispatch(&self, request: PyKernelRequest) -> PyResult<PyResponse> {
        let method = request.method_enum();
        let raw_path = request.path_owned();
        let body_is_none = request.body().is_none();
        let bearer = py_bearer_token(&request);
        let kernel = Arc::clone(&self.inner);
        let guard = self
            .wait(async move {
                kernel
                    .begin_request(method, &raw_path, body_is_none, bearer)
                    .await
            })
            .map_err(super::tc_error)?;
        {
            let guard = *guard;
            let contract = guard.body_contract();
            let request_body = request.body();
            let content_type = request.headers.iter().find_map(|(name, value)| {
                name.eq_ignore_ascii_case("content-type")
                    .then_some(value.as_str())
            });
            if matches!(contract, crate::BodyContract::Application { .. })
                && !matches!(
                    content_type,
                    None | Some("application/json") | Some("application/wasm")
                )
            {
                return Err(PyValueError::new_err(
                    "unsupported application content type",
                ));
            }
            let (body, _memory) = match contract {
                crate::BodyContract::Application { max_bytes }
                    if content_type == Some("application/wasm") =>
                {
                    let bytes = request_body_bytes(request_body)?;
                    if bytes.is_empty() || bytes.len() > max_bytes {
                        return Err(PyValueError::new_err("invalid WASM Library size"));
                    }
                    let permit = self
                        .wait(guard.admit_application_memory(bytes.len()))
                        .map_err(super::tc_error)?;
                    (
                        Some(crate::State::Tuple(vec![
                            crate::State::None,
                            crate::State::from(tc_value::Value::Bytes(bytes.into())),
                        ])),
                        Some(permit),
                    )
                }
                crate::BodyContract::Application { max_bytes } => {
                    if request_body.as_ref().is_some_and(PyStateHandle::is_native) {
                        (
                            state_from_handle(request_body, guard.txn().clone(), &self.runtime)?,
                            None,
                        )
                    } else {
                        let bytes = request_body_bytes(request_body)?;
                        if bytes.len() > max_bytes {
                            return Err(PyValueError::new_err(
                                "application body exceeds its request bound",
                            ));
                        }
                        let permit = self
                            .wait(guard.admit_application_memory(bytes.len()))
                            .map_err(super::tc_error)?;
                        let input = futures::stream::iter([Ok::<_, std::io::Error>(bytes.into())]);
                        let txn = guard.txn().clone();
                        let state = self
                            .wait(async move { destream_json::try_decode(txn, input).await })
                            .map_err(|error| PyValueError::new_err(error.to_string()))?;
                        (Some(state), Some(permit))
                    }
                }
                _ => (
                    state_from_handle(request_body, guard.txn().clone(), &self.runtime)?,
                    None,
                ),
            };
            let result = self.wait(guard.execute_bound(body));
            match result {
                Ok((_, _, guard))
                    if matches!(contract, crate::BodyContract::Application { .. }) =>
                {
                    self.wait(guard.finish_success()).map_err(super::tc_error)?;
                    Ok(PyResponse::new(204, None, None))
                }
                Ok((None, _, _guard)) => Ok(PyResponse::new(204, None, None)),
                Ok((Some(state), _, guard)) => Ok(PyResponse::new(
                    200,
                    None,
                    Some(PyStateHandle::from_terminal_state(
                        state,
                        guard,
                        Arc::clone(&self.runtime),
                    )),
                )),
                Err(error) => Err(super::tc_error(error)),
            }
        }
    }

    pub fn healthz(&self) -> PyResult<()> {
        let kernel = Arc::clone(&self.inner);
        kernel
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
