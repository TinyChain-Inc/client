use crate::Method;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::types::PyKernelRequest;

pub(super) fn parse_method(method: &str) -> PyResult<Method> {
    method.parse().map_err(PyValueError::new_err)
}

pub(crate) fn py_bearer_token(request: &PyKernelRequest) -> Option<String> {
    request.headers.iter().find_map(|(key, value)| {
        key.eq_ignore_ascii_case("authorization")
            .then(|| crate::auth::bearer_token(value))
            .flatten()
            .map(str::to_owned)
    })
}
