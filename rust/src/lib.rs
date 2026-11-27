#![deny(unsafe_code)]
// PyO3 emits unsafe operations only inside generated module glue.
#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
#[cfg(feature = "local-host")]
pub(crate) use tinychain::*;

mod codec;
#[cfg(feature = "local-host")]
mod runtime;

#[pymodule]
fn tinychain_local(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    if !cfg!(panic = "abort") {
        return Err(pyo3::exceptions::PyImportError::new_err(
            "tinychain_local requires panic=abort; rebuild with the application profile",
        ));
    }

    codec::register(module)?;
    #[cfg(feature = "local-host")]
    runtime::register_python_api(module)?;

    Ok(())
}
