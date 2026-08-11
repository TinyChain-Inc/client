#![deny(unsafe_code)]
// PyO3 0.21 emits Rust 2024 unsafe operations only inside generated module glue.
#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

#[pymodule]
fn tinychain_local(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    tinychain::pyo3_runtime::register_python_api(module)
}
