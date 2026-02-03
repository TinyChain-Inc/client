use pyo3::prelude::*;

#[pymodule]
fn tinychain_local(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    tinychain::pyo3_runtime::register_python_api(module)
}
