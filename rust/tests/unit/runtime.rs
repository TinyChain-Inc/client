use super::{runtime, wait};

#[test]
fn synchronous_wait_rejects_tokio_reentry() {
    let runtime = runtime().unwrap();
    runtime.block_on(async {
        assert!(wait(&runtime, async {}).is_err());
    });
}

#[test]
fn synchronous_wait_returns_and_propagates_unwinding() {
    pyo3::Python::initialize();
    let runtime = runtime().unwrap();
    assert_eq!(wait(&runtime, async { 42 }).unwrap(), 42);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = wait(&runtime, async { panic!("request panic") });
    }));

    assert!(result.is_err());
    assert_eq!(wait(&runtime, async { 43 }).unwrap(), 43);
}
