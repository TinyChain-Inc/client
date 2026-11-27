;; Regenerate the binary fixture with:
;; wasm-tools parse native_wasm.wat -o native_wasm.wasm
(module
    (memory (export "memory") 1)
    (data (i32.const 0) "{\"definition\":{\"/lib/example-devco/native-wasm/1.0.0\":{\"answer\":42}},\"routes\":[{\"path\":\"/answer\",\"export\":\"answer\"}]}")
    (data (i32.const 2048) "42")
    (func (export "alloc") (param i32) (result i32) i32.const 4096)
    (func (export "free") (param i32 i32))
    (func (export "tc_library_entry") (result i64) i64.const 502511173632)
    (func (export "answer") (param i32 i32 i32 i32) (result i64)
        i64.const 8589936640))
