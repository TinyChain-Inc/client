# Python Service behavior

`tc.Service` shares Library's identity metadata, decorators, lexical compiler,
installation, and backend selection. Declare members on the class using
`publisher`, `resource_name`, and `version`. Public inherited declarations follow
Python override order; changing an instance does not change its compiled definition.

Service specializes declaration discovery and symbolic member binding. Library
owns shared compilation and definition assembly. Compilation discovers Service
declarations once and binds members directly to `$self`; explicit dependencies
retain their application links and authority requirements.

Declare persistent members with `tc.chain.Sync(tc.collection.BTree(schema))` or
`tc.chain.Sync(tc.collection.Table(schema))`. This declares a native `SyncChain`
member; it is not Rust's `Sync` trait or a command to flush storage. Service owns
the application's name and methods, Chain records mutations for durable replay,
and the collection implements its local operations. The host manages transaction
decisions and storage synchronization.

Python's Sync declaration requires an empty, schema-bearing BTree or Table.
Native storage and Services support persistent dense and sparse Tensors, but
Python `Sync(Tensor)` declarations are not yet supported. Bound members use the
existing typed collection wrappers at their Service paths. Execute member
operations with `tc.execute`, or compose them inside decorated methods.

Every application mutation of a persistent mutable collection must pass through
its owning Chain. Returned views grant read access, not mutation authority.
Table selection conveniences send update/truncate to the original member with
explicit bounds, so Chain records them across transactions and restart. Further
selection or transformation is read-only; combine selectors at the owner.

## V1 provenance and boundaries

The reference is v1 revision `17ef342e8f7026e4c4a60d2044de9aeb1b145b91`,
`client/py/tinychain/service.py` and `chain.py`, and its Service, BTree, and Table
integration scenarios.

- Retained: scalar and method members, Sync-backed collections, symbolic self
  references, typed member operations, and install/mutate/restart behavior.
- Adapted: class declarations and explicit application identity replace constructor
  reflection and URI slicing. Native wrappers replace generated collection
  subclasses; installation and transport delegate to existing client owners.
- Unsupported by Python Service authoring: BlockChain, MetricsChain, Tensor
  declarations, and populated collection declarations. Damaged-storage repair is
  also unsupported. The single-host pilot does not establish distributed parity;
  broader replica scenarios remain separate integration work.

## Usage and validation

See the [Service example](examples/service.md) for local and HTTP use. From the
client checkout, with the PyO3 extension and HTTP example built from matching
source and dependency identities. Put the extension and `py/` on `PYTHONPATH`,
verify the imported artifact, and unset `RUST_MIN_STACK` for normal-worker-stack
coverage. Missing signing dependencies, launch failures, and authentication errors
must fail validation rather than skip a transport:

```bash
python -m pytest py/tests/test_service.py
python -m pytest py/tests/test_service_integration.py
python -m pytest py/tests
```

The authoring fixture preserves the canonical BTree Service definition without
requiring a sibling server checkout. Integration tests exercise both collection
kinds and transports, consume or drop responses before shutdown, and reopen
storage in separate processes while preserving host-control records. Execution
commits before response projection; shutdown drains host work before runtime
teardown. Restart phases include expired recovered transactions.

Both transport restart fixtures run on ordinary worker stacks without inherited
stack overrides. They verify selected mutations survive reopening and returned
views reject mutation; they do not establish power-loss recovery.
