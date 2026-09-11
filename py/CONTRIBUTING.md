# Contributing to the Python client

Read the repository-local [Python ownership rules](AGENTS.md). The TinyChain
[workspace contributor guide](https://github.com/TinyChain-Inc/tcv2/blob/main/CONTRIBUTING.md)
is non-normative integration context.

From the client repository root:

```bash
python -m pytest py/tests
```

When changing the local backend, also build and import the `tinychain_local`
extension owned by `client/rust`. When changing a public Python signature,
update its annotation/docstring and the concise user-facing example in
[README.md](README.md). Do not maintain a separate stub architecture unless the
package actually ships generated stubs.

Changes to symbolic forms require round-trip and structural tests. Changes to
Autograph require both accepted-form and fail-closed rejection tests. Changes to
HTTP or PyO3 projection require parity tests demonstrating one native semantic
path and serialization only at the real boundary.
