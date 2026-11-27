# Contributing to the Python client

Read the repository-local [Python ownership rules](AGENTS.md).

From the client repository root:

```bash
python -m pytest py/tests
```

Build and import a source-matched `tinychain_local` extension owned by
`client/rust` before testing. Its default feature set supports authoring and
wire codecs; enable `--features local-host` for embedded execution tests. See
[installation](README.md#install-and-test). When changing a public Python signature,
update its annotation/docstring and the concise user-facing example in
[README.md](README.md). Do not maintain a separate stub architecture unless the
package actually ships generated stubs.

Changes to symbolic forms require round-trip and structural tests. Changes to
Autograph require both accepted-form and fail-closed rejection tests. Changes to
HTTP or PyO3 projection require parity tests demonstrating one native semantic
path and serialization only at the real boundary.

Test behavior and executable examples, not documentation wording. Do not assert
prose keywords, phrases, headings, or line counts; paraphrasing must not break
tests. Link integrity and runnable examples may be checked mechanically. Review
explanations and claims against the owning contract and behavioral evidence.

For Service declarations and collection selection writes, follow the local
[SERVICE_PARITY.md](SERVICE_PARITY.md) contract and [Service example](examples/service.md).
Wrappers construct owner-addressed requests; the runtime enforces read-only
returned views. Preserve both the original-owner selector and rejection after
further transforms. From the client repository root, run the relevant authoring
and request-shape fixtures:

```bash
python -m pytest py/tests/test_service.py py/tests/test_state_table.py py/tests/test_collection_tensor.py
```

The [Service integration fixture](tests/test_service_integration.py) exercises
local and HTTP mutation/reopening and view rejection. Run it with the matching
artifacts and required dependencies described in SERVICE_PARITY; missing
prerequisites must fail rather than skip a transport.
