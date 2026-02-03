# Contributing to the Python client

TinyChain’s Python package combines generated PyO3 bindings with a small layer
of hand-authored stubs. Those stubs keep the public API indexable by Sphinx and
discoverable in IDEs even though the runtime logic lives in Rust. When you add
new `State` variants, handlers, or `/lib` routes, update the stubs alongside the
Rust change so the documentation stays accurate.

## When to touch the stubs

Create or refresh Python stubs whenever you:

- Expose a new `State`/`Collection` variant (e.g., tensors) through PyO3.
- Add a handler or endpoint that should be callable from Python (`/lib/...`,
  `/service/...`, queue helpers, etc.).
- Change the request/response shape of an existing handler or value type.

## Where the stubs live

Add stub modules under `py/tinychain/` (create the package if it does not
exist yet). Mirror the structure of the Rust API (e.g., `tinychain/state.py`,
`tinychain/kernel.py`, `tinychain/handlers.py`) and keep the following in mind:

- Each stub should define the public class/function with the correct signature
  and docstring. The body can be `...`/`raise NotImplementedError` because the
  real implementation is provided by PyO3 at runtime.
- Include any constants/enums that PyO3 exposes so type checkers can import
  them without talking to a running host.
- Reference the canonical URI helpers (`tc.uri.*`) instead of string-building to
  keep docs aligned with the kernel contract.

## Update checklist

1. **Add or edit the stub** file that corresponds to the new API surface.
2. **Document the change** inside `py/README.md` so users know how to
   call the endpoint/variant.
3. **Regenerate or adjust Sphinx docs** if you maintain API references.
4. **Run the Python tests** (`python -m pytest py/tests`) and the PyO3
   integration tests to ensure stubs and bindings agree.

Keeping the stubs current is what lets the Python package advertise every Rust
capability without re-implementing it. If you are unsure where a stub should go
or how to describe a new endpoint, mention it in your PR so reviewers can
double-check the Python surface.

## Rights and licensing

By contributing to this package you represent that (a) you authored the work (or
have the right to contribute it) and (b) you transfer and assign all right,
title, and interest in the contribution to the TinyChain Open-Source Project for
distribution under the TinyChain open-source license (Apache 2.0, see the root
`LICENSE`). Contributions must be free of third-party claims or encumbrances.
