"""Documentation-content tests for the training-step compiler's public surface.

These tests do not exercise behavior -- `training_step.py`'s implementation is
frozen by an earlier subtask. They pin what the module documentation and the
record documentation must state (NFR-129-005) and what neither may ever claim
unconditionally (Definition of Done item 12): no unconditional purity,
side-effect-freedom, or determinism claim, with the record described only as
the framework-owned envelope of §9.3 and Inv-13's determinism stated only in
its conditional form.

Each of the eight required-topic tests below looks for a literal anchor
heading in the module docstring. The headings are a documentation contract in
their own right -- readable section markers a maintainer can grep for -- not
merely a testing convenience.
"""

from __future__ import annotations

import re
from pathlib import Path

from tinychain.autodiff import training_step


def _module_doc() -> str:
    return training_step.__doc__ or ""


def _all_documented_texts() -> list[str]:
    texts = [_module_doc()]
    for name in ("ParameterCompilation", "TrainingStepProvenance", "CompiledTrainingStep"):
        texts.append(getattr(training_step, name).__doc__ or "")
    texts.append(training_step.compile_training_step.__doc__ or "")
    return texts


def test_module_documentation_states_the_compile_sequence() -> None:
    assert "Compile sequence (FR-129-001)" in _module_doc()


def test_module_documentation_states_why_expansion_follows_differentiation() -> None:
    assert "Why expansion follows differentiation (§7.4)" in _module_doc()


def test_module_documentation_states_the_four_artifacts() -> None:
    assert "The four artifacts (§6)" in _module_doc()


def test_module_documentation_states_the_capture_selection_rule() -> None:
    assert "Capture selection rule (§8.5)" in _module_doc()


def test_module_documentation_states_the_seed_contract() -> None:
    assert "Seed contract (§8.3)" in _module_doc()


def test_module_documentation_states_the_arity_rule() -> None:
    assert "Arity rule (§9.4)" in _module_doc()


def test_module_documentation_states_the_collaborator_failure_table() -> None:
    assert "Collaborator-failure table (§13.2)" in _module_doc()


def test_module_documentation_states_where_dependency_provenance_lives() -> None:
    assert "Dependency provenance (FR-129-013)" in _module_doc()


# Patterns that assert purity, side-effect-freedom, determinism, or the
# record's portability/comparability/serializability as an *unconditional*
# property -- the exact defect DoD item 12 forbids. The negative lookbehinds
# let the required negated forms of §9.3 ("not portable between backends",
# "not comparable across consumers", "not serializable") stand: those are
# what the record documentation must say, not what it must avoid.
_FORBIDDEN_UNCONDITIONAL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?<!not )(?<!never )\bis pure\b",
        r"\bpurely functional\b",
        r"(?<!not )(?<!never )\bhas no side effects\b",
        r"\bside-effect-free\b",
        r"\bside effect free\b",
        r"\balways produces the same\b",
        r"\balways deterministic\b",
        r"\bguaranteed deterministic\b",
        r"\bfully deterministic\b",
        r"(?<!not )(?<!never )\bis deterministic\b",
        r"\bbackend-neutral data\b",
        r"(?<!not )(?<!never )\bportable between backends\b",
        r"(?<!not )(?<!never )\bcomparable across consumers\b",
        r"(?<!not )(?<!never )\bis serializable\b",
    )
)


def test_no_unconditional_purity_or_determinism_claim_appears() -> None:
    for doc in _all_documented_texts():
        for pattern in _FORBIDDEN_UNCONDITIONAL_PATTERNS:
            match = pattern.search(doc)
            assert match is None, match.group(0) if match else pattern.pattern


def test_determinism_is_stated_only_in_its_conditional_form() -> None:
    doc = _module_doc().lower()
    assert "conditional determinism" in doc or (
        "given" in doc and "deterministic" in doc
    )


def test_record_is_described_as_a_framework_owned_envelope() -> None:
    doc = training_step.CompiledTrainingStep.__doc__ or ""
    assert "envelope" in doc.lower()


def test_readme_documents_training_step_end_to_end_example() -> None:
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    text = readme_path.read_text(encoding="utf-8")

    assert "compile_training_step" in text
    assert "tests.autodiff_reference_consumer" in text
    assert "test-tree" in text.lower()


def _readme_training_step_example_source() -> str:
    """The one fenced Python block under the training-step README heading.

    A reader copies exactly this block, so the block itself -- not a
    paraphrase of it -- is what must be extracted and executed.
    """
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    text = readme_path.read_text(encoding="utf-8")

    heading = "### Compiling a training step end to end"
    after_heading = text[text.index(heading) + len(heading) :]
    fence_start = after_heading.index("```python") + len("```python")
    fence_end = after_heading.index("```", fence_start)
    return after_heading[fence_start:fence_end]


def test_readme_training_step_example_runs_verbatim() -> None:
    """The copy-pasteable block must actually execute, not merely parse.

    Extracted and `exec`'d exactly as a reader would run it (as `__main__`,
    from `py/`, so its `from tests.autodiff_reference_consumer import ...`
    resolves the same way the test suite's own imports do) -- eyeballing the
    source is not proof it runs.
    """
    source = _readme_training_step_example_source()

    namespace: dict[str, object] = {"__name__": "__main__"}
    exec(compile(source, "py/README.md (training-step example)", "exec"), namespace)

    step = namespace["step"]
    assert step.forward.selected_outputs
    assert step.derivative.selected_outputs
    assert step.parameter("w").update.selected_outputs
