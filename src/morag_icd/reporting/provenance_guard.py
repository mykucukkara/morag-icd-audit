"""
Provenance guard: keep smoke-fixture numbers out of manuscript artifacts.

The repo carries a smoke fixture in which every experiment reports micro_f1 =
0.42105263157894735 (8/19) with success_count = 6. Those rows exist so the reporting
pipeline can be exercised without a real run, but they must NEVER reach a paper table — a
reviewer seeing them would (correctly) discard the whole manuscript. This module is the
shared guard that manuscript-table generators call before emitting a row (currently
scripts 34 and 25), so the check lives in one place instead of being re-implemented per
script. Any NEW paper-table emitter must call check_row() too — it is not automatic.

Two independent signals, either of which condemns a row:
  1. The sentinel micro_f1 constant (the fixture's tell).
  2. A sample count below `min_samples` — a real Top-N test run has thousands of notes;
     a fixture/canary has 5-100. This catches contamination the sentinel misses (e.g. a
     100-note canary with a plausible-looking F1).
"""
from __future__ import annotations

import math
from typing import Optional

# The fixture's micro_f1 == 8/19 to full float precision. Compared with a tight tolerance so
# a genuine run that happens to land near 0.421 is not falsely rejected.
FIXTURE_MICRO_F1 = 8.0 / 19.0
FIXTURE_TOL = 1e-9


class FixtureContaminationError(RuntimeError):
    """Raised when a smoke-fixture / under-powered row is about to enter a manuscript table."""


def is_fixture_value(micro_f1: Optional[float]) -> bool:
    return micro_f1 is not None and math.isfinite(micro_f1) and abs(micro_f1 - FIXTURE_MICRO_F1) < FIXTURE_TOL


def check_row(
    experiment: str,
    micro_f1: Optional[float],
    n_samples: Optional[int],
    min_samples: int,
    strict: bool = True,
) -> Optional[str]:
    """Return a human-readable reason the row is unfit for a manuscript table, or None.

    With strict=True a reason is raised as FixtureContaminationError; with strict=False it is
    returned so the caller can skip/annotate the row instead of aborting.
    """
    reason = None
    if is_fixture_value(micro_f1):
        reason = (f"{experiment}: micro_f1={micro_f1!r} is the smoke-fixture sentinel "
                  f"({FIXTURE_MICRO_F1:.6f}) — this is fixture data, not a real run.")
    elif n_samples is not None and n_samples < min_samples:
        reason = (f"{experiment}: n_samples={n_samples} < required {min_samples} — "
                  f"under-powered (pilot/canary/fixture), not a full-scale result.")
    if reason and strict:
        raise FixtureContaminationError(reason)
    return reason
