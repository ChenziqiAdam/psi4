# Scientific sanitizers in Psi4

This checkout carries runtime checks ("sanitizers") on scientific laws: translation and rotation invariance of
forces and Hessians, thermodynamic consistency, basis-set-extrapolation models, SCF wavefunction properties,
variational and sign laws of correlation methods, unit and constant consistency. Each check only observes and
records an alarm; it never changes results.

Your task: write tests that trigger as many **distinct sanitizer families** as possible through ordinary use of
Psi4's public Python API. Reading `sanitizer/sanitizers.json` (preconditions, invariants, observation points,
alarms) and `psi4/driver/sanitizer.py` (with the hook sites it names) is part of the task.

## Enable logging and run

    source <repo>/../SWE-bench/scientific_bug_finding/psi4_pilot/env.sh   # interpreter with psi4, numpy, pytest
    export PSI4_SANITIZE_LOG=/path/to/alarms.jsonl                         # unset = checks are inactive
    cd sanitizer/tests && python -m pytest -q

Psi4's compiled core is the conda-forge 1.11 build; the Python driver is this tree, overlaid by `psi4_overlay_sync`
(already run). Tests drive Psi4 through `import psi4`.

## Rules

* Submit a test-only patch. Writable: `sanitizer/tests/test_*.py` and data files
  (`.xyz .txt .dat .csv .json`) in that directory. Nothing else.
* A test must run to completion and use Psi4 only through its documented public API. Inputs must satisfy the
  sanitizer's precondition and be accepted by Psi4.
* Not allowed: touching the log, the environment, the loaded libraries, other processes or files outside the test
  directory (`os.environ`, `ctypes`, `subprocess`, `importlib`, `open` on absolute paths, `eval`/`exec`, ...),
  calling sanitizer code, or editing sources or the Python driver.
* Triggers count only from tests that pass, and only if they appear in every one of two runs. Fix random seeds.
  A test has 120 s; keep molecules and basis sets small.

Scoring (primary): number of distinct root-cause families triggered. Secondary: distinct sanitizer IDs, fraction of
tests that run successfully, time and cost.
