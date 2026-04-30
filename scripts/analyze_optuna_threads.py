"""
Analyze a threaded_tunable Optuna study DB after-the-fact.

Usage:
    python scripts/analyze_optuna_threads.py [path/to/optuna_study.db]

Loads the write/read/concurrent studies (study name prefix `threaded_tunable_`)
from the given SQLite DB and prints:
  - Trial count (completed / pruned / failed)
  - Best trial and its parameters
  - fANOVA parameter importance (requires scikit-learn)
  - Top-5 trials per mode

Defaults to results/optuna_study.db if no path given.
"""
import sys
from pathlib import Path

import optuna
from optuna.trial import TrialState


def analyze_study(db_path: str, study_name: str):
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
    )

    trials = study.trials
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in trials if t.state == TrialState.PRUNED]
    failed = [t for t in trials if t.state == TrialState.FAIL]

    mode = study_name.replace("threaded_tunable_", "")
    print(f"\n{'=' * 60}")
    print(f"  {mode.upper()} study")
    print(f"{'=' * 60}")
    print(f"  Trials: {len(complete)} complete, {len(pruned)} pruned, {len(failed)} failed")

    if not complete:
        print("  No completed trials — skipping.")
        return

    best = study.best_trial
    print(f"\n  Best trial #{best.number}: {best.value:.2f} GB/s")
    for key, value in best.params.items():
        print(f"    {key:20s} {value}")

    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"\n  Parameter importance (fANOVA):")
        for i, (param, imp) in enumerate(importance.items(), 1):
            print(f"    {i}. {param:20s} {imp:.1%}")
    except Exception as e:
        print(f"\n  fANOVA unavailable ({type(e).__name__}): {e}")

    top = sorted(complete, key=lambda t: t.value, reverse=True)[:5]
    print(f"\n  Top-5 trials:")
    for t in top:
        params_str = ", ".join(f"{k}={v}" for k, v in t.params.items())
        print(f"    #{t.number:3d}  {t.value:6.2f} GB/s  {params_str}")


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "results/optuna_study.db"
    if not Path(db_path).exists():
        print(f"ERROR: DB not found: {db_path}")
        sys.exit(1)

    print(f"Analyzing: {db_path}")

    for mode in ("write", "read", "concurrent"):
        try:
            analyze_study(db_path, f"threaded_tunable_{mode}")
        except KeyError:
            print(f"\n  [skip] Study 'threaded_tunable_{mode}' not found in DB")
        except Exception as e:
            print(f"\n  [error] {mode}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
