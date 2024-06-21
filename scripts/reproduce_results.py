"""Script for reproducing all results shown in the report."""

import argparse
import shlex
import subprocess
from typing import List


def run_command(cmd: List[str]) -> None:
    """
    Helper function to run a command via subprocess.
    """
    print(
        "---------------------------------------------------------------------"
    )
    print(f"Running command: {' '.join(cmd)}")
    print(
        "---------------------------------------------------------------------"
    )

    subprocess.run(cmd, check=True, shell=False)  # noqa: S603


def run_script(
    script_name,
    targets=None,
    include_structural=None,
    max_evals=None,
) -> None:
    """
    Helper function to run the Python scripts via subprocess.
    """
    cmd = ["python", script_name]

    if targets:
        targets = [shlex.quote(target) for target in targets]
        target_string = " ".join(targets)
        cmd.extend(["--target", target_string])

    if include_structural is not None:
        cmd.extend(["--include_structural", str(include_structural)])

    if max_evals is not None:
        cmd.extend(["--max_evals", str(max_evals)])

    run_command(cmd)


def run_one_target_experiments(max_evals: int) -> None:
    """
    Runs the experiments for single target predictions.
    """
    targets = ["metal", "X3_ligand", "E_ligand"]
    for target in targets:
        run_script(
            "./scripts/training/one_target.py", [target], True, max_evals
        )
        run_script(
            "./scripts/training/one_target.py", [target], False, max_evals
        )


def run_multi_target_experiments(max_evals: int) -> None:
    """
    Runs the experiments for multiple target predictions.
    """
    target_combinations = [
        ("metal", "E_ligand"),
        ("metal", "X3_ligand"),
        ("X3_ligand", "E_ligand"),
        ("metal", "E_ligand", "X3_ligand"),
    ]
    for targets in target_combinations:
        run_script(
            "./scripts/training/multi_targets.py", targets, False, max_evals
        )


def run_baselines() -> None:
    """
    Runs the baseline experiments.
    """
    run_command(["python", "scripts/training/baselines.py"])


def run_visualize_results(script_name: str, max_evals: int) -> None:
    """
    Runs the visualization script.
    """
    run_script(script_name, max_evals=max_evals)


def run_dataframe_statistics() -> None:
    """
    Runs the dataframe statistics script.
    """
    run_command(["python", "scripts/analysis/dataset_statistics.py"])
    run_command(["python", "scripts/analysis/pca_ligand_space.py"])


def run_accuracy_table() -> None:
    """
    Runs the accuracy table script.
    """
    run_command(["python", "scripts/analysis/accuracy_table.py"])


def main():
    parser = argparse.ArgumentParser(
        description="Run reproducibility script for all experiments."
    )
    parser.add_argument(
        "--max_evals",
        "-me",
        type=int,
        default=1,
        help="Max evaluations for hyperparameter tuning.",
    )
    args = parser.parse_args()

    run_baselines()
    run_dataframe_statistics()
    run_one_target_experiments(args.max_evals)
    run_multi_target_experiments(args.max_evals)
    run_visualize_results(
        "scripts/analysis/visualize_results.py", max_evals=args.max_evals
    )
    run_accuracy_table()


if __name__ == "__main__":
    main()
