"""Scripts for reproducing all results shown in the report.

The project consists of 3 main "experiments". that are all enabled by:
(i) loading, splitting and preprocessing data
(ii) declaring and hyperparameter-tune models by CV
(iii) training and evaluating models
(iv) plotting results
For three "experiments":
(i)
(ii)
(iii)
The following scipts are called in this script:
- analysis: analysing the dataset with PCA
- training: training single-target, multi-target and baseline models
- plotting: plotting the results as shown in the report
"""

import argparse
import shlex
import subprocess


def run_script(script_name, targets, include_structural, max_evals):
    """
    Helper function to run the Python scripts via subprocess, ensuring safety by escaping inputs.
    """
    # Sanitize each target to prevent shell injection, even though shell=False by default
    targets = [shlex.quote(target) for target in targets]
    target_string = " ".join(targets)

    # Safely prepare the command array
    cmd = [
        "python",
        script_name,
        "--target",
        target_string,
        "--include_structural",
        str(include_structural),
        "--max_evals",
        str(max_evals),
    ]

    # pylint: disable=subprocess-run-check
    subprocess.run(cmd, check=True, shell=False)  # noqa: S603


def run_one_target_experiments(max_evals):
    """
    Runs the experiments for single target predictions.
    """
    targets = ["metal", "X3_ligand", "E_ligand"]
    # Run with structural features False for all, but True for X3_ligand
    for target in targets:
        if target == "X3_ligand":
            include_structural = True
            run_script(
                "./scripts/training/one_target.py",
                [target],
                include_structural,
                max_evals,
            )
        include_structural = False
        run_script(
            "./scripts/training/one_target.py",
            [target],
            include_structural,
            max_evals,
        )


def run_multi_target_experiments(max_evals):
    """
    Runs the experiments for multiple target predictions.
    """
    target_combinations = [
        ("metal", "E_ligand"),
        ("metal", "X3_ligand"),
        ("X3_ligand", "E_ligand"),
        ("metal", "E_ligand", "X3_ligand"),
    ]
    # Run with and without structural features for the combination of all three targets
    for targets in target_combinations:
        if len(targets) > 2:
            include_structural = True
            run_script(
                "./scripts/training/multi_targets.py",
                targets,
                include_structural,
                max_evals,
            )
        include_structural = False
        run_script(
            "./scripts/training/multi_targets.py",
            targets,
            include_structural,
            max_evals,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run reproducibility script for all experiments."
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=50,
        help="Max evaluations for hyperparameter tuning.",
    )
    args = parser.parse_args()

    # run baselines
    run_one_target_experiments(args.max_evals)
    run_multi_target_experiments(args.max_evals)
    # plot the data
    # plot exp 1
    # plot exp 2
    # plot exp 3


if __name__ == "__main__":
    main()
