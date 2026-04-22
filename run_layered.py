from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run layered-DAG experiments and save figures and tables."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Base directory where a new run folder will be created.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom name for the run folder. If omitted, one is generated automatically.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[20, 40],
        help="List of layered-DAG dimensions.",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[3, 4],
        help="List of layered-DAG depths.",
    )
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=[0.2, 0.4],
        help="List of edge inclusion probabilities between adjacent layers.",
    )
    parser.add_argument(
        "--correlations",
        type=float,
        nargs="+",
        default=[0.3, 0.6],
        help="List of edge-weight scales used as a correlation proxy.",
    )
    parser.add_argument(
        "--observation-strategies",
        nargs="+",
        default=["uniform", "shallow", "deep"],
        help="Observation selection strategies.",
    )
    parser.add_argument(
        "--observed-fraction",
        type=float,
        default=0.25,
        help="Fraction of observed nodes.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per configuration.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Number of retained MCMC samples per run.",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=500,
        help="Burn-in iterations for each sampler.",
    )
    parser.add_argument(
        "--thinning",
        type=int,
        default=2,
        help="Thinning interval for each sampler.",
    )
    parser.add_argument(
        "--rw-mh-proposal-scale",
        type=float,
        default=0.6,
        help="Proposal scale for random-walk Metropolis-Hastings.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for the full experiment.",
    )
    return parser.parse_args()


def _compact_list(values: list[int] | list[float] | list[str]) -> str:
    if len(values) == 1:
        return str(values[0]).replace(".", "p")
    return f"{str(values[0]).replace('.', 'p')}-{str(values[-1]).replace('.', 'p')}"


def build_run_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        return f"{timestamp}_{args.run_name}"
    return (
        f"{timestamp}_"
        f"dim{_compact_list(args.dimensions)}_"
        f"dep{_compact_list(args.depths)}_"
        f"sp{_compact_list(args.sparsities)}_"
        f"corr{_compact_list(args.correlations)}_"
        f"obs{'-'.join(args.observation_strategies)}_"
        f"rep{args.repetitions}"
    )


def main() -> None:
    args = parse_args()

    run_dir = args.output_dir.resolve() / build_run_name(args)
    run_dir.mkdir(parents=True, exist_ok=False)
    base_dir = run_dir
    figures_dir = base_dir / "figures"
    tables_dir = base_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    mpl_config_dir = figures_dir / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    run_config = {
        "run_directory": str(run_dir),
        "dimensions": args.dimensions,
        "depths": args.depths,
        "sparsities": args.sparsities,
        "correlations": args.correlations,
        "observation_strategies": args.observation_strategies,
        "observed_fraction": args.observed_fraction,
        "repetitions": args.repetitions,
        "n_samples": args.n_samples,
        "burn_in": args.burn_in,
        "thinning": args.thinning,
        "rw_mh_proposal_scale": args.rw_mh_proposal_scale,
        "random_state": args.random_state,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(base_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    with open(base_dir / "run_command.txt", "w", encoding="utf-8") as f:
        f.write("python3 run_layered.py \\\n")
        f.write(f"  --dimensions {' '.join(map(str, args.dimensions))} \\\n")
        f.write(f"  --depths {' '.join(map(str, args.depths))} \\\n")
        f.write(f"  --sparsities {' '.join(map(str, args.sparsities))} \\\n")
        f.write(f"  --correlations {' '.join(map(str, args.correlations))} \\\n")
        f.write(f"  --observation-strategies {' '.join(args.observation_strategies)} \\\n")
        f.write(f"  --observed-fraction {args.observed_fraction} \\\n")
        f.write(f"  --repetitions {args.repetitions} \\\n")
        f.write(f"  --n-samples {args.n_samples} \\\n")
        f.write(f"  --burn-in {args.burn_in} \\\n")
        f.write(f"  --thinning {args.thinning} \\\n")
        f.write(f"  --rw-mh-proposal-scale {args.rw_mh_proposal_scale} \\\n")
        f.write(f"  --random-state {args.random_state}\n")

    from stats608_pipeline import (
        plot_error,
        plot_ess,
        plot_runtime,
        run_layered_experiment_grid,
        summarize_experiment_results,
    )

    def save_metric_family(results_df, x_axis: str) -> None:
        ess_summary = summarize_experiment_results(
            results_df,
            metric="ess_mean",
            x=x_axis,
            facet="observation_strategy",
        )
        runtime_summary = summarize_experiment_results(
            results_df,
            metric="runtime_seconds",
            x=x_axis,
            facet="observation_strategy",
        )
        mean_rmse_summary = summarize_experiment_results(
            results_df,
            metric="mean_rmse",
            x=x_axis,
            facet="observation_strategy",
        )
        variance_rmse_summary = summarize_experiment_results(
            results_df,
            metric="variance_rmse",
            x=x_axis,
            facet="observation_strategy",
        )
        covariance_error_summary = summarize_experiment_results(
            results_df,
            metric="covariance_frobenius_error",
            x=x_axis,
            facet="observation_strategy",
        )

        ess_summary.to_csv(tables_dir / f"ess_summary_by_{x_axis}.csv", index=False)
        runtime_summary.to_csv(tables_dir / f"runtime_summary_by_{x_axis}.csv", index=False)
        mean_rmse_summary.to_csv(tables_dir / f"mean_rmse_summary_by_{x_axis}.csv", index=False)
        variance_rmse_summary.to_csv(tables_dir / f"variance_rmse_summary_by_{x_axis}.csv", index=False)
        covariance_error_summary.to_csv(tables_dir / f"covariance_error_summary_by_{x_axis}.csv", index=False)

        ess_fig, _ = plot_ess(
            results_df,
            x=x_axis,
            facet="observation_strategy",
            title=f"ESS vs {x_axis.capitalize()}",
        )
        ess_fig.savefig(figures_dir / f"ess_vs_{x_axis}.png", dpi=200, bbox_inches="tight")

        runtime_fig, _ = plot_runtime(
            results_df,
            x=x_axis,
            facet="observation_strategy",
            title=f"Runtime vs {x_axis.capitalize()}",
        )
        runtime_fig.savefig(figures_dir / f"runtime_vs_{x_axis}.png", dpi=200, bbox_inches="tight")

        error_fig, _ = plot_error(
            results_df,
            metric="mean_rmse",
            x=x_axis,
            facet="observation_strategy",
            title=f"Posterior Mean RMSE vs {x_axis.capitalize()}",
        )
        error_fig.savefig(figures_dir / f"mean_rmse_vs_{x_axis}.png", dpi=200, bbox_inches="tight")

        variance_fig, _ = plot_error(
            results_df,
            metric="variance_rmse",
            x=x_axis,
            facet="observation_strategy",
            title=f"Posterior Variance RMSE vs {x_axis.capitalize()}",
        )
        variance_fig.savefig(figures_dir / f"variance_rmse_vs_{x_axis}.png", dpi=200, bbox_inches="tight")

        covariance_fig, _ = plot_error(
            results_df,
            metric="covariance_frobenius_error",
            x=x_axis,
            facet="observation_strategy",
            title=f"Covariance Error vs {x_axis.capitalize()}",
        )
        covariance_fig.savefig(figures_dir / f"covariance_error_vs_{x_axis}.png", dpi=200, bbox_inches="tight")

    results_df = run_layered_experiment_grid(
        dimensions=args.dimensions,
        depths=args.depths,
        sparsities=args.sparsities,
        correlations=args.correlations,
        observation_strategies=args.observation_strategies,
        observed_fraction=args.observed_fraction,
        n_repetitions=args.repetitions,
        n_samples=args.n_samples,
        burn_in=args.burn_in,
        thinning=args.thinning,
        rw_mh_proposal_scale=args.rw_mh_proposal_scale,
        random_state=args.random_state,
    )

    results_df.to_csv(tables_dir / "layered_experiment_results.csv", index=False)

    for x_axis in ("correlation", "dimension", "sparsity"):
        save_metric_family(results_df, x_axis)

    print(f"Saved run outputs to {run_dir}")
    print(f"Saved figures to {figures_dir}")
    print(f"Saved tables to {tables_dir}")


if __name__ == "__main__":
    main()
