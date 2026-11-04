"""Choose Pathway-Transformer hyperparameters by validation AUROC.

Each configuration (JSON overrides of DEFAULTS in pathway_transformer.py) is trained on the same pathways, tasks
and folds, and the configurations are ranked by mean validation AUROC. Test metrics are not read, so the test
folds play no part in the choice. The default search is the one used for the paper (Supplementary Section S1):
hidden dimension 32 or 64 (feed-forward dimension twice as large) and learning rate 1e-3 or 3e-4, on six
pathways, ER and PR, all four channels, five folds.

Usage:
    python tune.py --tune_name paper --gpus 0 1
    python tune.py --tune_name regularization --grid '{"dropout": [0.1, 0.2], "weight_decay": [0.01, 0.1]}'
    python tune.py --tune_name size --configs '[{"d_model": 64, "ffn_dim": 128}, {"d_model": 128, "ffn_dim": 256}]'

Writes results/<tune_name>/<configuration>/ (laid out as by run.py) and results/<tune_name>/tuning.csv.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

from run import HERE, launch

PAPER_CONFIGS = [{"d_model": 64, "ffn_dim": 128, "lr": 1e-3}, {"d_model": 64, "ffn_dim": 128, "lr": 3e-4},
                 {"d_model": 32, "ffn_dim": 64, "lr": 1e-3}, {"d_model": 32, "ffn_dim": 64, "lr": 3e-4}]
PAPER_PATHWAYS = ["hsa04915", "hsa04010", "hsa04060", "hsa04110", "hsa04630", "hsa05202"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tune_name", required=True)
    search = parser.add_mutually_exclusive_group()
    search.add_argument("--grid", help="JSON object of value lists; every combination is tried")
    search.add_argument("--configs", help="JSON list of configurations (default: the paper's four)")
    parser.add_argument("--tasks", nargs="+", default=["ER", "PR"])
    parser.add_argument("--features", nargs="+", default=["all"])
    parser.add_argument("--pathways", nargs="+", default=PAPER_PATHWAYS)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--workers_per_gpu", type=int, default=4)
    parser.add_argument("--data_dir", default=str(HERE / "data"))
    parser.add_argument("--clinical", default=str(HERE / "raw_data" / "brca_tcga_clinical_data.tsv"))
    parser.add_argument("--results_dir", default=str(HERE / "results"))
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def configurations(args):
    if args.grid:
        grid = json.loads(args.grid)
        return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    return json.loads(args.configs) if args.configs else PAPER_CONFIGS


def main():
    args = parse_args()
    tune_dir = Path(args.results_dir) / args.tune_name
    rows = []
    for config in configurations(args):
        name = "_".join(f"{key}{value}" for key, value in sorted(config.items())) or "defaults"
        codes = launch(tune_dir / name, args.tasks, args.features, args.pathways, args.folds, json.dumps(config),
                       args.gpus, args.workers_per_gpu, args.data_dir, args.clinical, args.python)
        if any(codes):
            sys.exit(f"some workers failed; see {tune_dir / name / 'logs'}")
        folds = pd.concat([pd.read_csv(f, usecols=["task", "valid_auroc"])
                           for f in sorted((tune_dir / name / "folds").glob("*_*_hsa*.csv"))])
        row = {"configuration": json.dumps(config), "valid_auroc": folds["valid_auroc"].mean(), "n_folds": len(folds)}
        row.update({f"valid_auroc_{task}": value for task, value in folds.groupby("task")["valid_auroc"].mean().items()})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("valid_auroc", ascending=False)
    table.to_csv(tune_dir / "tuning.csv", index=False)
    print(table.round(4).to_string(index=False))
    print(f"\nbest by validation AUROC: {table.iloc[0]['configuration']}")


if __name__ == "__main__":
    main()
