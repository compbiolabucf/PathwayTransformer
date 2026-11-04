"""Run Pathway-Transformer for many pathways in parallel and summarize the results.

The pathways are spread over GPU worker processes (pathway_transformer.py), balanced by pathway size. Finished
task/feature/pathway results are skipped, so an interrupted run can be resumed with the same command.

Usage:
    python run.py --run_name er_all --tasks ER --features all --gpus 0 1 --workers_per_gpu 4

Writes results/<run_name>/:
    folds/<task>_<features>_<pathway>.csv   one row per test fold (see pathway_transformer.py)
    logs/worker<k>.log
    per_pathway.csv                         mean over the folds run, per task, features and pathway
    summary.csv                             mean and SD over pathways, per task and features
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
METRICS = ["test_auroc", "test_auprc", "test_f1", "test_mcc", "valid_auroc"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--tasks", nargs="+", default=["ER"])
    parser.add_argument("--features", nargs="+", default=["all"])
    parser.add_argument("--pathways", nargs="*", default=None, help="default: every pathway in <data_dir>/structures")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--config", default="{}", help="JSON overrides of the hyperparameters")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--workers_per_gpu", type=int, default=4)
    parser.add_argument("--data_dir", default=str(HERE / "data"))
    parser.add_argument("--clinical", default=str(HERE / "raw_data" / "brca_tcga_clinical_data.tsv"))
    parser.add_argument("--results_dir", default=str(HERE / "results"))
    parser.add_argument("--python", default=sys.executable, help="interpreter for the workers")
    return parser.parse_args()


def launch(run_dir, tasks, features, pathways, folds, config, gpus, workers_per_gpu, data_dir, clinical, python):
    """Run all task x features x pathway jobs on len(gpus) x workers_per_gpu workers; returns their exit codes."""
    run_dir = Path(run_dir)
    structures = Path(data_dir) / "structures"
    pathways = pathways or sorted(p.stem for p in structures.glob("hsa*.npz"))
    size = {}
    for pathway in pathways:
        with np.load(structures / f"{pathway}.npz") as data:
            size[pathway] = int(data["n"])
    slots = [gpu for _ in range(workers_per_gpu) for gpu in gpus]   # consecutive workers on different GPUs
    assigned, load = [[] for _ in slots], [0] * len(slots)
    for pathway in sorted(pathways, key=lambda p: -size[p]):       # largest first, to the least loaded worker
        i = load.index(min(load))
        assigned[i].append(pathway)
        load[i] += size[pathway]
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    procs = []
    for k, (gpu, worker_pathways) in enumerate(zip(slots, assigned)):
        if not worker_pathways:
            continue
        command = [python, "-B", str(HERE / "pathway_transformer.py"), "--tasks", *tasks, "--features", *features,
                   "--pathways", *worker_pathways, "--folds", *map(str, folds), "--config", config,
                   "--data_dir", str(data_dir), "--clinical", str(clinical), "--output_dir", str(run_dir / "folds")]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2")
        log = open(run_dir / "logs" / f"worker{k}.log", "w")
        procs.append(subprocess.Popen(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT))
    print(f"{len(tasks) * len(features) * len(pathways)} jobs on {len(procs)} workers -> {run_dir}", flush=True)
    return [p.wait() for p in procs]


def summarize(run_dir):
    run_dir = Path(run_dir)
    files = sorted((run_dir / "folds").glob("*_*_hsa*.csv"))
    if not files:
        raise RuntimeError(f"no results in {run_dir / 'folds'}")
    folds = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    keys = ["task", "features", "pathway"]
    per_pathway = folds.groupby(keys)[METRICS].mean()
    per_pathway["n_folds"] = folds.groupby(keys).size()
    per_pathway.reset_index().to_csv(run_dir / "per_pathway.csv", index=False)
    summary = per_pathway.groupby(["task", "features"])[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_pathways"] = per_pathway.groupby(["task", "features"]).size()
    summary.reset_index().to_csv(run_dir / "summary.csv", index=False)
    return per_pathway.reset_index(), summary.reset_index()


def main():
    args = parse_args()
    json.loads(args.config)                     # fail early on malformed JSON
    run_dir = Path(args.results_dir) / args.run_name
    codes = launch(run_dir, args.tasks, args.features, args.pathways, args.folds, args.config, args.gpus,
                   args.workers_per_gpu, args.data_dir, args.clinical, args.python)
    print(f"workers finished, exit codes {codes}", flush=True)
    _, summary = summarize(run_dir)
    print(summary.round(4).to_string(index=False))
    if any(codes):
        sys.exit(f"some workers failed; see {run_dir / 'logs'}")


if __name__ == "__main__":
    main()
