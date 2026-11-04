"""Pathway-Transformer: hormone-receptor status prediction from transcript variants on KEGG pathway graphs.

One model is trained per KEGG pathway. Each gene of the pathway is a token, and a virtual pathway token is added
whose final representation gives the prediction.
* Gene features: token_g = b_g + sum_c z_gc * w_gc over the feature channels c (gene expression, CR-APA, UTR-APA,
  AS), plus a learned per-channel embedding when gene g has no value in channel c. Expression is log2(x + 1);
  every gene/channel is robustly scaled with training-fold statistics, (x - median) / (IQR / 1.349), and
  compressed with arcsinh.
* Gene centrality encoding: in-degree and out-degree embeddings are added to the gene tokens.
* Pathway spatial encoding: every attention head gets a bias from the directed shortest-path distance of each
  gene pair (with an index for unreachable pairs) and from the KEGG edge types along that path (up to 5 hops).
* Pre-LayerNorm Transformer blocks, binary cross-entropy, AdamW with linear warm-up and cosine decay, early
  stopping on validation AUROC, and the average of three seeds.

Evaluation: samples with a label and all four channels, sorted by ID; StratifiedKFold(5, shuffle, seed 4);
for test fold k, fold (k + 1) mod 5 is the validation fold and the other three are the training folds.
F1 and MCC use the Youden threshold of the test fold (the *_valthr columns use the validation-fold threshold).

Usage (jobs run one after another on one device; run.py spreads them over GPUs):
    python pathway_transformer.py --tasks ER --features all --pathways hsa04915 --folds 0
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
CHANNELS = ("gene", "crapa", "utrapa", "asquant")
FEATURES = {"gene": ["gene"], "crapa": ["crapa"], "gene_crapa": ["gene", "crapa"],
            "all": ["gene", "crapa", "utrapa", "asquant"]}
LABEL_COLUMNS = {"ER": "ER Status By IHC", "PR": "PR status by ihc"}
MAX_HOPS = 5          # edge types encoded along the first 5 hops of a shortest path
SPATIAL_MAX = 30      # distances are clipped to 30; index 31 = unreachable
DEGREE_MAX = 63       # degrees are clipped to 63
UNREACHABLE = 510     # distance stored for unreachable pairs (prepare_data.py)

# Hyperparameters (Supplementary Table S1); any of them can be overridden with --config
DEFAULTS = dict(d_model=64, n_heads=4, n_layers=2, ffn_dim=128, dropout=0.1, attn_dropout=0.1, lr=1e-3,
                weight_decay=1e-2, batch_size=64, max_epochs=150, warmup_epochs=5, patience=25, seeds=[0, 1, 2])


# ---- data --------------------------------------------------------------------

def read_channel(data_dir, channel):
    table = pd.read_csv(Path(data_dir) / "channels" / f"{channel}.csv.gz", index_col=0)
    table.index = table.index.astype(str)
    table.columns = table.columns.astype(str)
    return table


def load_task(task, data_dir, clinical_file):
    """Sorted samples with an ER/PR label and all four channels, their 0/1 labels, and the channel tables."""
    clinical = pd.read_csv(clinical_file, sep="\t")
    tables = {c: read_channel(data_dir, c) for c in CHANNELS}
    column = LABEL_COLUMNS[task]
    clinical = clinical[clinical[column].isin(["Positive", "Negative"])]
    clinical.index = clinical["Sample ID"].astype(str)
    samples = sorted(set(clinical.index).intersection(*(set(t.columns) for t in tables.values())))
    y = (clinical.loc[samples, column] == "Positive").astype(int).to_numpy()
    return samples, y, tables


def load_structure(data_dir, pathway):
    with np.load(Path(data_dir) / "structures" / f"{pathway}.npz") as data:
        return {k: data[k] for k in data.files}


def raw_features(tables, channels, genes, samples):
    """(samples, genes, channels) values and (genes, channels) presence mask; missing and negative values -> 0."""
    values, present = [], []
    for c in channels:
        table = tables[c]
        present.append(np.array([g in table.index for g in genes]))
        x = table.reindex(index=genes, columns=samples).fillna(0).to_numpy(float).T
        x = np.clip(x, 0, None)
        values.append(np.log2(x + 1) if c == "gene" else x)
    return np.stack(values, axis=2).astype(np.float32), np.stack(present, axis=1)


def robust_scale(raw, present, fit):
    """arcsinh((x - median) / scale) with statistics of the samples `fit`; scale = IQR / 1.349, or 1.2533 x the
    mean absolute deviation from the median when the IQR vanishes, or 1; 0 where a gene is absent."""
    median = np.median(raw[fit], 0)
    q1, q3 = np.percentile(raw[fit], [25, 75], axis=0)
    scale = (q3 - q1) / 1.349
    mean_abs = np.abs(raw[fit] - median).mean(0) * 1.2533
    scale = np.where(scale > 1e-6, scale, np.where(mean_abs > 1e-6, mean_abs, 1.0))
    return np.arcsinh((raw - median) / scale).astype(np.float32) * present[None]


def splits(y, n_splits=5, seed=4):
    folds = [t for _, t in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(y, y)]
    out = []
    for k in range(n_splits):
        test, valid = np.sort(folds[k]), np.sort(folds[(k + 1) % n_splits])
        train = np.setdiff1d(np.arange(len(y)), np.concatenate([test, valid]))
        out.append((k, train, valid, test))
    return out


# ---- model -------------------------------------------------------------------

class Block(nn.Module):
    """Pre-LayerNorm Transformer block whose attention scores receive the pathway bias."""

    def __init__(self, d, heads, ffn, dropout, attn_dropout):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d))
        self.attn_drop, self.drop = nn.Dropout(attn_dropout), nn.Dropout(dropout)

    def forward(self, x, bias):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh) + bias
        attn = self.attn_drop(scores.softmax(-1))
        x = x + self.drop(self.out((attn @ v).transpose(1, 2).reshape(b, t, d)))
        return x + self.drop(self.ffn(self.norm2(x)))


class PathwayTransformer(nn.Module):
    def __init__(self, structure, present, cfg):
        super().__init__()
        n, c = present.shape
        d, h = cfg["d_model"], cfg["n_heads"]
        bound = 1 / math.sqrt(d)
        self.weight = nn.Parameter(torch.empty(n, c, d).uniform_(-bound, bound))      # w_gc
        self.gene_bias = nn.Parameter(torch.empty(n, d).uniform_(-bound, bound))      # b_g (gene identity)
        self.missing = nn.Parameter(torch.empty(c, d).uniform_(-bound, bound))        # missing value, per channel
        self.register_buffer("present", torch.as_tensor(present, dtype=torch.float32))
        self.register_buffer("in_degree", torch.as_tensor(structure["in_degree"]).clamp(max=DEGREE_MAX))
        self.register_buffer("out_degree", torch.as_tensor(structure["out_degree"]).clamp(max=DEGREE_MAX))
        dist = torch.as_tensor(structure["dist"])
        spatial = torch.where(dist >= UNREACHABLE, torch.full_like(dist, SPATIAL_MAX + 1), dist.clamp(max=SPATIAL_MAX))
        self.register_buffer("spatial", spatial)
        self.register_buffer("hops", torch.as_tensor(structure["hops"]))
        self.register_buffer("path_len", dist.clamp(min=1, max=MAX_HOPS).float())
        self.in_deg_emb = nn.Embedding(DEGREE_MAX + 1, d)
        self.out_deg_emb = nn.Embedding(DEGREE_MAX + 1, d)
        self.spatial_emb = nn.Embedding(SPATIAL_MAX + 2, h)
        self.edge_emb = nn.Embedding(9, h, padding_idx=0)                              # 8 KEGG edge types, 0 = none
        self.hop_weight = nn.Parameter(torch.randn(MAX_HOPS, h, h) * 0.02)
        self.graph_token = nn.Parameter(torch.zeros(1, 1, d))
        self.virtual_distance = nn.Parameter(torch.zeros(h))
        self.blocks = nn.ModuleList(Block(d, h, cfg["ffn_dim"], cfg["dropout"], cfg["attn_dropout"])
                                    for _ in range(cfg["n_layers"]))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.input_drop = nn.Dropout(cfg["dropout"])

    def attention_bias(self):
        """(heads, genes + 1, genes + 1) bias: distance and multi-hop edge encodings; token 0 = pathway token."""
        n = self.spatial.size(0)
        h = self.virtual_distance.size(0)
        edge = torch.einsum("ijlh,lhk->ijk", self.edge_emb(self.hops), self.hop_weight) / self.path_len.unsqueeze(-1)
        bias = torch.zeros(h, n + 1, n + 1, device=self.spatial.device)
        bias[:, 1:, 1:] = (self.spatial_emb(self.spatial) + edge).permute(2, 0, 1)
        bias[:, 1:, 0] = self.virtual_distance.view(h, 1)
        bias[:, 0, :] = self.virtual_distance.view(h, 1)
        return bias

    def forward(self, x):   # x: (batch, genes, channels), scaled; 0 where missing
        tokens = self.gene_bias + torch.einsum("bnc,ncd->bnd", x * self.present, self.weight)
        tokens = tokens + (1 - self.present) @ self.missing
        tokens = tokens + self.in_deg_emb(self.in_degree) + self.out_deg_emb(self.out_degree)
        h = torch.cat([self.graph_token.expand(x.size(0), -1, -1), tokens], dim=1)
        h = self.input_drop(h)
        bias = self.attention_bias()
        for block in self.blocks:
            h = block(h, bias)
        h = self.norm(h)
        return self.head(h[:, 0]).squeeze(-1)


# ---- training and evaluation -------------------------------------------------

def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    t = float(thr[int((tpr - fpr).argmax())])
    return t if math.isfinite(t) else float(np.max(p)) + 1e-12


def metrics(y, p, threshold=None):
    threshold = youden_threshold(y, p) if threshold is None else threshold
    pred = (p >= threshold).astype(int)
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "f1": float(f1_score(y, pred, zero_division=0)), "mcc": float(matthews_corrcoef(y, pred)),
            "youden_threshold": threshold}


def predict(model, x, batch=256):
    model.eval()
    with torch.no_grad():
        return torch.cat([torch.sigmoid(model(x[i:i + batch])) for i in range(0, len(x), batch)]).cpu().numpy()


def train_seed(structure, present, x_tr, y_tr, x_va, y_va, cfg, seed, device):
    """Train one model; returns the checkpoint with the best validation AUROC, that AUROC and its epoch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PathwayTransformer(structure, present, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = math.ceil(len(y_tr) / cfg["batch_size"])
    total, warmup = cfg["max_epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / warmup if s < warmup else
                                                 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32, device=device)
    best_auroc, best_state, best_epoch = -math.inf, None, 0
    for epoch in range(cfg["max_epochs"]):
        model.train()
        order = torch.as_tensor(rng.permutation(len(y_tr)), device=device)
        for start in range(0, len(order), cfg["batch_size"]):
            idx = order[start:start + cfg["batch_size"]]
            loss = F.binary_cross_entropy_with_logits(model(x_tr[idx]), y_tr_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
        valid_auroc = roc_auc_score(y_va, predict(model, x_va))
        if valid_auroc > best_auroc:
            best_auroc, best_epoch = valid_auroc, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return model, best_auroc, best_epoch


def run_job(task, features, pathway, data, folds, cfg, device, data_dir):
    """Train and evaluate one task/feature/pathway combination on the given test folds; one row per fold."""
    samples, y, tables = data
    structure = load_structure(data_dir, pathway)
    genes = [str(g) for g in structure["genes"]]
    raw, present = raw_features(tables, FEATURES[features], genes, samples)
    rows = []
    for fold, tr, va, te in splits(y):
        if fold not in folds:
            continue
        x = torch.as_tensor(robust_scale(raw, present, tr), device=device)
        probs_va, probs_te, seeds = [], [], []
        for seed in cfg["seeds"]:
            start = time.time()
            model, valid_auroc, best_epoch = train_seed(structure, present, x[tr], y[tr], x[va], y[va], cfg,
                                                        seed + 100 * fold, device)
            probs_va.append(predict(model, x[va]))
            probs_te.append(predict(model, x[te]))
            seeds.append({"valid_auroc": valid_auroc, "best_epoch": best_epoch, "seconds": time.time() - start})
        p_va, p_te = np.mean(probs_va, 0), np.mean(probs_te, 0)
        valid_m, test_m = metrics(y[va], p_va), metrics(y[te], p_te)
        valthr = metrics(y[te], p_te, threshold=valid_m["youden_threshold"])
        row = {"task": task, "features": features, "pathway": pathway, "fold": fold, "n_genes": len(genes),
               "n_present": int(present.any(1).sum()), "train_samples": len(tr), "valid_samples": len(va),
               "test_samples": len(te), "mean_best_epoch": float(np.mean([s["best_epoch"] for s in seeds])),
               "seed_valid_aurocs": json.dumps([round(s["valid_auroc"], 4) for s in seeds]),
               "seconds": round(sum(s["seconds"] for s in seeds), 1)}
        row.update({f"valid_{k}": v for k, v in valid_m.items()})
        row.update({f"test_{k}": v for k, v in test_m.items()})
        row.update({"test_f1_valthr": valthr["f1"], "test_mcc_valthr": valthr["mcc"]})
        rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=["ER"], choices=sorted(LABEL_COLUMNS))
    parser.add_argument("--features", nargs="+", default=["all"], choices=list(FEATURES))
    parser.add_argument("--pathways", nargs="*", default=None, help="default: every pathway in <data_dir>/structures")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="test folds to run")
    parser.add_argument("--config", default="{}", help='JSON overrides of DEFAULTS, e.g. \'{"lr": 3e-4}\'')
    parser.add_argument("--data_dir", default=str(HERE / "data"))
    parser.add_argument("--clinical", default=str(HERE / "raw_data" / "brca_tcga_clinical_data.tsv"))
    parser.add_argument("--output_dir", default=str(HERE / "results" / "default" / "folds"))
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = {**DEFAULTS, **json.loads(args.config)}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown hyperparameters {sorted(unknown)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    pathways = args.pathways or sorted(p.stem for p in (Path(args.data_dir) / "structures").glob("hsa*.npz"))
    for task in args.tasks:
        data = None
        for features in args.features:
            for pathway in pathways:
                out = output_dir / f"{task}_{features}_{pathway}.csv"
                if out.exists():        # finished earlier: runs can be resumed
                    continue
                if data is None:
                    data = load_task(task, args.data_dir, args.clinical)
                start = time.time()
                rows = pd.DataFrame(run_job(task, features, pathway, data, args.folds, cfg, device, args.data_dir))
                rows.to_csv(out, index=False)
                print(f"{time.strftime('%H:%M:%S')} {task} {features} {pathway}: test AUROC "
                      f"{rows['test_auroc'].mean():.4f}, valid AUROC {rows['valid_auroc'].mean():.4f} "
                      f"({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
