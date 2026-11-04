"""Prepare the Pathway-Transformer inputs from the raw data.

Raw data (https://www.kaggle.com/datasets/sudiptobaul/data-for-pathway-transformer), in --raw_dir:
    brca_gene_expression.xlsx, brca_crapa.xlsx, brca_utrapa.xlsx, brca_asquant.xlsx   genes x samples
    brca_tcga_clinical_data.tsv                                                       ER/PR status
    pathway_csv/hsa*.csv                                                              KEGG relations of 89 pathways

Written to --out_dir:
    channels/{gene,crapa,utrapa,asquant}.csv.gz   values of the 3,244 pathway genes (genes x samples), untransformed;
                                                  the first row is kept for a gene listed twice
    structures/<pathway>.npz                      genes (sorted), in/out-degree, directed shortest-path distances
                                                  and the KEGG edge types along each shortest path (up to 5 hops)

Usage:
    python prepare_data.py --raw_dir raw_data --out_dir data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CHANNEL_FILES = {"gene": "brca_gene_expression.xlsx", "crapa": "brca_crapa.xlsx",
                 "utrapa": "brca_utrapa.xlsx", "asquant": "brca_asquant.xlsx"}
# KEGG relation subtypes and their edge-type index (1-8; 0 = no edge); rare subtypes are merged into "others"
EDGE_TYPES = ["others", "indirect effect", "compound", "expression", "phosphorylation", "inhibition",
              "binding/association", "activation"]
OTHERS = {"dissociation", "ubiquitination", "dephosphorylation", "missing interaction", "state change",
          "methylation", "repression"}
MAX_HOPS = 5
UNREACHABLE = 510


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", default=str(HERE / "raw_data"))
    parser.add_argument("--out_dir", default=str(HERE / "data"))
    for channel, name in CHANNEL_FILES.items():
        parser.add_argument(f"--{channel}_file", default=None, help=f"default: <raw_dir>/{name}")
    parser.add_argument("--pathway_dir", default=None, help="default: <raw_dir>/pathway_csv")
    return parser.parse_args()


def read_relations(path):
    """KEGG relations of one pathway, rows with any missing field removed."""
    relations = pd.read_csv(path, index_col=0).dropna()
    subtype = relations["subtype"].where(~relations["subtype"].isin(OTHERS), "others")
    unknown = sorted(set(subtype) - set(EDGE_TYPES))
    if unknown:
        raise ValueError(f"{path.name}: unknown relation subtypes {unknown}")
    return relations.assign(subtype=subtype)


def pathway_genes(relations):
    return sorted(set(relations["from_genesymbols"]) | set(relations["to_genesymbols"]))


def floyd_warshall(adjacency):
    """All-pairs shortest paths of an unweighted directed graph; path[i, j] is an intermediate node (0 = direct)."""
    dist = np.asarray(adjacency, dtype=np.int64).copy(order="C")
    path = np.zeros_like(dist)
    dist[dist == 0] = UNREACHABLE
    np.fill_diagonal(dist, 0)
    for k in range(len(dist)):
        through_k = dist[:, [k]] + dist[[k], :]
        shorter = through_k < dist
        path[shorter] = k
        dist = np.minimum(dist, through_k)
    unreachable = dist >= UNREACHABLE
    path[unreachable] = UNREACHABLE
    dist[unreachable] = UNREACHABLE
    return dist, path


def path_nodes(path, i, j):
    """Intermediate nodes on the shortest path from i to j."""
    k = int(path[i][j])
    if k == 0:
        return []
    return path_nodes(path, i, k) + [k] + path_nodes(path, k, j)


def build_structure(relations):
    genes = pathway_genes(relations)
    index = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    adjacency = np.zeros((n, n), dtype=np.int64)
    edge_type = np.zeros((n, n), dtype=np.int64)
    for source, target, subtype in relations[["from_genesymbols", "to_genesymbols", "subtype"]].itertuples(index=False):
        adjacency[index[source], index[target]] = 1
        edge_type[index[source], index[target]] = 1 + EDGE_TYPES.index(subtype)   # a repeated pair keeps the last type
    dist, path = floyd_warshall(adjacency)
    hops = np.zeros((n, n, MAX_HOPS), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i == j or dist[i, j] >= UNREACHABLE:
                continue
            nodes = [i] + path_nodes(path, i, j) + [j]
            for h in range(min(MAX_HOPS, len(nodes) - 1)):
                hops[i, j, h] = edge_type[nodes[h], nodes[h + 1]]
    return {"genes": np.array(genes), "n": np.array(n), "out_degree": adjacency.sum(1),
            "in_degree": adjacency.sum(0), "dist": dist, "hops": hops}


def main():
    args = parse_args()
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    pathway_dir = Path(args.pathway_dir or raw_dir / "pathway_csv")
    (out_dir / "channels").mkdir(parents=True, exist_ok=True)
    (out_dir / "structures").mkdir(parents=True, exist_ok=True)

    pathway_files = sorted(pathway_dir.glob("hsa*.csv"))
    genes = set()
    for path in pathway_files:
        relations = read_relations(path)
        genes.update(pathway_genes(relations))
        structure = build_structure(relations)
        np.savez_compressed(out_dir / "structures" / f"{path.stem}.npz", **structure)
        print(f"{path.stem}: {int(structure['n'])} genes, {int(structure['out_degree'].sum())} edges", flush=True)
    print(f"{len(pathway_files)} pathways, {len(genes)} genes", flush=True)

    for channel, name in CHANNEL_FILES.items():
        path = Path(getattr(args, f"{channel}_file") or raw_dir / name)
        print(f"reading {path} (a few minutes)", flush=True)
        matrix = pd.read_excel(path, index_col=0)
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)
        matrix = matrix[matrix.index.isin(genes)]
        matrix = matrix[~matrix.index.duplicated(keep="first")]
        matrix = matrix.apply(pd.to_numeric, errors="coerce")
        matrix.to_csv(out_dir / "channels" / f"{channel}.csv.gz")
        print(f"{channel}: {matrix.shape[0]} genes x {matrix.shape[1]} samples", flush=True)


if __name__ == "__main__":
    main()
