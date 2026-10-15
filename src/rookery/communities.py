import random
import time

import igraph as ig
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rookery import localgraph as lg
from rookery import sparkgraph as sg
from rookery.mine import node_path
from rookery.sparkctx import spark_session, stop
from rookery.store import graph_path, load_json, record_timing, save_json, site_key, timed


def to_igraph(g):
    rows = np.repeat(np.arange(g.n, dtype=np.int32), np.diff(g.indptr))
    keep = rows < g.indices
    h = ig.Graph(n=g.n, directed=False)
    h.add_edges(np.column_stack([rows[keep], g.indices[keep]]))
    return h, g.weights[keep].astype(np.float64)


def ground_truth(path, ids):
    table = pq.read_table(str(path))
    cid = table.column("cid").to_numpy()
    node = table.column("node").to_numpy()
    pos = np.searchsorted(ids, node)
    ok = (pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == node)
    return cid[ok], pos[ok]


def f1_scores(gt_cid, gt_node, membership):
    det_size = np.bincount(membership)
    _, gt_inv = np.unique(gt_cid, return_inverse=True)
    gt_size = np.bincount(gt_inv)
    labels = membership[gt_node]
    key = gt_inv.astype(np.int64) * (int(labels.max()) + 1) + labels
    pairs, overlap = np.unique(key, return_counts=True)
    g_idx = pairs // (int(labels.max()) + 1)
    d_idx = pairs % (int(labels.max()) + 1)
    f1 = 2.0 * overlap / (gt_size[g_idx] + det_size[d_idx])
    best_gt = np.zeros(len(gt_size))
    np.maximum.at(best_gt, g_idx, f1)
    uniq_d, d_inv = np.unique(d_idx, return_inverse=True)
    best_det = np.zeros(len(uniq_d))
    np.maximum.at(best_det, d_inv, f1)
    return {"f1_gt": float(best_gt.mean()), "f1_det": float(best_det.mean()),
            "f1": float((best_gt.mean() + best_det.mean()) / 2), "gt_communities": int(len(gt_size))}


def relabel(values):
    return np.unique(values, return_inverse=True)[1]


def spark_lpa(cfg, names):
    spark = spark_session(cfg)
    lp = cfg["lpa"]
    info = {}
    done = load_json(cfg, "lpa_runs", {})
    for name, path in names:
        if name in done and node_path(cfg, name, "lpa").exists():
            info[name] = done[name]
            continue
        edges = spark.read.parquet(str(path))
        with timed(cfg, "lpa_spark", name, edges.count()):
            labels, meta = sg.label_propagation(edges, lp["max_iter"], lp["self_weight"], lp["min_changed_fraction"],
                                                cfg["spark"]["check_every"])
            labels.write.mode("overwrite").parquet(str(node_path(cfg, name, "lpa")))
        sg.clear_scratch(spark)
        info[name] = meta
        done[name] = meta
        save_json(cfg, "lpa_runs", done)
    stop()
    return info


def lpa_membership(cfg, name, ids):
    table = pq.read_table(str(node_path(cfg, name, "lpa")))
    nodes = table.column("node").to_numpy()
    labels = table.column("label").to_numpy()
    out = np.zeros(len(ids), dtype=np.int64)
    out[np.searchsorted(ids, nodes)] = labels
    return relabel(out)


def run_algorithms(cfg, name, path, lpa_meta, references):
    lv = cfg["louvain"]
    g = lg.load(path, weighted=True)
    rows = {}
    start = time.perf_counter()
    memb, meta = lg.louvain(g, lv["resolution"], lv["max_levels"], lv["max_sweeps"], lv["min_gain"], lv["seed"])
    rows["louvain"] = {"membership": memb, "seconds": time.perf_counter() - start, **meta}
    rows["lpa_spark"] = {"membership": lpa_membership(cfg, name, g.ids), **lpa_meta}
    for row in rows.values():
        row["modularity"] = lg.modularity(g, row["membership"])
    ids, edges = g.ids, int(g.indptr[-1] // 2)
    if references:
        h, w = to_igraph(g)
        del g
        algos = [("igraph_leiden", lambda: h.community_leiden(objective_function="modularity", weights=w,
                                                              n_iterations=2).membership)]
        if edges <= lv["reference_max_edges"]:
            algos.insert(0, ("igraph_louvain", lambda: h.community_multilevel(weights=w).membership))
        if edges <= cfg["lpa"]["reference_max_edges"]:
            algos.append(("igraph_lpa", lambda: h.community_label_propagation(weights=w).membership))
        random.seed(lv["seed"])
        for label, fn in algos:
            start = time.perf_counter()
            found = fn()
            rows[label] = {"membership": np.asarray(found, dtype=np.int64), "seconds": time.perf_counter() - start,
                           "modularity": h.modularity(found, weights=w)}
        del h
    for row in rows.values():
        row["communities"] = int(row["membership"].max()) + 1
        row["largest_share"] = float(np.bincount(row["membership"]).max() / len(ids))
    return ids, edges, rows


def save_membership(cfg, name, algo, ids, membership):
    path = node_path(cfg, name, algo)
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"node": ids, "community": membership}), path / "part-0.parquet")


def tag_coherence(cfg, key, ids, membership, seed):
    table = pq.read_table(str(graph_path(cfg, key, "user_tag")), columns=["src", "dst", "weight"]).to_pandas()
    top = table.sort_values(["src", "weight", "dst"], ascending=[True, False, True]).drop_duplicates("src")
    pos = np.searchsorted(ids, top["src"].to_numpy())
    ok = (pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == top["src"].to_numpy())
    comm, tag = membership[pos[ok]], top["dst"].to_numpy()[ok]

    def purity(c):
        _, c_inv, c_size = np.unique(c, return_inverse=True, return_counts=True)
        key2 = c_inv.astype(np.int64) * (int(-tag.min()) + 2) + (-tag)
        pairs, counts = np.unique(key2, return_counts=True)
        best = np.zeros(len(c_size))
        np.maximum.at(best, pairs // (int(-tag.min()) + 2), counts)
        big = c_size >= 10
        return float(best[big].sum() / c_size[big].sum()) if big.any() else 0.0

    shuffled = np.random.default_rng(seed).permutation(comm)
    return {"users_with_tags": int(ok.sum()), "purity": purity(comm), "random_purity": purity(shuffled)}


def run(cfg):
    snap = [(n, graph_path(cfg, n, "graph")) for n in cfg["fetch"]["snap_graphs"]]
    sites = [(site_key(s), graph_path(cfg, site_key(s), "interaction")) for s in cfg["fetch"]["sites"]]
    lpa_meta = spark_lpa(cfg, snap + sites)
    lg.warm_up()
    out = load_json(cfg, "communities", {})
    for name, path in snap + sites:
        print(f"  communities on {name}", flush=True)
        with timed(cfg, "communities", name):
            ids, edges, rows = run_algorithms(cfg, name, path, lpa_meta[name], references=(name, path) in snap)
        for algo in ("louvain", "igraph_louvain", "igraph_leiden"):
            if algo in rows:
                record_timing(cfg, algo, name, edges, rows[algo]["seconds"])
        report = {}
        for algo, row in rows.items():
            entry = {k: v for k, v in row.items() if k != "membership"}
            if (name, path) in snap:
                cid, node = ground_truth(graph_path(cfg, name, "cmty"), ids)
                entry.update(f1_scores(cid, node, row["membership"]))
            else:
                entry.update(tag_coherence(cfg, name, ids, row["membership"], cfg["louvain"]["seed"]))
            report[algo] = entry
        save_membership(cfg, name, "louvain", ids, rows["louvain"]["membership"])
        if (name, path) in snap:
            cid, node = ground_truth(graph_path(cfg, name, "cmty"), ids)
            shuffled = np.random.default_rng(cfg["louvain"]["seed"]).permutation(rows["louvain"]["membership"])
            report["random_baseline"] = f1_scores(cid, node, shuffled)
        out[name] = report
        save_json(cfg, "communities", out)
    lv = cfg["louvain"]
    for site in cfg["fetch"]["sites"]:
        key = site_key(site)
        g = lg.load(graph_path(cfg, key, "tag_cooc"), weighted=True)
        memb, meta = lg.louvain(g, lv["resolution"], lv["max_levels"], lv["max_sweeps"], lv["min_gain"], lv["seed"])
        save_membership(cfg, f"{key}_tags", "louvain", g.ids, memb)
        out.setdefault("tag_graphs", {})[key] = meta
    save_json(cfg, "communities", out)
