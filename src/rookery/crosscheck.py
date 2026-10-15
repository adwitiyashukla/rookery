import time

import igraph as ig
import networkx as nx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from rookery import localgraph as lg
from rookery.mine import SNAP_PUBLISHED, is_real_snap, node_path, targets
from rookery.store import load_json, save_json


def edge_arrays(path, reverse=False):
    table = pq.read_table(str(path), columns=["src", "dst", "weight"])
    src = table.column("src").to_numpy()
    dst = table.column("dst").to_numpy()
    weight = table.column("weight").to_numpy().astype(np.float64)
    del table
    pa.default_memory_pool().release_unused()
    keep = src != dst
    if not keep.all():
        src, dst, weight = src[keep], dst[keep], weight[keep]
    return (dst, src, weight) if reverse else (src, dst, weight)


def metric(cfg, graph, name, column, ids):
    table = pq.read_table(str(node_path(cfg, graph, name)), columns=["node", column])
    nodes = table.column("node").to_numpy()
    values = table.column(column).to_numpy()
    pos = np.searchsorted(ids, nodes)
    out = np.full(len(ids), np.nan)
    out[pos] = values
    return out


def same_partition(a, b):
    pairs = np.unique(np.column_stack([a, b]), axis=0)
    return len(pairs) == len(np.unique(a)) == len(np.unique(b))


def compare_ranks(ours, ref):
    top_ours = set(np.argsort(-ours)[:100].tolist())
    top_ref = set(np.argsort(-ref)[:100].tolist())
    rho = spearmanr(ours, ref).statistic if len(ours) > 2 else 1.0
    return {"max_abs_diff": float(np.max(np.abs(ours - ref))),
            "max_rel_diff": float(np.max(np.abs(ours - ref) / ref)),
            "spearman": float(rho), "top100_overlap": len(top_ours & top_ref) / 100}


def check_graph(cfg, name, path, kind):
    alpha = cfg["pagerank"]["alpha"]
    src, dst, weight = edge_arrays(path, reverse=kind == "expertise")
    ids, a, b = lg.index_nodes(src, dst)
    del src, dst
    directed = kind == "expertise"
    h = ig.Graph(n=len(ids), directed=directed)
    h.add_edges(np.column_stack([a, b]))
    del a, b
    weighted = not np.all(weight == 1.0)
    if weighted:
        h.es["weight"] = weight
        h.simplify(combine_edges="sum")
        weight = np.asarray(h.es["weight"], dtype=np.float64)
    elif not h.is_simple():
        h.simplify()
        weight = np.ones(h.ecount())
    out = {"nodes": len(ids), "edges": h.ecount()}
    start = time.perf_counter()
    ref_pr = np.asarray(h.pagerank(damping=alpha, directed=directed, weights="weight" if weighted else None))
    out["igraph_seconds"] = {"pagerank": time.perf_counter() - start}
    out["pagerank"] = compare_ranks(metric(cfg, name, "pagerank", "rank", ids), ref_pr)
    if directed:
        return out
    start = time.perf_counter()
    comp = np.asarray(h.connected_components().membership)
    out["igraph_seconds"]["components"] = time.perf_counter() - start
    ours_comp = metric(cfg, name, "components", "component", ids)
    out["components"] = {"count": int(comp.max()) + 1, "same_partition": bool(same_partition(ours_comp, comp))}
    start = time.perf_counter()
    core = np.asarray(h.coreness())
    out["igraph_seconds"]["kcore"] = time.perf_counter() - start
    out["kcore"] = {"agree_fraction": float(np.mean(metric(cfg, name, "kcore", "core", ids) == core)),
                    "max_core": int(core.max())}
    start = time.perf_counter()
    cc = np.asarray(h.transitivity_local_undirected(mode="zero"))
    out["igraph_seconds"]["triangles"] = time.perf_counter() - start
    deg = np.asarray(h.degree(), dtype=np.float64)
    ref_tri = np.rint(cc * deg * (deg - 1) / 2)
    ours_tri = metric(cfg, name, "triangles", "triangles", ids)
    out["triangles"] = {"agree_fraction": float(np.mean(ours_tri == ref_tri)), "total": int(ref_tri.sum() // 3)}
    if (node_path(cfg, name, "triangles_spark")).exists():
        spark_tri = metric(cfg, name, "triangles_spark", "triangles", ids)
        out["triangles"]["spark_agree_fraction"] = float(np.mean(spark_tri == ref_tri))
    if h.ecount() <= cfg["crosscheck"]["networkx_max_edges"]:
        pairs = np.asarray(h.get_edgelist(), dtype=np.int64).reshape(-1, 2)
        out["networkx"] = networkx_checks(cfg, name, ids, pairs[:, 0], pairs[:, 1], weight, alpha)
    published = SNAP_PUBLISHED.get(name)
    if is_real_snap(cfg, name):
        out["published"] = {"nodes_equal": len(ids) == published["nodes"],
                            "edges_equal": h.ecount() == published["edges"],
                            "triangles_equal": out["triangles"]["total"] == published["triangles"]}
    return out


def networkx_checks(cfg, name, ids, a, b, weight, alpha):
    g = nx.Graph()
    g.add_nodes_from(range(len(ids)))
    g.add_weighted_edges_from(zip(a.tolist(), b.tolist(), weight.tolist(), strict=True))
    pr = nx.pagerank(g, alpha=alpha, weight="weight", tol=1e-12, max_iter=500)
    ours_pr = metric(cfg, name, "pagerank", "rank", ids)
    core = nx.core_number(g)
    tri = nx.triangles(g)
    return {
        "pagerank_max_abs_diff": float(np.max(np.abs(ours_pr - np.array([pr[i] for i in range(len(ids))])))),
        "kcore_agree_fraction": float(np.mean(metric(cfg, name, "kcore", "core", ids)
                                              == np.array([core[i] for i in range(len(ids))]))),
        "triangles_agree_fraction": float(np.mean(metric(cfg, name, "triangles", "triangles", ids)
                                                  == np.array([tri[i] for i in range(len(ids))]))),
        "components": nx.number_connected_components(g),
    }


def run(cfg):
    out = load_json(cfg, "crosscheck", {})
    for name, path, kind in targets(cfg):
        print(f"  cross checking {name}", flush=True)
        out[name] = check_graph(cfg, name, path, kind)
        save_json(cfg, "crosscheck", out)


def verify(cfg):
    checks = load_json(cfg, "crosscheck", {})
    mining = load_json(cfg, "mining", {})
    problems = [] if checks else ["no cross check results, run the crosscheck stage first"]
    for name, c in checks.items():
        pr = c["pagerank"]
        if pr["spearman"] < 0.99 or pr["top100_overlap"] < 0.9:
            problems.append(f"{name}: pagerank differs from igraph {pr}")
        if "kcore" in c and c["kcore"]["agree_fraction"] != 1.0:
            problems.append(f"{name}: k-core agrees on {c['kcore']['agree_fraction']:.4f} of nodes")
        if "components" in c and not c["components"]["same_partition"]:
            problems.append(f"{name}: components differ from igraph")
        if "triangles" in c and c["triangles"]["agree_fraction"] != 1.0:
            problems.append(f"{name}: triangles agree on {c['triangles']['agree_fraction']:.4f} of nodes")
        if "triangles" in c and c["triangles"].get("spark_agree_fraction", 1.0) != 1.0:
            problems.append(f"{name}: Spark triangles agree on {c['triangles']['spark_agree_fraction']:.4f} of nodes")
        for key, ok in c.get("published", {}).items():
            if not ok:
                problems.append(f"{name}: {key} is false against the SNAP published statistics")
    for name, m in mining.items():
        if m.get("triangles_agree") is False:
            problems.append(f"{name}: Spark and numba triangle totals differ")
    for line in problems:
        print(f"  FAIL {line}")
    print(f"  {len(checks)} graphs cross checked, {len(problems)} problems", flush=True)
    return not problems
