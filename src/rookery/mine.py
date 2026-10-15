import gzip
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import functions as F

from rookery import localgraph as lg
from rookery import sparkgraph as sg
from rookery.fetch import snap_files
from rookery.sparkctx import spark_session, stop
from rookery.store import graph_path, load_json, raw_dir, results_dir, save_json, site_key, timed

SNAP_PUBLISHED = {
    "amazon": {"nodes": 334_863, "edges": 925_872, "triangles": 667_129, "avg_clustering": 0.3967},
    "dblp": {"nodes": 317_080, "edges": 1_049_866, "triangles": 2_224_385, "avg_clustering": 0.6324},
    "youtube": {"nodes": 1_134_890, "edges": 2_987_624, "triangles": 3_056_386, "avg_clustering": 0.0808},
    "lj": {"nodes": 3_997_962, "edges": 34_681_189, "triangles": 177_820_130, "avg_clustering": 0.2843},
}


def is_real_snap(cfg, name):
    path = raw_dir(cfg, "snap") / snap_files(name)[0]
    if name not in SNAP_PUBLISHED or not path.exists():
        return False
    with gzip.open(path, "rt") as fh:
        return "synthetic" not in "".join(fh.readline() for _ in range(3))


def targets(cfg):
    out = []
    for name in cfg["fetch"]["snap_graphs"]:
        out.append((name, graph_path(cfg, name, "graph"), "undirected"))
    for site in cfg["fetch"]["sites"]:
        key = site_key(site)
        out.append((f"{key}", graph_path(cfg, key, "interaction"), "undirected"))
        out.append((f"{key}_answers", graph_path(cfg, key, "answers"), "expertise"))
    out.append(("network", graph_path(cfg, "network", "interaction"), "undirected"))
    return out


def node_path(cfg, graph, metric):
    return results_dir(cfg) / "nodes" / graph / metric


def write_nodes(cfg, graph, metric, df):
    path = node_path(cfg, graph, metric)
    df.write.mode("overwrite").parquet(str(path))
    return df.sparkSession.read.parquet(str(path))


def powerlaw_fit(values, min_tail=50):
    x = np.asarray(values, dtype=np.float64)
    x = x[x > 0]
    vals, counts = np.unique(x, return_counts=True)
    tail = counts[::-1].cumsum()[::-1]
    best = None
    for i, xmin in enumerate(vals):
        n = tail[i]
        if n < min_tail:
            break
        tv, tc = vals[i:], counts[i:]
        alpha = 1.0 + n / np.sum(tc * np.log(tv / (xmin - 0.5)))
        emp = np.cumsum(tc) / n
        model = 1.0 - ((tv + 0.5) / (xmin - 0.5)) ** (1.0 - alpha)
        ks = float(np.max(np.abs(emp - model)))
        if best is None or ks < best["ks"]:
            best = {"xmin": float(xmin), "alpha": float(alpha), "ks": ks, "n_tail": int(n)}
    return best or {"xmin": None, "alpha": None, "ks": None, "n_tail": 0}


def ccdf(values, points=200):
    x = np.sort(np.asarray(values, dtype=np.float64))
    x = x[x > 0]
    grid = np.unique(np.round(np.logspace(0, np.log10(max(x.max(), 1.0)), points)))
    frac = 1.0 - np.searchsorted(x, grid, side="left") / len(x)
    return {"x": grid.tolist(), "p": frac.tolist()}


def mine_spark(spark, cfg, name, path, kind, stats, save):
    edges = spark.read.parquet(str(path))
    if kind == "expertise":
        edges = edges.select(F.col("dst").alias("src"), F.col("src").alias("dst"), "weight")
    m = edges.count()
    stats.update(edges=m, kind=kind)
    pr = cfg["pagerank"]
    if "nodes" not in stats:
        with timed(cfg, "degrees", name, m):
            deg = write_nodes(cfg, name, "degree", sg.degrees(edges, directed=kind == "expertise"))
            values = np.asarray(deg.select("degree").toPandas()["degree"])
        stats.update(nodes=int(len(values)), max_degree=int(values.max()), mean_degree=float(values.mean()),
                     powerlaw=powerlaw_fit(values), ccdf=ccdf(values))
        save()
    if "pagerank" not in stats:
        with timed(cfg, "pagerank", name, m):
            ranks, info = sg.pagerank(edges, pr["alpha"], pr["tol"], pr["max_iter"], directed=kind == "expertise",
                                      check_every=cfg["spark"]["check_every"])
            ranks = write_nodes(cfg, name, "pagerank", ranks)
        top = ranks.orderBy(F.desc("rank")).limit(10).collect()
        stats["pagerank"] = {**info, "top": [[int(r["node"]), float(r["rank"])] for r in top]}
        save()
    if kind == "expertise":
        return
    if "components" not in stats:
        with timed(cfg, "components", name, m):
            labels, info = sg.connected_components(edges, cfg["components"]["max_iter"])
            labels = write_nodes(cfg, name, "components", labels)
        sizes = labels.groupBy("component").count().orderBy(F.desc("count"))
        biggest = sizes.first()["count"]
        stats["components"] = {**info, "count": sizes.count(), "largest": int(biggest),
                               "largest_fraction": biggest / stats["nodes"]}
        save()
    if "kcore" not in stats and m <= cfg["kcore"]["spark_max_edges"]:
        with timed(cfg, "kcore", name, m):
            core, info = sg.kcore(edges, cfg["kcore"]["max_iter"], cfg["spark"]["check_every"])
            core = write_nodes(cfg, name, "kcore", core)
        shells = {int(r["core"]): int(r["count"]) for r in core.groupBy("core").count().collect()}
        stats["kcore"] = {**info, "engine": "spark", "max_core": max(shells), "shells": shells}
        save()
    if "triangles_spark" not in stats and m <= cfg["triangles"]["spark_max_edges"]:
        with timed(cfg, "triangles_spark", name, m):
            counts, info = sg.triangles(edges)
            write_nodes(cfg, name, "triangles_spark", counts)
        stats["triangles_spark"] = info["triangles"]
        save()


def mine_local(cfg, name, path, stats):
    with timed(cfg, "triangles_local", name, stats["edges"]):
        g = lg.load(path, weighted=False)
        tri, total = lg.triangles(g)
    deg = g.degree().astype(np.float64)
    pairs = deg * (deg - 1) / 2
    local = np.divide(tri, pairs, out=np.zeros_like(pairs), where=pairs > 0)
    stats["triangles_local"] = int(total)
    stats["transitivity"] = float(tri.sum() / pairs.sum()) if pairs.sum() else 0.0
    stats["avg_clustering"] = float(local.mean())
    stats["avg_clustering_deg2"] = float(local[pairs > 0].mean()) if (pairs > 0).any() else 0.0
    if "triangles_spark" in stats:
        stats["triangles_agree"] = stats["triangles_spark"] == stats["triangles_local"]
    write_local(cfg, name, "triangles", {"node": g.ids, "degree": g.degree(), "triangles": tri})
    if "kcore" not in stats:
        with timed(cfg, "kcore_local", name, stats["edges"]):
            core = lg.kcore(g)
        write_local(cfg, name, "kcore", {"node": g.ids, "core": core})
        shells = np.bincount(core)
        stats["kcore"] = {"engine": "numba", "max_core": int(core.max()),
                          "shells": {int(k): int(c) for k, c in enumerate(shells) if c}}
    if is_real_snap(cfg, name):
        stats["published"] = SNAP_PUBLISHED[name]


def write_local(cfg, name, metric, columns):
    out = node_path(cfg, name, metric)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    pq.write_table(pa.table(columns), out / "part-0.parquet")


def run(cfg):
    spark = spark_session(cfg)
    results = load_json(cfg, "mining", {})
    for name, path, kind in targets(cfg):
        stats = results.setdefault(name, {})
        if stats.get("spark_done"):
            print(f"  {name} already mined, skipping (delete data/results to start over)", flush=True)
            continue
        print(f"  mining {name}", flush=True)
        mine_spark(spark, cfg, name, path, kind, stats, lambda: save_json(cfg, "mining", results))
        stats["spark_done"] = True
        sg.clear_scratch(spark)
        save_json(cfg, "mining", results)
    stop()
    lg.warm_up()
    for name, path, kind in targets(cfg):
        if kind == "undirected" and "triangles_local" not in results[name]:
            mine_local(cfg, name, path, results[name])
            save_json(cfg, "mining", results)
