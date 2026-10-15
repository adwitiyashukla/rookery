import shutil
import tempfile
import uuid
from pathlib import Path

from pyspark.sql import Window
from pyspark.sql import functions as F

BROADCAST_LIMIT = 6_000_000
SCRATCH_KEY = "spark.rookery.scratch"


class Scratch:
    def __init__(self, spark, label, keep_last=3):
        base = spark.conf.get(SCRATCH_KEY, "") or tempfile.gettempdir()
        self.root = Path(base) / f"{label}-{uuid.uuid4().hex[:10]}"
        self.spark = spark
        self.keep_last = keep_last
        self.paths = []

    def keep(self, df):
        path = self.root / f"step{len(self.paths):05d}"
        df.write.mode("overwrite").parquet(str(path))
        self.paths.append(path)
        if len(self.paths) > self.keep_last:
            shutil.rmtree(self.paths[-self.keep_last - 1], ignore_errors=True)
        return self.spark.read.parquet(str(path))


def clear_scratch(spark):
    base = spark.conf.get(SCRATCH_KEY, "")
    if base:
        shutil.rmtree(base, ignore_errors=True)


def simple_undirected(edges):
    return (edges.select(F.least("src", "dst").alias("src"), F.greatest("src", "dst").alias("dst"),
                         F.coalesce(F.col("weight"), F.lit(1.0)).alias("weight"))
            .filter(F.col("src") != F.col("dst"))
            .groupBy("src", "dst").agg(F.sum("weight").alias("weight")))


def symmetric(edges):
    return edges.select("src", "dst", "weight").unionByName(
        edges.select(F.col("dst").alias("src"), F.col("src").alias("dst"), "weight"))


def nodes_of(edges):
    return edges.select(F.col("src").alias("node")).union(edges.select(F.col("dst").alias("node"))).distinct()


def maybe_broadcast(df, rows):
    return F.broadcast(df) if rows <= BROADCAST_LIMIT else df


def degrees(edges, directed):
    if directed:
        out_deg = edges.groupBy(F.col("src").alias("node")).agg(F.count("*").alias("out_degree"))
        in_deg = edges.groupBy(F.col("dst").alias("node")).agg(F.count("*").alias("in_degree"))
        return out_deg.join(in_deg, "node", "outer").fillna(0).withColumn(
            "degree", F.col("out_degree") + F.col("in_degree"))
    sym = symmetric(simple_undirected(edges))
    return sym.groupBy(F.col("src").alias("node")).agg(F.count("*").alias("degree"),
                                                       F.sum("weight").alias("strength"))


def pagerank(edges, alpha, tol, max_iter, directed=True, teleport=None, check_every=4):
    base = edges.select("src", "dst", F.coalesce(F.col("weight"), F.lit(1.0)).alias("weight"))
    base = base.filter(F.col("src") != F.col("dst"))
    if not directed:
        base = symmetric(simple_undirected(base))
    base = base.groupBy("src", "dst").agg(F.sum("weight").alias("weight"))
    out_w = base.groupBy("src").agg(F.sum("weight").alias("out_w"))
    trans = (base.join(out_w, "src").select("src", "dst", (F.col("weight") / F.col("out_w")).alias("p"))
             .repartition("src").cache())
    nodes = nodes_of(base).join(out_w.select(F.col("src").alias("node"), F.lit(False).alias("out")), "node", "left")
    nodes = nodes.select("node", F.col("out").isNull().alias("dangling"))
    n = nodes.count()
    if teleport is None:
        state = nodes.withColumn("v", F.lit(1.0 / n))
    else:
        total = teleport.agg(F.sum("v")).first()[0]
        state = nodes.join(teleport.select("node", (F.col("v") / total).alias("v")), "node", "left").fillna(0.0)
    scratch = Scratch(edges.sparkSession, "pagerank")
    rank = scratch.keep(state.withColumn("rank", F.col("v")))
    delta, it = float("inf"), 0
    while it < max_iter:
        it += 1
        dm = rank.filter("dangling").agg(F.coalesce(F.sum("rank"), F.lit(0.0)).alias("dm"))
        src_rank = maybe_broadcast(rank.select(F.col("node").alias("src"), "rank"), n)
        contrib = (trans.join(src_rank, "src").groupBy(F.col("dst").alias("node"))
                   .agg(F.sum(F.col("p") * F.col("rank")).alias("c")))
        new = scratch.keep(rank.join(contrib, "node", "left").crossJoin(F.broadcast(dm))
                           .select("node", "dangling", "v", F.col("rank").alias("old"),
                                   (alpha * (F.coalesce(F.col("c"), F.lit(0.0)) + F.col("dm") * F.col("v"))
                                    + (1 - alpha) * F.col("v")).alias("rank")))
        rank = new.drop("old")
        if it % check_every == 0 or it == max_iter:
            delta = new.agg(F.sum(F.abs(F.col("rank") - F.col("old")))).first()[0]
            if delta < tol:
                break
    trans.unpersist()
    return rank.select("node", "rank"), {"iterations": it, "l1_delta": delta, "nodes": n}


def connected_components(edges, max_iter):
    scratch = Scratch(edges.sparkSession, "components")
    e = scratch.keep(edges.select(F.least("src", "dst").alias("u"), F.greatest("src", "dst").alias("v"))
                     .filter(F.col("u") != F.col("v")).distinct())
    rounds = 0
    while rounds < max_iter:
        rounds += 1
        sym = e.select("u", "v").union(e.select(F.col("v").alias("u"), F.col("u").alias("v")))
        low = sym.groupBy("u").agg(F.min("v").alias("mv")).select("u", F.least("u", "mv").alias("m"))
        large = (sym.join(low, "u").filter(F.col("v") > F.col("u"))
                 .select(F.least("v", "m").alias("u"), F.greatest("v", "m").alias("v"))
                 .filter(F.col("u") != F.col("v")).distinct())
        upper = large.select(F.col("v").alias("x"), F.col("u").alias("y"))
        low2 = upper.groupBy("x").agg(F.min("y").alias("m"))
        small = scratch.keep(upper.join(low2, "x").select(F.col("y").alias("a"), "m")
                             .union(low2.select(F.col("x").alias("a"), "m"))
                             .filter(F.col("a") != F.col("m"))
                             .select(F.least("a", "m").alias("u"), F.greatest("a", "m").alias("v")).distinct())
        diff = (small.withColumn("a", F.lit(1)).join(e.withColumn("b", F.lit(1)), ["u", "v"], "full_outer")
                .filter(F.col("a").isNull() | F.col("b").isNull()))
        changed = diff.limit(1).count()
        e = small
        if changed == 0:
            break
    labels = e.select(F.col("v").alias("node"), F.col("u").alias("component")).union(
        e.select(F.col("u").alias("node"), F.col("u").alias("component"))).distinct()
    return labels, {"rounds": rounds}


def kcore(edges, max_iter, check_every=4):
    sym = symmetric(simple_undirected(edges)).select("src", "dst").cache()
    scratch = Scratch(edges.sparkSession, "kcore")
    core = scratch.keep(sym.groupBy(F.col("src").alias("node")).agg(F.count("*").alias("core")))
    changed = core.select("node")
    by_value = Window.partitionBy("src").orderBy(F.desc("c"))
    it = 0
    while it < max_iter:
        it += 1
        touched = (sym.join(changed.withColumnRenamed("node", "dst"), "dst")
                   .select(F.col("src").alias("node")).distinct())
        vals = (sym.join(touched.withColumnRenamed("node", "src"), "src")
                .join(core.select(F.col("node").alias("dst"), F.col("core").alias("c")), "dst")
                .withColumn("r", F.row_number().over(by_value)).filter(F.col("c") >= F.col("r"))
                .groupBy(F.col("src").alias("node")).agg(F.max("r").alias("h")))
        upd = scratch.keep(core.join(vals, "node", "left").select(
            "node", F.least("core", F.coalesce("h", "core")).alias("core"), F.col("core").alias("old")))
        changed = upd.filter(F.col("core") < F.col("old")).select("node")
        core = upd.select("node", "core")
        if (it % check_every == 0 or it == max_iter) and changed.limit(1).count() == 0:
            break
    sym.unpersist()
    return core, {"iterations": it}


def label_propagation(edges, max_iter, self_weight, min_changed_fraction, check_every=2):
    sym = symmetric(simple_undirected(edges)).cache()
    scratch = Scratch(edges.sparkSession, "lpa")
    labels = scratch.keep(nodes_of(sym).withColumn("label", F.col("node")))
    n = labels.count()
    it, frac = 0, 1.0
    while it < max_iter:
        it += 1
        votes = (sym.join(labels.select(F.col("node").alias("dst"), "label"), "dst")
                 .select(F.col("src").alias("node"), "label", F.col("weight").alias("score")))
        own = labels.select("node", "label", F.lit(float(self_weight)).alias("score"))
        best = (votes.unionByName(own).groupBy("node", "label").agg(F.sum("score").alias("score"))
                .groupBy("node").agg(F.max_by("label", F.struct("score", F.xxhash64("label", F.lit(it)).alias("tie")))
                                     .alias("new")))
        moved = scratch.keep(labels.join(best, "node").select("node", F.col("new").alias("label"),
                                                              (F.col("new") != F.col("label")).alias("moved")))
        labels = moved.select("node", "label")
        if it % check_every == 0 or it == max_iter:
            frac = moved.filter("moved").count() / n
            if frac < min_changed_fraction:
                break
    sym.unpersist()
    return labels, {"iterations": it, "last_changed_fraction": frac}


def triangles(edges):
    e = simple_undirected(edges).select("src", "dst").cache()
    deg = symmetric(e.withColumn("weight", F.lit(1.0))).groupBy(F.col("src").alias("node")).agg(
        F.count("*").alias("d"))
    forward = (F.col("ds") < F.col("dd")) | ((F.col("ds") == F.col("dd")) & (F.col("src") < F.col("dst")))
    oriented = (e.join(deg.select(F.col("node").alias("src"), F.col("d").alias("ds")), "src")
                .join(deg.select(F.col("node").alias("dst"), F.col("d").alias("dd")), "dst")
                .select(F.when(forward, F.col("src")).otherwise(F.col("dst")).alias("s"),
                        F.when(forward, F.col("dst")).otherwise(F.col("src")).alias("t"))
                .cache())
    x, y = oriented.alias("x"), oriented.alias("y")
    wedges = x.join(y, (F.col("x.s") == F.col("y.s")) & (F.col("x.t") < F.col("y.t"))).select(
        F.col("x.s").alias("a"), F.col("x.t").alias("b"), F.col("y.t").alias("c"))
    tri = wedges.join(e, (F.col("src") == F.least("b", "c")) & (F.col("dst") == F.greatest("b", "c"))).select(
        "a", "b", "c").cache()
    total = tri.count()
    per_node = (tri.select(F.explode(F.array("a", "b", "c")).alias("node")).groupBy("node")
                .agg(F.count("*").alias("triangles")))
    counts = deg.join(per_node, "node", "left").select("node", F.col("d").alias("degree"),
                                                       F.coalesce("triangles", F.lit(0)).alias("triangles"))
    counts = Scratch(edges.sparkSession, "triangles").keep(counts)
    tri.unpersist()
    oriented.unpersist()
    e.unpersist()
    return counts, {"triangles": total}


def unit(*cols):
    return F.xxhash64(*cols).bitwiseAND(F.lit(2147483647)).cast("double") / 2147483648.0


def random_walks(edges, num_walks, walk_length, p, q, max_tries, seed, bipartite, out_path):
    spark = edges.sparkSession
    sym = symmetric(simple_undirected(edges)).select("src", "dst").cache()
    adj = sym.groupBy(F.col("src").alias("cur")).agg(F.collect_list("dst").alias("nbrs")).cache()
    reps = spark.range(num_walks).withColumnRenamed("id", "r")
    start = (adj.select("cur").crossJoin(F.broadcast(reps))
             .select((F.col("cur") * num_walks + F.col("r")).alias("walk"), F.col("cur").alias("node"),
                     F.lit(None).cast("long").alias("prev")))
    start.write.mode("overwrite").parquet(f"{out_path}/step=0")
    walkers = spark.read.parquet(f"{out_path}/step=0").withColumnRenamed("node", "cur")
    top = max(1.0 / p, 1.0, 1.0 / q)
    biased = not (p == 1.0 and q == 1.0)
    for step in range(1, walk_length):
        pending, done = walkers, []
        for attempt in range(max_tries + 1):
            salt = F.lit(seed * 1_000_003 + step * 1_009 + attempt * 2)
            pick = (F.floor(unit("walk", salt) * F.size("nbrs")) + 1).cast("int")
            cand = pending.join(adj, "cur").select("walk", "cur", "prev", F.element_at("nbrs", pick).alias("nxt"))
            if not biased or attempt == max_tries:
                done.append(cand)
                break
            if bipartite:
                near = F.lit(False)
            else:
                cand = cand.join(sym.select(F.col("src").alias("prev"), F.col("dst").alias("nxt"),
                                            F.lit(True).alias("is_nb")), ["prev", "nxt"], "left")
                near = F.coalesce(F.col("is_nb"), F.lit(False))
            weight = (F.when(F.col("prev").isNull(), F.lit(top))
                      .when(F.col("nxt") == F.col("prev"), F.lit(1.0 / p))
                      .when(near, F.lit(1.0)).otherwise(F.lit(1.0 / q)))
            cand = cand.withColumn("ok", unit("walk", salt + 1) * top < weight)
            done.append(cand.filter("ok").select("walk", "cur", "prev", "nxt"))
            pending = cand.filter(~F.col("ok")).select("walk", "cur", "prev")
        moved = done[0]
        for extra in done[1:]:
            moved = moved.unionByName(extra)
        moved.select("walk", F.col("nxt").alias("node"), F.col("cur").alias("prev")).write.mode("overwrite").parquet(
            f"{out_path}/step={step}")
        walkers = spark.read.parquet(f"{out_path}/step={step}").withColumnRenamed("node", "cur")
    sym.unpersist()
    adj.unpersist()
    return spark.read.parquet(str(out_path)).select("walk", F.col("step").cast("int").alias("step"), "node")
