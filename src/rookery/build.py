from pyspark.sql import Window
from pyspark.sql import functions as F

from rookery.sparkctx import spark_session
from rookery.store import graph_path, lake, save_json, site_key, timed, update_json


def table(spark, cfg, name, key):
    return spark.read.parquet(str(lake(cfg, name))).filter(F.col("site") == key).drop("site")


def directed(pairs, min_weight):
    return (pairs.groupBy("src", "dst")
            .agg(F.count("*").cast("double").alias("weight"), F.min("ts").alias("first_ts"),
                 F.max("ts").alias("last_ts"))
            .filter(F.col("weight") >= min_weight))


def undirected(edges):
    return (edges.select(F.least("src", "dst").alias("src"), F.greatest("src", "dst").alias("dst"), "weight",
                         "first_ts", "last_ts")
            .groupBy("src", "dst")
            .agg(F.sum("weight").alias("weight"), F.min("first_ts").alias("first_ts"),
                 F.max("last_ts").alias("last_ts")))


def save(df, cfg, name, kind):
    path = graph_path(cfg, name, kind)
    df.write.mode("overwrite").parquet(str(path))
    return df.sparkSession.read.parquet(str(path))


def growth(edges):
    month = F.date_format("first_ts", "yyyy-MM")
    edge_months = edges.groupBy(month.alias("month")).agg(F.count("*").alias("new_edges"))
    ends = edges.select(F.col("src").alias("node"), "first_ts").union(edges.select("dst", "first_ts"))
    node_months = ends.groupBy("node").agg(F.min("first_ts").alias("first_ts")).groupBy(month.alias("month")).agg(
        F.count("*").alias("new_nodes"))
    rows = edge_months.join(node_months, "month", "outer").fillna(0).orderBy("month").collect()
    out, n_tot, e_tot = [], 0, 0
    for r in rows:
        n_tot += r["new_nodes"]
        e_tot += r["new_edges"]
        out.append({"month": r["month"], "nodes": n_tot, "edges": e_tot})
    return out


def build_site(spark, cfg, site):
    key = site_key(site)
    min_w = cfg["graphs"]["min_weight"]
    posts = table(spark, cfg, "posts", key)
    comments = table(spark, cfg, "comments", key)
    links = table(spark, cfg, "postlinks", key)
    questions = posts.filter(F.col("post_type") == 1).select(
        F.col("id").alias("qid"), F.col("owner").alias("asker"), "tags", "initial_tags")
    answers = posts.filter(F.col("post_type") == 2).select(
        F.col("parent_id").alias("qid"), F.col("owner").alias("answerer"), F.col("created").alias("ts"))
    qa = answers.join(questions, "qid")
    ans_pairs = qa.filter((F.col("answerer") > 0) & (F.col("asker") > 0) & (F.col("answerer") != F.col("asker")))
    ans = save(directed(ans_pairs.select(F.col("answerer").alias("src"), F.col("asker").alias("dst"), "ts"), min_w),
               cfg, key, "answers")
    owners = posts.select(F.col("id").alias("post_id"), F.col("owner").alias("post_owner"))
    com_pairs = (comments.join(owners, "post_id")
                 .filter((F.col("user_id") > 0) & (F.col("post_owner") > 0) & (F.col("user_id") != F.col("post_owner")))
                 .select(F.col("user_id").alias("src"), F.col("post_owner").alias("dst"), F.col("created").alias("ts")))
    com = save(directed(com_pairs, min_w), cfg, key, "comments")
    inter = save(undirected(ans.unionByName(com)), cfg, key, "interaction")
    final = questions.select("qid", F.explode("tags").alias("tag"))
    first = questions.select("qid", F.explode("initial_tags").alias("tag"))
    tag_names = (final.union(first).distinct().groupBy("tag").agg(F.count("*").alias("questions"))
                 .withColumn("tag_id", F.row_number().over(Window.orderBy(F.desc("questions"), "tag")).cast("long")))
    tag_names = save(tag_names, cfg, key, "tags")
    qtags = final.join(tag_names.select("tag", "tag_id"), "tag")
    ut_pairs = (answers.filter(F.col("answerer") > 0).join(first.join(tag_names.select("tag", "tag_id"), "tag"), "qid")
                .select(F.col("answerer").alias("src"), (-F.col("tag_id")).alias("dst"), "ts"))
    user_tag = save(directed(ut_pairs, 1), cfg, key, "user_tag")
    a, b = qtags.alias("a"), qtags.alias("b")
    cooc = (a.join(b, (F.col("a.qid") == F.col("b.qid")) & (F.col("a.tag_id") < F.col("b.tag_id")))
            .groupBy(F.col("a.tag_id").alias("src"), F.col("b.tag_id").alias("dst"))
            .agg(F.count("*").cast("double").alias("weight")))
    cooc = save(cooc, cfg, key, "tag_cooc")
    qids = questions.select("qid")
    plinks = (links.filter(F.col("link_type").isin(1, 3))
              .join(qids.withColumnRenamed("qid", "post_id"), "post_id")
              .join(qids.withColumnRenamed("qid", "related_post_id"), "related_post_id")
              .filter(F.col("post_id") != F.col("related_post_id"))
              .select(F.col("post_id").alias("src"), F.col("related_post_id").alias("dst"), "link_type")
              .dropDuplicates(["src", "dst"]).withColumn("weight", F.lit(1.0)))
    plinks = save(plinks, cfg, key, "post_links")
    sizes = {kind: df.count() for kind, df in
             [("answers", ans), ("comments", com), ("interaction", inter), ("user_tag", user_tag),
              ("tag_cooc", cooc), ("post_links", plinks), ("tags", tag_names)]}
    update_json(cfg, "graphs", key, sizes)
    update_json(cfg, "growth", key, growth(inter))
    print(f"  {key}: " + ", ".join(f"{k} {v:,}" for k, v in sizes.items()), flush=True)


def build_network(spark, cfg):
    keys = [site_key(s) for s in cfg["fetch"]["sites"]]
    users = spark.read.parquet(str(lake(cfg, "users"))).filter(F.col("account_id").isNotNull()).select(
        "site", F.col("id").alias("uid"), "account_id")
    parts = []
    for key in keys:
        inter = spark.read.parquet(str(graph_path(cfg, key, "interaction"))).withColumn("site", F.lit(key))
        parts.append(inter)
    edges = parts[0]
    for p in parts[1:]:
        edges = edges.unionByName(p)
    u1 = users.withColumnRenamed("uid", "src").withColumnRenamed("account_id", "a_src")
    u2 = users.withColumnRenamed("uid", "dst").withColumnRenamed("account_id", "a_dst")
    mapped = (edges.join(u1, ["site", "src"]).join(u2, ["site", "dst"])
              .select(F.col("a_src").alias("src"), F.col("a_dst").alias("dst"), "weight", "first_ts", "last_ts")
              .filter(F.col("src") != F.col("dst")))
    net = save(undirected(mapped), cfg, "network", "interaction")
    posts = spark.read.parquet(str(lake(cfg, "posts"))).select("site", F.col("owner").alias("uid"))
    coms = spark.read.parquet(str(lake(cfg, "comments"))).select("site", F.col("user_id").alias("uid"))
    active = (posts.union(coms).filter(F.col("uid") > 0).distinct().join(users, ["site", "uid"])
              .select("account_id", "site").distinct()).cache()
    x, y = active.alias("x"), active.alias("y")
    pairs = (x.join(y, F.col("x.account_id") == F.col("y.account_id"))
             .groupBy(F.col("x.site").alias("a"), F.col("y.site").alias("b")).count().collect())
    per_account = active.groupBy("account_id").agg(F.count("*").alias("k")).groupBy("k").count().collect()
    save_json(cfg, "cross_site", {
        "sites": keys,
        "shared": {f"{r['a']}|{r['b']}": r["count"] for r in pairs},
        "sites_per_account": {int(r["k"]): r["count"] for r in per_account},
        "network_edges": net.count(),
    })
    active.unpersist()


def run(cfg):
    spark = spark_session(cfg)
    for site in cfg["fetch"]["sites"]:
        with timed(cfg, "build", site_key(site)):
            build_site(spark, cfg, site)
    with timed(cfg, "build", "network"):
        build_network(spark, cfg)
