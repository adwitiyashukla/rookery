import shutil

import numpy as np
from gensim.models import Word2Vec
from pyspark.sql import functions as F

from rookery import sparkgraph as sg
from rookery.sparkctx import spark_session
from rookery.store import graph_path, load_json, results_dir, save_json, site_key, timed


def embeddings_dir(cfg):
    path = results_dir(cfg) / "embeddings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_corpus(walks, target):
    lines = (walks.groupBy("walk").agg(F.sort_array(F.collect_list(F.struct("step", "node"))).alias("seq"))
             .select(F.concat_ws(" ", F.transform("seq", lambda x: x["node"].cast("string"))).alias("value")))
    parts = target.with_name(target.name + ".parts")
    lines.write.mode("overwrite").text(str(parts))
    with open(target, "wb") as out:
        for part in sorted(parts.glob("part-*")):
            with open(part, "rb") as fh:
                shutil.copyfileobj(fh, out)
    shutil.rmtree(parts, ignore_errors=True)


def vectors_path(cfg, key, cutoff):
    return embeddings_dir(cfg) / f"{key}_{cutoff}.npz"


def load_vectors(cfg, key, cutoff):
    path = vectors_path(cfg, key, cutoff)
    if not path.exists():
        return None
    data = np.load(path)
    order = np.argsort(data["nodes"])
    return data["nodes"][order], data["vectors"][order], data["context"][order], data["counts"][order]


def run(cfg):
    spark = spark_session(cfg)
    wk, em = cfg["walks"], cfg["embed"]
    out = load_json(cfg, "embeddings", {})
    base = embeddings_dir(cfg)
    for site in cfg["fetch"]["sites"]:
        key = site_key(site)
        for cutoff in (cfg["routing"]["train_start"], cfg["routing"]["test_start"]):
            name = f"{key}_{cutoff}"
            if name in out and vectors_path(cfg, key, cutoff).exists():
                print(f"  {name} already embedded, skipping (delete data/results to start over)", flush=True)
                continue
            edges = spark.read.parquet(str(graph_path(cfg, key, "user_tag"))).filter(
                F.col("first_ts") < F.to_timestamp(F.lit(cutoff)))
            m = edges.count()
            if m == 0:
                print(f"  {key}: no user-tag edges before {cutoff}, skipping", flush=True)
                continue
            walk_dir = cfg["paths"]["data"] / "tmp" / "walks" / name
            corpus = base / f"{name}.walks.txt"
            with timed(cfg, "walks", name, m):
                walks = sg.random_walks(edges, wk["num_walks"], wk["walk_length"], wk["p"], wk["q"],
                                        wk["max_tries"], wk["seed"], True, str(walk_dir))
                write_corpus(walks, corpus)
            with timed(cfg, "word2vec", name, m):
                model = Word2Vec(corpus_file=str(corpus), vector_size=em["dim"], window=em["window"], min_count=1,
                                 sg=1, negative=em["negative"], workers=em["workers"], epochs=em["epochs"],
                                 seed=em["seed"])
            keys = model.wv.index_to_key
            nodes = np.array([int(k) for k in keys], dtype=np.int64)
            counts = np.array([model.wv.get_vecattr(k, "count") for k in keys], dtype=np.int64)
            np.savez_compressed(vectors_path(cfg, key, cutoff), nodes=nodes, counts=counts,
                                vectors=model.wv.vectors.astype(np.float32), context=model.syn1neg.astype(np.float32))
            out[name] = {"nodes": int(len(nodes)), "edges": m, "tokens": int(model.corpus_total_words),
                         "tags": int((nodes < 0).sum()), "users": int((nodes > 0).sum())}
            corpus.unlink()
            shutil.rmtree(walk_dir, ignore_errors=True)
            save_json(cfg, "embeddings", out)
            print(f"  {name}: {out[name]}", flush=True)
