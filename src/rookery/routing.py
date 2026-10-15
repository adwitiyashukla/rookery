import igraph as ig
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pyspark.sql import functions as F

from rookery import localgraph as lg
from rookery.embed import load_vectors
from rookery.sparkctx import spark_session, stop
from rookery.store import graph_path, lake, save_json, site_key, timed

FEATURES = [
    "ppr", "ppr_rank", "tag_answers", "tag_max", "tag_share", "tag_decay", "decay_activity", "answers_30d",
    "answers_90d", "answers_365d", "answers_window", "days_since_last", "hour_share", "expertise_pr", "coreness",
    "asker_cn", "asker_aa", "asker_direct", "emb_cos", "emb_link", "q_tags", "asker_questions",
]
BASELINES = {"recent_activity": "answers_90d", "tag_activity": "tag_answers", "tag_ppr": "ppr", "embedding": "emb_link"}


def load_site(spark, cfg, key):
    ro = cfg["routing"]
    start = (pd.Timestamp(ro["train_start"]) - pd.Timedelta(days=ro["history_days"])).strftime("%Y-%m-%d")
    lo, hi = F.to_timestamp(F.lit(start)), F.to_timestamp(F.lit(ro["test_end"]))
    posts = spark.read.parquet(str(lake(cfg, "posts"))).filter(F.col("site") == key)
    owner = F.coalesce(F.col("owner"), F.lit(-1))
    q = posts.filter("post_type = 1").select(F.col("id").alias("qid"), owner.alias("asker"),
                                             F.col("created").alias("q_time"), F.col("initial_tags").alias("tags"))
    a = (posts.filter("post_type = 2").filter((F.col("created") >= lo) & (F.col("created") < hi))
         .select(F.col("parent_id").alias("qid"), owner.alias("user"), F.col("created").alias("a_time")))
    answers = a.join(q, "qid").toPandas()
    questions = q.filter((F.col("q_time") >= lo) & (F.col("q_time") < hi)).toPandas()
    tags = spark.read.parquet(str(graph_path(cfg, key, "tags"))).select("tag", "tag_id").toPandas()
    inter = spark.read.parquet(str(graph_path(cfg, key, "interaction"))).select("src", "dst", "first_ts").toPandas()
    return answers, questions, tags, inter


def tag_ids(tag_lists, lookup):
    return [np.array([lookup[t] for t in tags if t in lookup], dtype=np.int64) for tags in tag_lists]


def pagerank_scipy(src, dst, weight, n, alpha=0.85, iters=60):
    m = sp.csr_matrix((weight, (src, dst)), shape=(n, n))
    out = np.asarray(m.sum(axis=1)).ravel()
    inv = np.divide(1.0, out, out=np.zeros(n), where=out > 0)
    p = sp.diags(inv) @ m
    pt = p.T.tocsr()
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        dang = r[out == 0].sum()
        r = alpha * (pt @ r + dang / n) + (1 - alpha) / n
    return r


class Snapshot:
    def __init__(self, cfg, cutoff, answers, questions, lookup, inter, vectors):
        ro = cfg["routing"]
        self.cutoff = cutoff
        win = answers[(answers["a_time"] >= cutoff - pd.Timedelta(days=ro["history_days"]))
                      & (answers["a_time"] < cutoff) & (answers["user"] > 0)].reset_index(drop=True)
        self.users = np.unique(win["user"].to_numpy())
        nu = len(self.users)
        ui = np.searchsorted(self.users, win["user"].to_numpy())
        age = (cutoff - win["a_time"]).dt.total_seconds().to_numpy() / 86400.0
        self.answers_window = np.bincount(ui, minlength=nu).astype(np.float64)
        self.answers_30d = np.bincount(ui[age <= 30], minlength=nu).astype(np.float64)
        self.answers_90d = np.bincount(ui[age <= 90], minlength=nu).astype(np.float64)
        self.answers_365d = np.bincount(ui[age <= 365], minlength=nu).astype(np.float64)
        last = np.full(nu, np.inf)
        np.minimum.at(last, ui, age)
        self.days_since_last = last
        hours = win["a_time"].dt.hour.to_numpy()
        self.hours = sp.csr_matrix((np.ones(len(ui)), (ui, hours)), shape=(nu, 24)).toarray()
        tids = tag_ids(win["tags"], lookup)
        rows = np.repeat(ui, [len(t) for t in tids])
        cols = np.concatenate(tids) if len(tids) else np.zeros(0, dtype=np.int64)
        self.n_tags = max(lookup.values(), default=0) + 1
        self.ut = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nu, self.n_tags))
        self.ut.sum_duplicates()
        decay = np.exp(-age / ro["decay_days"])
        self.decay_activity = np.bincount(ui, weights=decay, minlength=nu)
        weights = np.repeat(decay, [len(t) for t in tids])
        self.ut_decay = sp.csr_matrix((weights, (rows, cols)), shape=(nu, self.n_tags))
        self.ut_decay.sum_duplicates()
        self.tag_top = {}
        utc = self.ut.tocsc()
        for t in range(self.n_tags):
            col = utc.getcol(t)
            if col.nnz:
                order = np.argsort(-col.data, kind="stable")[: ro["candidates"]]
                self.tag_top[t] = col.indices[order]
        self.global_top = np.argsort(-self.answers_90d, kind="stable")[: ro["candidates"]]
        coo = self.ut.tocoo()
        self.ppr_graph = lg.from_arrays(coo.row.astype(np.int64), (nu + coo.col).astype(np.int64), coo.data)
        self.ppr = lg.PushPPR(self.ppr_graph, ro["ppr_alpha"], ro["ppr_eps"])
        asker = win["asker"].to_numpy()
        ok = (asker > 0) & (asker != win["user"].to_numpy())
        nodes = np.unique(np.concatenate([self.users, asker[ok]]))
        pr = pagerank_scipy(np.searchsorted(nodes, asker[ok]), np.searchsorted(nodes, win["user"].to_numpy()[ok]),
                            np.ones(ok.sum()), len(nodes))
        self.expertise_pr = pr[np.searchsorted(nodes, self.users)] * len(nodes)
        snap = inter[inter["first_ts"] < cutoff]
        self.int_nodes = np.unique(np.concatenate([snap["src"].to_numpy(), snap["dst"].to_numpy()]))
        a = np.searchsorted(self.int_nodes, snap["src"].to_numpy())
        b = np.searchsorted(self.int_nodes, snap["dst"].to_numpy())
        ni = len(self.int_nodes)
        adj = sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(ni, ni))
        self.adj = ((adj + adj.T) > 0).astype(np.float64).tocsr()
        deg = np.asarray(self.adj.sum(axis=1)).ravel()
        self.inv_log = np.divide(1.0, np.log(deg), out=np.zeros(ni), where=deg > 1)
        self.coreness = np.zeros(nu)
        if ni:
            h = ig.Graph(n=ni)
            h.add_edges(np.column_stack([a, b]))
            core = np.asarray(h.simplify().coreness(), dtype=np.float64)
            pos = np.minimum(np.searchsorted(self.int_nodes, self.users), ni - 1)
            self.coreness = np.where(self.int_nodes[pos] == self.users, core[pos], 0.0)
        since = cutoff - pd.Timedelta(days=ro["history_days"])
        recent = (questions["q_time"] >= since) & (questions["q_time"] < cutoff)
        prior = questions[recent & (questions["asker"] > 0)]["asker"].to_numpy()
        self.asker_counts = pd.Series(prior).value_counts()
        self.vectors = vectors
        self.lookup = lookup

    def user_index(self, user_ids):
        pos = np.searchsorted(self.users, user_ids)
        ok = (pos < len(self.users)) & (self.users[np.minimum(pos, len(self.users) - 1)] == user_ids)
        return np.where(ok, pos, -1)

    def candidates(self, qtags, asker_idx, limit):
        seeds = np.searchsorted(self.ppr_graph.ids, len(self.users) + qtags)
        seeds = seeds[(seeds < self.ppr_graph.n) & (self.ppr_graph.ids[np.minimum(seeds, self.ppr_graph.n - 1)]
                                                    == len(self.users) + qtags)]
        ppr_scores = np.zeros(len(self.users))
        if len(seeds):
            nodes, vals = self.ppr.run(seeds, np.ones(len(seeds)))
            ids = self.ppr_graph.ids[nodes]
            users = ids < len(self.users)
            ppr_scores[ids[users]] = vals[users]
        positive = np.flatnonzero(ppr_scores > 0)
        if len(positive) > limit:
            positive = positive[np.argpartition(-ppr_scores[positive], limit)[:limit]]
        by_ppr = positive[np.argsort(-ppr_scores[positive], kind="stable")]
        pools = [by_ppr] + [self.tag_top[t] for t in qtags if t in self.tag_top] + [self.global_top]
        merged = pd.unique(np.concatenate(pools))
        merged = merged[merged != asker_idx][:limit]
        return merged, ppr_scores

    def features(self, cand, ppr_scores, qtags, q_time, asker):
        n = len(cand)
        f = {}
        f["ppr"] = ppr_scores[cand]
        ordered = np.sort(ppr_scores[ppr_scores > 0])
        above = len(ordered) - np.searchsorted(ordered, f["ppr"], side="right")
        f["ppr_rank"] = np.where(f["ppr"] > 0, above + 1.0, np.nan)
        sub = self.ut[cand][:, qtags] if len(qtags) else sp.csr_matrix((n, 1))
        f["tag_answers"] = np.asarray(sub.sum(axis=1)).ravel()
        f["tag_max"] = np.asarray(sub.max(axis=1).todense()).ravel() if sub.shape[1] else np.zeros(n)
        f["tag_share"] = f["tag_answers"] / np.maximum(self.answers_window[cand], 1)
        decay_sub = self.ut_decay[cand][:, qtags] if len(qtags) else sp.csr_matrix((n, 1))
        f["tag_decay"] = np.asarray(decay_sub.sum(axis=1)).ravel()
        f["decay_activity"] = self.decay_activity[cand]
        for name in ("answers_30d", "answers_90d", "answers_365d", "answers_window", "expertise_pr", "coreness"):
            f[name] = getattr(self, name)[cand]
        gap = (q_time - self.cutoff).total_seconds() / 86400.0
        f["days_since_last"] = self.days_since_last[cand] + gap
        h = q_time.hour
        near = self.hours[cand][:, [(h - 1) % 24, h, (h + 1) % 24]].sum(axis=1)
        f["hour_share"] = near / np.maximum(self.answers_window[cand], 1)
        f["asker_cn"], f["asker_aa"], f["asker_direct"] = (np.zeros(n) for _ in range(3))
        ni = len(self.int_nodes)
        pa = np.searchsorted(self.int_nodes, asker) if ni else 0
        if ni and pa < ni and self.int_nodes[pa] == asker:
            pos = np.searchsorted(self.int_nodes, self.users[cand])
            ok = (pos < ni) & (self.int_nodes[np.minimum(pos, ni - 1)] == self.users[cand])
            if ok.any():
                rows = self.adj[pos[ok]]
                arow = self.adj[pa]
                f["asker_cn"][ok] = np.asarray((rows @ arow.T).todense()).ravel()
                f["asker_aa"][ok] = np.asarray((rows @ arow.multiply(self.inv_log).T).todense()).ravel()
                f["asker_direct"][ok] = np.asarray(arow[:, pos[ok]].todense()).ravel()
        f["emb_cos"], f["emb_link"] = np.full(n, np.nan), np.full(n, np.nan)
        if self.vectors is not None and len(qtags):
            nodes, vecs, ctx, counts = self.vectors
            tpos = np.searchsorted(nodes, -qtags)
            tok = (tpos < len(nodes)) & (nodes[np.minimum(tpos, len(nodes) - 1)] == -qtags)
            upos = np.searchsorted(nodes, self.users[cand])
            uok = (upos < len(nodes)) & (nodes[np.minimum(upos, len(nodes) - 1)] == self.users[cand])
            if tok.any() and uok.any():
                center = vecs[tpos[tok]].mean(axis=0)
                uv = vecs[upos[uok]]
                f["emb_cos"][uok] = uv @ center / (np.linalg.norm(uv, axis=1) * np.linalg.norm(center) + 1e-12)
                f["emb_link"][uok] = (uv @ ctx[tpos[tok]].T).mean(axis=1) + np.log(counts[upos[uok]])
        f["q_tags"] = np.full(n, len(qtags), dtype=np.float64)
        f["asker_questions"] = np.full(n, float(self.asker_counts.get(asker, 0)))
        return f


def build_rows(snapshot, questions, answers, lo, hi, limit, cap, seed):
    window = questions[(questions["q_time"] >= lo) & (questions["q_time"] < hi)]
    helped = answers[(answers["user"] > 0) & (answers["user"] != answers["asker"]) & (answers["a_time"] < hi)]
    who = helped.groupby("qid")["user"].apply(lambda s: np.unique(s.to_numpy()))
    window = window[window["qid"].isin(who.index)]
    known = window["qid"].map(lambda q: bool((snapshot.user_index(who[q]) >= 0).any()))
    eligible = window[known.to_numpy(dtype=bool)]
    sample = eligible.sample(n=min(cap, len(eligible)), random_state=seed).sort_values("q_time")
    frames, stats = [], {"answered": int(len(window)), "eligible": int(len(eligible)), "sampled": int(len(sample)),
                         "with_positive": 0}
    for row in sample.itertuples(index=False):
        qtags = np.array([snapshot.lookup[t] for t in row.tags if t in snapshot.lookup], dtype=np.int64)
        asker = int(row.asker)
        aidx = snapshot.user_index(np.array([asker]))[0]
        cand, ppr_scores = snapshot.candidates(qtags, aidx, limit)
        if len(cand) == 0:
            continue
        feats = snapshot.features(cand, ppr_scores, qtags, row.q_time, asker)
        label = np.isin(snapshot.users[cand], who[row.qid]).astype(np.int8)
        stats["with_positive"] += int(label.any())
        frame = pd.DataFrame(feats)
        frame["label"] = label
        frame["qid"] = row.qid
        frame["q_time"] = row.q_time
        frame["order"] = np.arange(len(cand))
        frames.append(frame)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), stats


def rank_metrics(df, score_col, ks, seed=0):
    out = {f"hit@{k}": [] for k in ks}
    out["mrr"], out["ndcg@10"] = [], []
    df = df.assign(tie=np.random.default_rng(seed).random(len(df)))
    for _, g in df.groupby("qid", sort=False):
        order = np.lexsort((g["tie"].to_numpy(), -g[score_col].to_numpy()))
        labels = g["label"].to_numpy()[order]
        hits = np.flatnonzero(labels)
        first = hits[0] + 1 if len(hits) else None
        for k in ks:
            out[f"hit@{k}"].append(1.0 if first is not None and first <= k else 0.0)
        out["mrr"].append(1.0 / first if first else 0.0)
        gains = 1.0 / np.log2(np.arange(2, 12))
        dcg = float((labels[:10] * gains[: min(10, len(labels))]).sum())
        ideal = float(gains[: min(10, int(labels.sum()))].sum())
        out["ndcg@10"].append(dcg / ideal if ideal else 0.0)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in out.items()}


def train_ranker(train, ro):
    train = train.sort_values(["q_time", "qid", "order"])
    qids = train["qid"].drop_duplicates().to_numpy()
    cut = int(len(qids) * 0.8) if len(qids) > 5 else len(qids)
    fit = train[train["qid"].isin(qids[:cut])]
    val = train[train["qid"].isin(qids[cut:])]
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10], "learning_rate": ro["learning_rate"],
              "num_leaves": ro["num_leaves"], "min_data_in_leaf": 20, "bagging_fraction": 0.8, "bagging_freq": 1,
              "feature_fraction": 0.9, "seed": ro["seed"], "deterministic": True, "verbosity": -1}
    fit_set = lgb.Dataset(fit[FEATURES], fit["label"], group=fit.groupby("qid", sort=False).size().to_numpy())
    kwargs = {}
    if len(val):
        val_set = lgb.Dataset(val[FEATURES], val["label"], group=val.groupby("qid", sort=False).size().to_numpy(),
                              reference=fit_set)
        kwargs = {"valid_sets": [val_set], "callbacks": [lgb.early_stopping(50, verbose=False)]}
    return lgb.train(params, fit_set, num_boost_round=ro["n_estimators"], **kwargs)


def run(cfg):
    ro = cfg["routing"]
    spark = spark_session(cfg)
    loaded = {}
    for site in cfg["fetch"]["sites"]:
        key = site_key(site)
        loaded[key] = load_site(spark, cfg, key)
    stop()
    t1, t2, t3 = (pd.Timestamp(ro[k]) for k in ("train_start", "test_start", "test_end"))
    report = {"sites": {}, "features": FEATURES}
    tests, trains = [], []
    for key, (answers, questions, tags, inter) in loaded.items():
        lookup = dict(zip(tags["tag"], tags["tag_id"].astype(int), strict=True))
        with timed(cfg, "routing_features", key):
            snap1 = Snapshot(cfg, t1, answers, questions, lookup, inter, load_vectors(cfg, key, ro["train_start"]))
            train, s1 = build_rows(snap1, questions, answers, t1, t2, ro["candidates"], ro["max_train_questions"],
                                   ro["seed"])
            del snap1
            snap2 = Snapshot(cfg, t2, answers, questions, lookup, inter, load_vectors(cfg, key, ro["test_start"]))
            test, s2 = build_rows(snap2, questions, answers, t2, t3, ro["candidates"], ro["max_test_questions"],
                                  ro["seed"])
            del snap2
        report["sites"][key] = {"train": s1, "test": s2}
        if len(train) and len(test):
            train["site"], test["site"] = key, key
            trains.append(train)
            tests.append(test)
        print(f"  {key}: train {s1}, test {s2}", flush=True)
    if not trains:
        save_json(cfg, "routing", report)
        return
    train, test = pd.concat(trains, ignore_index=True), pd.concat(tests, ignore_index=True)
    train["qid"] = train["site"] + ":" + train["qid"].astype(str)
    test["qid"] = test["site"] + ":" + test["qid"].astype(str)
    with timed(cfg, "routing_train", "all", len(train)):
        model = train_ranker(train, ro)
    test["model"] = model.predict(test[FEATURES])
    ks = ro["k_values"]
    methods = {"lightgbm_ranker": "model", **BASELINES}
    report["pooled"] = {m: rank_metrics(test, col, ks, ro["seed"]) for m, col in methods.items()}
    for key in report["sites"]:
        part = test[test["site"] == key]
        if len(part):
            report["sites"][key]["metrics"] = {m: rank_metrics(part, col, ks, ro["seed"]) for m, col in methods.items()}
    report["candidate_recall"] = float(test.groupby("qid")["label"].max().mean())
    tested = [s["test"] for s in report["sites"].values()]
    report["eligible_share"] = sum(s["eligible"] for s in tested) / max(sum(s["answered"] for s in tested), 1)
    report["rows"] = {"train": int(len(train)), "test": int(len(test))}
    report["best_iteration"] = int(model.best_iteration or ro["n_estimators"])
    gain = model.feature_importance(importance_type="gain")
    report["feature_gain"] = dict(zip(FEATURES, (gain / gain.sum()).round(5).tolist(), strict=True))
    save_json(cfg, "routing", report)
    print(f"  pooled: { {m: round(v['hit@10'], 3) for m, v in report['pooled'].items()} }", flush=True)
