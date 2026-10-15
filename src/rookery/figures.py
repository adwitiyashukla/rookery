import random

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

from rookery.embed import load_vectors
from rookery.mine import node_path
from rookery.store import graph_path, lake, load_json, save_json, site_key

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95",
                                                    "#0d366b"])
OTHER = "#c3c2b7"
FEATURE_NAMES = {
    "ppr": "tag PPR score", "ppr_rank": "tag PPR rank", "tag_answers": "answers in the question's tags",
    "tag_max": "answers in the strongest tag", "tag_share": "share of answers in these tags",
    "tag_decay": "recent answers in these tags", "decay_activity": "recent answers overall",
    "answers_30d": "answers last 30 days", "answers_90d": "answers last 90 days",
    "answers_365d": "answers last year", "answers_window": "answers in history window",
    "days_since_last": "days since last answer", "hour_share": "active at this hour",
    "expertise_pr": "expertise PageRank", "coreness": "k-core number",
    "asker_cn": "common neighbors with asker", "asker_aa": "Adamic-Adar with asker",
    "asker_direct": "talked to asker before", "emb_cos": "node2vec cosine", "emb_link": "node2vec link score",
    "q_tags": "tags on question",
    "asker_questions": "asker's past questions",
}


def style(dpi):
    plt.switch_backend("Agg")
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 9, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED,
        "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "legend.frameon": False, "legend.labelcolor": INK2, "lines.linewidth": 2, "font.family": "DejaVu Sans",
        "figure.dpi": dpi, "savefig.dpi": dpi, "text.color": INK, "axes.axisbelow": True,
    })


def save(fig, cfg, name):
    out = cfg["paths"]["figures"]
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  wrote {name}.png", flush=True)


def undirected_graphs(cfg):
    return list(cfg["fetch"]["snap_graphs"]) + [site_key(s) for s in cfg["fetch"]["sites"]]


def degree_ccdf(cfg, mining):
    names = [n for n in undirected_graphs(cfg) if n in mining]
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10.5, 3.0 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names, strict=False):
        m = mining[name]
        x, p = np.array(m["ccdf"]["x"]), np.array(m["ccdf"]["p"])
        keep = p > 0
        ax.loglog(x[keep], p[keep], "o", ms=3.5, color=SERIES[0], label="observed")
        fit = m["powerlaw"]
        if fit["alpha"]:
            xmin = fit["xmin"]
            at = np.interp(np.log(xmin), np.log(x[keep]), np.log(p[keep]))
            xs = np.logspace(np.log10(xmin), np.log10(x[keep].max()), 30)
            ax.loglog(xs, np.exp(at) * (xs / xmin) ** (1 - fit["alpha"]), color=SERIES[1], lw=1.6,
                      label="power law fit")
            ax.text(0.97, 0.95, f"alpha {fit['alpha']:.2f}\nxmin {int(xmin)}", transform=ax.transAxes, ha="right",
                    va="top", fontsize=8, color=INK2)
        ax.set_title(f"{name}  ({m['nodes']:,} nodes)", fontsize=9.5)
        ax.set_xlabel("degree k")
        ax.set_ylabel("P(degree >= k)")
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    axes.flat[0].legend(loc="lower left")
    fig.suptitle("Degree distributions are heavy tailed on every graph", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, cfg, "degree_ccdf")


def kcore_shells(cfg, mining):
    groups = [("SNAP networks", list(cfg["fetch"]["snap_graphs"])),
              ("Stack Exchange interaction graphs", [site_key(s) for s in cfg["fetch"]["sites"]])]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, (title, names) in zip(axes, groups, strict=True):
        for i, name in enumerate(n for n in names if n in mining and "kcore" in mining[n]):
            shells = {int(k): v for k, v in mining[name]["kcore"]["shells"].items()}
            ks = np.arange(1, max(shells) + 1)
            ax.semilogy(ks, [shells.get(k, np.nan) for k in ks], color=SERIES[i % 8], lw=1.6, marker="o", ms=2.5,
                        label=f"{name} (max core {max(shells)})")
        ax.set_title(title)
        ax.set_xlabel("k-shell")
        ax.set_ylabel("nodes in shell")
        ax.legend()
    fig.tight_layout()
    save(fig, cfg, "kcore_shells")


def graph_growth(cfg, growth):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, site in enumerate(cfg["fetch"]["sites"]):
        key = site_key(site)
        rows = growth.get(key, [])
        if len(rows) < 3:
            continue
        n = np.array([r["nodes"] for r in rows], dtype=float)
        e = np.array([r["edges"] for r in rows], dtype=float)
        ok = (n > 0) & (e > 0)
        slope = np.polyfit(np.log(n[ok]), np.log(e[ok]), 1)[0]
        ax.loglog(n[ok], e[ok], color=SERIES[i % 8], lw=1.8, label=f"{key}  exponent {slope:.2f}")
    ax.set_xlabel("nodes so far")
    ax.set_ylabel("edges so far")
    ax.set_title("Densification: edges grow faster than nodes")
    ax.legend()
    save(fig, cfg, "graph_growth")


def pagerank_vs_reputation(cfg, mining):
    sizes = {site_key(s): mining.get(site_key(s), {}).get("edges", 0) for s in cfg["fetch"]["sites"]}
    key = max(sizes, key=sizes.get)
    pr = pq.read_table(str(node_path(cfg, f"{key}_answers", "pagerank"))).to_pandas()
    deg = pq.read_table(str(node_path(cfg, f"{key}_answers", "degree")), columns=["node", "in_degree"]).to_pandas()
    users = pq.read_table(str(lake(cfg, "users")), filters=[("site", "=", key)], columns=["id", "reputation"])
    users = users.to_pandas().rename(columns={"id": "node"})
    df = pr.merge(deg, on="node").merge(users, on="node")
    df = df[(df["reputation"] > 0) & (df["in_degree"] > 0)]
    rho_pr = spearmanr(df["rank"], df["reputation"]).statistic
    rho_deg = spearmanr(df["in_degree"], df["reputation"]).statistic
    top = df.nlargest(1000, "rank")
    rho_top = spearmanr(top["rank"], top["reputation"]).statistic
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    hb = ax.hexbin(df["rank"] * len(pr), df["reputation"], xscale="log", yscale="log", gridsize=45, cmap=BLUES,
                   mincnt=1, bins="log", linewidths=0.2, edgecolors=SURFACE)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("users per cell (log)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_xlabel("expertise PageRank x number of users")
    ax.set_ylabel("final reputation")
    ax.set_title(f"{key}: expertise PageRank vs reputation ({len(df):,} answerers)")
    ax.text(0.03, 0.96, f"Spearman rho, PageRank: {rho_pr:.2f}\nSpearman rho, askers helped: {rho_deg:.2f}\n"
            f"rho inside PageRank top 1,000: {rho_top:.2f}", transform=ax.transAxes, va="top", color=INK2, fontsize=8.5)
    ax.grid(False)
    save(fig, cfg, "pagerank_vs_reputation")
    return {"site": key, "answerers": int(len(df)), "rho_pagerank": float(rho_pr), "rho_in_degree": float(rho_deg),
            "rho_top1000": float(rho_top)}


def tag_frame(cfg, key):
    tags = pq.read_table(str(graph_path(cfg, key, "tags"))).to_pandas()
    memb = pq.read_table(str(node_path(cfg, f"{key}_tags", "louvain"))).to_pandas()
    return tags.merge(memb.rename(columns={"node": "tag_id"}), on="tag_id", how="left")


def community_colors(frame, slots):
    sizes = frame.groupby("community")["questions"].sum().sort_values(ascending=False)
    lead = {c: SERIES[i] for i, c in enumerate(sizes.index[:slots])}
    names = {c: frame[frame["community"] == c].sort_values("questions", ascending=False)["tag"].iloc[0]
             for c in sizes.index[:slots]}
    return lead, names


def place_labels(ax, xy, names, limit, size=7.5):
    ax.get_xlim()
    pos = ax.transAxes.inverted().transform(ax.transData.transform(xy))
    box = ax.get_position()
    width, height = ax.figure.get_size_inches()
    char = 0.6 * size / 72 / (width * box.width)
    line = 1.6 * size / 72 / (height * box.height)
    taken = []
    for i, name in enumerate(names):
        if len(taken) >= limit:
            break
        x, y = pos[i]
        w = char * (len(name) + 1)
        if all(x + w < a or a + b < x or abs(y - c) > line for a, b, c in taken):
            ax.annotate(name, xy[i], fontsize=size, color=INK, xytext=(4, 3), textcoords="offset points", zorder=4)
            taken.append((x, w, y))


def tag_map(cfg):
    key = site_key(cfg["figures"]["tag_map_site"])
    top_n = cfg["figures"]["tag_map_top"]
    frame = tag_frame(cfg, key).sort_values("questions", ascending=False).head(top_n).reset_index(drop=True)
    edges = pq.read_table(str(graph_path(cfg, key, "tag_cooc"))).to_pandas()
    index = {t: i for i, t in enumerate(frame["tag_id"])}
    edges = edges[edges["src"].isin(index) & edges["dst"].isin(index)].copy()
    edges["a"], edges["b"] = edges["src"].map(index), edges["dst"].map(index)
    both = pd.concat([edges[["a", "b", "weight"]], edges.rename(columns={"a": "b", "b": "a"})[["a", "b", "weight"]]])
    keep = both.sort_values("weight", ascending=False).groupby("a").head(4)
    pairs = keep.assign(lo=keep[["a", "b"]].min(axis=1), hi=keep[["a", "b"]].max(axis=1)).drop_duplicates(["lo", "hi"])
    g = ig.Graph(n=len(frame), edges=list(zip(pairs["lo"], pairs["hi"], strict=True)))
    g.vs["name"] = list(range(len(frame)))
    giant = g.connected_components().giant()
    alive = np.array(giant.vs["name"])
    frame = frame.loc[alive].reset_index(drop=True)
    remap = {old: new for new, old in enumerate(alive)}
    pairs = pairs[pairs["lo"].isin(remap) & pairs["hi"].isin(remap)]
    lo, hi = pairs["lo"].map(remap).to_numpy(), pairs["hi"].map(remap).to_numpy()
    comm = frame["community"].to_numpy()
    w = np.log1p(pairs["weight"].to_numpy()) * np.where(comm[lo] == comm[hi], 3.0, 0.4)
    h = ig.Graph(n=len(frame), edges=list(zip(lo, hi, strict=True)))
    random.seed(cfg["louvain"]["seed"])
    layout = np.array(h.layout_fruchterman_reingold(weights=w.tolist(), niter=2000))
    lead, names = community_colors(frame, 6)
    colors = [lead.get(c, OTHER) for c in frame["community"]]
    size = 18 + 260 * np.sqrt(frame["questions"] / frame["questions"].max())
    fig, ax = plt.subplots(figsize=(10, 7.4))
    strength = np.log1p(pairs["weight"].to_numpy())
    for a, b, s_ in zip(lo, hi, strength / strength.max(), strict=True):
        ax.plot(layout[[a, b], 0], layout[[a, b], 1], color=AXIS, lw=0.3 + 1.2 * s_, alpha=0.55, zorder=1)
    ax.scatter(layout[:, 0], layout[:, 1], s=size, c=colors, edgecolors=SURFACE, linewidths=1.5, zorder=2)
    place_labels(ax, layout, frame["tag"].tolist(), 28)
    handles = [plt.Line2D([], [], marker="o", ls="", color=col, ms=8, label=f"community of '{names[c]}'")
               for c, col in lead.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=OTHER, ms=8, label="smaller communities"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.set_title(f"{key}: top {len(frame)} tags, strongest co-occurrence edges, colors = Louvain communities")
    ax.axis("off")
    save(fig, cfg, "tag_communities")


def embedding_map(cfg):
    key = site_key(cfg["figures"]["tag_map_site"])
    vectors = load_vectors(cfg, key, cfg["routing"]["test_start"])
    if vectors is None:
        return
    nodes, vecs = vectors[:2]
    frame = tag_frame(cfg, key)
    frame = frame[frame["tag_id"].isin(-nodes[nodes < 0])].sort_values("questions", ascending=False).head(400)
    pos = np.searchsorted(nodes, -frame["tag_id"].to_numpy())
    unit = vecs[pos] / np.maximum(np.linalg.norm(vecs[pos], axis=1, keepdims=True), 1e-12)
    x = unit - unit.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    xy = x @ vt[:2].T
    lead, names = community_colors(frame, 3)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    other = ~frame["community"].isin(lead).to_numpy()
    ax.scatter(xy[other, 0], xy[other, 1], s=16, color=OTHER, edgecolors=SURFACE, linewidths=0.8,
               label="other tag communities")
    for c, col in lead.items():
        sel = (frame["community"] == c).to_numpy()
        ax.scatter(xy[sel, 0], xy[sel, 1], s=26, color=col, edgecolors=SURFACE, linewidths=1.2,
                   label=f"community of '{names[c]}'")
    place_labels(ax, xy, frame["tag"].tolist(), 14)
    ax.set_xlabel("principal component 1")
    ax.set_ylabel("principal component 2")
    ax.set_title(f"{key}: node2vec tag embeddings (PCA), colored by co-occurrence community")
    ax.legend(loc="best")
    save(fig, cfg, "embedding_map")


def community_quality(cfg, comms):
    graphs = [g for g in cfg["fetch"]["snap_graphs"] if g in comms]
    algos = [("louvain", "Louvain (ours, numba)"), ("lpa_spark", "label propagation (ours, Spark)"),
             ("igraph_leiden", "Leiden (igraph)"), ("random_baseline", "random partition")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    width = 0.8 / len(algos)
    for i, (algo, label) in enumerate(algos):
        xs = np.arange(len(graphs)) + (i - (len(algos) - 1) / 2) * width
        f1 = [comms[g].get(algo, {}).get("f1", np.nan) for g in graphs]
        axes[0].bar(xs, f1, width * 0.92, color=SERIES[i], label=label)
        if algo != "random_baseline":
            q = [comms[g].get(algo, {}).get("modularity", np.nan) for g in graphs]
            axes[1].bar(xs, q, width * 0.92, color=SERIES[i], label=label)
    for ax, title, ylabel in ((axes[0], "Match with SNAP ground truth", "average F1"),
                              (axes[1], "Modularity of the partition", "modularity Q")):
        ax.set_xticks(np.arange(len(graphs)), graphs)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    fig.tight_layout()
    save(fig, cfg, "community_quality")


def routing_curves(cfg, routing):
    pooled = routing.get("pooled")
    if not pooled:
        return
    ks = cfg["routing"]["k_values"]
    labels = {"lightgbm_ranker": "LightGBM ranker on graph features", "tag_ppr": "tag personalized PageRank",
              "tag_activity": "answers in the question's tags", "recent_activity": "most active recently",
              "embedding": "node2vec link score"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for i, (method, label) in enumerate(labels.items()):
        if method in pooled:
            axes[0].plot(ks, [pooled[method][f"hit@{k}"] for k in ks], marker="o", ms=5, color=SERIES[i],
                         label=f"{label}  MRR {pooled[method]['mrr']:.3f}")
    axes[0].set_xlabel("k (users suggested per question)")
    axes[0].set_ylabel("hit@k")
    axes[0].set_xticks(ks)
    axes[0].set_title("Who will answer? held out last six months")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.16))
    gain = routing.get("feature_gain", {})
    top = sorted(gain.items(), key=lambda kv: kv[1])[-12:]
    axes[1].barh([FEATURE_NAMES.get(k, k) for k, _ in top], [v for _, v in top], color=SERIES[0], height=0.62)
    axes[1].set_xlabel("share of total gain")
    axes[1].set_title("What the ranker relies on")
    axes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, cfg, "routing")


def runtime_scaling(cfg, timings):
    spark_algos = [("pagerank", "PageRank"), ("components", "connected components"), ("kcore", "k-core"),
                   ("lpa_spark", "label propagation"), ("triangles_spark", "triangles")]
    local_algos = [("triangles_local", "triangles (numba)"), ("kcore_local", "k-core (numba)"),
                   ("louvain", "Louvain (numba)"), ("igraph_louvain", "Louvain (igraph, C)"),
                   ("igraph_leiden", "Leiden (igraph, C)")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, algos, title in ((axes[0], spark_algos, "Spark, local mode"),
                             (axes[1], local_algos, "single machine engine")):
        for i, (stage, label) in enumerate(algos):
            rows = sorted((r["edges"], r["seconds"]) for r in timings if r["stage"] == stage and r["edges"])
            if rows:
                e, s = zip(*rows, strict=True)
                ax.loglog(e, s, marker="o", ms=4.5, lw=1.6, color=SERIES[i], label=label)
        ax.set_xlabel("edges")
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel("seconds")
    fig.tight_layout()
    save(fig, cfg, "runtime_scaling")


def cross_site(cfg, data):
    if not data:
        return
    sites = data["sites"]
    shared = data["shared"]
    n = len(sites)
    totals = np.array([shared.get(f"{s}|{s}", 0) for s in sites], dtype=float)
    mat = np.zeros((n, n))
    for i, a in enumerate(sites):
        for j, b in enumerate(sites):
            both = shared.get(f"{a}|{b}", 0)
            union = totals[i] + totals[j] - both
            mat[i, j] = 100.0 * both / union if union and i != j else np.nan
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(mat, cmap=BLUES)
    for i in range(n):
        for j in range(n):
            if i != j:
                dark = mat[i, j] > np.nanmax(mat) * 0.6
                ax.text(j, i, f"{mat[i, j]:.1f}%", ha="center", va="center", fontsize=8,
                        color=SURFACE if dark else INK)
    ax.set_xticks(range(n), sites, rotation=30, ha="right")
    ax.set_yticks(range(n), sites)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax)
    cb.outline.set_visible(False)
    cb.set_label("Jaccard overlap of active accounts (%)", color=INK2)
    ax.set_title("Same people across sites")
    save(fig, cfg, "cross_site_overlap")


def run(cfg):
    style(cfg["figures"]["dpi"])
    mining = load_json(cfg, "mining", {})
    if mining:
        degree_ccdf(cfg, mining)
        kcore_shells(cfg, mining)
        save_json(cfg, "reputation", pagerank_vs_reputation(cfg, mining))
    graph_growth(cfg, load_json(cfg, "growth", {}))
    comms = load_json(cfg, "communities", {})
    if comms:
        community_quality(cfg, comms)
        tag_map(cfg)
        embedding_map(cfg)
    routing_curves(cfg, load_json(cfg, "routing", {}))
    runtime_scaling(cfg, load_json(cfg, "timings", []))
    cross_site(cfg, load_json(cfg, "cross_site", {}))
