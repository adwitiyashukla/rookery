import networkx as nx
import numpy as np
import pandas as pd
import pytest

from rookery.communities import f1_scores
from rookery.mine import powerlaw_fit
from rookery.routing import pagerank_scipy, rank_metrics


@pytest.mark.parametrize("alpha", [2.2, 2.7])
def test_powerlaw_fit_recovers_exponent(alpha):
    rng = np.random.default_rng(4)
    u = rng.random(40_000)
    x = np.floor((5 - 0.5) * (1 - u) ** (-1 / (alpha - 1)) + 0.5)
    fit = powerlaw_fit(np.concatenate([x, rng.integers(1, 5, 3000)]))
    assert abs(fit["alpha"] - alpha) < 0.12
    assert fit["xmin"] <= 8


def test_f1_perfect_and_random():
    gt_cid = np.repeat(np.arange(10), 30)
    gt_node = np.arange(300)
    perfect = f1_scores(gt_cid, gt_node, np.repeat(np.arange(10), 30))
    assert perfect["f1"] == pytest.approx(1.0)
    shuffled = np.random.default_rng(1).permutation(np.repeat(np.arange(10), 30))
    assert f1_scores(gt_cid, gt_node, shuffled)["f1"] < 0.5


def test_f1_handles_overlap():
    gt_cid = np.array([0, 0, 0, 1, 1, 1])
    gt_node = np.array([0, 1, 2, 2, 3, 4])
    memb = np.array([0, 0, 0, 1, 1])
    out = f1_scores(gt_cid, gt_node, memb)
    assert out["f1_gt"] == pytest.approx((1.0 + 0.8) / 2)


def test_rank_metrics():
    df = pd.DataFrame({
        "qid": [1, 1, 1, 2, 2, 2],
        "order": [0, 1, 2, 0, 1, 2],
        "label": [0, 1, 0, 0, 0, 0],
        "score": [0.9, 0.8, 0.1, 0.5, 0.4, 0.3],
    })
    out = rank_metrics(df, "score", [1, 2])
    assert out["hit@1"] == 0.0
    assert out["hit@2"] == 0.5
    assert out["mrr"] == pytest.approx(0.25)
    assert out["ndcg@10"] == pytest.approx(0.5 * (1 / np.log2(3)))


def test_rank_metrics_breaks_ties_at_random():
    n = 3000
    df = pd.DataFrame({
        "qid": np.repeat(np.arange(n), 3),
        "order": np.tile([0, 1, 2], n),
        "label": np.tile([1, 0, 0], n),
        "score": np.zeros(3 * n),
    })
    out = rank_metrics(df, "score", [1, 3], seed=5)
    assert out["hit@1"] == pytest.approx(1 / 3, abs=0.03)
    assert out["hit@3"] == 1.0
    assert rank_metrics(df, "score", [1], seed=5) == rank_metrics(df, "score", [1], seed=5)


def test_pagerank_scipy_matches_networkx():
    g = nx.gnm_random_graph(80, 300, seed=2, directed=True)
    src, dst = (np.array(x) for x in zip(*g.edges(), strict=True))
    ours = pagerank_scipy(src, dst, np.ones(len(src)), 80, iters=200)
    ref = nx.pagerank(g, alpha=0.85, tol=1e-12, max_iter=1000)
    assert max(abs(ours[i] - ref[i]) for i in range(80)) < 1e-9
