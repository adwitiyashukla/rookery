import igraph as ig
import networkx as nx
import numpy as np
import pytest

from rookery import localgraph as lg


def graph_of(g, weight=False):
    edges = list(g.edges(data=True))
    src = np.array([u for u, _, _ in edges], dtype=np.int64)
    dst = np.array([v for _, v, _ in edges], dtype=np.int64)
    w = np.array([d.get("weight", 1.0) for _, _, d in edges]) if weight else None
    return lg.from_arrays(src, dst, w)


def test_csr_degrees_and_negative_ids():
    g = nx.Graph([(-3, 5), (5, 9), (-3, 9), (-7, 5)])
    csr = graph_of(g)
    assert sorted(csr.ids.tolist()) == sorted(g.nodes())
    deg = dict(zip(csr.ids.tolist(), csr.degree().tolist(), strict=True))
    assert deg == dict(g.degree())


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_kcore_matches_networkx(seed):
    g = nx.powerlaw_cluster_graph(600, 6, 0.3, seed=seed)
    g.add_edges_from([(1000, 1001), (1001, 1002)])
    csr = graph_of(g)
    ref = nx.core_number(g)
    assert dict(zip(csr.ids.tolist(), lg.kcore(csr).tolist(), strict=True)) == {n: ref[n] for n in csr.ids.tolist()}


def test_triangles_match_networkx():
    g = nx.gnm_random_graph(400, 3000, seed=4)
    csr = graph_of(g)
    tri, total = lg.triangles(csr)
    ref = nx.triangles(g)
    assert dict(zip(csr.ids.tolist(), tri.tolist(), strict=True)) == {n: ref[n] for n in csr.ids.tolist()}
    assert total == sum(ref.values()) // 3


def test_modularity_matches_igraph():
    g = nx.gnm_random_graph(300, 1500, seed=8)
    csr = graph_of(g)
    memb = np.random.default_rng(1).integers(0, 7, csr.n)
    h = ig.Graph(n=csr.n, edges=[(int(a), int(b)) for a, b in zip(*np.nonzero(np.triu(nx.to_numpy_array(
        g, nodelist=csr.ids.tolist()))), strict=True)])
    assert lg.modularity(csr, memb) == pytest.approx(h.modularity(memb.tolist()), abs=1e-12)


def test_louvain_recovers_planted_partition():
    g = nx.planted_partition_graph(6, 40, 0.4, 0.01, seed=2)
    csr = graph_of(g)
    memb, info = lg.louvain(csr, 1.0, 10, 50, 1e-9, 7)
    truth = np.array([n // 40 for n in csr.ids.tolist()])
    nmi = ig.compare_communities(truth.tolist(), memb.tolist(), method="nmi")
    h = ig.Graph(n=csr.n, edges=[(int(np.searchsorted(csr.ids, u)), int(np.searchsorted(csr.ids, v)))
                                 for u, v in g.edges()])
    ref = h.community_multilevel().modularity
    assert nmi > 0.95
    assert info["modularity"] > ref - 0.01


def test_louvain_weighted_graph():
    rng = np.random.default_rng(3)
    g = nx.gnm_random_graph(250, 1200, seed=5)
    for u, v in g.edges():
        g[u][v]["weight"] = float(rng.uniform(0.2, 4.0))
    csr = graph_of(g, weight=True)
    memb, info = lg.louvain(csr, 1.0, 10, 50, 1e-9, 7)
    h = ig.Graph(n=csr.n, edges=[(int(np.searchsorted(csr.ids, u)), int(np.searchsorted(csr.ids, v)))
                                 for u, v in g.edges()])
    ref = h.community_multilevel(weights=[d["weight"] for _, _, d in g.edges(data=True)]).modularity
    assert info["modularity"] == pytest.approx(lg.modularity(csr, memb))
    assert info["modularity"] > ref - 0.02


def test_push_ppr_close_to_exact():
    g = nx.gnm_random_graph(300, 1800, seed=6)
    csr = graph_of(g)
    ppr = lg.PushPPR(csr, 0.15, 1e-9)
    seeds = [int(np.searchsorted(csr.ids, 4)), int(np.searchsorted(csr.ids, 77))]
    nodes, vals = ppr.run(seeds, [1.0, 3.0])
    ref = nx.pagerank(g, alpha=0.85, personalization={4: 0.25, 77: 0.75}, tol=1e-14, max_iter=5000)
    ours = dict(zip(csr.ids[nodes].tolist(), vals.tolist(), strict=True))
    assert max(abs(ours.get(n, 0.0) - ref[n]) for n in g.nodes()) < 1e-5
    nodes2, vals2 = ppr.run(seeds, [1.0, 3.0])
    assert np.allclose(np.sort(vals), np.sort(vals2))
