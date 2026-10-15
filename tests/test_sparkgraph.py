import random

import networkx as nx
import pytest
from conftest import to_df

from rookery import sparkgraph as sg


def as_dict(df, key, value):
    return {r[key]: r[value] for r in df.collect()}


def test_pagerank_directed_weighted(spark):
    rng = random.Random(5)
    g = nx.gnm_random_graph(120, 500, seed=5, directed=True)
    for u, v in g.edges():
        g[u][v]["weight"] = rng.uniform(0.5, 3.0)
    g.add_edge(500, 501, weight=1.0)
    ranks, info = sg.pagerank(to_df(spark, g, weight=True), 0.85, 1e-12, 200)
    ours = as_dict(ranks, "node", "rank")
    ref = nx.pagerank(g, alpha=0.85, tol=1e-13, max_iter=1000)
    assert max(abs(ours[n] - ref[n]) for n in ref) < 1e-8
    assert info["iterations"] < 200


def test_pagerank_undirected(spark, small_graph):
    ranks, _ = sg.pagerank(to_df(spark, small_graph), 0.85, 1e-12, 200, directed=False)
    ours = as_dict(ranks, "node", "rank")
    ref = nx.pagerank(small_graph, alpha=0.85, tol=1e-13, max_iter=1000)
    assert max(abs(ours[n] - ref[n]) for n in ref) < 1e-8


def test_personalized_pagerank(spark):
    g = nx.gnm_random_graph(90, 300, seed=9, directed=True)
    seeds = {3: 2.0, 17: 1.0}
    tele = spark.createDataFrame(list(seeds.items()), "node long, v double")
    ranks, _ = sg.pagerank(to_df(spark, g), 0.85, 1e-12, 300, teleport=tele)
    ours = as_dict(ranks, "node", "rank")
    ref = nx.pagerank(g, alpha=0.85, personalization=seeds, tol=1e-13, max_iter=1000)
    assert max(abs(ours[n] - ref[n]) for n in ref) < 1e-8


def test_connected_components(spark, small_graph):
    labels, info = sg.connected_components(to_df(spark, small_graph), 50)
    ours = as_dict(labels, "node", "component")
    for comp in nx.connected_components(small_graph):
        assert {ours[n] for n in comp} == {min(comp)}
    assert len(ours) == small_graph.number_of_nodes()
    assert info["rounds"] < 50


def test_connected_components_long_path(spark):
    g = nx.path_graph(300)
    nx.relabel_nodes(g, {i: (i * 7919) % 1009 for i in g.nodes()}, copy=False)
    labels, _ = sg.connected_components(to_df(spark, g), 60)
    assert set(as_dict(labels, "node", "component").values()) == {min(g.nodes())}


def test_kcore(spark, small_graph):
    core, _ = sg.kcore(to_df(spark, small_graph), 200)
    assert as_dict(core, "node", "core") == nx.core_number(small_graph)


def test_triangles(spark, small_graph):
    counts, info = sg.triangles(to_df(spark, small_graph))
    ref = nx.triangles(small_graph)
    assert as_dict(counts, "node", "triangles") == ref
    assert info["triangles"] == sum(ref.values()) // 3


def test_label_propagation_two_cliques(spark):
    g = nx.disjoint_union(nx.complete_graph(8), nx.complete_graph(8))
    g.add_edge(0, 8)
    labels, _ = sg.label_propagation(to_df(spark, g), 20, 1.0, 0.0)
    lab = as_dict(labels, "node", "label")
    assert len({lab[n] for n in range(1, 8)}) == 1
    assert len({lab[n] for n in range(9, 16)}) == 1
    assert lab[1] != lab[9]


@pytest.mark.parametrize(("p", "q", "bipartite"), [(1.0, 1.0, False), (1.0, 0.5, False), (2.0, 0.5, True)])
def test_random_walks_follow_edges(spark, small_graph, p, q, bipartite, tmp_path):
    g = nx.complete_bipartite_graph(5, 7) if bipartite else small_graph
    walks = sg.random_walks(to_df(spark, g), 2, 6, p, q, 3, 11, bipartite, str(tmp_path / "walks"))
    rows = walks.collect()
    seq = {}
    for r in rows:
        seq.setdefault(r["walk"], {})[r["step"]] = r["node"]
    nodes_with_edges = {n for n in g.nodes() if g.degree(n) > 0}
    assert len(seq) == 2 * len(nodes_with_edges)
    for steps in seq.values():
        assert sorted(steps) == list(range(6))
        for i in range(5):
            assert g.has_edge(steps[i], steps[i + 1])
    again = sg.random_walks(to_df(spark, g), 2, 6, p, q, 3, 11, bipartite, str(tmp_path / "again"))
    assert sorted(map(tuple, again.collect())) == sorted(map(tuple, rows))
