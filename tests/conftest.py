import networkx as nx
import pytest
from pyspark.sql import SparkSession

from rookery.sparkctx import prepare_env


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    prepare_env()
    session = (SparkSession.builder.master("local[2]").appName("rookery-tests")
               .config("spark.ui.enabled", "false")
               .config("spark.sql.shuffle.partitions", "4")
               .config("spark.sql.session.timeZone", "UTC")
               .config("spark.rookery.scratch", str(tmp_path_factory.mktemp("scratch")))
               .getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    session.sparkContext.setCheckpointDir(str(tmp_path_factory.mktemp("checkpoints")))
    yield session
    session.stop()


def to_df(spark, graph, weight=False):
    rows = [(int(u), int(v), float(d.get("weight", 1.0)) if weight else 1.0) for u, v, d in graph.edges(data=True)]
    return spark.createDataFrame(rows, "src long, dst long, weight double")


@pytest.fixture(scope="session")
def small_graph():
    g = nx.gnm_random_graph(160, 520, seed=3)
    g.add_edges_from([(200, 201), (201, 202), (202, 200), (300, 301)])
    return g
