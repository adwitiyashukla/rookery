import gzip
import json
import shutil
from contextlib import ExitStack

import py7zr
from py7zr.io import Py7zIO, WriterFactory
from pyspark.sql import functions as F
from pyspark.sql import types as T

from rookery.fetch import snap_files
from rookery.sparkctx import spark_session
from rookery.store import graph_path, lake, raw_dir, site_key, timed, update_json, xml_dir

LONG, INT, STR = T.LongType(), T.IntegerType(), T.StringType()

TABLES = {
    "Posts": [
        ("Id", LONG), ("PostTypeId", INT), ("ParentId", LONG), ("AcceptedAnswerId", LONG), ("CreationDate", STR),
        ("Score", INT), ("OwnerUserId", LONG), ("Tags", STR), ("AnswerCount", INT),
    ],
    "Comments": [("Id", LONG), ("PostId", LONG), ("Score", INT), ("CreationDate", STR), ("UserId", LONG)],
    "Users": [("Id", LONG), ("Reputation", INT), ("CreationDate", STR), ("AccountId", LONG)],
    "PostLinks": [("Id", LONG), ("CreationDate", STR), ("PostId", LONG), ("RelatedPostId", LONG), ("LinkTypeId", INT)],
    "Tags": [("Id", LONG), ("TagName", STR), ("Count", INT)],
}
HISTORY = [("Id", LONG), ("PostId", LONG), ("Text", STR)]
FIRST_TAGS = ("PostHistory.xml", "InitialTags.xml", b'PostHistoryTypeId="3"')


class Sink(Py7zIO):
    def __init__(self, fh, keep=None):
        self.fh = fh
        self.keep = keep
        self.tail = b""
        self.length = 0
        self.done = False
        if keep:
            self.fh.write(b"<rows>\n")

    def write(self, s):
        self.length += len(s)
        if self.keep is None:
            self.fh.write(s)
            return len(s)
        lines = bytes(s).split(b"\n")
        lines[0] = self.tail + lines[0]
        self.tail = lines.pop()
        self.fh.writelines(line + b"\n" for line in lines if self.keep in line)
        return len(s)

    def finish(self):
        if self.done:
            return
        if self.keep:
            if self.keep in self.tail:
                self.fh.write(self.tail + b"\n")
            self.fh.write(b"</rows>\n")
        self.fh.flush()
        self.done = True

    def close(self):
        self.finish()

    def read(self, size=None):
        return b""

    def seek(self, offset, whence=0):
        return 0

    def seekable(self):
        return False

    def flush(self):
        self.fh.flush()

    def size(self):
        return self.length


class Sinks(WriterFactory):
    def __init__(self, sinks):
        self.sinks = sinks

    def create(self, filename):
        return self.sinks[filename.split("/")[-1]]


def extract(cfg, site):
    archive = raw_dir(cfg, "stackexchange") / f"{site}.7z"
    out = xml_dir(cfg, site)
    marker = out / "extracted.json"
    if marker.exists():
        return out
    if not archive.exists():
        raise FileNotFoundError(f"missing {archive}, run the fetch step first")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    wanted = {f"{name}.xml" for name in TABLES} | {FIRST_TAGS[0]}
    with ExitStack() as stack, py7zr.SevenZipFile(archive, mode="r") as zf:
        targets = [n for n in zf.getnames() if n.split("/")[-1] in wanted]
        missing = wanted - {n.split("/")[-1] for n in targets}
        if missing:
            raise RuntimeError(f"{archive.name} is missing {sorted(missing)}")
        sinks = {}
        for name in {n.split("/")[-1] for n in targets}:
            first = name == FIRST_TAGS[0]
            fh = stack.enter_context(open(out / (FIRST_TAGS[1] if first else name), "wb"))
            sinks[name] = Sink(fh, FIRST_TAGS[2] if first else None)
        zf.extract(path=out, targets=targets, factory=Sinks(sinks))
        for sink in sinks.values():
            sink.finish()
    found = {p.name: p.stat().st_size for p in sorted(out.glob("*.xml"))}
    marker.write_text(json.dumps(found), encoding="utf-8")
    return out


def tags_array(col):
    pipes = F.split(F.regexp_replace(col, r"^\|+|\|+$", ""), r"\|")
    angles = F.regexp_extract_all(col, F.lit(r"<([^>]+)>"), 1)
    return F.when(col.isNull() | (F.length(col) == 0), F.array().cast("array<string>")).when(
        col.startswith("|"), pipes).otherwise(angles)


def read_xml(spark, path, columns):
    schema = T.StructType([T.StructField(f"_{col}", kind) for col, kind in columns])
    df = spark.read.format("xml").option("rowTag", "row").schema(schema).load(str(path))
    for col, _ in columns:
        df = df.withColumnRenamed(f"_{col}", col)
    if "CreationDate" in df.columns:
        df = df.withColumn("created", F.to_timestamp("CreationDate")).drop("CreationDate")
    return df


def read_table(spark, folder, name):
    return read_xml(spark, folder / f"{name}.xml", TABLES[name])


def with_initial_tags(posts, history):
    first = history.groupBy(F.col("PostId").alias("id")).agg(F.min_by("Text", "Id").alias("text"))
    first = first.select("id", tags_array(F.col("text")).alias("first_tags"))
    return (posts.join(first, "id", "left").withColumn("initial_tags", F.coalesce("first_tags", "tags"))
            .drop("first_tags"))


def shape(name, df):
    if name == "Posts":
        return df.select(
            F.col("Id").alias("id"), F.col("PostTypeId").alias("post_type"), F.col("ParentId").alias("parent_id"),
            F.col("AcceptedAnswerId").alias("accepted_answer_id"), "created", F.col("Score").alias("score"),
            F.col("OwnerUserId").alias("owner"), tags_array(F.col("Tags")).alias("tags"),
            F.col("AnswerCount").alias("answer_count"),
        ).filter(F.col("post_type").isin(1, 2))
    if name == "Comments":
        return df.select(F.col("Id").alias("id"), F.col("PostId").alias("post_id"), F.col("Score").alias("score"),
                         "created", F.col("UserId").alias("user_id"))
    if name == "Users":
        return df.select(F.col("Id").alias("id"), F.col("Reputation").alias("reputation"), "created",
                         F.col("AccountId").alias("account_id"))
    if name == "PostLinks":
        return df.select(F.col("PostId").alias("post_id"), F.col("RelatedPostId").alias("related_post_id"),
                         F.col("LinkTypeId").alias("link_type"), "created")
    return df.select(F.col("Id").alias("tag_id"), F.col("TagName").alias("tag"), F.col("Count").alias("count"))


def ingest_site(spark, cfg, site):
    folder = extract(cfg, site)
    key = site_key(site)
    counts = {}
    for name in TABLES:
        df = shape(name, read_table(spark, folder, name))
        if name == "Posts":
            df = with_initial_tags(df, read_xml(spark, folder / FIRST_TAGS[1], HISTORY))
        out = lake(cfg, name.lower())
        df.withColumn("site", F.lit(key)).write.mode("overwrite").partitionBy("site").parquet(str(out))
        counts[name.lower()] = spark.read.parquet(str(out)).filter(F.col("site") == key).count()
    posts = spark.read.parquet(str(lake(cfg, "posts"))).filter((F.col("site") == key) & (F.col("post_type") == 1))
    counts["retagged_questions"] = posts.filter(F.sort_array("tags") != F.sort_array("initial_tags")).count()
    if not cfg["ingest"]["keep_xml"]:
        shutil.rmtree(folder, ignore_errors=True)
    update_json(cfg, "ingest", key, counts)
    print(f"  {site}: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()), flush=True)


def ingest_snap(spark, cfg, name):
    edges_file, cmty_file = (raw_dir(cfg, "snap") / f for f in snap_files(name))
    lines = spark.read.text(str(edges_file)).filter(~F.col("value").startswith("#"))
    parts = F.split(F.trim(F.col("value")), r"\s+")
    pairs = lines.select(parts[0].cast("long").alias("a"), parts[1].cast("long").alias("b"))
    edges = (pairs.filter(F.col("a").isNotNull() & F.col("b").isNotNull() & (F.col("a") != F.col("b")))
             .select(F.least("a", "b").alias("src"), F.greatest("a", "b").alias("dst")).distinct()
             .withColumn("weight", F.lit(1.0)))
    edges.write.mode("overwrite").parquet(str(graph_path(cfg, name, "graph")))
    cmty = (spark.read.text(str(cmty_file)).filter(F.length(F.trim("value")) > 0)
            .withColumn("cid", F.monotonically_increasing_id())
            .select("cid", F.explode(F.split(F.trim(F.col("value")), r"\s+")).alias("node"))
            .select("cid", F.col("node").cast("long").alias("node")).filter(F.col("node").isNotNull()))
    cmty.write.mode("overwrite").parquet(str(graph_path(cfg, name, "cmty")))
    stored = spark.read.parquet(str(graph_path(cfg, name, "graph")))
    nodes = stored.select(F.col("src").alias("n")).union(stored.select("dst")).distinct().count()
    with gzip.open(edges_file, "rt") as fh:
        header = "".join(fh.readline() for _ in range(3))
    info = {"nodes": nodes, "edges": stored.count(), "synthetic": "synthetic" in header,
            "communities": spark.read.parquet(str(graph_path(cfg, name, "cmty"))).select("cid").distinct().count()}
    update_json(cfg, "ingest", name, info)
    print(f"  {name}: {info['nodes']:,} nodes, {info['edges']:,} edges, {info['communities']:,} communities",
          flush=True)


def run(cfg):
    spark = spark_session(cfg)
    for name in cfg["fetch"]["snap_graphs"]:
        with timed(cfg, "ingest", name):
            ingest_snap(spark, cfg, name)
    for site in cfg["fetch"]["sites"]:
        with timed(cfg, "ingest", site_key(site)):
            ingest_site(spark, cfg, site)
