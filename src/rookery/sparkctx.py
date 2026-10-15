import glob
import os
import re
import sys

from pyspark.sql import SparkSession

_session = None
SUPPORTED_JAVA = (17, 21)


def java_major(home):
    release = os.path.join(home, "release")
    if not os.path.isfile(release):
        return None
    with open(release, encoding="utf-8", errors="ignore") as fh:
        found = re.search(r'JAVA_VERSION="(\d+)', fh.read())
    return int(found.group(1)) if found else None


def pick_java(configured):
    if configured:
        return configured
    current = os.environ.get("JAVA_HOME", "")
    if current and java_major(current) in SUPPORTED_JAVA:
        return current
    roots = ["C:/Program Files/*/jdk-*", "C:/Program Files/*/jdk*"] if os.name == "nt" else ["/usr/lib/jvm/*"]
    for pattern in roots:
        for home in sorted(glob.glob(pattern), reverse=True):
            if java_major(home) in SUPPORTED_JAVA:
                print(f"  using Java {java_major(home)} at {home}", flush=True)
                return home
    return current


def prepare_env(java_home="", hadoop_home=""):
    java = pick_java(java_home)
    if java:
        os.environ["JAVA_HOME"] = java
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    hadoop = hadoop_home or os.environ.get("HADOOP_HOME", "")
    if os.name == "nt":
        if not os.path.isfile(os.path.join(hadoop, "bin", "winutils.exe")):
            raise RuntimeError("Spark on Windows needs winutils.exe and hadoop.dll, set HADOOP_HOME (or "
                               "spark.hadoop_home) to a folder whose bin has both")
        os.environ["HADOOP_HOME"] = hadoop
        os.environ["PATH"] = os.path.join(hadoop, "bin") + os.pathsep + os.environ.get("PATH", "")


def spark_session(cfg):
    global _session
    if _session is not None:
        return _session
    sp = cfg["spark"]
    prepare_env(sp["java_home"], sp["hadoop_home"])
    data = cfg["paths"]["data"]
    local_dir = sp["local_dir"] or str(data / "tmp" / "spark")
    os.makedirs(local_dir, exist_ok=True)
    builder = (
        SparkSession.builder.master(sp["master"])
        .appName("rookery")
        .config("spark.driver.memory", sp["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(sp["shuffle_partitions"]))
        .config("spark.local.dir", local_dir)
        .config("spark.sql.adaptive.enabled", "true" if sp["adaptive"] else "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.rookery.scratch", str(data / "tmp" / "scratch"))
    )
    _session = builder.getOrCreate()
    _session.sparkContext.setLogLevel(sp["log_level"])
    checkpoints = data / "tmp" / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    _session.sparkContext.setCheckpointDir(str(checkpoints))
    return _session


def stop():
    global _session
    if _session is not None:
        _session.stop()
        _session = None
