import re
import tomllib
from datetime import date
from pathlib import Path

SNAP_GRAPHS = {"amazon", "dblp", "youtube", "lj"}

SPEC = {
    "paths": {"data_dir": str, "figures_dir": str},
    "spark": {
        "master": str,
        "driver_memory": str,
        "shuffle_partitions": int,
        "java_home": str,
        "hadoop_home": str,
        "local_dir": str,
        "check_every": int,
        "adaptive": bool,
        "log_level": str,
    },
    "fetch": {
        "stackexchange_url": str,
        "snap_url": str,
        "sites": list,
        "snap_graphs": list,
        "retries": int,
        "timeout_seconds": int,
    },
    "ingest": {"keep_xml": bool},
    "graphs": {"min_weight": int},
    "pagerank": {"alpha": float, "tol": float, "max_iter": int},
    "components": {"max_iter": int},
    "kcore": {"max_iter": int, "spark_max_edges": int},
    "lpa": {"max_iter": int, "self_weight": float, "min_changed_fraction": float, "reference_max_edges": int},
    "triangles": {"spark_max_edges": int},
    "louvain": {"resolution": float, "max_levels": int, "max_sweeps": int, "min_gain": float, "seed": int,
                "reference_max_edges": int},
    "walks": {"num_walks": int, "walk_length": int, "p": float, "q": float, "max_tries": int, "seed": int},
    "embed": {"dim": int, "window": int, "negative": int, "epochs": int, "workers": int, "seed": int},
    "routing": {
        "train_start": str,
        "test_start": str,
        "test_end": str,
        "history_days": int,
        "decay_days": float,
        "candidates": int,
        "max_train_questions": int,
        "max_test_questions": int,
        "k_values": list,
        "ppr_alpha": float,
        "ppr_eps": float,
        "num_leaves": int,
        "learning_rate": float,
        "n_estimators": int,
        "seed": int,
    },
    "crosscheck": {"networkx_max_edges": int},
    "figures": {"dpi": int, "tag_map_site": str, "tag_map_top": int},
    "synth": {"scale": float, "snap_scale": float, "seed": int},
}

POSITIVE = [
    ("spark", "shuffle_partitions"),
    ("spark", "check_every"),
    ("fetch", "retries"),
    ("fetch", "timeout_seconds"),
    ("graphs", "min_weight"),
    ("pagerank", "max_iter"),
    ("components", "max_iter"),
    ("kcore", "max_iter"),
    ("kcore", "spark_max_edges"),
    ("lpa", "max_iter"),
    ("lpa", "reference_max_edges"),
    ("triangles", "spark_max_edges"),
    ("louvain", "max_levels"),
    ("louvain", "max_sweeps"),
    ("louvain", "reference_max_edges"),
    ("walks", "num_walks"),
    ("walks", "max_tries"),
    ("embed", "dim"),
    ("embed", "window"),
    ("embed", "negative"),
    ("embed", "epochs"),
    ("embed", "workers"),
    ("routing", "history_days"),
    ("routing", "decay_days"),
    ("routing", "candidates"),
    ("routing", "max_train_questions"),
    ("routing", "max_test_questions"),
    ("routing", "n_estimators"),
    ("crosscheck", "networkx_max_edges"),
    ("synth", "scale"),
    ("synth", "snap_scale"),
    ("pagerank", "tol"),
    ("louvain", "resolution"),
    ("louvain", "min_gain"),
    ("walks", "p"),
    ("walks", "q"),
    ("routing", "ppr_eps"),
]


class ConfigError(ValueError):
    pass


def check_types(raw):
    for section, keys in SPEC.items():
        if section not in raw:
            raise ConfigError(f"missing section [{section}]")
        extra = set(raw[section]) - set(keys)
        if extra:
            raise ConfigError(f"unknown keys in [{section}]: {sorted(extra)}")
        for key, kind in keys.items():
            if key not in raw[section]:
                raise ConfigError(f"missing key {section}.{key}")
            value = raw[section][key]
            if kind is float and isinstance(value, int) and not isinstance(value, bool):
                raw[section][key] = float(value)
                continue
            if kind is int and isinstance(value, bool):
                raise ConfigError(f"{section}.{key} must be an integer")
            if not isinstance(value, kind):
                raise ConfigError(f"{section}.{key} must be {kind.__name__}, got {type(value).__name__}")
    extra_sections = set(raw) - set(SPEC)
    if extra_sections:
        raise ConfigError(f"unknown sections: {sorted(extra_sections)}")


def check_values(cfg):
    for section, key in POSITIVE:
        if cfg[section][key] <= 0:
            raise ConfigError(f"{section}.{key} must be positive")
    sp = cfg["spark"]
    if not sp["master"].strip():
        raise ConfigError("spark.master must not be empty")
    if not re.fullmatch(r"\d+[mMgG]", sp["driver_memory"]):
        raise ConfigError("spark.driver_memory must look like 4g or 2048m")
    if sp["log_level"] not in {"ALL", "DEBUG", "INFO", "WARN", "ERROR", "FATAL", "OFF"}:
        raise ConfigError("spark.log_level is not a log4j level")
    if sp["java_home"] and not Path(sp["java_home"]).is_dir():
        raise ConfigError(f"spark.java_home does not exist: {sp['java_home']}")
    if sp["hadoop_home"] and not (Path(sp["hadoop_home"]) / "bin").is_dir():
        raise ConfigError(f"spark.hadoop_home has no bin folder: {sp['hadoop_home']}")
    fe = cfg["fetch"]
    if not fe["sites"] or not all(isinstance(s, str) and re.fullmatch(r"[a-z0-9.\-]+\.[a-z]+", s) for s in fe["sites"]):
        raise ConfigError("fetch.sites must be a non-empty list of site domains")
    if len(set(fe["sites"])) != len(fe["sites"]):
        raise ConfigError("fetch.sites has duplicates")
    if not fe["snap_graphs"] or not set(fe["snap_graphs"]) <= SNAP_GRAPHS:
        raise ConfigError(f"fetch.snap_graphs must be a non-empty subset of {sorted(SNAP_GRAPHS)}")
    for url_key in ("stackexchange_url", "snap_url"):
        if not fe[url_key].startswith(("http://", "https://")):
            raise ConfigError(f"fetch.{url_key} must be an http url")
    if not 0 < cfg["pagerank"]["alpha"] < 1:
        raise ConfigError("pagerank.alpha must be in (0, 1)")
    if cfg["lpa"]["self_weight"] < 0:
        raise ConfigError("lpa.self_weight must not be negative")
    if not 0 <= cfg["lpa"]["min_changed_fraction"] < 1:
        raise ConfigError("lpa.min_changed_fraction must be in [0, 1)")
    if cfg["walks"]["walk_length"] < 2:
        raise ConfigError("walks.walk_length must be at least 2")
    check_routing(cfg["routing"])
    fig = cfg["figures"]
    if fig["dpi"] < 50:
        raise ConfigError("figures.dpi must be at least 50")
    if fig["tag_map_site"] not in fe["sites"]:
        raise ConfigError("figures.tag_map_site must be one of fetch.sites")
    if fig["tag_map_top"] < 10:
        raise ConfigError("figures.tag_map_top must be at least 10")


def check_routing(ro):
    try:
        start = date.fromisoformat(ro["train_start"])
        mid = date.fromisoformat(ro["test_start"])
        end = date.fromisoformat(ro["test_end"])
    except ValueError as exc:
        raise ConfigError(f"routing dates must be YYYY-MM-DD: {exc}") from exc
    if not start < mid < end:
        raise ConfigError("routing dates must satisfy train_start < test_start < test_end")
    ks = ro["k_values"]
    if not ks or not all(isinstance(k, int) and not isinstance(k, bool) and k > 0 for k in ks):
        raise ConfigError("routing.k_values must be positive integers")
    if sorted(set(ks)) != ks:
        raise ConfigError("routing.k_values must be strictly increasing")
    if ro["candidates"] < ks[-1]:
        raise ConfigError("routing.candidates must be at least the largest k")
    if not 0 < ro["ppr_alpha"] < 1:
        raise ConfigError("routing.ppr_alpha must be in (0, 1)")
    if ro["num_leaves"] < 2:
        raise ConfigError("routing.num_leaves must be at least 2")
    if not 0 < ro["learning_rate"] <= 1:
        raise ConfigError("routing.learning_rate must be in (0, 1]")


def load(path):
    path = Path(path).resolve()
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    check_types(raw)
    check_values(raw)
    base = path.parent
    raw["paths"]["data"] = (base / raw["paths"]["data_dir"]).resolve()
    raw["paths"]["figures"] = (base / raw["paths"]["figures_dir"]).resolve()
    return raw
