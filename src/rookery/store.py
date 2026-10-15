import json
import time
from contextlib import contextmanager


def site_key(site):
    return site.split(".")[0]


def raw_dir(cfg, kind):
    path = cfg["paths"]["data"] / "raw" / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def xml_dir(cfg, site):
    return cfg["paths"]["data"] / "xml" / site_key(site)


def lake(cfg, table):
    return cfg["paths"]["data"] / "lake" / table


def graph_path(cfg, name, kind):
    return cfg["paths"]["data"] / "graphs" / name / kind


def results_dir(cfg):
    path = cfg["paths"]["data"] / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(cfg, name, obj):
    path = results_dir(cfg) / f"{name}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def load_json(cfg, name, default=None):
    path = results_dir(cfg) / f"{name}.json"
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def update_json(cfg, name, key, value):
    data = load_json(cfg, name, {})
    data[key] = value
    save_json(cfg, name, data)


def record_timing(cfg, stage, graph, edges, seconds):
    rows = load_json(cfg, "timings", [])
    rows = [r for r in rows if not (r["stage"] == stage and r["graph"] == graph)]
    rows.append({"stage": stage, "graph": graph, "edges": edges, "seconds": round(seconds, 3)})
    save_json(cfg, "timings", rows)


@contextmanager
def timed(cfg, stage, graph, edges=None):
    start = time.perf_counter()
    yield
    seconds = time.perf_counter() - start
    record_timing(cfg, stage, graph, edges, seconds)
    print(f"  {stage} on {graph}: {seconds:.1f}s", flush=True)
