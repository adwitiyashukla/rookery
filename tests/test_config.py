import tomllib
from pathlib import Path

import pytest

from rookery import config

ROOT = Path(__file__).resolve().parents[1]


def write(tmp_path, raw):
    lines = []
    for section, values in raw.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            if isinstance(value, bool):
                text = "true" if value else "false"
            elif isinstance(value, str):
                text = f'"{value}"'
            elif isinstance(value, list):
                text = "[" + ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in value) + "]"
            else:
                text = repr(value)
            lines.append(f"{key} = {text}")
    path = tmp_path / "config.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def base():
    with open(ROOT / "config.ci.toml", "rb") as fh:
        return tomllib.load(fh)


@pytest.mark.parametrize("name", ["config.toml", "config.ci.toml"])
def test_shipped_configs_load(name):
    cfg = config.load(ROOT / name)
    assert cfg["paths"]["data"].is_absolute()


@pytest.mark.parametrize(("section", "key", "value", "message"), [
    ("pagerank", "alpha", 1.5, "alpha"),
    ("spark", "driver_memory", "lots", "driver_memory"),
    ("routing", "test_start", "2022-01-01", "train_start < test_start"),
    ("routing", "k_values", [10, 5], "increasing"),
    ("routing", "candidates", 5, "largest k"),
    ("fetch", "snap_graphs", ["orkut"], "subset"),
    ("figures", "tag_map_site", "example.com", "tag_map_site"),
    ("lpa", "max_iter", True, "integer"),
    ("walks", "q", 0.0, "positive"),
])
def test_bad_values_are_rejected(tmp_path, section, key, value, message):
    raw = base()
    raw[section][key] = value
    with pytest.raises(config.ConfigError, match=message):
        config.load(write(tmp_path, raw))


def test_unknown_and_missing_keys(tmp_path):
    raw = base()
    raw["pagerank"]["damping"] = 0.85
    with pytest.raises(config.ConfigError, match="unknown keys"):
        config.load(write(tmp_path, raw))
    raw = base()
    del raw["kcore"]["max_iter"]
    with pytest.raises(config.ConfigError, match="missing key"):
        config.load(write(tmp_path, raw))
