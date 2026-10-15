import argparse
import importlib
import sys
import time

from rookery import config

STAGES = {
    "fetch": "rookery.fetch",
    "synth": "rookery.synth",
    "ingest": "rookery.ingest",
    "build": "rookery.build",
    "mine": "rookery.mine",
    "crosscheck": "rookery.crosscheck",
    "communities": "rookery.communities",
    "embed": "rookery.embed",
    "route": "rookery.routing",
    "figures": "rookery.figures",
}

PIPELINE = ["ingest", "build", "mine", "crosscheck", "communities", "embed", "route", "figures"]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rookery", description="graph mining on Stack Exchange and SNAP networks")
    parser.add_argument("stage", choices=[*STAGES, "all", "check", "verify"])
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args(argv)
    cfg = config.load(args.config)
    if args.stage == "check":
        print(f"config ok: {args.config}")
        return
    if args.stage == "verify":
        from rookery.crosscheck import verify

        sys.exit(0 if verify(cfg) else 1)
    stages = PIPELINE if args.stage == "all" else [args.stage]
    try:
        for stage in stages:
            print(f"[{stage}]", flush=True)
            start = time.perf_counter()
            importlib.import_module(STAGES[stage]).run(cfg)
            print(f"[{stage}] done in {time.perf_counter() - start:.1f}s", flush=True)
    finally:
        from rookery.sparkctx import stop

        stop()
