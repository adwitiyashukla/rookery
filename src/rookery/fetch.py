import shutil
import time
from urllib.request import Request, urlopen

from rookery.store import raw_dir


def remote_size(url, timeout):
    req = Request(url, method="HEAD", headers={"User-Agent": "rookery"})
    with urlopen(req, timeout=timeout) as resp:
        length = resp.headers.get("Content-Length")
    return int(length) if length else None


def download(url, dest, retries, timeout):
    total = remote_size(url, timeout)
    if dest.exists() and total is not None and dest.stat().st_size == total:
        print(f"  ok {dest.name} ({total / 1e6:.1f} MB, already here)", flush=True)
        return
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if total is not None and have > total:
            dest.unlink()
            have = 0
        headers = {"User-Agent": "rookery"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as resp:
                mode = "ab" if have and resp.status == 206 else "wb"
                with open(dest, mode) as fh:
                    shutil.copyfileobj(resp, fh, 1 << 20)
        except OSError as exc:
            print(f"  attempt {attempt} for {dest.name} failed: {exc}", flush=True)
            time.sleep(min(30, 2 ** attempt))
            continue
        size = dest.stat().st_size
        if total is None or size == total:
            print(f"  ok {dest.name} ({size / 1e6:.1f} MB)", flush=True)
            return
    raise RuntimeError(f"could not download {url} after {retries} attempts")


def snap_files(name):
    return [f"com-{name}.ungraph.txt.gz", f"com-{name}.top5000.cmty.txt.gz"]


def run(cfg):
    fe = cfg["fetch"]
    snap = raw_dir(cfg, "snap")
    for name in fe["snap_graphs"]:
        for fname in snap_files(name):
            download(f"{fe['snap_url']}/{fname}", snap / fname, fe["retries"], fe["timeout_seconds"])
    se = raw_dir(cfg, "stackexchange")
    for site in fe["sites"]:
        download(f"{fe['stackexchange_url']}/{site}.7z", se / f"{site}.7z", fe["retries"], fe["timeout_seconds"])
