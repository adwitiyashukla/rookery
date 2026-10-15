import gzip
import shutil

import numpy as np
import py7zr

from rookery.fetch import snap_files
from rookery.store import raw_dir

DUMP_END = np.datetime64("2024-03-31T23:00:00", "s")
TOPICS = 40

SITE_FIELDS = ("users", "questions", "answers", "comments", "tags", "start")

SITES = {
    "unix.stackexchange.com": (420_000, 232_000, 330_000, 900_000, 2_700, 2010),
    "serverfault.com": (640_000, 320_000, 540_000, 1_100_000, 3_900, 2009),
    "askubuntu.com": (1_500_000, 400_000, 550_000, 1_300_000, 3_100, 2010),
    "superuser.com": (1_300_000, 480_000, 720_000, 1_700_000, 5_300, 2009),
    "math.stackexchange.com": (1_100_000, 1_670_000, 2_450_000, 7_000_000, 1_950, 2010),
}

SNAP = {
    "amazon": (334_863, 925_872),
    "dblp": (317_080, 1_049_866),
    "youtube": (1_134_890, 2_987_624),
    "lj": (3_997_962, 34_681_189),
}

WORDS = [
    "kernel", "shell", "package", "config", "driver", "network", "disk", "boot", "update", "install", "error",
    "proof", "limit", "integral", "matrix", "series", "prime", "group", "space", "vector", "server", "proxy",
    "route", "mount", "cache", "script", "loop", "file",
]


def timeline(rng, n, start_year):
    start = np.datetime64(f"{start_year}-08-01T00:00:00", "s")
    span = int((DUMP_END - start).astype(np.int64))
    offsets = np.sort((rng.random(n) ** 0.75 * span).astype(np.int64))
    return start + offsets.astype("timedelta64[s]")


def stamp(ts):
    return np.datetime_as_string(ts.astype("datetime64[ms]"), unit="ms")


def bodies(rng, n):
    pool = []
    for length in (120, 300, 600, 1200, 2400):
        words = " ".join(rng.choice(WORDS, size=length // 6))
        pool.append(f"&lt;p&gt;{words[:length]}&lt;/p&gt;")
    return np.array(pool, dtype=object)[rng.integers(0, len(pool), n)]


def gumbel_top(rng, weights, k):
    keys = np.log(weights) - np.log(-np.log(rng.random(len(weights))))
    return np.argpartition(-keys, k)[:k]


def weighted_prefix(rng, cum, limit):
    limit = np.maximum(limit, 1)
    r = rng.random(len(limit)) * cum[limit - 1]
    return np.minimum(np.searchsorted(cum, r, side="right"), limit - 1)


def write_xml(path, root, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f'<?xml version="1.0" encoding="utf-8"?>\n<{root}>\n')
        for chunk in rows:
            fh.write("".join(chunk))
        fh.write(f"</{root}>\n")


def attr(name, values, mask=None):
    out = np.array([f' {name}="{v}"' for v in values], dtype=object)
    if mask is not None:
        out[~mask] = ""
    return out


def site_tables(rng, spec, scale, site_index, pool_weights):
    nu = max(int(spec["users"] * scale), 50)
    nq = max(int(spec["questions"] * scale), 40)
    na = max(int(spec["answers"] * scale), 40)
    nc = max(int(spec["comments"] * scale), 40)
    ntag = max(int(spec["tags"] * min(1.0, scale * 20)), 30)
    topic_p = rng.dirichlet(np.full(TOPICS, 0.8))
    u_created = timeline(rng, nu, spec["start"])
    answerer = rng.random(nu) < 0.12
    ans_w = np.where(answerer, rng.pareto(1.1, nu) + 1.0, 0.0)
    ask_w = rng.lognormal(0.0, 1.0, nu) * (rng.random(nu) < 0.55) + 1e-3
    com_w = 0.3 * ans_w + rng.lognormal(-1.0, 1.0, nu) * (rng.random(nu) < 0.4) + 1e-3
    u_topic = rng.choice(TOPICS, nu, p=topic_p)
    u_topic2 = rng.choice(TOPICS, nu, p=topic_p)
    account = 100_000_000 * (site_index + 1) + np.arange(1, nu + 1)
    shared = np.flatnonzero(rng.random(nu) < 0.18)[: len(pool_weights) - 1]
    if len(shared):
        account[shared] = gumbel_top(rng, pool_weights, len(shared)) + 1
    tag_topic = rng.choice(TOPICS, ntag, p=topic_p)
    tag_names = np.array([f"{rng.choice(WORDS)}-{i}" for i in range(ntag)], dtype=object)
    q_time = timeline(rng, nq, spec["start"])
    u_sec = u_created.astype(np.int64)
    eligible_q = np.searchsorted(u_sec, q_time.astype(np.int64), side="right")
    asker = weighted_prefix(rng, np.cumsum(ask_w), eligible_q)
    q_topic = np.where(rng.random(nq) < 0.6, u_topic[asker], rng.choice(TOPICS, nq, p=topic_p))
    ntags = rng.choice([1, 2, 3, 4, 5], nq, p=[0.15, 0.25, 0.3, 0.18, 0.12])
    q_tags = np.empty((nq, 5), dtype=np.int64)
    for t in range(TOPICS):
        idx = np.flatnonzero(q_topic == t)
        own = np.flatnonzero(tag_topic == t)
        if len(own) == 0:
            own = np.arange(ntag)
        w = np.cumsum(1.0 / (np.arange(len(own)) + 1.0) ** 1.1)
        picks = np.searchsorted(w, rng.random((len(idx), 5)) * w[-1], side="right")
        q_tags[idx] = own[np.minimum(picks, len(own) - 1)]
    mean_ans = na / nq
    n_ans = rng.negative_binomial(2, 2 / (2 + mean_ans), nq)
    a_q = np.repeat(np.arange(nq), n_ans)
    delay = rng.lognormal(8.5, 2.2, len(a_q)).astype(np.int64)
    a_time = q_time[a_q] + delay.astype("timedelta64[s]")
    keep = a_time <= DUMP_END
    a_q, a_time = a_q[keep], a_time[keep]
    order = np.argsort(a_time, kind="stable")
    a_q, a_time = a_q[order], a_time[order]
    eligible_a = np.searchsorted(u_sec, a_time.astype(np.int64), side="right")
    a_user = weighted_prefix(rng, np.cumsum(ans_w + 1e-9), eligible_a)
    a_topic = q_topic[a_q]
    expert = rng.random(len(a_q)) < 0.85
    for t in range(TOPICS):
        experts = np.flatnonzero(answerer & ((u_topic == t) | (u_topic2 == t)))
        sel = np.flatnonzero(expert & (a_topic == t))
        if len(experts) == 0 or len(sel) == 0:
            continue
        lim = np.searchsorted(experts, eligible_a[sel] - 1, side="right")
        ok = lim > 0
        cum = np.cumsum(ans_w[experts])
        a_user[sel[ok]] = experts[weighted_prefix(rng, cum, lim[ok])]
    a_score = rng.poisson(0.5 + 1.5 * np.log1p(ans_w[a_user]))
    q_score = rng.poisson(1.2, nq)
    accepted = np.full(nq, -1, dtype=np.int64)
    answered = np.flatnonzero(np.bincount(a_q, minlength=nq) > 0)
    chosen = answered[rng.random(len(answered)) < 0.5]
    first_answer = np.full(nq, -1, dtype=np.int64)
    uq, first_idx = np.unique(a_q, return_index=True)
    first_answer[uq] = first_idx
    accepted[chosen] = first_answer[chosen]
    post_time = np.concatenate([q_time, a_time])
    post_owner = np.concatenate([asker, a_user])
    post_score = np.concatenate([q_score, a_score])
    counterpart = np.concatenate([np.where(first_answer >= 0, a_user[np.maximum(first_answer, 0)], -1), asker[a_q]])
    c_post = np.searchsorted(np.cumsum(1.0 + post_score), rng.random(nc) * (1.0 + post_score).sum(), side="right")
    c_post = np.minimum(c_post, len(post_time) - 1)
    c_time = post_time[c_post] + rng.lognormal(7.5, 2.0, nc).astype(np.int64).astype("timedelta64[s]")
    keep = c_time <= DUMP_END
    c_post, c_time = c_post[keep], c_time[keep]
    eligible_c = np.searchsorted(u_sec, c_time.astype(np.int64), side="right")
    c_user = weighted_prefix(rng, np.cumsum(com_w), eligible_c)
    use_cp = (rng.random(len(c_post)) < 0.3) & (counterpart[c_post] >= 0)
    c_user[use_cp] = counterpart[c_post][use_cp]
    links = []
    for link_type, frac in ((3, 0.03), (1, 0.07)):
        src = rng.choice(nq, int(nq * frac), replace=False)
        for t in range(TOPICS):
            pool = np.flatnonzero(q_topic == t)
            sel = src[q_topic[src] == t]
            if len(pool) < 2 or len(sel) == 0:
                continue
            pos = np.searchsorted(pool, sel)
            ok = pos > 0
            tgt = pool[(rng.random(ok.sum()) * pos[ok]).astype(np.int64)]
            links.append(np.column_stack([sel[ok], tgt, np.full(ok.sum(), link_type)]))
    links = np.concatenate(links) if links else np.zeros((0, 3), dtype=np.int64)
    return {
        "nu": nu, "u_created": u_created, "account": account, "ans_w": ans_w, "q_time": q_time, "asker": asker,
        "q_tags": q_tags, "ntags": ntags, "q_score": q_score, "a_q": a_q, "a_time": a_time, "a_user": a_user,
        "a_score": a_score, "accepted": accepted, "n_ans": np.bincount(a_q, minlength=nq), "c_post": c_post,
        "c_time": c_time, "c_user": c_user, "post_owner": post_owner, "links": links, "tag_names": tag_names,
        "post_time": post_time,
    }


def chunks(n, size=100_000):
    for start in range(0, n, size):
        yield start, min(n, start + size)


def write_site(rng, site, spec, scale, site_index, pool_weights, workdir, out_file):
    d = site_tables(rng, spec, scale, site_index, pool_weights)
    nu, nq, na = d["nu"], len(d["q_time"]), len(d["a_q"])
    folder = workdir / site
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    order = np.argsort(d["post_time"], kind="stable")
    post_id = np.empty(nq + na, dtype=np.int64)
    post_id[order] = np.arange(1, nq + na + 1)
    q_id, a_id = post_id[:nq], post_id[nq:]
    created = stamp(d["u_created"])
    rep = (1 + d["ans_w"] * 40).astype(np.int64)

    account = d["account"]

    def users():
        yield [f'  <row Id="-1" Reputation="1" CreationDate="{created[0]}" DisplayName="Community" AccountId="-1" />\n']
        for s, e in chunks(nu):
            yield [f'  <row Id="{i + 1}" Reputation="{rep[i]}" CreationDate="{created[i]}" DisplayName="user{i + 1}" '
                   f'LastAccessDate="{created[i]}" Views="0" AccountId="{account[i]}" />\n' for i in range(s, e)]

    write_xml(folder / "Users.xml", "users", users())
    p_time = stamp(d["post_time"])
    p_owner = d["post_owner"] + 1
    alive = rng.random(nq + na) >= 0.02
    owner_attr = np.where(alive, np.char.add(np.char.add(' OwnerUserId="', p_owner.astype(str)), '"'),
                          ' OwnerDisplayName="deleted"').astype(object)
    body = bodies(rng, nq + na)
    names = d["tag_names"]
    acc = d["accepted"]

    def qrow(i):
        tags = "|" + "|".join(dict.fromkeys(names[d["q_tags"][i, : d["ntags"][i]]])) + "|"
        accepted = f' AcceptedAnswerId="{a_id[acc[i]]}"' if acc[i] >= 0 else ""
        return (f'  <row Id="{q_id[i]}" PostTypeId="1"{accepted} CreationDate="{p_time[i]}" '
                f'Score="{d["q_score"][i]}" ViewCount="{10 + 7 * d["q_score"][i]}" Body="{body[i]}"{owner_attr[i]} '
                f'LastActivityDate="{p_time[i]}" Title="question {q_id[i]}" Tags="{tags}" '
                f'AnswerCount="{d["n_ans"][i]}" CommentCount="0" ContentLicense="CC BY-SA 4.0" />\n')

    def arow(j):
        i = nq + j
        return (f'  <row Id="{a_id[j]}" PostTypeId="2" ParentId="{q_id[d["a_q"][j]]}" CreationDate="{p_time[i]}" '
                f'Score="{d["a_score"][j]}" Body="{body[i]}"{owner_attr[i]} CommentCount="0" '
                f'ContentLicense="CC BY-SA 4.0" />\n')

    def posts():
        for s, e in chunks(nq + na):
            yield [qrow(i) if i < nq else arow(i - nq) for i in order[s:e]]

    write_xml(folder / "Posts.xml", "posts", posts())
    first = d["q_tags"].copy()
    retag = np.flatnonzero(rng.random(nq) < 0.13)
    slot = (rng.random(len(retag)) * d["ntags"][retag]).astype(np.int64)
    first[retag, slot] = rng.integers(0, len(names), len(retag))
    delay = rng.lognormal(9.0, 1.5, len(retag)).astype(np.int64).astype("timedelta64[s]")
    later = np.minimum(d["q_time"][retag] + delay, DUMP_END)
    user_attr = np.where(alive, np.char.add(np.char.add(' UserId="', p_owner.astype(str)), '"'), "").astype(object)

    def tag_text(table, i):
        return "|" + "|".join(dict.fromkeys(names[table[i, : d["ntags"][i]]])) + "|"

    def hrow(n, kind, i, when, user, text):
        return (f'  <row Id="{n}" PostHistoryTypeId="{kind}" PostId="{post_id[i]}" CreationDate="{when}"{user} '
                f'Text="{text}" ContentLicense="CC BY-SA 4.0" />\n')

    def history():
        n = 0
        for s, e in chunks(nq + na):
            out = []
            for i in order[s:e]:
                kinds = [(2, body[i])]
                if i < nq:
                    kinds += [(1, f"question {q_id[i]}"), (3, tag_text(first, i))]
                for kind, text in kinds:
                    n += 1
                    out.append(hrow(n, kind, i, p_time[i], user_attr[i], text))
            yield out
        yield [hrow(n + k + 1, 6, i, stamp(t), "", tag_text(d["q_tags"], i))
               for k, (i, t) in enumerate(zip(retag, later, strict=True))]

    write_xml(folder / "PostHistory.xml", "posthistory", history())
    c_order = np.argsort(d["c_time"], kind="stable")
    c_time = stamp(d["c_time"])
    c_alive = rng.random(len(c_order)) >= 0.01
    c_text = bodies(rng, len(c_order))

    def comments():
        for s, e in chunks(len(c_order)):
            out = []
            for n, i in enumerate(c_order[s:e], start=s + 1):
                user = f' UserId="{d["c_user"][i] + 1}"' if c_alive[i] else ' UserDisplayName="gone"'
                out.append(f'  <row Id="{n}" PostId="{post_id[d["c_post"][i]]}" Score="0" Text="{c_text[i][:300]}" '
                           f'CreationDate="{c_time[i]}"{user} ContentLicense="CC BY-SA 4.0" />\n')
            yield out

    write_xml(folder / "Comments.xml", "comments", comments())
    links = d["links"]
    l_time = stamp(d["q_time"][links[:, 0]]) if len(links) else []
    write_xml(folder / "PostLinks.xml", "postlinks", [[
        f'  <row Id="{n + 1}" CreationDate="{l_time[n]}" PostId="{q_id[r[0]]}" RelatedPostId="{q_id[r[1]]}" '
        f'LinkTypeId="{r[2]}" />\n' for n, r in enumerate(links)]])
    counts = np.bincount(d["q_tags"][np.arange(5)[None, :] < d["ntags"][:, None]], minlength=len(names))
    write_xml(folder / "Tags.xml", "tags", [[
        f'  <row Id="{i + 1}" TagName="{names[i]}" Count="{counts[i]}" />\n' for i in range(len(names))]])
    for extra, root in (("Badges.xml", "badges"), ("Votes.xml", "votes")):
        write_xml(folder / extra, root, [[f'  <row Id="1" CreationDate="{p_time[0]}" />\n']])
    if out_file.exists():
        out_file.unlink()
    with py7zr.SevenZipFile(out_file, "w", filters=[{"id": py7zr.FILTER_LZMA2, "preset": 1}]) as zf:
        for f in sorted(folder.iterdir()):
            zf.write(f, arcname=f.name)
    shutil.rmtree(folder)
    return {"users": nu, "questions": nq, "answers": na, "comments": len(c_order), "links": len(links),
            "retagged": len(retag)}


def snap_graph(rng, n_nodes, n_edges):
    member = np.minimum(1 + rng.geometric(0.55, n_nodes), 8)
    slots = np.repeat(np.arange(n_nodes), member)
    rng.shuffle(slots)
    sizes = []
    total = 0
    while total < len(slots):
        batch = np.minimum((3 * (1 - rng.random(10_000)) ** (-1 / 1.3)).astype(np.int64), 2000)
        sizes.append(batch)
        total += int(batch.sum())
    sizes = np.concatenate(sizes)
    cut = np.searchsorted(np.cumsum(sizes), len(slots))
    sizes = sizes[: cut + 1]
    sizes[-1] -= int(sizes.sum()) - len(slots)
    sizes = sizes[sizes > 0]
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    theta = rng.pareto(1.6, n_nodes) + 1.0
    cum_theta = np.cumsum(theta)
    p_comm = np.cumsum(sizes.astype(float) ** 1.1)
    keys = np.zeros(0, dtype=np.int64)
    want = n_edges
    while len(keys) < want:
        need = int((want - len(keys)) * 1.15) + 1000
        n_in = int(need * 0.85)
        c = np.searchsorted(p_comm, rng.random(n_in) * p_comm[-1], side="right")
        c = np.minimum(c, len(sizes) - 1)
        u = slots[ptr[c] + (rng.random(n_in) * sizes[c]).astype(np.int64)]
        v = slots[ptr[c] + (rng.random(n_in) * sizes[c]).astype(np.int64)]
        n_bg = need - n_in
        bu = np.searchsorted(cum_theta, rng.random(n_bg) * cum_theta[-1], side="right")
        bv = np.searchsorted(cum_theta, rng.random(n_bg) * cum_theta[-1], side="right")
        a = np.concatenate([u, np.minimum(bu, n_nodes - 1)])
        b = np.concatenate([v, np.minimum(bv, n_nodes - 1)])
        ok = a != b
        lo, hi = np.minimum(a[ok], b[ok]), np.maximum(a[ok], b[ok])
        keys = np.unique(np.concatenate([keys, lo * n_nodes + hi]))
    keys = keys[rng.permutation(len(keys))[:want]]
    ids = np.sort(rng.choice(int(n_nodes * 1.6), n_nodes, replace=False)) + 1
    src, dst = ids[keys // n_nodes], ids[keys % n_nodes]
    big = np.flatnonzero(sizes >= 3)
    top = big[rng.permutation(len(big))[:5000]]
    cmty = [ids[np.unique(slots[ptr[c]: ptr[c + 1]])] for c in top]
    return src, dst, cmty


def write_snap(rng, name, n_nodes, n_edges, folder):
    src, dst, cmty = snap_graph(rng, n_nodes, n_edges)
    edge_file, cmty_file = (folder / f for f in snap_files(name))
    nodes = len(np.unique(np.concatenate([src, dst])))
    with gzip.open(edge_file, "wt", compresslevel=4, newline="\n") as fh:
        fh.write(f"# Undirected graph: {name}.ungraph.txt\n# synthetic replica\n"
                 f"# Nodes: {nodes} Edges: {len(src)}\n# FromNodeId\tToNodeId\n")
        for s, e in chunks(len(src), 1_000_000):
            fh.write("".join(f"{a}\t{b}\n" for a, b in zip(src[s:e].tolist(), dst[s:e].tolist(), strict=True)))
    with gzip.open(cmty_file, "wt", compresslevel=6, newline="\n") as fh:
        for members in cmty:
            fh.write("\t".join(map(str, members.tolist())) + "\n")
    return {"nodes": nodes, "edges": len(src), "communities": len(cmty)}


def run(cfg):
    sy = cfg["synth"]
    rng = np.random.default_rng(sy["seed"])
    snap_out = raw_dir(cfg, "snap")
    for name in cfg["fetch"]["snap_graphs"]:
        n, m = SNAP[name]
        info = write_snap(rng, name, max(int(n * sy["snap_scale"]), 200), max(int(m * sy["snap_scale"]), 600),
                          snap_out)
        print(f"  synthetic {name}: {info}", flush=True)
    se_out = raw_dir(cfg, "stackexchange")
    work = cfg["paths"]["data"] / "tmp" / "synth"
    pool = 1.0 / (np.arange(2_000_000) + 50.0) ** 0.6
    for index, site in enumerate(cfg["fetch"]["sites"]):
        spec = dict(zip(SITE_FIELDS, SITES.get(site, SITES["unix.stackexchange.com"]), strict=True))
        info = write_site(rng, site, spec, sy["scale"], index, pool, work, se_out / f"{site}.7z")
        print(f"  synthetic {site}: {info}", flush=True)
