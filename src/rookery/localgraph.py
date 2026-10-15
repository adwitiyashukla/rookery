from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit


@dataclass
class Graph:
    ids: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray

    @property
    def n(self):
        return len(self.ids)

    def degree(self):
        return np.diff(self.indptr)

    def strength(self):
        return _row_sums(self.n, self.indptr, self.weights)


def index_nodes(src, dst):
    lo = int(min(src.min(), dst.min()))
    hi = int(max(src.max(), dst.max()))
    if hi - lo < 300_000_000:
        present = np.zeros(hi - lo + 1, dtype=np.bool_)
        present[src - lo] = True
        present[dst - lo] = True
        ids = np.flatnonzero(present).astype(np.int64) + lo
        del present
        lookup = np.full(hi - lo + 1, -1, dtype=np.int32)
        lookup[ids - lo] = np.arange(len(ids), dtype=np.int32)
        return ids, lookup[src - lo], lookup[dst - lo]
    ids, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
    return ids, inv[: len(src)].astype(np.int32), inv[len(src):].astype(np.int32)


@njit(cache=True)
def _csr(n, a, b, w, sym):
    deg = np.zeros(n + 1, dtype=np.int64)
    for i in range(len(a)):
        deg[a[i] + 1] += 1
        if sym:
            deg[b[i] + 1] += 1
    indptr = np.cumsum(deg)
    pos = indptr[:-1].copy()
    indices = np.empty(indptr[-1], dtype=np.int32)
    weights = np.empty(indptr[-1], dtype=w.dtype)
    for i in range(len(a)):
        u, v = a[i], b[i]
        indices[pos[u]] = v
        weights[pos[u]] = w[i]
        pos[u] += 1
        if sym:
            indices[pos[v]] = u
            weights[pos[v]] = w[i]
            pos[v] += 1
    return indptr, indices, weights


def from_arrays(src, dst, weight=None, symmetric=True):
    keep = src != dst
    if not keep.all():
        src, dst = src[keep], dst[keep]
        weight = None if weight is None else np.asarray(weight)[keep]
    del keep
    ids, a, b = index_nodes(src, dst)
    w = np.ones(len(a), dtype=np.float32) if weight is None else np.asarray(weight, dtype=np.float32)
    indptr, indices, weights = _csr(len(ids), a, b, w, symmetric)
    return Graph(ids, indptr, indices, weights)


def load(path, weighted=True, symmetric=True):
    cols = ["src", "dst", "weight"] if weighted else ["src", "dst"]
    table = pq.read_table(str(path), columns=cols)
    src = table.column("src").to_numpy()
    dst = table.column("dst").to_numpy()
    weight = table.column("weight").to_numpy() if weighted else None
    del table
    pa.default_memory_pool().release_unused()
    return from_arrays(src, dst, weight, symmetric)


@njit(cache=True)
def _triangles(n, indptr, indices):
    deg = indptr[1:] - indptr[:-1]
    optr = np.zeros(n + 1, dtype=np.int64)
    for u in range(n):
        c = 0
        for e in range(indptr[u], indptr[u + 1]):
            v = indices[e]
            if deg[u] < deg[v] or (deg[u] == deg[v] and u < v):
                c += 1
        optr[u + 1] = c
    optr = np.cumsum(optr)
    out = np.empty(optr[-1], dtype=np.int32)
    for u in range(n):
        k = optr[u]
        for e in range(indptr[u], indptr[u + 1]):
            v = indices[e]
            if deg[u] < deg[v] or (deg[u] == deg[v] and u < v):
                out[k] = v
                k += 1
    mark = np.full(n, -1, dtype=np.int64)
    tri = np.zeros(n, dtype=np.int64)
    total = 0
    for u in range(n):
        for e in range(optr[u], optr[u + 1]):
            mark[out[e]] = u
        for e in range(optr[u], optr[u + 1]):
            v = out[e]
            for f in range(optr[v], optr[v + 1]):
                w = out[f]
                if mark[w] == u:
                    tri[u] += 1
                    tri[v] += 1
                    tri[w] += 1
                    total += 1
    return tri, total


def triangles(g):
    return _triangles(g.n, g.indptr, g.indices)


@njit(cache=True)
def _kcore(n, indptr, indices):
    deg = indptr[1:] - indptr[:-1]
    top = deg.max() if n else 0
    start = np.zeros(top + 2, dtype=np.int64)
    for v in range(n):
        start[deg[v] + 1] += 1
    start = np.cumsum(start)
    nxt = start[:-1].copy()
    pos = np.empty(n, dtype=np.int64)
    vert = np.empty(n, dtype=np.int64)
    for v in range(n):
        pos[v] = nxt[deg[v]]
        vert[pos[v]] = v
        nxt[deg[v]] += 1
    head = start[:-1].copy()
    for i in range(n):
        v = vert[i]
        for e in range(indptr[v], indptr[v + 1]):
            u = indices[e]
            if deg[u] > deg[v]:
                du = deg[u]
                pu = pos[u]
                pw = head[du]
                w = vert[pw]
                if u != w:
                    pos[u] = pw
                    vert[pu] = w
                    pos[w] = pu
                    vert[pw] = u
                head[du] += 1
                deg[u] -= 1
    return deg


def kcore(g):
    return _kcore(g.n, g.indptr, g.indices)


@njit(cache=True)
def _one_level(n, indptr, indices, weights, k, m2, resolution, max_sweeps, min_gain, seed):
    np.random.seed(seed)
    comm = np.arange(n)
    tot = k.copy()
    order = np.random.permutation(n)
    acc = np.zeros(n)
    seen = np.zeros(n, dtype=np.bool_)
    lst = np.empty(n, dtype=np.int64)
    moved_any = False
    for _ in range(max_sweeps):
        moves = 0
        for idx in range(n):
            i = order[idx]
            ci = comm[i]
            ki = k[i]
            cnt = 0
            for e in range(indptr[i], indptr[i + 1]):
                j = indices[e]
                if j == i:
                    continue
                cj = comm[j]
                if not seen[cj]:
                    seen[cj] = True
                    lst[cnt] = cj
                    cnt += 1
                acc[cj] += weights[e]
            tot[ci] -= ki
            best_c = ci
            best = acc[ci] - resolution * tot[ci] * ki / m2
            for t in range(cnt):
                c = lst[t]
                gain = acc[c] - resolution * tot[c] * ki / m2
                if gain > best + min_gain:
                    best = gain
                    best_c = c
            tot[best_c] += ki
            if best_c != ci:
                comm[i] = best_c
                moves += 1
            for t in range(cnt):
                acc[lst[t]] = 0.0
                seen[lst[t]] = False
        if moves == 0:
            break
        moved_any = True
    return comm, moved_any


@njit(cache=True)
def _aggregate(n, indptr, indices, weights, new, nc):
    cnt = np.zeros(nc + 1, dtype=np.int64)
    for i in range(n):
        cnt[new[i] + 1] += 1
    start = np.cumsum(cnt)
    pos = start[:-1].copy()
    members = np.empty(n, dtype=np.int64)
    for i in range(n):
        members[pos[new[i]]] = i
        pos[new[i]] += 1
    seen = np.zeros(nc, dtype=np.bool_)
    lst = np.empty(nc, dtype=np.int64)
    acc = np.zeros(nc)
    optr = np.zeros(nc + 1, dtype=np.int64)
    for c in range(nc):
        m = 0
        for t in range(start[c], start[c + 1]):
            i = members[t]
            for e in range(indptr[i], indptr[i + 1]):
                d = new[indices[e]]
                if not seen[d]:
                    seen[d] = True
                    lst[m] = d
                    m += 1
        optr[c + 1] = m
        for t in range(m):
            seen[lst[t]] = False
    optr = np.cumsum(optr)
    out_idx = np.empty(optr[-1], dtype=np.int32)
    out_w = np.empty(optr[-1], dtype=np.float64)
    for c in range(nc):
        m = 0
        for t in range(start[c], start[c + 1]):
            i = members[t]
            for e in range(indptr[i], indptr[i + 1]):
                d = new[indices[e]]
                if not seen[d]:
                    seen[d] = True
                    lst[m] = d
                    m += 1
                acc[d] += weights[e]
        for t in range(m):
            d = lst[t]
            out_idx[optr[c] + t] = d
            out_w[optr[c] + t] = acc[d]
            acc[d] = 0.0
            seen[d] = False
    return optr, out_idx, out_w


@njit(cache=True)
def _row_sums(n, indptr, weights):
    k = np.zeros(n)
    for i in range(n):
        for e in range(indptr[i], indptr[i + 1]):
            k[i] += weights[e]
    return k


@njit(cache=True)
def _modularity(n, indptr, indices, weights, memb, resolution):
    nc = memb.max() + 1
    inside = np.zeros(nc)
    tot = np.zeros(nc)
    m2 = 0.0
    for i in range(n):
        ci = memb[i]
        for e in range(indptr[i], indptr[i + 1]):
            w = weights[e]
            m2 += w
            tot[ci] += w
            if memb[indices[e]] == ci:
                inside[ci] += w
    q = 0.0
    for c in range(nc):
        q += inside[c] / m2 - resolution * (tot[c] / m2) ** 2
    return q


def modularity(g, membership, resolution=1.0):
    return float(_modularity(g.n, g.indptr, g.indices, g.weights, membership.astype(np.int64), resolution))


def louvain(g, resolution, max_levels, max_sweeps, min_gain, seed):
    n, indptr, indices, weights = g.n, g.indptr, g.indices, g.weights
    k = _row_sums(n, indptr, weights)
    m2 = k.sum()
    membership = np.arange(g.n)
    sizes = []
    for level in range(max_levels):
        comm, moved = _one_level(n, indptr, indices, weights, k, m2, resolution, max_sweeps, min_gain, seed + level)
        if not moved:
            break
        uniq, new = np.unique(comm, return_inverse=True)
        membership = new[membership]
        sizes.append(len(uniq))
        indptr, indices, weights = _aggregate(n, indptr, indices, weights, new, len(uniq))
        n = len(uniq)
        k = _row_sums(n, indptr, weights)
    return membership, {"levels": len(sizes), "communities_per_level": sizes,
                        "modularity": modularity(g, membership, resolution)}


@njit(cache=True)
def _push(indptr, indices, weights, strength, seeds, seed_w, alpha, eps, p, r, inq, queue, touched):
    nq = len(queue)
    head, tail, size, nt = 0, 0, 0, 0
    for s in range(len(seeds)):
        u = seeds[s]
        if r[u] == 0.0 and p[u] == 0.0:
            touched[nt] = u
            nt += 1
        r[u] += seed_w[s]
    for s in range(len(seeds)):
        u = seeds[s]
        if not inq[u] and strength[u] > 0 and r[u] >= eps * strength[u]:
            inq[u] = True
            queue[tail] = u
            tail = (tail + 1) % nq
            size += 1
    while size > 0:
        u = queue[head]
        head = (head + 1) % nq
        size -= 1
        inq[u] = False
        ru = r[u]
        if ru < eps * strength[u]:
            continue
        p[u] += alpha * ru
        r[u] = 0.0
        share = (1.0 - alpha) * ru / strength[u]
        for e in range(indptr[u], indptr[u + 1]):
            v = indices[e]
            if r[v] == 0.0 and p[v] == 0.0:
                touched[nt] = v
                nt += 1
            r[v] += share * weights[e]
            if not inq[v] and r[v] >= eps * strength[v]:
                inq[v] = True
                queue[tail] = v
                tail = (tail + 1) % nq
                size += 1
    nodes = touched[:nt].copy()
    vals = np.empty(nt)
    for t in range(nt):
        x = nodes[t]
        vals[t] = p[x]
        p[x] = 0.0
        r[x] = 0.0
        inq[x] = False
    return nodes, vals


class PushPPR:
    def __init__(self, g, alpha, eps):
        self.g = g
        self.alpha = alpha
        self.eps = eps
        self.strength = _row_sums(g.n, g.indptr, g.weights)
        self.p = np.zeros(g.n)
        self.r = np.zeros(g.n)
        self.inq = np.zeros(g.n, dtype=np.bool_)
        self.queue = np.empty(g.n, dtype=np.int64)
        self.touched = np.empty(g.n, dtype=np.int64)

    def run(self, seeds, seed_w):
        seed_w = np.asarray(seed_w, dtype=np.float64)
        seed_w = seed_w / seed_w.sum()
        return _push(self.g.indptr, self.g.indices, self.g.weights, self.strength, np.asarray(seeds, dtype=np.int64),
                     seed_w, self.alpha, self.eps, self.p, self.r, self.inq, self.queue, self.touched)


def warm_up():
    g = from_arrays(np.array([0, 1, 2, 0, 3]), np.array([1, 2, 0, 3, 4]))
    triangles(g)
    kcore(g)
    louvain(g, 1.0, 2, 2, 1e-9, 1)
    PushPPR(g, 0.15, 1e-4).run([0], [1.0])
