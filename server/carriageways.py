"""A motorway's carriageway as one profile through its roads and its bridges (BNE_INTERCHANGE, stage 2 of the
interchange proposal; the user, 2026-11-01: "keep the road deck smooth through bridges so they transition to the next
bridge smoothly").

Grading planned a road up to the mapped end of its bridge from the elevation there, and roads.BridgeProfiles laid the
deck from the ground at its two ends and lifted it over what it crosses: two answers that often did not meet. Mapped
ends sit on embankments' slopes, at the edge of a cutting or down a gully, so the road dived into the bridge, and a deck
clearing a railway just past its end started metres over the road (Perth's Mitchell Freeway through Northbridge:
1.8-4 m steps at its bridges).

Here a carriageway is a chain of parts of one class family (motorway, trunk and their links), ground and bridge, joined
end to end, between branch points (where three or more such parts meet) and loose ends (a tunnel's portal, the map's
edge). Its profile is the smoothest one near the ground it can be trusted for:

    minimise  sum w_i (z_i - t_i)^2 + LAMBDA * sum (z_{i-1} - 2 z_i + z_{i+1})^2
    subject to lo_i <= z_i <= hi_i

t: the ground along it, median-filtered and smoothed as a major road's plan is; w: 1 on the ground, 0 on a bridge, rising
from 0 at a bridge's end to 1 TRUST_M from it (the slopes and cuttings by an abutment are not the road's); lo: over a
railway or another carriageway passing under one of its bridges, that line's height + the clearance; hi: under another
carriageway's deck, that deck less the clearance. The bounds are met by a few rounds of the solve with the violated ones
pinned. The ends are held at their ground.

Streets under its bridges give way (grading's underpass dips, measured from this profile): only tracks and other
carriageways bound it. The deck then runs between ends that lie on the profile, so roads and decks meet.
"""
import math

import numpy as np

STEP_M = 4.0            # m between profile samples
TRUST_M = 40.0          # m from a bridge's end over which the ground's weight rises from 0 to 1
LAMBDA = 6.0e4          # smoothness: ~ (50 m / STEP_M) ** 4, a vertical curve of about 50 m per metre of bend
END_W = 1.0e4           # weight holding a chain's ends at their ground
ROUNDS = 6              # bound rounds
TOUCH_M = 0.75          # m: parts whose ends are this close are joined
FAMILY = ("motorway", "trunk")
MAX_DIG_M = 3.0         # m under its ground another carriageway's deck may push one at most (the deck rises for the rest)
LEVELS = 3              # rounds of the carriageways crossing each other bounding each other


def family(cls):
    """'motorway' for motorway and motorway_link, 'trunk' likewise; None for anything else."""
    c = str(cls or "")
    base = c[:-5] if c.endswith("_link") else c
    return base if base in FAMILY else None


def chains(parts):
    """parts: [(key, line (k, 2), is_bridge, cls)] (one family's parts near a place, each once). Returns the chains:
    [[(part index, reversed)]] in order along each, each a maximal run joined end to end through points where exactly
    two parts meet. Deterministic: a chain starts from the end of its lowest-keyed part that is a chain end, or for a
    loop, from its lowest-keyed part. Pure."""
    n = len(parts)
    if not n:
        return []
    ends = np.array([[p[1][0, :2], p[1][-1, :2]] for p in parts], np.float64).reshape(-1, 2)    # 2i, 2i+1

    def meets(e):
        d = np.hypot(ends[:, 0] - ends[e, 0], ends[:, 1] - ends[e, 1])
        return [int(k) for k in np.flatnonzero(d <= TOUCH_M) if k != e]
    nb = [meets(e) for e in range(2 * n)]
    order = sorted(range(n), key=lambda i: str(parts[i][0]))
    used = np.zeros(n, bool)
    out = []

    def walk(i, rev):
        seq = []
        while True:
            used[i] = True
            seq.append((i, rev))
            e = 2 * i + (0 if rev else 1)              # the far end as walked
            if len(nb[e]) != 1:
                return seq
            k = nb[e][0]
            j = k // 2
            if used[j]:
                return seq
            i, rev = j, (k % 2 == 1)                    # (entered at its last point: walked reversed)
    for i in order:
        if used[i]:
            continue
        # walk back to this chain's start
        j, rev = i, False
        seen = {i}
        while True:
            e = 2 * j + (1 if rev else 0)               # the near end as walked
            if len(nb[e]) != 1:
                break
            k = nb[e][0]
            jj = k // 2
            if jj in seen or used[jj]:
                break
            seen.add(jj)
            j, rev = jj, (k % 2 == 0)                   # (its end at k is its first point: we walk it reversed)
        out.append(walk(j, rev))
    return out


def sample(parts, chain, step=STEP_M):
    """The chain's samples: (xy (m, 2), s (m,), part index (m,), bridge (m,) bool, from_end (m,): m to the nearest
    bridge end along it). Pure."""
    xs, pi, br = [], [], []
    for i, rev in chain:
        ln = np.asarray(parts[i][1], np.float64)[:, :2]
        if rev:
            ln = ln[::-1]
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(ln, axis=0).T))]
        m = max(2, int(math.ceil(s[-1] / step)) + 1)
        a = np.linspace(0.0, s[-1], m)
        pts = np.stack([np.interp(a, s, ln[:, 0]), np.interp(a, s, ln[:, 1])], axis=1)
        if xs:
            pts = pts[1:]                               # (the joint once)
        xs.append(pts)
        pi += [i] * len(pts)
        br += [bool(parts[i][2])] * len(pts)
    xy = np.concatenate(xs)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    pi, br = np.array(pi), np.array(br)
    # bridge ends along the chain: where the bridge flag changes
    cut = s[np.flatnonzero(br[1:] != br[:-1])] if len(br) > 1 else np.zeros(0)
    cut = np.r_[cut, s[np.flatnonzero(br[1:] != br[:-1]) + 1]] if len(cut) else cut
    from_end = np.min(np.abs(s[:, None] - cut[None, :]), axis=1) if len(cut) else np.full(len(s), np.inf)
    return xy, s, pi, br, from_end


def weights(br, from_end, trust=TRUST_M):
    """0 on a bridge, rising (smoothstep) to 1 trust m from a bridge's end on the ground. Pure."""
    u = np.clip(from_end / trust, 0.0, 1.0)
    w = u * u * (3.0 - 2.0 * u)
    return np.where(br, 0.0, w)


def _solve_banded(w, t, lam):
    """argmin sum w (z - t)^2 + lam sum (D2 z)^2: (diag(w) + lam D2'D2) z = w t, pentadiagonal, by Cholesky. Pure."""
    n = len(t)
    if n < 3:
        return np.asarray(t, np.float64).copy()
    # bands of D2'D2: main, first, second
    d0 = np.zeros(n)
    d1 = np.zeros(n - 1)
    d2 = np.zeros(n - 2)
    c = np.array([1.0, -2.0, 1.0])
    for k in range(n - 2):          # each row of D2 touches k, k+1, k+2
        for a in range(3):
            d0[k + a] += c[a] * c[a]
        d1[k] += c[0] * c[1]
        d1[k + 1] += c[1] * c[2]
        d2[k] += c[0] * c[2]
    A0 = np.asarray(w, np.float64) + lam * d0
    A1 = lam * d1
    A2 = lam * d2
    # banded Cholesky (L lower, bandwidth 2): L[i,i], L[i,i-1], L[i,i-2]
    l0 = np.zeros(n)
    l1 = np.zeros(n)
    l2 = np.zeros(n)
    for i in range(n):
        if i >= 2:
            l2[i] = A2[i - 2] / l0[i - 2]
        if i >= 1:
            l1[i] = (A1[i - 1] - (l2[i] * l1[i - 1] if i >= 2 else 0.0)) / l0[i - 1]
        v = A0[i] - l1[i] ** 2 - l2[i] ** 2
        l0[i] = math.sqrt(max(v, 1e-12))
    b = np.asarray(w, np.float64) * np.asarray(t, np.float64)
    y = np.zeros(n)
    for i in range(n):
        acc = b[i]
        if i >= 1:
            acc -= l1[i] * y[i - 1]
        if i >= 2:
            acc -= l2[i] * y[i - 2]
        y[i] = acc / l0[i]
    z = np.zeros(n)
    for i in range(n - 1, -1, -1):
        acc = y[i]
        if i + 1 < n:
            acc -= l1[i + 1] * z[i + 1]
        if i + 2 < n:
            acc -= l2[i + 2] * z[i + 2]
        z[i] = acc / l0[i]
    return z


def profile(t, w, lo=None, hi=None, lam=LAMBDA, end_w=END_W, rounds=ROUNDS):
    """The smooth profile (see the module): t, w per sample; lo, hi bounds (or None, -inf/inf where none); the two ends
    held at t. Pure."""
    t = np.asarray(t, np.float64)
    n = len(t)
    w = np.asarray(w, np.float64).copy()
    # (an end on the ground is held there; one on a bridge, a joint with another carriageway's deck, runs on free)
    if w[0] > 0:
        w[0] = end_w
    if w[-1] > 0:
        w[-1] = end_w
    t = t.copy()
    lo = np.full(n, -np.inf) if lo is None else np.asarray(lo, np.float64)
    hi = np.full(n, np.inf) if hi is None else np.asarray(hi, np.float64)
    pin_w = np.zeros(n)
    pin_t = np.zeros(n)
    z = _solve_banded(w, t, lam)
    for _ in range(rounds):
        under = z < lo - 0.02
        over = z > hi + 0.02
        if not under.any() and not over.any():
            break
        pin_w = np.where(under | over, end_w, pin_w)
        pin_t = np.where(under, lo, np.where(over, hi, pin_t))
        ww = w + pin_w
        tt = np.where(pin_w > 0, (w * t + pin_w * pin_t) / np.maximum(ww, 1e-12), t)
        z = _solve_banded(ww, tt, lam)
    return z
