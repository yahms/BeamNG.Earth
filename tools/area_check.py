#!/usr/bin/env python3
"""Plan an area's roads offline from a cache folder and check the freeway / ramp profiles (the interchange solver's
quality and speed, without the game).

  BNE_INTERCHANGE=1 python3 tools/area_check.py --bbox -31.955,115.842,-31.940,115.856 --cache /path/to/cache
  python3 tools/area_check.py --bbox ... --cache ... --report-only      # the last run's plans again, no planning

--cache: a server cache folder (dem/, mapdata/, roadwidth/ ...) with the area's map data and elevation already in it
(a played area's own cache, or a copy); nothing is downloaded (offline). Each run plans into a run folder of its own
under it (plans/<run>), so the plans of a code change are made fresh. The report lists, per road class, the roads
over their grade or vertical-curve limit and the joins where two of them meet with a step or a break in grade; with
--fid, one road's profile. --json writes the numbers for comparing runs (tools/area_check.py --compare a.json b.json).
"""
import argparse
import glob
import json
import multiprocessing
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(os.path.dirname(HERE), "server")
sys.path.insert(0, SERVER)

# (grade %, K m/%) a road of the class should keep: a report line where one is past GRADE_OVER x its grade or under
# K_UNDER x its K (carriageways.LIMITS' spirit; a street's are the junction-level ones)
LIMITS = {"motorway": (5.0, 30.0), "motorway_link": (7.0, 12.0), "trunk": (6.0, 20.0), "trunk_link": (8.0, 10.0),
          "primary": (8.0, 8.0), "primary_link": (8.0, 6.0), "secondary": (10.0, 5.0)}
GRADE_OVER = 1.3
K_UNDER = 0.5
STEP_M = 0.15           # m: a join with a step over this is reported
BREAK_PCT = 4.0         # %: ... or a break in grade over this
WIN_M = 10.0            # m: grades are measured over this (a road's samples are ~4 m apart)


def _worker(args):
    cache, cells, wid, run = args
    import logging
    import numpy as np
    import blockworker as bw
    import mapcells
    import plancache
    os.environ[plancache.RUN_ENV] = run
    bw.init(cache, True, True, True, True, True, None, getattr(logging, os.environ.get("AREA_LOG", "WARNING")), False,
            True, 0, None, False)
    g, _ = bw._s.anchor(1, cells[0][3], cells[0][4])
    times = []
    for z, x, y, _la, _lo in cells:
        k = mapcells.CellKey(z, x, y)
        t0 = time.perf_counter()
        try:
            w, s, e, n = k.bounds()
            LA, LO = np.meshgrid(np.linspace(s, n, 3), np.linspace(w, e, 3))
            g.prepare(LA.ravel(), LO.ravel())
            err = None
        except Exception as ex:
            err = f"{type(ex).__name__}: {ex}"
        times.append(((z, x, y), time.perf_counter() - t0, err))
        print(f"  worker {wid}: cell {z}/{x}/{y} {times[-1][1]:.1f} s" + (f" ({err})" if err else ""), flush=True)
    return times


def plan(bbox, cache, procs):
    os.environ.setdefault("BNE_REVISIT", "off")    # (a place's kept plans would be read instead of planned again)
    import mapcells
    import plancache
    s, w, n, e = bbox
    keys = mapcells.cells_for_bbox(s, w, n, e, max_cells=100000)
    cells = [(k.z, k.x, k.y, *k.center()) for k in keys]
    plancache.start_run(cache)
    run = os.environ[plancache.RUN_ENV]
    print(f"{len(cells)} cells, {procs} processes, run {run}", flush=True)
    t0 = time.perf_counter()
    jobs = [(cache, cells[i::procs], i, run) for i in range(procs) if cells[i::procs]]
    if len(jobs) == 1:
        times = _worker(jobs[0])
    else:
        with multiprocessing.get_context("spawn").Pool(len(jobs)) as pool:
            times = [t for ts in pool.map(_worker, jobs) for t in ts]
    wall = time.perf_counter() - t0
    cell_s = sorted(t for _k, t, _e in times)
    print(f"planned in {wall:.1f} s wall; per cell median {cell_s[len(cell_s) // 2]:.1f} s, max {cell_s[-1]:.1f} s, "
          f"sum {sum(cell_s):.1f} s", flush=True)
    return run, {"wall_s": wall, "cells": len(cells), "cell_s": {f"{k[0]}/{k[1]}/{k[2]}": t for k, t, _e in times},
                 "errors": {f"{k[0]}/{k[1]}/{k[2]}": e for k, _t, e in times if e}}


def _classes(keys):
    """{fid: (class, name)} of the roads of cells keys, from the cached map data (their v2 cells, read directly:
    no fetcher, nothing downloaded)."""
    import numpy as np
    out = {}
    for path in _CELL_FILES:
        try:
            zf = np.load(path, allow_pickle=False)
        except OSError:
            continue
        cl, sub, nm = zf["attr.class"], zf["attr.subclass"], zf["attr.name"]
        for i, fid in enumerate(zf["ids"]):
            out[str(fid)] = (str(cl[i]) + ("_link" if str(sub[i]) == "link" else ""), str(nm[i]))
    return out


_CELL_FILES = []


def report(cache, run, bbox, fids=(), as_json=None, timing=None):
    import numpy as np
    import geotiles
    import mapcells
    import plancache
    s, w, n, e = bbox
    keys = mapcells.cells_for_bbox(s, w, n, e, max_cells=100000)
    pc = plancache.PlanCache(os.path.join(cache, "plans", run), "")
    roads = {}                  # fid: (xy in one frame, z, frame)
    fr0 = geotiles.region_frame(*geotiles.region_of(*keys[0].center()))
    sigs = glob.glob(os.path.join(cache, "plans", run, plancache.FRAMES, "*"))
    missing = []
    for k in keys:
        got = None
        for sd in sigs:
            pc.dir = sd
            got = pc.load((k.z, k.x, k.y))
            if got is not None:
                break
        if got is None:
            missing.append(k)
            continue
        fr = geotiles.region_frame(*geotiles.region_of_cell(k))
        for fid, p in got[0]:
            xy = np.asarray(p.xy, np.float64)[:, :2]
            if fid in roads and len(roads[fid][1]) >= len(xy):
                continue
            if fr is not fr0:
                la, lo, _ = fr.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
                ex, ny, _ = fr0.geodetic_to_enu(np.asarray(la), np.asarray(lo), 0.0)
                xy = np.c_[ex, ny]
            roads[fid] = (xy, np.asarray(p.z, np.float64))
    _CELL_FILES[:] = [f for k in keys for f in glob.glob(os.path.join(cache, "mapdata", "v2", "overture", "*", "roads.*", str(k.z), str(k.x), f"{k.y}.npz"))]
    cls = _classes(keys)

    def ll(p):
        la, lo, _ = fr0.enu_to_geodetic(np.array([p[0]]), np.array([p[1]]), 0.0)
        return f"{float(la[0]):.5f},{float(lo[0]):.5f}"

    def resample(xy, z, h=2.0):
        sd = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        if sd[-1] < 3 * h:
            return None
        t = np.arange(0, sd[-1] + 1e-9, h)
        return t, np.interp(t, sd, z), np.c_[np.interp(t, sd, xy[:, 0]), np.interp(t, sd, xy[:, 1])]

    over = []
    stats = {}
    for fid, (xy, z) in roads.items():
        c, name = cls.get(fid, ("?", ""))
        if c not in LIMITS:
            continue
        r = resample(xy, z)
        if r is None:
            continue
        t, zz, pxy = r
        m = int(WIN_M / 2.0)
        if len(zz) <= 2 * m:
            continue
        g = (zz[m:] - zz[:-m]) / WIN_M
        dg = np.abs(g[m:] - g[:-m]) * 100.0
        K = WIN_M / np.maximum(dg, 1e-9)
        gl, kl = LIMITS[c]
        gi, ki = int(np.argmax(np.abs(g))), int(np.argmin(K))
        st = stats.setdefault(c, {"roads": 0, "m": 0.0, "over_grade_m": 0.0, "under_K_m": 0.0})
        st["roads"] += 1
        st["m"] += float(t[-1])
        st["over_grade_m"] += float((np.abs(g) * 100 > gl * GRADE_OVER).sum() * 2.0)
        st["under_K_m"] += float((K < kl * K_UNDER).sum() * 2.0)
        if abs(g[gi]) * 100 > gl * GRADE_OVER or K[ki] < kl * K_UNDER:
            over.append((c, name, fid, float(t[-1]), float(abs(g[gi]) * 100), float(K[ki]),
                         ll(pxy[gi + m // 2]), ll(pxy[ki + m])))
    print(f"\n{len(roads)} roads planned, {len(missing)} cells without plans"
          + (": " + " ".join(f"{k.z}/{k.x}/{k.y}" for k in missing) if missing else ""))
    print("\nper class: roads, km, m over the grade limit, m under the K limit")
    for c in LIMITS:
        if c in stats:
            st = stats[c]
            print(f"  {c:14s} {st['roads']:4d} {st['m'] / 1000:6.2f} km  {st['over_grade_m']:6.0f} m  "
                  f"{st['under_K_m']:6.0f} m")
    print(f"\nroads past {GRADE_OVER:g} x their grade or under {K_UNDER:g} x their K (worst first):")
    over.sort(key=lambda o: -max(o[4] / LIMITS[o[0]][0], LIMITS[o[0]][1] / max(o[5], 1e-9)))
    for c, name, fid, L, gm, km, at_g, at_k in over[:40]:
        print(f"  {c:14s} {name[:20]:20s} {fid[:8]} {L:5.0f} m  grade {gm:5.1f} % at {at_g}  K {km:5.1f} at {at_k}")

    ends = []
    for fid, (xy, z) in roads.items():
        c = cls.get(fid, ("?", ""))[0]
        if not (c.startswith("motorway") or c.startswith("trunk")) or len(z) < 3:
            continue
        for a, b in ((0, 1), (-1, -2)):
            run_ = max(float(np.hypot(*(xy[a] - xy[b]))), 1e-6)
            ends.append((fid, c, xy[a], float(z[a]), float(z[a] - z[b]) / run_))
    joins = []
    P = np.array([e_[2] for e_ in ends]) if ends else np.zeros((0, 2))
    for i in range(len(ends)):
        if not len(P):
            break
        d = np.hypot(*(P[i + 1:] - P[i]).T)
        for j in np.flatnonzero(d < 1.5) + i + 1:
            A, B = ends[i], ends[j]
            if A[0] == B[0]:
                continue
            dz, br = A[3] - B[3], (A[4] + B[4]) * 100
            if abs(dz) > STEP_M or abs(br) > BREAK_PCT:
                joins.append((A, B, dz, br))
    joins.sort(key=lambda j_: -max(abs(j_[2]) / STEP_M, abs(j_[3]) / BREAK_PCT))
    print(f"\nfreeway / ramp joins with a step over {STEP_M} m or a grade break over {BREAK_PCT} % ({len(joins)}):")
    for A, B, dz, br in joins[:25]:
        print(f"  {A[0][:8]} {A[1]:14s} / {B[0][:8]} {B[1]:14s} step {dz:+.2f} m  break {br:+5.1f} %  at {ll(A[2])}")

    for pre in fids:
        for fid, (xy, z) in roads.items():
            if fid.startswith(pre):
                r = resample(xy, z, 4.0)
                if r is None:
                    continue
                t, zz, _ = r
                print(f"\n{fid} ({cls.get(fid, ('?', ''))[0]}) {t[-1]:.0f} m from {ll(xy[0])} to {ll(xy[-1])}")
                print("  z  " + " ".join(f"{v:.1f}" for v in zz))
                print("  g% " + " ".join(f"{v:.0f}" for v in np.r_[np.diff(zz) / 4.0 * 100, 0]))
    if as_json:
        with open(as_json, "w") as fh:
            json.dump({"run": run, "bbox": bbox, "stats": stats, "timing": timing,
                       "over": [list(o) for o in over], "joins": [[A[0], B[0], dz, br] for A, B, dz, br in joins]},
                      fh, indent=1)


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    print(f"{'class':14s} {'over grade m':>16s} {'under K m':>16s}")
    for c in sorted(set(A["stats"]) | set(B["stats"])):
        sa, sb = A["stats"].get(c, {}), B["stats"].get(c, {})
        print(f"{c:14s} {sa.get('over_grade_m', 0):7.0f} -> {sb.get('over_grade_m', 0):6.0f} "
              f"{sa.get('under_K_m', 0):7.0f} -> {sb.get('under_K_m', 0):6.0f}")
    print(f"roads over a limit {len(A['over'])} -> {len(B['over'])}, joins {len(A['joins'])} -> {len(B['joins'])}")
    if A.get("timing") and B.get("timing"):
        print(f"planning wall {A['timing']['wall_s']:.1f} s -> {B['timing']['wall_s']:.1f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", help="S,W,N,E")
    ap.add_argument("--cache")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--report-only", action="store_true", help="report the last run's plans, no planning")
    ap.add_argument("--run", help="report this run folder's plans (plans/<run>)")
    ap.add_argument("--fid", nargs="*", default=[], help="print these roads' profiles (id prefixes)")
    ap.add_argument("--json", help="write the numbers here")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    if not (a.bbox and a.cache):
        ap.error("--bbox and --cache")
    bbox = tuple(map(float, a.bbox.split(",")))
    timing = None
    if a.run:
        run = a.run
    elif a.report_only:
        runs = sorted(os.listdir(os.path.join(a.cache, "plans")),
                      key=lambda r: os.path.getmtime(os.path.join(a.cache, "plans", r)))
        run = runs[-1]
    else:
        run, timing = plan(bbox, a.cache, a.procs)
    report(a.cache, run, bbox, a.fid, a.json, timing)


if __name__ == "__main__":
    main()
