"""Asphalt colour from the aerial photo (GitHub #466: red asphalt, drawn the same grey as every other road).

A paved road may be painted in a colour category instead of the default grey (surface.PHOTO_ROAD_RGBA "paved") when
the aerial photo shows that colour along it. Three rules (the user, 2026-10-31):

- Accuracy, OpenStreetMap decides first. A road whose OSM surface tag (osmtags "osm_surface") or Overture's surface
  rule names anything but asphalt is never coloured (concrete, paving stones, ... keep their own look). An OSM
  surface:colour value where the map data carries one (attrs "osm_surface_colour") is taken as it is, photo or not.
  Layercake's highways file, osmtags' source, has no surface:colour column (its sql/highways.sql, read 2026-10-31),
  so until it does the photo alone decides.
- Consistency: the categories are OpenStreetMap surface:colour values (CATEGORIES), nothing OSM would not say.
- Safety, in doubt the default: too few clean samples, a red share short of RED_SHARE, a photo from a coarse source
  (Sentinel-2's 10 m pixel is wider than a street) or one with gaps.

One decision per road (by feature id) for every photo it crosses, so a road is never red in one block and grey in the
next: the first fine photo that sees enough of it decides (learn), and the decision is kept in a small SQLite file in
the server's cache folder (Store), shared by the server and its worker processes, the first writer winning. Photos
painted before a road is decided paint it the default.

Cost: sampling reads three pixels a metre of each undecided candidate road in a photo that may learn, every road of
the photo at once (numpy, no fetch): ~60 ms for 400 new roads in a 1 km town block photo (2026-10-31, one core), once;
after that, and for every other photo, a dictionary lookup a road (~0.3 ms for 400).
"""
import logging
import os
import sqlite3
import threading

import numpy as np

log = logging.getLogger("earthd.roadcolour")

VERSION = 1             # bump when the rules below change: decisions made under the old ones are dropped (Store path)

# category (an OpenStreetMap surface:colour value) -> the colour it is painted (sRGB, beside "paved"'s (95, 95, 94))
CATEGORIES = {"red": (122, 74, 66)}
# surface rule tokens (roads.surface_token) that are asphalt, so may carry a colour; anything else keeps its own look
ASPHALT = frozenset({"asphalt", "paved", "tar", "bitumen", "chipseal"})

# Which photos may decide: every source in it z17 or finer (about 1 m a pixel; Sentinel-2 and USGS's z16 are not), no
# holes, and texels no coarser than LEARN_MAX_TEXEL_M (a far ring rendered at 2 m from a fine source would blur the
# road into its verge).
FINE_ZOOM = 17
LEARN_MAX_TEXEL_M = 1.5
NOT_SOURCES = frozenset({"colormatch", "deshadow", "leafon"})     # render()'s `used` keys that are no source
HOLES = frozenset({"none", "polar"})

# Sampling: points every SAMPLE_STEP_M along the road, on its centre line and LANE_SHARE of its width to each side
# (the middle third: the map's line is often a metre or two off the photo's road, and the verge must not be read).
SAMPLE_STEP_M = 1.0
LANE_SHARE = 1.0 / 6.0
# A decision needs MIN_SEEN_M of the road in the photo, or the whole road when it is shorter (SEEN_SHARE of it).
MIN_SEEN_M = 50.0
SEEN_SHARE = 0.8
# Clean samples: neither shadow nor glare / painted lines (mean brightness), at least MIN_CLEAN_SHARE of those read.
DARK, BRIGHT = 40.0, 200.0
MIN_CLEAN_SHARE = 0.5
# A red sample: red the largest channel, saturated, its hue within RED_HUE of pure red (brown soil and sand are
# around 25-40 degrees). The road is red when RED_SHARE of its clean samples are (cars, patches and shade stay under).
RED_SAT = 0.20
RED_HUE = (-20.0, 22.0)
RED_SHARE = 0.6


def fine_photo(used, chain, texel_m):
    """Whether a photo may decide colours: imagery.render's `used` (source name -> texels, and its other counts), the
    chain it was rendered from (imagery.Source), its texel size in metres."""
    if not used or texel_m > LEARN_MAX_TEXEL_M or any(used.get(k) for k in HOLES):
        return False
    zoom = {s.name: s.max_zoom for s in chain}
    srcs = [k for k, n in used.items() if n and k not in NOT_SOURCES]
    return bool(srcs) and all(zoom.get(k, 0) >= FINE_ZOOM for k in srcs)


def tag_colour(feats, i):
    """The road's OpenStreetMap surface:colour (lower case, "gray" as "grey"), None without one. Any value decides:
    a category is painted, any other (grey, black, ...) is the default, and the photo is not asked."""
    tags = feats.attrs.get("osm_surface_colour")
    if tags is None or tags[i] is None:
        return None
    v = str(tags[i]).strip().lower()
    if not v or v == "none":
        return None
    return "grey" if v == "gray" else v


def candidate(feats, i, spec):
    """Whether the photo may colour this road: a paved carriageway that neither OpenStreetMap's surface tag nor
    Overture's surface rule says is something other than asphalt; with neither saying anything, only in town (out of
    town an untagged "paved" road may be a red dirt track). spec: its roads.road_specs entry."""
    import roads as roadsmod
    if spec is None or spec.kind != "paved":
        return False
    osm = feats.attrs.get("osm_surface")
    osm = str(osm[i]).strip() if osm is not None and osm[i] is not None else ""
    osm = roadsmod.surface_token(osm) if osm and osm != "None" else None     # (osmtags: '' where the way has none)
    if osm is not None:                 # OpenStreetMap's own tag first
        return osm in ASPHALT
    if spec.surf is not None:
        return spec.surf in ASPHALT
    return bool(spec.urban)


def sample_classes(pixels):
    """Per sample of (k, 3) uint8: (clean, red) bool arrays (k,). Pure."""
    p = np.asarray(pixels, np.float32).reshape(-1, 3)
    mean = p.mean(axis=1)
    clean = (mean >= DARK) & (mean <= BRIGHT)
    r, g, b = p[:, 0], p[:, 1], p[:, 2]
    mx, mn = p.max(axis=1), p.min(axis=1)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    hue = 60.0 * (g - b) / np.maximum(mx - mn, 1e-3)        # degrees from red, where red is the largest channel
    red = clean & (r >= mx) & (sat >= RED_SAT) & (hue >= RED_HUE[0]) & (hue <= RED_HUE[1])
    return clean, red


def decide(n, n_clean, n_red):
    """A category, "" (the default, decided) or None (not decided) from a road's sample counts."""
    if n_clean == 0 or n_clean < MIN_CLEAN_SHARE * n:
        return None
    return "red" if n_red >= RED_SHARE * n_clean else ""


def classify(pixels):
    """One road's samples (k, 3) uint8 -> decide()'s answer. Pure."""
    clean, red = sample_classes(pixels)
    return decide(len(clean), int(clean.sum()), int(red.sum()))


def sample_points(rings, px_m):
    """Sample positions along polylines in pixels: rings [(cols (n,), rows (n,), half width px)]. Every SAMPLE_STEP_M
    on each centre line and LANE_SHARE of its width to each side. (positions (k, 2) col, row; ring of each (k,); each
    ring's length in metres (len(rings),)), all rings at once."""
    p0, d, hw, ring = [], [], [], []
    for k, (c, r, half_w) in enumerate(rings):
        p = np.column_stack((np.asarray(c, np.float64), np.asarray(r, np.float64)))
        p0.append(p[:-1])
        d.append(p[1:] - p[:-1])
        hw.append(np.full(len(p) - 1, float(half_w)))
        ring.append(np.full(len(p) - 1, k, np.int64))
    if not p0:
        return np.empty((0, 2)), np.empty(0, np.int64), np.zeros(0)
    p0, d, hw, ring = np.concatenate(p0), np.concatenate(d), np.concatenate(hw), np.concatenate(ring)
    seg = np.hypot(d[:, 0], d[:, 1])
    lengths = np.bincount(ring, seg, minlength=len(rings)) / px_m
    keep = seg > 0.0
    p0, d, hw, ring, seg = p0[keep], d[keep], hw[keep], ring[keep], seg[keep]
    n = np.maximum(1, (seg / (SAMPLE_STEP_M * px_m)).astype(np.int64))     # samples per segment
    which = np.repeat(np.arange(len(seg)), n)
    first = np.cumsum(n) - n
    t = (np.arange(len(which)) - first[which] + 0.5) / n[which]
    mid = p0[which] + t[:, None] * d[which]
    nrm = np.c_[-d[:, 1], d[:, 0]][which] / seg[which, None] * (2.0 * hw[which, None] * LANE_SHARE)
    return np.concatenate([mid - nrm, mid, mid + nrm]), np.tile(ring[which], 3), lengths


class Store:
    """The decisions (feature id -> category, or "" for the default), kept in memory and, with a folder, in
    roadcolour_v<VERSION>.sqlite there for every process: the first decision for a road stands (INSERT OR IGNORE).
    Any trouble with the file leaves the decisions in memory only (never a failed photo)."""

    def __init__(self, folder=None):
        self._lock = threading.Lock()
        self._known = {}
        self._db = None
        if folder:
            try:
                os.makedirs(folder, exist_ok=True)
                db = sqlite3.connect(os.path.join(folder, f"roadcolour_v{VERSION}.sqlite"), timeout=5.0,
                                     check_same_thread=False, isolation_level=None)
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("CREATE TABLE IF NOT EXISTS decided (fid TEXT PRIMARY KEY, colour TEXT NOT NULL)")
                self._db = db
            except sqlite3.Error as e:
                log.info("road colours kept in memory only: %s", e)

    def get(self, fids):
        """{fid: category or ""} of the decided ones among fids."""
        with self._lock:
            out = {f: self._known[f] for f in fids if f in self._known}
            ask = [f for f in fids if f not in self._known]
            if ask and self._db is not None:
                try:
                    for k in range(0, len(ask), 500):
                        part = ask[k:k + 500]
                        rows = self._db.execute(f"SELECT fid, colour FROM decided WHERE fid IN "
                                                f"({','.join('?' * len(part))})", part).fetchall()
                        for f, col in rows:
                            self._known[f] = out[f] = col
                except sqlite3.Error as e:
                    log.debug("road colours: %s", e)
            return out

    def decide(self, fid, colour):
        """Keep a decision unless one was made before (here or in another process); the one that stands."""
        return self.decide_many({fid: colour})[fid]

    def decide_many(self, new):
        """decide() for {fid: colour} in one transaction: {fid: the decision that stands}."""
        with self._lock:
            out = {f: self._known[f] for f in new if f in self._known}
            ask = {f: c for f, c in new.items() if f not in out}
            if ask and self._db is not None:
                try:
                    self._db.execute("BEGIN")
                    self._db.executemany("INSERT OR IGNORE INTO decided (fid, colour) VALUES (?, ?)",
                                         list(ask.items()))
                    self._db.execute("COMMIT")
                    keys = list(ask)
                    for k in range(0, len(keys), 500):
                        part = keys[k:k + 500]
                        for f, col in self._db.execute(f"SELECT fid, colour FROM decided WHERE fid IN "
                                                       f"({','.join('?' * len(part))})", part):
                            ask[f] = col
                except sqlite3.Error as e:
                    log.debug("road colours: %s", e)
                    try:
                        self._db.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
            self._known.update(ask)
            out.update(ask)
            return out


_store = (None, None)          # (folder, Store)
_store_lock = threading.Lock()


def store():
    """This process's Store, in the server's cache folder (revisit.CACHE_DIR) when it has one."""
    global _store
    import revisit
    with _store_lock:
        if _store[1] is None or _store[0] != revisit.CACHE_DIR:
            _store = (revisit.CACHE_DIR, Store(revisit.CACHE_DIR))
        return _store[1]


def colours(feats, specs, ids, learn=None):
    """The category of each road of feats (one Features) as painted: {fid: category} for the coloured ones (the rest
    are the default). ids: [(index in feats, fid)] to answer for; specs: roads.road_specs(feats). learn: None, or
    (rgb, rings, px_m) to decide the undecided candidates from: the photo before roads are painted on it, rings(i) ->
    [(cols, rows, half width px)], each ground ring of feature i in its pixels, and its pixels a metre."""
    st = store()
    out = {}
    cand = {}
    for i, fid in ids:
        if specs[i] is None or specs[i].kind != "paved":
            continue
        t = tag_colour(feats, i)
        if t is not None:                       # OpenStreetMap's colour: no photo needed
            if t in CATEGORIES:
                out[fid] = t
            continue
        if candidate(feats, i, specs[i]):
            cand[fid] = i
    if not cand:
        return out
    known = st.get(list(cand))
    for fid, col in known.items():
        if col:
            out[fid] = col
    if learn is None:
        return out
    rgb, rings, px_m = learn
    h, w = rgb.shape[:2]
    todo = [(fid, i) for fid, i in cand.items() if fid not in known]
    all_rings, road_of = [], []
    for k, (fid, i) in enumerate(todo):
        for ring in rings(i):
            all_rings.append(ring)
            road_of.append(k)
    if not all_rings:
        return out
    # every undecided road's samples at once (per-road numpy calls cost ~0.2 ms each: 80 ms for a town's block photo)
    m = len(todo)
    road_of = np.asarray(road_of, np.int64)
    p, ring, lengths = sample_points(all_rings, px_m)
    total = np.bincount(road_of, lengths, minlength=m)
    inside = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    p, road = p[inside].astype(np.int64), road_of[ring[inside]]
    clean, red = sample_classes(rgb[p[:, 1], p[:, 0]])
    n = np.bincount(road, minlength=m)
    n_clean = np.bincount(road, clean, minlength=m)
    n_red = np.bincount(road, red, minlength=m)
    new = {}
    for k, (fid, _) in enumerate(todo):
        if n[k] / 3.0 * SAMPLE_STEP_M < min(MIN_SEEN_M, SEEN_SHARE * total[k]):
            continue                            # not enough of it here: a later photo decides
        col = decide(int(n[k]), int(n_clean[k]), int(n_red[k]))
        if col is not None:
            new[fid] = col
    if new:
        for fid, col in st.decide_many(new).items():
            if col:
                out[fid] = col
                log.debug("road %s: %s", fid, col)
    return out
