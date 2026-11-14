"""T junction cross slopes (grading.tee_crossfall, TEE_*): a T's stem lies on the through road's surface over its core
and eases back to its own cross slope over a runoff its speed and width set, so a side street on a hill no longer pulls
the through road's edge into a hump on one side of its mouth and a dip on the other."""
import math
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import grading  # noqa: E402
import roads  # noqa: E402

TRUNK = roads.Spec(16.0, "paved", False, "trunk", False)
PRIMARY = roads.Spec(14.0, "paved", True, "primary", False)
RESIDENTIAL = roads.Spec(7.0, "paved", True, "residential", False)
SERVICE = roads.Spec(5.0, "paved", True, "service", False)
GRADE = 0.08


def line(x0, y0, x1, y1, step=4.0):
    n = int(round(math.hypot(x1 - x0, y1 - y0) / step)) + 1
    return np.c_[np.linspace(x0, x1, n), np.linspace(y0, y1, n)]


def plan(xy, z, spec, rid):
    p = grading.Plan(np.asarray(xy, float), np.asarray(z, float), spec.width / 2 + 1.0, road_half=spec.width / 2)
    p.rid = rid
    p.rank = grading.grade_rank(spec)
    return p


def hill(xy):                   # the through road climbs GRADE along x; the ground also rises 3 % to the north
    return 50.0 + GRADE * xy[:, 0] + 0.03 * xy[:, 1]


def tee(through, stem, split=True, stem_end=(0.0, 150.0)):
    """A through road along x (one part, or two meeting at the node), a stem from the node to stem_end."""
    ps, sp = [], []
    if split:
        for k, (a, b) in enumerate(((-200.0, 0.0), (0.0, 200.0))):
            ln = line(a, 0.0, b, 0.0)
            ps.append(plan(ln, hill(ln), through, 1 + k))
            sp.append(through)
    else:
        ln = line(-200.0, 0.0, 200.0, 0.0)
        ps.append(plan(ln, hill(ln), through, 1))
        sp.append(through)
    st = line(0.0, 0.0, *stem_end)
    ps.append(plan(st, hill(st) + 0.3, stem, 9))      # (its own smoothing: 0.3 m off at the node)
    sp.append(stem)
    return ps, sp


class TeeNodesTest(unittest.TestCase):
    def test_a_t_is_found_with_its_stem(self):
        for split in (True, False):
            ps, sp = tee(PRIMARY, RESIDENTIAL, split)
            got = grading.tee_nodes(ps, sp)
            self.assertEqual(len(got), 1, f"split={split}")
            _node, stem, through, u = got[0]
            self.assertEqual(stem[0], len(ps) - 1)
            self.assertNotIn(stem[0], [b[0] for b in through])
            self.assertAlmostEqual(abs(float(u[0])), 1.0, places=6)

    def test_a_crossroads_is_no_t(self):
        ps, sp = tee(RESIDENTIAL, RESIDENTIAL)
        st = line(0.0, 0.0, 0.0, -150.0)
        ps.append(plan(st, hill(st), RESIDENTIAL, 10))
        sp.append(RESIDENTIAL)
        self.assertEqual(grading.tee_nodes(ps, sp), [])

    def test_a_merge_at_a_shallow_angle_is_no_t(self):
        ps, sp = tee(PRIMARY, RESIDENTIAL, stem_end=(150.0, 40.0))     # (15 degrees off the through road)
        self.assertEqual(grading.tee_nodes(ps, sp), [])

    def test_a_stem_ranked_over_the_through_road_is_no_t(self):
        ps, sp = tee(RESIDENTIAL, PRIMARY)
        self.assertEqual(grading.tee_nodes(ps, sp), [])


class EaseTest(unittest.TestCase):
    def test_the_runoff_rule(self):
        # (pi / 2) x 3.5 m x 0.05 / 0.70 % at 40 km/h (a residential's class speed): 39 m
        self.assertAlmostEqual(grading.tee_ease_m(RESIDENTIAL, 3.5, 0.05), 0.5 * math.pi * 3.5 * 0.05 / 0.0070, 6)

    def test_faster_and_wider_stems_warp_over_longer(self):
        slow = grading.tee_ease_m(SERVICE, 2.5, 0.06)
        street = grading.tee_ease_m(RESIDENTIAL, 3.5, 0.06)
        fast = grading.tee_ease_m(RESIDENTIAL._replace(kmh=70.0), 3.5, 0.06)
        self.assertLess(slow, street)
        self.assertLess(street, fast)

    def test_held_to_its_bounds_and_its_share_of_the_part(self):
        self.assertEqual(grading.tee_ease_m(SERVICE, 2.5, 0.001), grading.TEE_MIN_M)
        self.assertEqual(grading.tee_ease_m(PRIMARY, 7.0, 0.2), grading.TEE_MAX_M)
        self.assertAlmostEqual(grading.tee_ease_m(PRIMARY, 7.0, 0.2, room=50.0), grading.TEE_SHARE * 50.0, 6)


class TeeCrossfallTest(unittest.TestCase):
    def setUp(self):
        for name, val in (("TEE_ON", True), ("PAD_ON", False)):
            patcher = mock.patch.object(grading, name, val)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_case(self, through, stem, tilt, split=True):
        ps, sp = tee(through, stem, split)
        grading.snap_junctions(ps, sp)
        if tilt:
            grading.tee_crossfall(ps, sp)
        ps = list(ps)
        pack = grading.merge_plan_set(ps)
        xs = np.arange(-40.0, 40.01, 0.5)
        out = []
        for off in (through.width / 2 - 0.5, through.width / 2 + 0.5):
            pts = np.c_[xs, np.full(len(xs), off)]
            z = grading.grade_points(pts[:, 0], pts[:, 1], hill(pts), pack)
            out.append(float(np.abs(z - (50.0 + GRADE * xs)).max()))
        return out, pack

    def test_the_through_road_is_never_changed(self):
        ps, sp = tee(PRIMARY, RESIDENTIAL)
        grading.snap_junctions(ps, sp)
        before = [(p.z.copy(), p.bank) for p in ps[:2]]
        self.assertEqual(grading.tee_crossfall(ps, sp), 1)
        for p, (z, b) in zip(ps[:2], before):
            np.testing.assert_array_equal(p.z, z)
            self.assertIs(p.bank, b)

    def test_equal_streets_lanes_no_longer_humped(self):
        (lane0, _s0), _ = self.run_case(RESIDENTIAL, RESIDENTIAL, False)
        (lane1, _s1), _ = self.run_case(RESIDENTIAL, RESIDENTIAL, True)
        self.assertGreater(lane0, 0.10, "level stem: a hump and a dip in the through road's lanes")
        self.assertLess(lane1, 0.02, "tilted stem: the lanes on their own surface")

    def test_a_held_carriageways_shoulder_is_closer(self):
        (_l0, sh0), _ = self.run_case(PRIMARY, RESIDENTIAL, False)
        (_l1, sh1), _ = self.run_case(PRIMARY, RESIDENTIAL, True)
        self.assertLess(sh1, 0.5 * sh0)

    def test_a_trunks_lanes_are_closer(self):
        (l0, _s0), _ = self.run_case(TRUNK, RESIDENTIAL, False)
        (l1, _s1), _ = self.run_case(TRUNK, RESIDENTIAL, True)
        self.assertLess(l1, 0.5 * l0)

    def test_the_stem_warps_no_faster_than_its_speed_allows(self):
        _r, pack = self.run_case(PRIMARY, RESIDENTIAL, True)
        rh = RESIDENTIAL.width / 2 - 0.3
        ys = np.arange(PRIMARY.width / 2 + 1.0, 80.0, 1.0)
        left = grading.grade_points(np.full(len(ys), -rh), ys, hill(np.c_[np.full(len(ys), -rh), ys]), pack)
        right = grading.grade_points(np.full(len(ys), rh), ys, hill(np.c_[np.full(len(ys), rh), ys]), pack)
        rel = np.abs(np.diff(left - right)) / 2.0
        self.assertLess(float(rel.max()), 0.0075, "the edge against the centreline at most ~AASHTO's 0.70 % at 40 km/h")
        self.assertLess(float(np.abs(np.diff(left)).max()), 0.12, "no step where the stem leaves the through road")

    def test_on_level_ground_nothing_changes(self):
        ln = line(-200.0, 0.0, 200.0, 0.0)
        st = line(0.0, 0.0, 0.0, 150.0)
        ps = [plan(ln, np.full(len(ln), 50.0), PRIMARY, 1), plan(st, 50.0 + 0.03 * st[:, 1], RESIDENTIAL, 9)]
        sp = [PRIMARY, RESIDENTIAL]
        z = ps[1].z.copy()
        self.assertEqual(grading.tee_crossfall(ps, sp), 0)
        np.testing.assert_array_equal(ps[1].z, z)
        self.assertIsNone(ps[1].bank)

    def test_on_level_ground_an_equal_stem_left_standing_is_brought_down(self):
        # (a street split at the node, as Overture maps every junction: the snap takes an equal-ranked stem for its
        # continuation and leaves its end 0.3 m up, which pulled the lanes up)
        ln_w, ln_e, st = line(-200.0, 0.0, 0.0, 0.0), line(0.0, 0.0, 200.0, 0.0), line(0.0, 0.0, 0.0, 150.0)
        ps = [plan(ln_w, np.full(len(ln_w), 50.0), RESIDENTIAL, 1), plan(ln_e, np.full(len(ln_e), 50.0), RESIDENTIAL, 2),
              plan(st, np.full(len(st), 50.3), RESIDENTIAL, 9)]
        sp = [RESIDENTIAL] * 3
        grading.snap_junctions(ps, sp)
        self.assertAlmostEqual(float(ps[2].z[0]), 50.3, 6)
        self.assertEqual(grading.tee_crossfall(ps, sp), 1)
        self.assertAlmostEqual(float(ps[2].z[0]), 50.0, 6)
        self.assertLess(float(np.abs(np.diff(ps[2].z)).max()) / 4.0, 0.05, "eased out, not a step")

    def test_a_slip_road_leaving_a_trunk_is_untouched(self):
        ln_a, ln_b = line(-200.0, 0.0, 0.0, 0.0), line(0.0, 0.0, 200.0, 0.0)
        slip = line(0.0, 0.0, 120.0, 25.0)              # (12 degrees off the trunk)
        link = roads.Spec(7.0, "paved", False, "trunk_link", True, link=True)
        ps = [plan(ln_a, hill(ln_a), TRUNK, 1), plan(ln_b, hill(ln_b), TRUNK, 2), plan(slip, hill(slip), link, 9)]
        sp = [TRUNK, TRUNK, link]
        z = ps[2].z.copy()
        self.assertEqual(grading.tee_crossfall(ps, sp), 0)
        np.testing.assert_array_equal(ps[2].z, z)

    def steep_stem(self, climb):
        """A dirt road leaving an 8 % street at a right angle, climbing `climb` away from it (snapped to it first)."""
        ln_w, ln_e, st = line(-200.0, 0.0, 0.0, 0.0), line(0.0, 0.0, 200.0, 0.0), line(0.0, 0.0, 0.0, 120.0)
        track = roads.Spec(4.0, "unpaved", False, "track", False)
        ps = [plan(ln_w, hill(ln_w), RESIDENTIAL, 1), plan(ln_e, hill(ln_e), RESIDENTIAL, 2),
              plan(st, 50.0 + climb * st[:, 1], track, 9)]
        sp = [RESIDENTIAL, RESIDENTIAL, track]
        grading.snap_junctions(ps, sp)
        return ps, sp

    def check_steep(self, climb):
        ps, sp = self.steep_stem(climb)
        self.assertEqual(grading.tee_crossfall(ps, sp), 1)
        p = ps[2]
        d = grading._arcs(p)
        g = np.abs(np.gradient(p.z, d))
        core = d <= RESIDENTIAL.width / 2 + 1.0
        # over the through road: on its surface (an 8 % grade across the stem)
        np.testing.assert_allclose(np.abs(p.bank[core]), GRADE, atol=1e-6)
        # past it: grade and tilt together within what the mesh can follow
        past = ~core & (np.abs(p.bank) > 1e-9)
        if past.any():
            self.assertLessEqual(float(np.hypot(g, p.bank)[past].max()), grading.TEE_MAX_SURFACE + 1e-6)
        return p, core

    def test_a_steep_stem_is_tilted_no_further_than_its_mesh_can_follow(self):
        # (Canning Mills Road, Kelmscott: a dirt road falling 33 % from the asphalt, tilted 18 % on top, its strip
        # under the ground: roadmesh holds a mesh to a cell's lowest corner past FOLLOW_STEP_M)
        self.check_steep(0.20)

    def test_a_very_steep_stem_still_meets_the_road_then_twists_back(self):
        # (Hall Road, Roleystone: driveways at ~50 % left level put 13-22 cm humps in the kerb)
        p, core = self.check_steep(0.50)
        self.assertLess(float(np.abs(p.bank[~core]).max()), GRADE, "past the kerb, less than the full tilt")

    def test_off_and_beside_the_pads_as_before(self):
        for patch in (("TEE_ON", False), ("PAD_ON", True)):
            ps, sp = tee(PRIMARY, RESIDENTIAL)
            z = [p.z.copy() for p in ps]
            with mock.patch.object(grading, patch[0], patch[1]):
                self.assertEqual(grading.tee_crossfall(ps, sp), 0)
            for p, zz in zip(ps, z):
                np.testing.assert_array_equal(p.z, zz)
            self.assertIsNone(ps[-1].bank)


if __name__ == "__main__":
    unittest.main()
