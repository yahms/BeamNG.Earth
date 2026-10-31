"""Asphalt colour from the aerial photo (roadcolour.py, GitHub #466): the sample classifier, which photos may decide,
which roads may be coloured (OpenStreetMap first), one decision per road kept for every photo, and the photo painted."""
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import geodesy  # noqa: E402
import mapgeom  # noqa: E402
import revisit  # noqa: E402
import roadcolour as rc  # noqa: E402
import roads  # noqa: E402
import surface as sf  # noqa: E402
import test_roads  # noqa: E402

GREY = (92, 92, 90)             # worn asphalt
RED = (140, 82, 72)             # red asphalt in an aerial photo
BROWN = (150, 120, 80)          # sandy soil (hue ~25 degrees)
PINDAN = (165, 95, 60)          # red soil beside a road
GRASS = (70, 95, 50)
PAINT_RED = rc.CATEGORIES["red"]


def pixels(*parts):
    """(k, 3) samples: [(colour, count)]."""
    return np.concatenate([np.tile(np.array(c, np.uint8), (n, 1)) for c, n in parts])


class ClassifyTest(unittest.TestCase):
    def test_grey_asphalt_is_the_default(self):
        self.assertEqual(rc.classify(pixels((GREY, 100))), "")

    def test_red_asphalt_is_red(self):
        self.assertEqual(rc.classify(pixels((RED, 100))), "red")

    def test_red_with_cars_and_patches_is_still_red(self):
        self.assertEqual(rc.classify(pixels((RED, 70), (GREY, 20), ((230, 230, 235), 10))), "red")

    def test_brown_soil_is_not_red(self):
        self.assertEqual(rc.classify(pixels((BROWN, 100))), "")

    def test_a_red_minority_is_the_default(self):
        self.assertEqual(rc.classify(pixels((RED, 50), (GREY, 50))), "", "in doubt: the default")

    def test_shade_and_glare_decide_nothing(self):
        self.assertIsNone(rc.classify(pixels(((20, 18, 18), 80), (RED, 20))), "mostly shadow: not decided")
        self.assertIsNone(rc.classify(pixels(((240, 240, 240), 100))))
        self.assertIsNone(rc.classify(np.empty((0, 3), np.uint8)))


class FinePhotoTest(unittest.TestCase):
    chain = [SimpleNamespace(name="naip", max_zoom=17), SimpleNamespace(name="usgs", max_zoom=16),
             SimpleNamespace(name="eox2016", max_zoom=14)]

    def test_fine_sources_only(self):
        self.assertTrue(rc.fine_photo({"naip": 900, "colormatch": 900}, self.chain, 1.0))
        self.assertFalse(rc.fine_photo({"naip": 800, "usgs": 100}, self.chain, 1.0), "part filled from z16")
        self.assertFalse(rc.fine_photo({"eox2016": 900}, self.chain, 1.0), "Sentinel-2 alone (Perth)")

    def test_holes_and_coarse_texels(self):
        self.assertFalse(rc.fine_photo({"naip": 800, "none": 100}, self.chain, 1.0))
        self.assertFalse(rc.fine_photo({"naip": 900}, self.chain, 2.0), "a fine source rendered at 2 m")
        self.assertFalse(rc.fine_photo({}, self.chain, 1.0))


def road(surfaces=("asphalt",), cls="residential", osm_surface=None, osm_colour=None, ids=None):
    """One straight road along north = 0 from east -60 to 60 m."""
    f = test_roads.features([[(-60, 0), (60, 0)]], [cls], surfaces=[(0, s) for s in surfaces if s is not None],
                            ids=ids)
    if osm_surface is not None:
        f.attrs["osm_surface"] = mapgeom.text_column([osm_surface])
    if osm_colour is not None:
        f.attrs["osm_surface_colour"] = mapgeom.text_column([osm_colour])
    return f


class CandidateTest(unittest.TestCase):
    def spec(self, f, urban=False):
        return roads.road_specs(f)[0]._replace(urban=urban)

    def test_osm_surface_decides_first(self):
        f = road(surfaces=("asphalt",), osm_surface="concrete")
        self.assertFalse(rc.candidate(f, 0, self.spec(f)), "OSM's concrete over Overture's asphalt")
        f = road(surfaces=(None,), osm_surface="asphalt")
        self.assertTrue(rc.candidate(f, 0, self.spec(f)))
        f = road(surfaces=(None,), osm_surface="paving_stones")
        self.assertFalse(rc.candidate(f, 0, self.spec(f, urban=True)))

    def test_overture_surface_without_osm(self):
        f = road(surfaces=("asphalt",), osm_surface="")
        self.assertTrue(rc.candidate(f, 0, self.spec(f)))
        f = road(surfaces=("concrete",))
        self.assertFalse(rc.candidate(f, 0, self.spec(f, urban=True)))

    def test_untagged_only_in_town(self):
        f = road(surfaces=(None,))
        self.assertFalse(rc.candidate(f, 0, self.spec(f)), "a rural untagged road may be a red dirt track")
        self.assertTrue(rc.candidate(f, 0, self.spec(f, urban=True)))

    def test_unpaved_never(self):
        f = road(surfaces=("gravel",))
        self.assertFalse(rc.candidate(f, 0, roads.road_specs(f)[0]))


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_first_decision_stands_across_processes(self):
        a, b = rc.Store(self.dir), rc.Store(self.dir)
        self.assertEqual(a.decide("w1", "red"), "red")
        self.assertEqual(b.decide("w1", ""), "red", "another process's earlier decision")
        self.assertEqual(b.get(["w1", "w2"]), {"w1": "red"})

    def test_memory_only_without_a_folder(self):
        s = rc.Store(None)
        self.assertEqual(s.decide("w1", ""), "")
        self.assertEqual(s.get(["w1"]), {"w1": ""})


class PhotoTest(unittest.TestCase):
    """photo_roads with a 1 m photo of 128 x 128 m around the road (row 0 north: the road on row 64)."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        old = revisit.CACHE_DIR
        revisit.CACHE_DIR = self.dir
        self.addCleanup(setattr, revisit, "CACHE_DIR", old)
        self.frame = geodesy.EnuFrame(test_roads.LAT, test_roads.LON)

    def photo(self, road_rgb, ground=GRASS, half=3):
        rgb = np.empty((128, 128, 3), np.uint8)
        rgb[:] = ground
        rgb[64 - half:64 + half] = road_rgb
        return rgb

    def paint(self, f, rgb, learn=True):
        return sf.photo_roads(rgb, self.frame, -64.0, -64.0, 128.0, [f], learn=rgb if learn else None)

    def test_red_road_painted_red(self):
        out = self.paint(road(), self.photo(RED))
        self.assertEqual(tuple(out[64, 64]), PAINT_RED)

    def test_grey_road_stays_grey(self):
        out = self.paint(road(), self.photo(GREY))
        self.assertEqual(tuple(out[64, 64]), sf.PHOTO_ROAD_RGBA["paved"][:3])

    def test_red_verges_do_not_make_a_red_road(self):
        out = self.paint(road(), self.photo(GREY, ground=PINDAN, half=2))
        self.assertEqual(tuple(out[64, 64]), sf.PHOTO_ROAD_RGBA["paved"][:3])

    def test_decision_kept_for_later_photos(self):
        f = road(ids=["w7"])
        self.paint(f, self.photo(RED))
        later = sf.photo_roads(self.photo(GREY), self.frame, -64.0, -64.0, 128.0, [road(ids=["w7"])])
        self.assertEqual(tuple(later[64, 64]), PAINT_RED, "decided once: the next photo agrees")
        again = self.paint(road(ids=["w7"]), self.photo(GREY))
        self.assertEqual(tuple(again[64, 64]), PAINT_RED, "the first decision stands")

    def test_no_learning_without_a_fine_photo(self):
        out = self.paint(road(ids=["w8"]), self.photo(RED), learn=False)
        self.assertEqual(tuple(out[64, 64]), sf.PHOTO_ROAD_RGBA["paved"][:3])
        self.assertEqual(rc.store().get(["w8"]), {}, "undecided: a fine photo decides later")

    def test_too_little_seen_decides_nothing(self):
        f = road(ids=["w9"])
        rgb = np.empty((28, 28, 3), np.uint8)       # 28 m square, east 36..64, north -14..14: the road on row 14
        rgb[:] = GRASS
        rgb[11:17] = RED
        out = sf.photo_roads(rgb, self.frame, 36.0, -14.0, 28.0, [f], learn=rgb)
        self.assertEqual(rc.store().get(["w9"]), {}, "24 m of a 120 m road")
        self.assertEqual(tuple(out[14, 5]), sf.PHOTO_ROAD_RGBA["paved"][:3])

    def test_osm_colour_overrules_the_photo(self):
        out = self.paint(road(osm_colour="grey"), self.photo(RED))
        self.assertEqual(tuple(out[64, 64]), sf.PHOTO_ROAD_RGBA["paved"][:3], "OSM says grey: no photo asked")
        out = self.paint(road(osm_colour="red", surfaces=("concrete",), ids=["w5"]), self.photo(GREY))
        self.assertEqual(tuple(out[64, 64]), PAINT_RED, "OSM says red")

    def test_road_mask_counts_coloured_asphalt(self):
        f = road(ids=["w6"])
        self.paint(f, self.photo(RED))
        got = sf._road_segments(self.frame, -64.0, -64.0, 128.0, 128, 128, [f], coloured=True)
        self.assertEqual(set(got[2].tolist()), {list(sf.PHOTO_ROAD_RGBA).index("paved:red")})
        sur = SimpleNamespace(urban=None, gas_pads=False, extra_pads=None,
                              roads=lambda la, lo: ([f], None), lots=lambda la, lo: [], fuel_pads=lambda la, lo: [])
        mask = sf.road_mask(self.frame, -64.0, -64.0, 128.0, 128, sur)
        self.assertEqual(int(mask[64, 64]), 255, "the coloured road is road in the mask")


if __name__ == "__main__":
    unittest.main()
