from server.pdf_cache import PDFCache, PDFGrid
from std.testing import assert_true, TestSuite


def test_pdf_cache_save_and_load() raises:
    var cache = PDFCache()
    var pdf: List[List[Float64]] = []
    pdf.append([1.0, 2.0])
    pdf.append([3.0, 4.0])
    var s_points: List[Float64] = [10.0, 20.0]
    var v_points: List[Float64] = [0.1, 0.2]
    var ds: List[Float64] = []
    var dv: List[Float64] = []
    var grid = PDFGrid(pdf=pdf^, s_points=s_points^, v_points=v_points^, T=1.0, ds_weights=ds^, dv_weights=dv^)
    grid.precompute_weights()
    cache.store(12345, grid^)

    var tmp_dir = "/tmp"
    var saved = cache.save_to_disk(tmp_dir)
    assert_true(saved, "save_to_disk should return True")

    var cache2 = PDFCache()
    var loaded = cache2.load_from_disk(tmp_dir)
    assert_true(loaded, "load_from_disk should return True")

    var retrieved = cache2.get(12345)
    assert_true(retrieved is not None, "should retrieve cached grid")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
