import importlib
import pytest

@pytest.mark.skipif(not importlib.util.find_spec("pycddlib"),
                                  reason="Skipping module polyhedron-analysis as pycddlib is not installed.")
class TestPolyhedronAnalysis:
    def test_should_fail(self):
        assert False