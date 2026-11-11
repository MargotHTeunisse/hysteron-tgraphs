import pytest
polyhedron_analysis = pytest.importorskip("hysteron_tgraphs.polyhedron_analysis",
                    reason="Could not import optional polyhedron-analysis module; skipping tests.")

from hysteron_tgraphs.model import *

import numpy as np

class TestPolyhedronAnalysis:
    @pytest.mark.parametrize("stateA, i, stateB, j", [
        ((0,), 0, (1, ), 0),
        ((0, 0), 0, (0, 0), 1),
        ((1, 0), 1, (1, 1), 0)
    ])
    def test_volume_is_one_half_for_single_inequality(self, stateA:State, i:int, stateB:State, j:int):
        """
        Check that polyhedron volume makes sense for n=1 and n=2; n >= 3 takes too long for testing.
        """
        switching_field_order = SwitchingFieldOrder(len(stateA))
        switching_field_order.add(stateA, i, stateB, j)

        volume: float = polyhedron_analysis.calculate_polyhedron_volume(switching_field_order)

        assert np.isclose(volume, 0.5)

    def test_volume_is_one_quarter_for_two_independent_inequalities(self):
        switching_field_order = SwitchingFieldOrder(2)
        switching_field_order.add((0, 0), 0, (0, 0), 1)
        switching_field_order.add((1, 1), 0, (1, 1), 1)

        volume: float = polyhedron_analysis.calculate_polyhedron_volume(switching_field_order)

        assert np.isclose(volume, 0.25)

    def test_volume_is_one_eight_for_three_independent_inequalities(self):
        switching_field_order = SwitchingFieldOrder(2)
        switching_field_order.add((0, 0), 0, (0, 0), 1)
        switching_field_order.add((1, 1), 0, (1, 1), 1)
        switching_field_order.add((0, 1), 0, (0, 1), 1)

        volume: float = polyhedron_analysis.calculate_polyhedron_volume(switching_field_order)

        assert np.isclose(volume, 0.125)