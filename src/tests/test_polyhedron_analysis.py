import pytest
polyhedron_analysis = pytest.importorskip("hysteron_tgraphs.polyhedron_analysis",
                    reason="Could not import optional polyhedron-analysis module; skipping tests.")

from hysteron_tgraphs.model import *

import numpy as np
import math

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

    def test_volume_matches_value_from_linear_extension_counting(self):
        """
        Check that volume is correct for actual design inequalities by comparing against linear extension count.
        """
        graph = Graph(2)
        graph.add((0, 0), (1, 0))
        graph.add((0, 1), (0, 1))
        graph.add((0, 1), (1,))
        graph.add((1, 1), (1,))
        graph.add((1, 0), (0,))
        graph.add((1, 0), (1,))
        switching_field_order = make_design_inequalities(graph)

        volume: float = polyhedron_analysis.calculate_polyhedron_volume(switching_field_order)

        # Calculate volume from linear extensions for comparison; these two volumes should be the same.
        linear_extension_count = count_linear_extensions(switching_field_order)
        total_linear_extensions = math.factorial(8)
        volume_from_linear_extensions = linear_extension_count / total_linear_extensions
        assert np.isclose(volume, volume_from_linear_extensions)