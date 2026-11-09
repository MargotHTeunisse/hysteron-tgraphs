from hysteron_tgraphs import candidategraphs
from hysteron_tgraphs.model import *
import math

import pytest

class TestScaffoldGeneration:
    @pytest.mark.parametrize("num_hysts, expected", [
        (1, 1),
        (2, 2),
        (3, 96)
    ])
    def test_scaffold_count_is_correct(self, num_hysts:int, expected:int):
        scaffold_count = len([scaffold 
                for scaffold in candidategraphs.make_all_scaffolds(num_hysts)])

        assert scaffold_count == expected

    @pytest.mark.parametrize("num_hysts", [1, 2, 3, 4])
    def test_preisach_scaffold_count_is_correct(self, num_hysts:int):
        preisach_scaffold_count = len([scaffold 
                for scaffold in candidategraphs.make_all_preisach_scaffolds(num_hysts)])

        assert preisach_scaffold_count == math.factorial(num_hysts)


    @pytest.mark.parametrize("num_hysts, expected", [
        (1, 1),
        (2, 2),
        (3, 96)
    ])
    def test_analytical_scaffold_count_is_correct(self, num_hysts:int, expected:int):
        analytical_scaffold_count = candidategraphs.count_scaffolds(num_hysts)

        assert analytical_scaffold_count == expected

class TestAvalancheForest:
    @pytest.mark.parametrize("state0, direction, expected", [
        ((0, 0), 1, {(1,), (1, 0)}),
        ((0, 1), 1, {(0,)}),
        ((0, 1), -1, {(1,)}),
        ((1, 0), 1, {(1,), (1, 0), (1, 0, 1)}),
        ((1, 0), -1, {(0,), (0, 1), (0, 1, 0)}),
        ((1, 1), -1, {(0,), (0, 1)})
    ])
    def test_transitions_in_avalanche_forest_are_correct(self,
                                                         state0:State,
                                                         direction:int,
                                                         expected:Set[Tuple[int]]):
        """
        Check that the avalanche forest is correct for the main loop (1, 0, 0, 1).
        """
        scaffold = Scaffold(2)
        scaffold[(0, 0), 1] = 1
        scaffold[(0, 1), 1] = 0
        scaffold[(0, 1), -1] = 1
        scaffold[(1, 0), 1] = 1
        scaffold[(1, 0), -1] = 0
        scaffold[(1, 1), -1] = 0

        avalanche_forest = candidategraphs.make_avalanche_forest(scaffold)

        assert set(avalanche_forest[(state0, direction)]) == expected

    @pytest.mark.parametrize("state0, direction, expected", [
        ((0, 0), 1, {(1,), (1, 0)}),
        ((0, 1), 1, {(0,)}),
        ((0, 1), -1, {(1,)}),
        ((1, 0), 1, {(1,)}),
        ((1, 0), -1, {(0,)}),
        ((1, 1), -1, {(1,), (1, 0)})
    ])
    def test_transitions_in_ferro_avalanche_forest_are_correct(self,
                                                               state0:State,
                                                               direction:int,
                                                               expected:Set[Tuple[int]]):
        """
        Check that the purely ferro avalanche forest is correct for the main loop (1, 0, 1, 0).
        """
        scaffold = Scaffold(2)
        scaffold[(0, 0), 1] = 1
        scaffold[(0, 1), 1] = 0
        scaffold[(0, 1), -1] = 1
        scaffold[(1, 0), 1] = 1
        scaffold[(1, 0), -1] = 0
        scaffold[(1, 1), -1] = 1

        avalanche_forest = candidategraphs.make_avalanche_forest_ferro(scaffold)

        assert set(avalanche_forest[(state0, direction)]) == expected

    @pytest.mark.parametrize("state0, direction, expected", [
        ((0, 0), 1, {(1,)}),
        ((0, 1), 1, {(0,), (0, 1)}),
        ((0, 1), -1, {(1,)}),
        ((1, 0), 1, {(1,)}),
        ((1, 0), -1, {(0,), (0, 1)}),
        ((1, 1), -1, {(1,)})
    ])
    def test_transitions_in_antiferro_avalanche_forest_are_correct(self,
                                                               state0:State,
                                                               direction:int,
                                                               expected:Set[Tuple[int]]):
        """
        Check that the purely antiferro avalanche forest is correct for the main loop (1, 0, 1, 0).
        """
        scaffold = Scaffold(2)
        scaffold[(0, 0), 1] = 1
        scaffold[(0, 1), 1] = 0
        scaffold[(0, 1), -1] = 1
        scaffold[(1, 0), 1] = 1
        scaffold[(1, 0), -1] = 0
        scaffold[(1, 1), -1] = 1

        avalanche_forest = candidategraphs.make_avalanche_forest_antiferro(scaffold)

        assert set(avalanche_forest[(state0, direction)]) == expected

    @pytest.mark.parametrize("critical_down_hysteron,expected", [
        (0, 36),
        (1, 81)
    ])
    def test_number_of_combinations_per_scaffold_is_correct(self,
                                               critical_down_hysteron:int,
                                               expected:int):
        """
        Check that for each n=2 scaffold and corresponding avalanche forest,
        the number of resulting candidate graphs is correct.
        See fig.8 in Teunisse & Van Hecke (2025).
        """
        scaffold = Scaffold(2)
        scaffold[(0, 0), 1] = 1
        scaffold[(0, 1), 1] = 0
        scaffold[(0, 1), -1] = 1
        scaffold[(1, 0), 1] = 1
        scaffold[(1, 0), -1] = 0
        scaffold[(1, 1), -1] = critical_down_hysteron
        avalanche_forest = candidategraphs.make_avalanche_forest(scaffold)

        number_of_combinations = candidategraphs.count_candidate_graphs(avalanche_forest)

        assert number_of_combinations == expected

    def test_total_number_of_combinations_is_correct(self):
        """Check that the total number of n=2 combinations is 117 without generating graphs."""
        number_of_combinations = candidategraphs.count_all_candidate_graphs(2)

        assert number_of_combinations == 117

    def test_total_number_of_ferro_combinations_is_correct(self):
        """Check that the total number of purely ferro n=2 combinations is 8 without generating graphs."""
        number_of_combinations = candidategraphs.count_all_candidate_graphs(2,
                                                                            model='ferro')

        assert number_of_combinations == 8

    def test_total_number_of_antiferro_combinations_is_correct(self):
        """Check that the total number of purely antiferro n=2 combinations is 8 without generating graphs."""
        number_of_combinations = candidategraphs.count_all_candidate_graphs(2, 
                                                                            model='antiferro')

        assert number_of_combinations == 8

class TestGraphGeneration:
    def test_two_hysteron_candidate_graph_count_is_correct(self):
        """
        Check that the total candidate graph count for two hysterons is 36 + 81 = 117.
        """
        candidate_graph_counter = 0
        for _ in candidategraphs.make_all_candidate_graphs(2):
            candidate_graph_counter += 1

        assert candidate_graph_counter == 117

    def test_two_hysteron_ferro_candidate_graph_count_is_correct(self):
        """
        Check that the number of purely ferro candidate graphs for two hysterons is 4 + 4 = 8.
        """
        candidate_graph_counter = 0
        for _ in candidategraphs.make_all_candidate_graphs(2, model='ferro'):
            candidate_graph_counter += 1

        assert candidate_graph_counter == 8

    def test_two_hysteron_antiferro_candidate_graph_count_is_correct(self):
        """
        Check that the number of purely antiferro candidate graphs for two hysterons is 4 + 4 = 8.
        """
        candidate_graph_counter = 0
        for _ in candidategraphs.make_all_candidate_graphs(2, model='antiferro'):
            candidate_graph_counter += 1

        assert candidate_graph_counter == 8

    @pytest.mark.parametrize("num_hysts", [1, 2, 3, 4, 5])
    def test_produces_consistent_graph(self, num_hysts:int):
        """
        Check that a single graph can be produced which has a complete and consistent scaffold.
        """ 
        states: Iterator[State] = itertools.product([0, 1], repeat=num_hysts)
        keys = [(state, 1) for state in states if 0 in state] \
                + [(state, -1) for state in states if 1 in state]

        # The generated graph is random, but the outcome of this test should not be.
        graph = candidategraphs.make_random_graph(num_hysts)

        scaffold = graph.scaffold
        assert all([scaffold[(state, direction)] != None for (state, direction) in keys])