import importlib

import pytest
pytestmark = pytest.mark.skipif(not importlib.util.find_spec("networkx"),
                                reason="NetworkX is not installed.")

from hysteron_tgraphs.model import *
from hysteron_tgraphs.graph_analysis import remove_goe, convert_graph_to_networkx

import copy

@pytest.mark.parametrize("state0, flipped", [
    ((0,), (0,)),
    ((0, 0), (1, 0)), 
    ((0, 1), (0, 1, 0)),
    ((1, 0), (1,)),
    ((0, 0), (1, 0, 1, 0)), # this is a self-loop, but should be allowed
    ((0, 0, 1), (1, 0))
])
class TestNetworkXConversion:
    """
    Check for various transitions that they are correctly converted to a NetworkX edge.
    """
    def test_networkx_edge_has_transition_length(self, state0:State, flipped:Tuple[int]):
        graph = Graph(len(state0))
        graph.add(state0, flipped)
        transition = Transition(state0, flipped)

        networkx_graph = convert_graph_to_networkx(graph)

        assert networkx_graph.edges[state0, transition.final_state, 0]['transition_length'] == transition.length

    def test_networkx_edge_has_direction(self, state0:State, flipped:Tuple[int]):
        graph = Graph(len(state0))
        graph.add(state0, flipped)
        transition = Transition(state0, flipped)

        networkx_graph = convert_graph_to_networkx(graph)

        assert networkx_graph.edges[state0, transition.final_state, 0]['direction'] == transition.direction
    

    def test_networkx_edge_tracks_flipped_hysterons(self, state0:State, flipped:Tuple[int]):
        graph = Graph(len(state0))
        graph.add(state0, flipped)
        transition = Transition(state0, flipped)

        networkx_graph = convert_graph_to_networkx(graph)

        assert networkx_graph.edges[state0, transition.final_state, 0]['flipped'] == flipped

class TestGardenOfEdenStateRemoval:
    def test_removes_garden_of_eden_states_which_are_inaccessible_in_scaffold(self):
        """
        Check that Garden-of-Eden states can be removed which are inaccessible in the scaffold.
        """
        graph = Graph(2)
        # Add states which are reachable from (0, 0) and (1, 1).
        graph.add((0, 0), (1,))
        graph.add((0, 1), (0,))
        graph.add((0, 1), (1,))
        graph.add((1, 1), (0,))
        expected = copy.deepcopy(graph)
        # Add the Garden-of-Eden state (1,0)
        graph.add((1, 0), (0,))
        graph.add((1, 0), (1,))

        reduced_graph = remove_goe(graph)

        assert reduced_graph == expected

    def test_removes_garden_of_eden_states_which_are_accessible_in_scaffold(self):
        """
        Check that Garden-of-Eden states can be removed when they are accessed by avalanches.
        """
        graph = Graph(2)
        # Add reachable states
        graph.add((0, 0), (1,))
        graph.add((0, 1), (0,))
        graph.add((0, 1), (1,))
        graph.add((1, 1), (1, 0)) # this avalanche passes through (1, 0)
        expected = copy.deepcopy(graph)
        # Add garden-of-Eden state (1, 0)
        graph.add((1, 0), (0,))
        graph.add((1, 0), (1,))

        reduced_graph = remove_goe(graph)

        assert reduced_graph == expected