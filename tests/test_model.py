import sys
sys.path.append("./src")
from hysteron_tgraphs.model import *
import pytest

def test_make_single_hysteron_graph():
    switching_fields = SwitchingFields(1)
    switching_fields[(0,)] = (2,)
    switching_fields[(1,)] = (1,)
    graph = make_graph(switching_fields)

    check_graph = Graph(1)
    check_graph.add((0,), (0,))
    check_graph.add((1,), (0,))

    assert graph == check_graph

def test_check_self_loop_single_hysteron():
    assert Transition((0,), (0, 0)).is_loop

def test_check_self_loop_two_hysterons():
    assert Transition((0, 0), (1, 0, 1, 0)).is_loop

def test_self_loop_raises_exception_by_default():
    switching_fields = SwitchingFields(1)
    switching_fields[(0,)] = (1,)
    switching_fields[(1,)] = (2,)

    with pytest.RaisesExc(Exception):
        make_graph(switching_fields)

def test_self_loop_allowance():
    switching_fields = SwitchingFields(1)
    switching_fields[(0,)] = (1,)
    switching_fields[(1,)] = (2,)
    graph = make_graph(switching_fields, allow_self=True)

    check_graph = Graph(1)
    check_graph.add((0,), (0, 0))
    check_graph.add((1,), (0, 0))
    
    assert graph == check_graph

def test_make_two_hysteron_graph():
    """
    Checks that the correct graph is generated for an example two-hysteron system.
    """
    switching_fields = SwitchingFields(2)
    switching_fields[(0, 0)] = (2, 1)
    switching_fields[(0, 1)] = (0, -1)
    switching_fields[(1, 1)] = (-2, 1)
    switching_fields[(1, 0)] = (0, 3)
    graph = make_graph(switching_fields)

    check_graph = Graph(2)
    check_graph.add((0, 0), (1, 0))
    check_graph.add((0, 1), (0, 1))
    check_graph.add((0, 1), (1,))
    check_graph.add((1, 1), (1,))
    check_graph.add((1, 0), (0,))
    check_graph.add((1, 0), (1,))

    assert graph == check_graph

def test_race_conditions_raise_exception_by_default():
    switching_fields = SwitchingFields(2)
    switching_fields[(0, 0)] = (2, 1)
    switching_fields[(0, 1)] = (-1, 2)
    switching_fields[(1, 1)] = (-2, 1)
    switching_fields[(1, 0)] = (0, 3)

    with pytest.RaisesExc(Exception):
        make_graph(switching_fields, allow_self=True)

def test_resolve_race_conds_via_most_unstable_hysteron():
    switching_fields = SwitchingFields(2)
    switching_fields[(0, 0)] = (2, 1)
    switching_fields[(0, 1)] = (-1, 2)
    switching_fields[(1, 1)] = (-2, 1)
    switching_fields[(1, 0)] = (0, 3)
    graph = make_graph(switching_fields, allow_self=True, resolve_race=True)

    check_graph = Graph(2)
    check_graph.add((0, 0), (1, 0))
    check_graph.add((0, 1), (0, 1, 0))
    check_graph.add((0, 1), (1, 1))
    check_graph.add((1, 1), (1,))
    check_graph.add((1, 0), (0,))
    check_graph.add((1, 0), (1,))

    assert graph == check_graph

def test_valid_graph():
    graph = Graph(2)
    graph.add((0, 0), (1, 0))
    graph.add((0, 1), (0, 1))
    graph.add((0, 1), (1,))
    graph.add((1, 1), (1,))
    graph.add((1, 0), (0,))
    graph.add((1, 0), (1,))

    switching_field_order = make_design_inequalities(graph)
    
    assert switching_field_order.valid

def test_valid_self_loop():
    """
    Check that the conditions for a self-loop are valid.
    Note that these conditions are well-defined, so no exception is raised.
    """
    graph = Graph(2)
    graph.add((0, 0), (1, 0, 1, 0))

    switching_field_order = make_design_inequalities(graph)

    assert switching_field_order.valid


def test_invalid_graph_direct_contradiction():
    """
    Check that a direct conflict between inequalities is identified correctly.
    See also Teunisse & Van Hecke (2025), Fig. 7a-b. 
    """
    graph = Graph(2)
    graph.add((0, 0), (1,))
    graph.add((0, 1), (0, 1, 0))

    switching_field_order = make_design_inequalities(graph)
    assert switching_field_order.valid == False

def test_invalid_graph_indirect_contradiction():
    """
    Check that an indirect direct conflict between inequalities is identified correctly.
    See also Teunisse & Van Hecke (2025), Fig. 7c-d. 
    """
    graph = Graph(2)
    graph.add((0, 0), (1, 0))
    graph.add((1, 0), (0, 1))

    switching_fields = make_design_inequalities(graph)
    assert switching_fields.valid == False
    
def test_irresolvible_race_conditions():
    """
    Check that a graph is marked as invalid if it requires race conditions where both the critical up and down hysteron are unstable.
    This is done because the design inequalities otherwise grow more complex than a simple partial order.
    """
    graph = Graph(2)
    graph.add((0, 0), (1, 0))
    graph.add((0, 1), (0, 1, 0))
    graph.add((0, 1), (1, 1))
    graph.add((1, 1), (1,))
    graph.add((1, 0), (0,))
    graph.add((1, 0), (1,))

    switching_field_order = make_design_inequalities(graph, resolve_race=True)

    assert switching_field_order.valid
