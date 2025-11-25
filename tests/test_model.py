from model import *

def test_make_graph():
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