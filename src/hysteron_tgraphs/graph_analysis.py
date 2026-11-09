from hysteron_tgraphs.model import *
import networkx as nx
import copy

"""This module is separated because it depends on the networkx package.
Currently it is only used to remove Garden-of-Eden states. 
"""

def convert_graph_to_networkx(graph:Graph) -> nx.MultiDiGraph:
    """
    Converts a hysteron graph to a graph of nodes and edges of NetworkX.
    This can be useful if NetworkX capabilities are needed which cannot easily be implemented in hysteron_tgraphs.Graph.
    """
    networkx_graph = nx.MultiDiGraph()
    for (state, direction) in graph:
        networkx_graph.add_node(state)
        transition = graph[(state, direction)]
        networkx_graph.add_edge(state, transition.final_state, flipped=transition.flipped,
                                direction=direction, transition_length=transition.length)
    return networkx_graph
    
def remove_goe(graph:Graph) -> Graph:
    #Use NetworkX to quickly identify GoE states, and remove these from graphs
    graph_reduced = copy.deepcopy(graph)
    G = convert_graph_to_networkx(graph)
    reachable = nx.dag.descendants(G, (0,)*graph.num_hysts)
    unreachable = set(G.nodes) - reachable - {(0,)*graph.num_hysts}
    for state in unreachable:
        if 0 in state:
            graph_reduced.remove(state, 1)
        if 1 in state:
            graph_reduced.remove(state, -1)
    return graph_reduced
