from .model import *
import networkx as nx

"""This module is separated because it depends on the networkx package.
Currently it is only used to remove Garden-of-Eden states. 
"""

def convert_graph_to_networkx(graph:Graph) -> nx.MultiDiGraph:
    networkx_graph = nx.MultiDiGraph()
    for (state, direction) in graph:
        networkx_graph.add_node(state)
        transition = graph[(state, direction)]
        networkx_graph.add_edge(state, transition.final_state, flipped=transition.flipped, direction=direction, transition_length=transition.length)
    return networkx_graph
    
def remove_goe(graph:Graph) -> Graph:
    #Use NetworkX to quickly identify GoE states, and remove these from graphs
    graph_reduced = copy.deepcopy(graph)
    G = nx.DiGraph()
    for (state, direction) in graph:
        transition = graph[(state, direction)]
        G.add_edge(transition.state, transition.final_state)
    reachable = nx.dag.descendants(G, (0,)*graph.num_hysts)
    unreachable = set(G.nodes) - reachable - {(0,)*graph.num_hysts}
    for state in unreachable:
        if any(state):
            graph_reduced.remove(state, -1)
        if not all(state):
            graph_reduced.remove(state, 1)
    return graph_reduced
