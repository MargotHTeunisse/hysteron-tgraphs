from model import *
import numpy as np
import scipy.special
import itertools
from typing import Dict, Tuple

AvalancheForest = Dict[Tuple[State, int], Tuple[Tuple[int]]]

def make_all_scaffolds(num_hysts:int)-> Iterator[Scaffold]:
    possible_critical_hysterons = {(state, 1-2*sign):tuple(i for i in range(num_hysts) if state[i]==sign) for state in itertools.product([0, 1], repeat=num_hysts) for sign in [0, 1] if sign in state}

    #Set up boundary of main loop to get rid of permutation symmetry
    for i in range(num_hysts):
        state = (0,)*(num_hysts-i) + (1,)*i
        possible_critical_hysterons[(state, 1)] = (num_hysts-i-1,)

    #Make iterator for possible combinations of critical hysterons (i.e., possible_scaffolds)
    keys = [key for key in possible_critical_hysterons]
    values = [possible_critical_hysterons[key] for key in possible_critical_hysterons]
    iterator = (dict(zip(keys, comb))for comb in itertools.product(*values))
    
    for combination_of_critical_hysterons in iterator:
        #Construct scaffold for given combination
        scaffold = Scaffold(num_hysts)
        for (state, direction) in combination_of_critical_hysterons:
            scaffold[(state, direction)] = combination_of_critical_hysterons[(state, direction)]

        yield scaffold

def make_all_preisach_scaffolds(num_hysts:int) -> Iterator[Scaffold]:
    for down_ordering in itertools.permutations(np.arange(num_hysts)):
        scaffold = Scaffold(num_hysts)
        for state in itertools.product([0, 1], repeat=num_hysts):
            #The rightmost hysteron which is in state 0 is critical for each state.
            if 0 in state:
                scaffold[(state, 1)] = num_hysts - 1 - state[::-1].index(0)
            for i in down_ordering:
                if state[i] == 1: 
                    scaffold[(state, -1)] = i
                    break
                
        yield scaffold

def make_random_scaffold(num_hysts:int) -> Scaffold:
    possible_critical_hysterons = {(state, 1-2*sign):tuple(i for i in range(num_hysts) if state[i]==sign) for state in itertools.product([0, 1], repeat=num_hysts) for sign in [0, 1] if sign in state}

    #Set up boundary of main loop to get rid of permutation symmetry
    for i in range(num_hysts):
        state = (0,)*(num_hysts-i) + (1,)*i
        possible_critical_hysterons[(state, 1)] = (num_hysts-i-1,)

    scaffold = Scaffold(num_hysts)
    for (state, direction) in possible_critical_hysterons:
        scaffold[(state, direction)] = np.random.choice(possible_critical_hysterons[(state, direction)])

    return scaffold

def make_random_graph(num_hysts:int) -> Graph:
    graph = Graph(num_hysts)

    scaffold = make_random_scaffold(num_hysts)

    avalanche_forest = make_avalanche_forest(scaffold)

    for (state, direction) in avalanche_forest:
        flipped = avalanche_forest[(state, direction)][np.random.choice(len(avalanche_forest[(state, direction)]))]
        graph.add(state, flipped)

    return graph

def make_avalanche_forest(scaffold:Scaffold) -> AvalancheForest:
    avalanche_forest = {(state, direction):tuple() for (state, direction) in scaffold}
    
    for (state, direction) in scaffold:
        critical_hysteron = scaffold[(state, direction)]
        queue = [(critical_hysteron,)]
        while queue:
            queue_new = []
            for flipped in queue:
                transition = Transition(state, flipped)
                if not transition.is_loop:
                    avalanche_forest[(state, direction)] += (flipped,)
                    final_state = transition.final_state
                    if 0 in final_state:
                        kappa =  scaffold[(final_state, 1)]
                        queue_new.append(flipped + (kappa,))
                    if 1 in final_state:
                        kappa = scaffold[(final_state, -1)]
                        queue_new.append(flipped + (kappa,))
            queue = queue_new

    return avalanche_forest

def make_avalanche_forest_serial(scaffold:Scaffold) -> AvalancheForest:
    avalanche_forest = {(state, direction):tuple() for (state, direction) in scaffold}

    for (state, direction) in scaffold:
        critical_hysteron = scaffold[(state, direction)]
        queue = [(critical_hysteron,)]
        while queue:
            queue_new = []
            for flipped in queue:
                transition = Transition(state, flipped)

                #Ensure directions (up/down) alternate by checking the direction of the last flipped hysteron.
                if not transition.is_loop:
                    avalanche_forest[(state, direction)] += (flipped,)
                    final_state = transition.final_state
                    kappa =  scaffold[(final_state, 1-2*final_state[flipped[-1]])]
                    queue_new.append(flipped + (kappa,))

            queue = queue_new
 
    return avalanche_forest

def make_candidate_graphs(scaffold:Scaffold, model:str='general') -> Iterator[Graph]:

    mapping =  {
        'general': make_avalanche_forest,
        'serial': make_avalanche_forest_serial
    }

    avalanche_forest = mapping[model](scaffold)
    
    #Make iterator over all possible combinations of transitions
    keys = [key for key in avalanche_forest]
    values = [avalanche_forest[key] for key in avalanche_forest]
    iterator = (dict(zip(keys, comb))for comb in itertools.product(*values))

    for combination_of_transitions in iterator:
        #Construct graph for given combination
        candidate_graph = Graph(scaffold.num_hysts)
        for (state, direction) in combination_of_transitions:
            flipped = combination_of_transitions[(state, direction)]
            candidate_graph.add(state, flipped)

        yield candidate_graph

def count_scaffolds(num_hysts:int) -> int:
    #Analytical expression to count number of scaffolds
    return int(np.prod([magnetisation**scipy.special.binom(num_hysts, magnetisation) for magnetisation in range(1, num_hysts+1)])**2/np.math.factorial(num_hysts))

def count_candidate_graphs(avalanche_forest:AvalancheForest)-> int:
    return np.product([len(avalanche_forest[(state, direction)]) for (state, direction) in avalanche_forest], dtype=np.int64)

def make_all_candidate_graphs(num_hysts:int, model:str='general') -> Iterator[Graph]:
    for scaffold in make_all_scaffolds(num_hysts):
        for graph in make_candidate_graphs(scaffold, model):
            yield graph
            
def count_all_candidate_graphs(num_hysts:int) -> int:
    total_count = 0
    for scaffold in make_all_scaffolds(num_hysts):
        avalanche_forest = make_avalanche_forest(scaffold)
        total_count += count_candidate_graphs(avalanche_forest)
    return total_count