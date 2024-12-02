import numpy as np
from typing import Tuple, List, Iterator
from dataclasses import dataclass
import copy
import itertools

State = Tuple[bool, ...]

@dataclass(frozen=True)
class Transition():
    state:State
    flipped:Tuple[int, ...]

    def __post_init__(self) -> None:
        if not all([i < self.num_hysts for i in self.flipped]):
            raise ValueError("Hysteron indices cannot exceed number of hysterons.")
        
        #Check if transition continues after looping; if so, raise an error.
        visited = set()
        state = self.state
        lamb = 0
        for i in self.flipped:
            visited.add(state)
            lamb += 1
            state = state[:i] + (1-state[i],) + state[i+1:]
            if state in visited:
                if lamb < len(self.flipped):
                    raise Exception("Transition cannot continue after looping.")

    def intermediate_state(self, lamb:int) -> State:
        if not (lamb > 0 and lamb < self.length):
            raise ValueError("Not an intermediate state; index must have 1 < lambda < l-1.")
        return tuple(s if self.flipped[:lamb].count(i)%2 == 0 else 1-s for i, s in enumerate(self.state))

    @property
    def is_loop(self) -> bool:
        final_state = self.final_state
        for _, state in zip(range(self.length-1), self.path()):
            if state == final_state:
                return True
        return False

    @property 
    def final_state(self) -> State:
        return tuple(s if self.flipped.count(i)%2 == 0 else 1-s for i, s in enumerate(self.state))

    @property
    def num_hysts(self) -> int:
        return len(self.state)

    @property
    def length(self) -> int:
        return len(self.flipped)
        
    @property
    def critical_hysteron(self) -> int:
       return self.flipped[0]
       
    @property
    def sign(self) -> int:
        return self.state[self.flipped[0]]
        
    @property
    def direction(self) -> int:
        return 1-2*self.state[self.flipped[0]]

    def path(self) -> Iterator[State]:
        state = self.state
        yield state
        
        for i in self.flipped:
            state = state[:i] + (1-state[i],) + state[i+1:]
            yield state

class Scaffold():
    def __init__(self, num_hysts:int):
        self.num_hysts = num_hysts
        iterator = ((state, 1-2*sign) for state in itertools.product([0, 1], repeat=num_hysts) for sign in [0, 1] if sign in state)
        self._mapping = {(state, direction):no for no, (state, direction) in enumerate(iterator)}
        self._critical_hysterons = np.array([None,]*len(self._mapping))

    def __eq__(self, other):
        return all([self[state, direction] == other[state, direction] for (state, direction) in self._mapping])

    def __iter__(self):
        return ((state, direction) for (state, direction) in self._mapping if self[state, direction] != None)

    def __setitem__(self, key:Tuple[State, int], critical_hysteron:int) -> None:
        state, direction = key
        if not 1-2*state[critical_hysteron] == direction:
            raise ValueError("Critical hysteron index and direction do not match.")
        self._critical_hysterons[self._mapping[key]] = critical_hysteron

    def __getitem__(self, key:Tuple[State, int]) -> int:
        return self._critical_hysterons[self._mapping[key]]

    def landing_state(self, state, direction) -> State:
        return tuple(s if i != self[(state, direction)] else not s for i, s in enumerate(state))

class SwitchingFields():
    def __init__(self, num_hysts:int):
        self.num_hysts = num_hysts
        self._mapping = {state:state_no for state_no, state in enumerate(itertools.product([0, 1], repeat=num_hysts))}
        self._values = np.zeros((2**num_hysts, num_hysts))

    def __iter__(self):
        return (state for state in self._mapping)

    def scaffold(self, exclude: List[Tuple[State, int]]=[]) -> Scaffold:
        scaffold = Scaffold(self.num_hysts)
        for state in self._mapping:
            if 0 in state and not (state, 1) in exclude:
                I0 = set(i for i, s in enumerate(state) if s == 0)
                k = np.argmin([self[state][i] if s == 0 else np.infty for i, s in enumerate(state)])
                if any(np.isclose(self[state][k], [self[state][i] for i in I0-{k}])):
                    raise Exception("Marginal scaffold.")
                scaffold[(state, 1)] = k
            if 1 in state and not (state, -1) in exclude:
                I1 = set(i for i, s in enumerate(state) if s == 1)
                k = np.argmax([self[state][i] if s == 1 else -np.infty for i, s in enumerate(state)])
                if any(np.isclose(self[state][k], [self[state][i] for i in I1-{k}])):
                    raise Exception("Marginal scaffold.")
                scaffold[(state, -1)] = k
        return scaffold

    def __getitem__(self, state:State):
        return self._values[self._mapping[state]]

    def __setitem__(self, state:State, row):
        self._values[self._mapping[state]] = row

class SwitchingFieldOrder():
    def __init__(self, num_hysts:int):
        self.num_hysts = num_hysts 
        self._mapping = {(state, i):num_hysts*state_no + i for state_no, state in enumerate(itertools.product([0, 1], repeat=num_hysts)) for i in range(num_hysts)}
        self._matrix = np.zeros((len(self._mapping), len(self._mapping)), dtype=bool)
        self._apply_transitive_closure()

    def to_coeffs(self):
        return np.array([np.identity(len(self._mapping))[self._mapping[(stateA, i)]] - np.identity(len(self._mapping))[self._mapping[(stateB, j)]] for ((stateA, i), (stateB, j)) in self.get()])
        
    @property
    def open_entries(self):
        return {((stateA, i), (stateB, j)) for (stateA, i) in self._mapping for (stateB, j) in self._mapping if (self._matrix[self._mapping[(stateA, i)], self._mapping[(stateB, j)]] == 0
                and self._matrix[self._mapping[(stateB, j)], self._mapping[(stateA, i)]] == 0) and (stateA, i) != (stateB, j)}

    @property
    def num_ineqs(self):
        return np.sum(self._matrix)

    def __iter__(self):
        return ((state, i) for (state, i) in self._mapping)

    def add(self, stateA:State, i:int, stateB:State, j:int)-> None:
        p = self._mapping[(stateA, i)]
        q = self._mapping[(stateB, j)]
        self._matrix[p, q] = 1
        self._apply_transitive_closure()

    def get(self):
        return {((stateA, i), (stateB, j)) for (stateA, i) in self._mapping for (stateB, j) in self._mapping if self._matrix[self._mapping[(stateA, i)], self._mapping[(stateB, j)]] == 1}
        
    def get_transitive_reduction(self):
        #Uses a transitive reduction algorithm (Aho et al., SIAM Journal of Computing, 1972) to get a minimal representation of the partial order.
        if not self.valid:
            raise Exception("Cannot construct transitive reduction for invalid partial order.")
        
        transitive_reduction_matrix = self._matrix*(1-(np.matmul(self._matrix, self._matrix)))
        return {((stateA, i), (stateB, j)) for (stateA, i) in self._mapping for (stateB, j) in self._mapping if transitive_reduction_matrix[self._mapping[(stateA, i)], self._mapping[(stateB, j)]] == 1}

    def equate(self, stateA:State, i:int, stateB:State, j:int) -> None:
        p = self._mapping[(stateA, i)]
        q = self._mapping[(stateB, j)]
        if p != q:
            self._matrix[p] = np.logical_or(self._matrix[p], self._matrix[q])
            self._matrix[:, p] = np.logical_or(self._matrix[:, p], self._matrix[:, q])
            self._matrix = np.delete(self._matrix, q, axis=0)
            self._matrix = np.delete(self._matrix, q, axis=1)
            self._apply_transitive_closure()
            self._mapping = {(state, i): p - (p > q) if self._mapping[(state, i)] == q else self._mapping[(state, i)] - (self._mapping[(state, i)] > q) for (state, i) in self._mapping}

    def enforce_symmetry_restriction(self):
        #Set the order of flips in the up boundary of the main loop to be n, n-1, ..., 1.
        for i in range(self.num_hysts):
            state = (0,)*(self.num_hysts-i) + (1,)*i
            k = self.num_hysts - i - 1
            for j in range(k):
                self.add(state, j, state, k)

    def _apply_transitive_closure(self) -> None:
        for i in range(self._matrix.shape[0]):
            row = self._matrix[i]
            col = self._matrix[:, i]
            induced = (row*col[:, np.newaxis]).astype(bool)
            self._matrix[induced] = 1

    @property
    def valid(self)-> bool:
        return np.all(self._matrix[np.identity(self._matrix.shape[0], dtype=bool)] == 0)
        
    def __iadd__(self, other):
        if self.num_hysts != other.num_hysts:
            raise ValueError("Numbers of hysterons in partial orders must match.")
        elif self._mapping != other._mapping:
            raise ValueError("Switching field mappings must match.")
        self._matrix = np.logical_or(self._matrix, other._matrix)
        self._apply_transitive_closure()
        return self
        
        
class Graph():
    def __init__(self, num_hysts:int):
        self.num_hysts = num_hysts
        iterator = ((state, 1-2*sign) for state in itertools.product([0, 1], repeat=num_hysts) for sign in [0, 1] if sign in state)
        self._mapping = {(state, direction):no for no, (state, direction) in enumerate(iterator)}
        self._flipped = np.array([None,]*len(self._mapping))

    def add(self, state:State, flipped:Tuple[int, ...]) -> None:
        direction = int(1-2*state[flipped[0]])
        self._flipped[self._mapping[(state, direction)]] = flipped

    def remove(self, state:State, direction:int):
        self._flipped[self._mapping[(state, direction)]] = None

    def __eq__(self, other):
        return all([self[state, direction] == other[state, direction] for (state, direction) in self._mapping])

    def __iter__(self):
        return ((state, direction) for (state, direction) in self._mapping if self[state, direction] != None)

    def __getitem__(self, key:Tuple[State, int]):
        state, direction = key
        flipped = self._flipped[self._mapping[(state, direction)]]
        if flipped == None:
            return None
        else:
            return Transition(state, flipped)

    def __eq__(self, other):
        if self.num_hysts == other.num_hysts:
            return np.all(self._flipped == other._flipped)
        else:
            return False

    @property
    def scaffold(self) -> Scaffold:
        scaffold = Scaffold(self.num_hysts)
        for (state, direction) in self:
            flipped = self._flipped[self._mapping[(state, direction)]]
            scaffold[(state, direction)] = flipped[0]
        return scaffold

def flip_state(state:State, index:int) -> State:
    return tuple(1-s if index == i else s for i, s in enumerate(state))

def make_initial_inequalities(scaffold:Scaffold, stable=False) -> SwitchingFieldOrder:
    switching_field_order = SwitchingFieldOrder(scaffold.num_hysts)
    for (state, direction) in scaffold:
        k = scaffold[(state, direction)]

        if direction == 1:
            I0_excl_k = (i for i in range(scaffold.num_hysts) if (state[i] == 0 and i != k))
            for i in I0_excl_k:
                switching_field_order.add(state, i, state, k)
               
            if stable:
                I1 = (i for i in range(scaffold.num_hysts) if state[i] == 1)
                for i in I1:
                    switching_field_order.add(state, k, state, i)
        else:
            I1_excl_k = (i for i in range(scaffold.num_hysts) if (state[i] == 1 and i != k))
            for i in I1_excl_k:
                switching_field_order.add(state, k, state, i)
                
            if stable:
                I0 = (i for i in range(scaffold.num_hysts) if state[i] == 0)
                for i in I0:
                    switching_field_order.add(state, i, state, k)

    return switching_field_order

def make_design_inequalities(graph:Graph, resolve_race = False) -> SwitchingFieldOrder:
    switching_field_order = make_initial_inequalities(graph.scaffold, stable=True)
    
    for (state, direction) in graph:
        transition = graph[(state, direction)]

        stateA = transition.state
        k = transition.critical_hysteron
            
        #Intermediate stability inequalities
        for lamb in range(1, transition.length):
            stateB = transition.intermediate_state(lamb)
            kappa = transition.flipped[lamb]
            I0_excl_kappa = (i for i in range(transition.num_hysts) if (stateB[i] == 0 and i != kappa))
            I1_excl_kappa = (i for i in range(transition.num_hysts) if (stateB[i] == 1 and i != kappa))
            if resolve_race:
                for i in I0_excl_kappa:
                    switching_field_order.add(stateB, i, stateB, kappa)
                for i in I1_excl_kappa:
                    switching_field_order.add(stateB, kappa, stateB, i)
            else:
                for i in I0_excl_kappa:
                    switching_field_order.add(stateB, i, stateA, k)
                for i in I1_excl_kappa:
                    switching_field_order.add(stateA, k, stateB, i)
            if stateB[kappa] == 0:
                switching_field_order.add(stateA, k, stateB, kappa)
            else:
                switching_field_order.add(stateB, kappa, stateA, k)
        
        #Final stability inequalities - exclude if transition is a loop. 
        if not transition.is_loop:
            stateB = transition.final_state
            I0 = (i for i in range(transition.num_hysts) if stateB[i] == 0)
            I1 = (i for i in range(transition.num_hysts) if stateB[i] == 1)
            for i in I0:
                switching_field_order.add(stateB, i, stateA, k)
            for i in I1:
                switching_field_order.add(stateA, k, stateB, i)
        
    return switching_field_order
    
def make_graph(switching_fields:SwitchingFields, resolve_race=False, allow_self = False, exclude: List[Tuple[State, int]]=[])-> Graph:
    graph = Graph(switching_fields.num_hysts)

    scaffold = switching_fields.scaffold(exclude)
    iterator = ((state, direction) for (state, direction) in scaffold if not (state, direction) in exclude)
    for (state, direction) in scaffold:
        state0 = state
        flipped = tuple()
        transition_path = [state0,]

        k = scaffold[(state0, direction)]
        critical_driving = switching_fields[state0][k]

        unstable_hysterons = (k,)

        while len(unstable_hysterons) > 0:
            if not len(unstable_hysterons) > 1:
                kappa = unstable_hysterons[0]
            elif resolve_race:
                diffs = [abs(switching_fields[state][i] - switching_fields[state0][k]) for i in unstable_hysterons]
                kappa = unstable_hysterons[np.argmax(diffs)]
            else:
                raise Exception("Ill-defined transition: multiple unstable hysterons.")
            flipped += (kappa,)
            state = flip_state(state, kappa)
            if state in transition_path:
                if allow_self:
                    break
                raise Exception("Ill-defined transition: self-loop.")
            if state != state0 and np.isclose(critical_driving, switching_fields[state][kappa]):
                raise Exception("Ill-defined transition: marginal avalanche.")
            transition_path += [state]

            unstable_hysterons = tuple(i for i, val in enumerate(switching_fields[state]) if (state[i] == 0 and val < critical_driving) or (state[i] == 1 and val > critical_driving))
        
        graph.add(state0, flipped)

    return graph

def make_graph_from_scaffold(scaffold:Scaffold) -> Graph:
    graph = Graph(scaffold.num_hysts)
    for (state, direction) in scaffold:
        graph.add(state, (scaffold[(state, direction)],))
    return graph

def generate_general_solution(switching_field_order:SwitchingFieldOrder, a=1, b=1) -> SwitchingFields:
    #Check that a solution exists
    if not switching_field_order.valid:
        raise Exception("Not a valid partial order.")

    #Produce a random linear extension.
    num_rows = switching_field_order._matrix.shape[0]
    ordering = []
    freshness = np.ones(num_rows, dtype=bool)
    while any(freshness):
        row = np.random.choice(np.where(freshness)[0])
        chain = [row]
        while len(chain) > 0:
            row = chain[-1]
            if np.any(switching_field_order._matrix[row]*freshness):
                chain += [np.random.choice(np.where(switching_field_order._matrix[row]*freshness)[0])]
            else:
                ordering += [row]
                freshness[row] = False
                chain = chain[:-1]

    #Construct switching fields based on linear extension
    switching_fields = SwitchingFields(switching_field_order.num_hysts)
    for state in switching_fields:
        for i in range(switching_fields.num_hysts):
            switching_fields[state][i] = a*ordering.index(switching_field_order._mapping[(state, i)]) + b

    return switching_fields
    
def count_linear_extensions(switching_field_order:SwitchingFieldOrder) -> int:
     #Count linear extensions based on algorithm described by Peczarski (2004).
    keys = {(state, i) for (state, i) in switching_field_order}
    counts = {tuple():1}
     
    for _ in keys:
        counts_new = {}
        for downset in counts:
            downset = set(downset)
            for (stateA, i) in keys - downset:
                if not any (((stateA, i), (stateB, j)) in switching_field_order.get() for (stateB, j) in keys - downset):
                    upset = downset.copy()
                    upset.add((stateA, i))
                    if tuple(sorted(list(upset))) in counts_new:
                        counts_new[tuple(sorted(list(upset)))] += counts[tuple(sorted(list(downset)))]
                    else:
                        counts_new[tuple(sorted(list(upset)))] = counts[tuple(sorted(list(downset)))]
        counts = counts_new

    return counts[tuple(sorted(list(keys)))]