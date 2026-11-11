import numpy as np
from typing import Tuple, List, Iterator, Set
from dataclasses import dataclass
import itertools
from hysteron_tgraphs import exceptions
State = Tuple[bool, ...]

@dataclass(frozen=True)
class Transition():
    """
   A data class which represents a single transition.
    A transition consists out of a state and a sequence of flipped hysterons.
    """
    state:State
    flipped:Tuple[int, ...]

    def __post_init__(self) -> None:
        """Checks whether the input state and transition form a valid transition. 

        Raises:
            ValueError: Hysteron indices cannot exceed number of hysterons.
            This occurs if 'flipped' contains indices which exceed the number of hysterons in 'state'. 

            InvalidTransitionException: Transition cannot continue after looping.
            To prevent this error, truncate 'flipped'.
        """        
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
                    raise exceptions.InvalidTransitionPathException("Transition cannot continue after looping.")

    def intermediate_state(self, lamb:int) -> State:
        """Get the lambda'th intermediate state for this transition.

        Args:
            lamb (int): The index 'lambda' of the intermediate state within the avalanche.

        Raises:
            ValueError: Not an intermediate state; index must have 1 < lambda < l-1.
            If you are seeing this error, please ensure you are looking at the right transition.

        Returns:
            State: The target intermediate state.
        """        
        if not (lamb > 0 and lamb < self.length):
            raise ValueError("Not an intermediate state; index must have 1 < lambda < l-1.")
        return tuple(s if self.flipped[:lamb].count(i)%2 == 0 else 1-s for i, s in enumerate(self.state))

    @property
    def is_loop(self) -> bool:
        """Check if transition is a self-loop.

        Returns:
            bool: True if transition is a self-loop.
        """        
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
    """
    An object class representing the 'scaffold' of a hysteron system.
    Scaffolds were introduced as an underlying structure to understand the structure of t-graphs.
    See also Teunisse & van Hecke (2025).
    """
    def __init__(self, num_hysts:int):
        """Creates a new scaffold object.

        Args:
            num_hysts (int): The number of hysterons corresponding to the scaffold.
        """
        self.num_hysts = num_hysts
        iterator = ((state, 1-2*sign) for state in itertools.product([0, 1], repeat=num_hysts) for sign in [0, 1] if sign in state)
        self._mapping = {(state, direction):no for no, (state, direction) in enumerate(iterator)}
        self._critical_hysterons = np.array([None,]*len(self._mapping))

    def __eq__(self, other):
        """
        Magic method; defines the command '==' for the scaffold.

        Args:
            other: The scaffold to compare to. 

        Returns:
            bool: True if the two scaffolds are equal.
        """
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
    """
    An object class representing the switching fields of a hysteron system in the most general model.
    These are the defining attributes of a hysteron system.
    """
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
                k = np.argmin([self[state][i] if s == 0 else np.inf for i, s in enumerate(state)])
                if any(np.isclose(self[state][k], [self[state][i] for i in I0-{k}])):
                    raise exceptions.MarginalSwitchingFieldsException("Marginal scaffold.")
                scaffold[(state, 1)] = k
            if 1 in state and not (state, -1) in exclude:
                I1 = set(i for i, s in enumerate(state) if s == 1)
                k = np.argmax([self[state][i] if s == 1 else -np.inf for i, s in enumerate(state)])
                if any(np.isclose(self[state][k], [self[state][i] for i in I1-{k}])):
                    raise exceptions.MarginalSwitchingFieldsException("Marginal scaffold.")
                scaffold[(state, -1)] = k
        return scaffold

    def __getitem__(self, state:State):
        return self._values[self._mapping[state]]

    def __setitem__(self, state:State, row):
        self._values[self._mapping[state]] = row

class SwitchingFieldOrder():
    """
    A class representing a partial order of the switching fields.
    """
    def __init__(self, num_hysts:int):
        self.num_hysts = num_hysts 
        self._mapping = {(state, i):num_hysts*state_no + i for state_no, state in enumerate(itertools.product([0, 1], repeat=num_hysts)) for i in range(num_hysts)}
        self._matrix = np.zeros((len(self._mapping), len(self._mapping)), dtype=bool)
        self._apply_transitive_closure()

    def to_coeffs(self):
        return np.array([np.identity(len(self._mapping))[self._mapping[(stateA, i)]] - np.identity(len(self._mapping))[self._mapping[(stateB, j)]] for ((stateA, i), (stateB, j)) in self.get()])
        
    @property
    def open_entries(self) -> Set[Tuple[Tuple[State, int], Tuple[State, int]]]:
        """A class property that gives all pairs of switching fields which are not restricted by the partial order.

        Returns:
            Set[Tuple[Tuple[State, int], Tuple[State, int]]]: A list of pairs of switching fields, indicated by their state and hysteron index.
        """    
    
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

    def get(self) -> Set[Tuple[Tuple[State, int], Tuple[State, int]]]:
        """Retrieves the set of switching field orderings that defines the partial order.
        This is a maximal set ('transitive closure').  To get a minimal set of inequalities, use get_transitive_reduction().

        Returns:
            Set[Tuple[Tuple[State, int], Tuple[State, int]]]: A set of pairs of switching fields, represented by their state and hysteron index.
            The first entry in each pair is larger than the second entry.
        """        
        return {((stateA, i), (stateB, j)) for (stateA, i) in self._mapping for (stateB, j) in self._mapping if self._matrix[self._mapping[(stateA, i)], self._mapping[(stateB, j)]] == 1}
        
    def get_transitive_reduction(self) -> Set[Tuple[Tuple[State, int], Tuple[State, int]]]:
        """Uses a transitive reduction algorithm (Aho et al., SIAM Journal of Computing, 1972) to get a minimal representation of the partial order.

        Raises:
            InvalidPartialOrderException: Cannot construct transitive reduction for invalid partial order.
            To avoid this error, check beforehand if the partial order is valid using self.valid. 

        Returns:
            Set[Tuple[State, int], Tuple[State, int]]: A set of pairs of switching fields, represented by their state and hysteron index.
            The first entry in each pair is larger than the second entry.
        """
        #
        if not self.valid:
            raise exceptions.InvalidPartialOrderException(
                "Cannot construct transitive reduction for invalid partial order.")
        
        transitive_reduction_matrix = self._matrix*(1-(np.matmul(self._matrix, self._matrix)))
        return {((stateA, i), (stateB, j)) for (stateA, i) in self._mapping for (stateB, j) in self._mapping if transitive_reduction_matrix[self._mapping[(stateA, i)], self._mapping[(stateB, j)]] == 1}

    def equate(self, stateA:State, i:int, stateB:State, j:int) -> None:
        """Enforce that two switching fields are equal to each other.
        This eliminates the second switching field from the partial order.

        Args:
            stateA (State): State corresponding to the first switching field.
            i (int): Hysteron index for the first switching field.
            stateB (State): State corresponding to the second switching field.
            j (int): Hysteron index for the second switching field.
        """
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
        if self._mapping != other._mapping:
            raise ValueError("Switching field mappings must match.")
        self._matrix = np.logical_or(self._matrix, other._matrix)
        self._apply_transitive_closure()
        return self
        
        
class Graph():
    """
    An object class representing a transition graph associated with a hysteron system.
    """
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
        for (state0, direction) in self:
            transition: Transition = self[(state0, direction)]
            state = state0
            for kappa in transition.flipped:
                key = (state, 1-2*state[kappa])
                if scaffold[key] == None:
                    scaffold[key] = kappa
                elif scaffold[key] != kappa:
                    raise exceptions.InconsistentScaffoldException(
                        "Scaffold is inconsistent for given graph.")
                state = flip_state(state, kappa)
        return scaffold

def flip_state(state:State, index:int) -> State:
    return tuple(1-s if index == i else s for i, s in enumerate(state))

def make_initial_inequalities(scaffold:Scaffold, stable=False) -> SwitchingFieldOrder:
    """Constructs the design inequalities associated with a given scaffold.

    Args:
        scaffold (Scaffold): The target scaffold.
        stable (bool, optional): Indicates whether the design inequalities should require states to be stable. Defaults to False.

    Returns:
        SwitchingFieldOrder: The design inequalities of the scaffold in the most general model
    """    
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
    """Constructs the design inequalities associated with a given hysteron t-graph.

    Args:
        graph (Graph): The target t-graph.
        resolve_race (bool, optional): Indicates whether race conditions are 'resolved' by flipping the most unstable hysteron.
        If True, the inequalitieis associated with race conditions are left out.
        Defaults to False.

    Returns:
        SwitchingFieldOrder: The design inequalities of the target t-graph in the most general model.
    """    
    # Scaffold inequalities
    switching_field_order = make_initial_inequalities(graph.scaffold, stable=False)
    
    for (state, direction) in graph:
        transition = graph[(state, direction)]

        stateA = transition.state
        k = transition.critical_hysteron

        # Initial stability inequalities
        for i in range(transition.num_hysts):
            if stateA[k] == 0 and stateA[i] == 1:
                switching_field_order.add(stateA, k, stateA, i)
            if stateA[k] == 1 and stateA[i] == 0:
                switching_field_order.add(stateA, i, stateA, k)
            
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
    """Constructs the t-graph for a given hysteron system in the most general model of interacting hysterons.

    Args:
        switching_fields (SwitchingFields): The switching fields for the target hysteron system in the general interacting hysteron model.
        To construct the t-graph for a specific hysteron, first convert these to the general model.
        resolve_race (bool, optional): Indicates whether race conditions are 'resolved' by flipping the most unstable hysteron.
        Defaults to False.
        allow_self (bool, optional): Indicates whether self-loops are considered well-defined transitions. Defaults to False.
        exclude (List[Tuple[State, int]], optional): Indicates transitions that should be excluded from the list.
        This can be necessary when, for example, the full t-graph would include degeneracies. Defaults to an empty list.

    Raises:
        AmbiguousTransitionException: To prevent this error, set resolve_race to True.
        SelfLoopException: To prevent this error, set allow_self to True.
        MarginalSwitchingFieldsException: cannot determine avalanche because switching fields are marginal. 
        This is raised if, at some point in an avalanche, there is a degeneracy in the critical hysterons.
        This degeneracy cannot be resolved by this code; instead, please check the input switching fields.

    Returns:
        Graph: The transition graph corresponding to the target hysteron system.
    """    
    graph = Graph(switching_fields.num_hysts)

    scaffold = switching_fields.scaffold(exclude)
    iterator = ((state, direction) for (state, direction) in scaffold if not (state, direction) in exclude)
    for (state, direction) in iterator:
        state0 = state
        flipped = tuple()
        transition_path = [state0,]

        kappa = scaffold[(state0, direction)]
        critical_driving = switching_fields[state0][kappa]

        while True:
            flipped += (kappa,)
            state = flip_state(state, kappa)

            # Before anything else, check if the transition path forms a self-loop.
            if state in transition_path:
                if allow_self:
                    break
                raise exceptions.SelfLoopException(
                    f"{['Up', 'Down'][(1-direction)//2]} transition from state {state0} is a self-loop.")
            transition_path += [state]

            # Now check for avalanches, starting from the critical hysterons at the landing state.
            kappa_up = scaffold[(state, 1)] if 0 in state else None
            kappa_down = scaffold[((state, -1))] if 1 in state else None
            critical_up_field = switching_fields[state][kappa_up] if kappa_up != None else np.inf
            critical_down_field = switching_fields[state][kappa_down] if kappa_down != None else -np.inf

            # If both critical hysterons are stable, the transition terminates.
            if critical_up_field >= critical_driving and critical_down_field <= critical_driving:
                if (np.isclose(critical_up_field, critical_driving)
                    or np.isclose(critical_down_field, critical_driving)):
                # Check for marginal stability; we should avoid this.
                    raise exceptions.MarginalSwitchingFieldsException(
                        f"Cannot construct  {['up', 'down'][(1-direction)//2]} transition "
                        + f"from state {state0} " 
                        + f"due to marginal stability of state {state}.")
                break

            # We now know that at least one hysteron is unstable. 
            # The following steps depend on whether we resolve race conditions or not.
            if resolve_race:
                # If race conditions are resolved, we only need to worry about the critical hysterons.
                # First, check for the case where only the critical up hysteron is unstable.
                if critical_down_field < critical_driving:
                    kappa = kappa_up      

                # Now check for the case where only the critical down hysteron is unstable.
                elif critical_up_field > critical_driving:
                    kappa = kappa_down

                # If none of the conditions apply, then both critical hysterons are unstable.
                # We enforce that the hysteron which is furthest from stability should flip.
                else:
                    diff_up = critical_driving - critical_up_field
                    diff_down = critical_down_field - critical_driving
                    kappa = (kappa_up if diff_up > diff_down else kappa_down)

            else:
                # If we do not resolve race conditions, then only a single hysteron can be unstable.
                unstable_hysterons = [i for (i, val) in enumerate(switching_fields[state])
                                        if state[i] == 0 and val < critical_driving
                                        or state[i] == 1 and val > critical_driving]
                if len(unstable_hysterons) > 1:
                        raise exceptions.AmbiguousTransitionException(
                            f"Cannot construct {['up', 'down'][(1-direction)//2]} avalanche from state {state0} "
                            + f"because multiple hysterons in state {state} are unstable. " 
                            + f"Unstable hysterons: {unstable_hysterons}")
                kappa = unstable_hysterons[0]

            #Having determined that hysteron kappa should flip, check that it is not marginally unstable.
            if np.isclose(switching_fields[state][kappa], critical_driving):
                raise exceptions.MarginalSwitchingFieldsException(
                    f"Cannot construct {['up', 'down'][(1-direction)//2]} transition from state {state0} "
                    f"due to marginal instability of hysteron {kappa} at state {state}.")
        
        graph.add(state0, flipped)

    return graph

def make_graph_from_scaffold(scaffold:Scaffold) -> Graph:
    """Constructs the avalanche-free graph for a given scaffold.

    Args:
        scaffold (Scaffold): The target scaffold.

    Returns:
        Graph: The t-graph without any avalanches corresponding to the given scaffold.
    """    
    graph = Graph(scaffold.num_hysts)
    for (state, direction) in scaffold:
        graph.add(state, (scaffold[(state, direction)],))
    return graph

def generate_general_solution(switching_field_order:SwitchingFieldOrder, a=1, b=1) -> SwitchingFields:
    """Constructs a random total ordering of the switching fields that obeys a given partial order.

    Args:
        switching_field_order (SwitchingFieldOrder): A partial order of the switching fields.
        a (int, optional): The distance between switching fields. Defaults to 1.
        b (int, optional): The value of the lowest switching field. Defaults to 1.

    Raises:
        InvalidPartialOrderException: Cannot generate solution for invalid partial order.

    Returns:
        SwitchingFields: A set of switching fields that follows the input partial order.
        Values range from b to ax + b, where x is the number of switching fields.
    """    

    #Check that a solution exists
    if not switching_field_order.valid:
        raise exceptions.InvalidPartialOrderException("Cannot generate solution for invalid partial order.")

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
    """Counts the number of total orders (or 'linear extensions') corresponding to a given partial order of the switching fields.

    Args:
        switching_field_order (SwitchingFieldOrder): A partial order of the switching fields.

    Returns:
        int: The number of total orders for the given partial order.
    """    
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