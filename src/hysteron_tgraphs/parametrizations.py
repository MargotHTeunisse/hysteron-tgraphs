import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from typing import List, Dict
from hysteron_tgraphs.model import *
from scipy import optimize

Inequalities = Dict[str, List[np.float64]]
Solution = Dict[str, np.float64]

class Parametrization(ABC):
    num_hysts:int
    variables:List[str]

    @abstractmethod
    def transform(self, state:State, i:int) -> Dict[str, np.float64]:
        pass

class Constraints():
    def __init__(self, variables:List[str]):
        self.variables = variables
        self.equalities = {var:[] for var in variables}
        self.inequalities = {var:[] for var in variables}
        self.rhs_eq = []
        self.rhs_ineq = []

    def add_inequality(self, inequality:Dict[str, np.float64], b=0):
        if set(inequality.keys()) <= set(self.variables):
            for var in inequality:
                self.inequalities[var].append(inequality[var])
            for var in set(self.variables) - set(inequality.keys()):
                self.inequalities[var].append(0)
            self.rhs_ineq.append(b)

    def add_equality(self, equality:Dict[str, np.float64], b=0):
        if set(equality.keys()) <= set(self.variables):
            for var in equality:
                self.equalities[var].append(equality[var])
            for var in set(self.variables) - set(equality.keys()):
                self.equalities[var].append(0)
            self.rhs_eq.append(b)

class PairwiseAdditive(Parametrization):
    def __init__(self, num_hysts:int):
        self.num_hysts = num_hysts
        self.variables = ['u_{}^+'.format(i+1) for i in range(num_hysts)] + ['u_{}^-'.format(i+1) for i in range(num_hysts)] + ['c_{}{}'.format(i+1, j+1) for i in range(num_hysts) for j in range(num_hysts) if i != j]

    def transform(self, state:State, i:int) -> Dict[str, np.float64]:
        coeffs = {var:0 for var in self.variables}
        coeffs['u_{}^{}'.format(i+1, ['+', '-'][state[i]])] = 1
        coeffs.update({'c_{}{}'.format(i+1, k+1): -state[k] for k in range(self.num_hysts) if k != i})
        return coeffs

    def solve(self, inequalities:Inequalities, constraints:Constraints=None) -> Solution:
        #Find Chebyshev center.
        if constraints == None:
            constraints = Constraints(self.variables)

        A_ub = -np.array([inequalities[var]+constraints.inequalities[var] for var in self.variables]).T
        A_ub = np.concatenate((np.linalg.norm(A_ub, axis=1)[:, np.newaxis], A_ub), axis=1, dtype=float)
        A_eq = np.array([constraints.equalities[var] for var in self.variables]).T
        A_eq = np.insert(A_eq, 0, 0, axis=1)
        b_ub = -np.array([0]*len(inequalities[self.variables[0]]) + constraints.rhs_ineq)
        b_eq = constraints.rhs_eq
        
        x = optimize.linprog([-1] + [0]*len(self.variables), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds = [(0, None)] + [(-1, 1)]*len(self.variables))['x']
        
        R = x[0]
        accuracy = int(np.log10(R/np.sqrt(len(self.variables)))//1)
        #Round solution based on distance to the nearest boundary
        solution = {var:np.round(val, -accuracy) for (var, val) in zip(self.variables, x[1:])}
        
        return solution
    
class PairwiseAdditiveGlobal(Parametrization):
    #Parametrization for 'global' coupling, where c_ij = c_j. 
    def __init__(self, num_hysts:int):
        self.num_hysts = num_hysts
        self.variables = ['u_{}^+'.format(i+1) for i in range(num_hysts)] + ['u_{}^-'.format(i+1) for i in range(num_hysts)] + ['c_{}'.format(j+1) for j in range(num_hysts)]

    def transform(self, state:State, i:int) -> Dict[str, np.float64]:
        coeffs = {var:0 for var in self.variables}
        coeffs['u_{}^{}'.format(i+1, ['+', '-'][state[i]])] = 1
        coeffs.update({'c_{}'.format(k+1): -state[k] for k in range(self.num_hysts) if k != i})
        return coeffs
        
    def generate_constraints_antiferro(self) -> Constraints:
        #Generate constraints such that all coupling is antiferromagnetic; we often look at coupling in series (global + antiferro), so useful to have a function for this. 
        constraints = Constraints(self.variables)
        
        for j in range(self.num_hysts):
            #Set antiferro coupling
            constraints.add_inequality({"c_{}".format(j+1):-1})

        return constraints 

#Translation functions between general model and specific parametrizations
def convert_to_specific_inequalities(switching_field_order:SwitchingFieldOrder, param:Parametrization) -> Inequalities:
    inequalities = {var:[] for var in param.variables}
    for no, ((stateA, i), (stateB, j)) in enumerate(switching_field_order.get_transitive_reduction()):
        for var in param.variables:
            inequalities[var].append(0)
        for (var, val) in param.transform(stateA, i).items():
            inequalities[var][-1] += val
        for (var, val) in param.transform(stateB, j).items():
            inequalities[var][-1] -= val
    return inequalities

def convert_to_general_sfs(solution:Solution, param:Parametrization) -> SwitchingFields:
    switching_fields = SwitchingFields(param.num_hysts)
    for state in switching_fields:
        switching_fields[state] = [np.sum([val*solution[var] for (var, val) in param.transform(state, i).items()]) for i in range(param.num_hysts)]
    return switching_fields

def check_solvable(inequalities:Inequalities, constraints:Constraints=None, epsilon=1):
    #Uses linear programming to check if a set of inequalities is consistent, given any constraints added. This is independent of the specific model that the inequalities represent.
    variables = list(inequalities.keys())
    if constraints == None:
        constraints = Constraints(variables)

    A_ub = -np.array([inequalities[var]+constraints.inequalities[var] for var in variables]).T
    A_eq = np.array([constraints.equalities[var] for var in variables]).T
    b_ub = -epsilon*np.ones(len(inequalities[variables[0]]))
    b_ub = np.concatenate((b_ub, -1*np.array(constraints.rhs_ineq)))
    b_eq = constraints.rhs_eq

    return optimize.linprog([0]*len(inequalities.keys()), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds = (None, None))['success']
