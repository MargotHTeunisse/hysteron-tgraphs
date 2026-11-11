
from hysteron_tgraphs.model import *
from hysteron_tgraphs import parametrizations

import pytest

class TestCustomParametrization:
    class CustomParametrization(parametrizations.Parametrization):
        """
        A simple example parametrization, which sets all thresholds equal to a single constant.
        No graphs are realizable for this parametrization.
        Note the value of the constant is not set; this must be done using a constraint.
        """
        def __init__(self, num_hysts:int):
            self.num_hysts = num_hysts
            self.variables = ['constant']
            
        def transform(self, state:State, i:int) -> dict[str, np.float64]:
            coeffs = {'constant':1}
            return coeffs
        
    @pytest.mark.parametrize("stateA, i, stateB, j", [
            ((0,), 0, (1,), 0),
            ((0, 0), 0, (0, 0), 1),
            ((0, 0, 1, 1), 2, (0, 1, 0, 1), 1) # precise inequality does not matter
    ])
    def test_reduces_inequality_to_zero_greater_than_zero(self, stateA:State, i:int, stateB:State, j:int):
            param = self.CustomParametrization(len(stateA))
            switching_field_order = SwitchingFieldOrder(len(stateA))
            switching_field_order.add(stateA, i, stateB, j)

            inequalities: dict[str, List[int]] = parametrizations.convert_to_specific_inequalities(switching_field_order,
                                                                                                   param)

            assert inequalities == {'constant':[0,]}

    @pytest.mark.parametrize("state0, flipped", [
         ((0,), (0,)),
         ((0, 0), (1,)),
         ((0, 0), (1, 0)),
         ((0, 1), (0, 1))
    ])
    def test_transitions_are_not_realizable(self, state0:State, flipped:List[int]):
         """
         Check that no transition is realizable, since all inequalities reduce to 0 > 0.
         """
         param = self.CustomParametrization(len(state0))
         graph = Graph(len(state0))
         graph.add(state0, flipped)
         switching_field_order = make_design_inequalities(graph)
         inequalities = parametrizations.convert_to_specific_inequalities(switching_field_order, param)

         solvable = parametrizations.check_solvable(inequalities)

         assert solvable == False

    def test_conversion_to_general_switching_fields(self):
        param = self.CustomParametrization(1)
        solution = {'constant':5}

        general_sfs: SwitchingFields = parametrizations.convert_to_general_sfs(solution, param)

        assert all([switching_field == solution['constant'] for state in general_sfs
                    for switching_field in general_sfs[state]])

class TestPairwiseAdditiveParametrization:
    @pytest.mark.parametrize("num_hysts,expected", [
        (1, {'u_1^+', 'u_1^-'}),
        (2, {'u_1^+', 'u_2^+', 'u_1^-', 'u_2^-', 'c_12', 'c_21'}),
        (3, {'u_1^+', 'u_2^+', 'u_3^+', 'u_1^-', 'u_2^-', 'u_3^-',
             'c_12', 'c_13', 'c_21', 'c_23', 'c_31', 'c_32'})
    ])
    def test_has_correct_variable_names(self, num_hysts:int, expected:set[str]):
        param = parametrizations.PairwiseAdditive(num_hysts)

        assert set(param.variables) == expected

    def test_ferro_graph_is_realizable_for_ferro_constraints(self):
        param = parametrizations.PairwiseAdditive(2)
        graph = Graph(2)
        graph.add((0, 0), (1, 0))
        graph.add((1, 1), (1, 0))
        switching_field_order = make_design_inequalities(graph)
        inequalities = parametrizations.convert_to_specific_inequalities(switching_field_order, param)

        solvable = parametrizations.check_solvable(inequalities, param.generate_constraints_ferro())

        assert solvable == True

    def test_antiferro_transition_is_not_realizable_for_ferro_constraints(self):
        param = parametrizations.PairwiseAdditive(2)
        graph = Graph(2)
        graph.add((0, 1), (0, 1))
        switching_field_order = make_design_inequalities(graph)
        inequalities = parametrizations.convert_to_specific_inequalities(switching_field_order, param)

        solvable = parametrizations.check_solvable(inequalities, param.generate_constraints_ferro())

        assert solvable == False

    def test_antiferro_graph_is_realizable_for_antiferro_constraints(self):
        param = parametrizations.PairwiseAdditive(2)
        graph = Graph(2)
        graph.add((0, 1), (0, 1))
        graph.add((1, 0), (0, 1))
        switching_field_order = make_design_inequalities(graph)
        inequalities = parametrizations.convert_to_specific_inequalities(switching_field_order, param)

        solvable = parametrizations.check_solvable(inequalities, param.generate_constraints_antiferro())

        assert solvable == True

    def test_ferro_transition_is_not_realizable_for_antiferro_constraints(self):
        param = parametrizations.PairwiseAdditive(2)
        graph = Graph(2)
        graph.add((0, 0), (1, 0))
        switching_field_order = make_design_inequalities(graph)
        inequalities = parametrizations.convert_to_specific_inequalities(switching_field_order, param)

        solvable = parametrizations.check_solvable(inequalities, param.generate_constraints_antiferro())

        assert solvable == False

    def test_preisach_graph_is_realizable_for_preisach_constraints(self):
        param = parametrizations.PairwiseAdditive(2)
        graph = Graph(2)
        graph.add((0, 0), (1,))
        graph.add((0, 1), (0,))
        graph.add((0, 1), (1,))
        graph.add((1, 1), (1,))
        graph.add((1, 0), (0,))
        graph.add((1, 0), (1,))
        switching_field_order = make_design_inequalities(graph)
        inequalities = parametrizations.convert_to_specific_inequalities(switching_field_order, param)

        solvable = parametrizations.check_solvable(inequalities, param.generate_constraints_preisach())

        assert solvable == True

    def test_ferro_transition_is_not_realizable_for_preisach_constraints(self):
        param = parametrizations.PairwiseAdditive(2)
        graph = Graph(2)
        graph.add((0, 0), (1, 0))
        switching_field_order = make_design_inequalities(graph)
        inequalities = parametrizations.convert_to_specific_inequalities(switching_field_order, param)

        solvable = parametrizations.check_solvable(inequalities, param.generate_constraints_preisach())

        assert solvable == False

    def test_antiferro_transition_is_not_realizable_for_preisach_constraints(self):
        param = parametrizations.PairwiseAdditive(2)
        graph = Graph(2)
        graph.add((0, 1), (0, 1))
        switching_field_order = make_design_inequalities(graph)
        inequalities = parametrizations.convert_to_specific_inequalities(switching_field_order, param)

        solvable = parametrizations.check_solvable(inequalities, param.generate_constraints_preisach())

        assert solvable == False

    def test_scrambled_graph_is_not_realizable_for_preisach_constraints(self):
        param = parametrizations.PairwiseAdditive(3)
        graph = Graph(3)
        graph.add((0, 0, 0), (2,))
        graph.add((1, 0, 0), (1,))
        switching_field_order = make_design_inequalities(graph)
        inequalities = parametrizations.convert_to_specific_inequalities(switching_field_order, param)

        solvable = parametrizations.check_solvable(inequalities, param.generate_constraints_preisach())

        assert solvable == False

    def test_can_produce_valid_solution(self):
        param = parametrizations.PairwiseAdditive(2)
        graph = Graph(2)
        graph.add((0, 0), (1, 0))
        graph.add((0, 1), (0, 1))
        graph.add((0, 1), (1,))
        graph.add((1, 1), (1,))
        graph.add((1, 0), (0,))
        graph.add((1, 0), (1,))
        switching_field_order = make_design_inequalities(graph)
        inequalities = parametrizations.convert_to_specific_inequalities(switching_field_order, param)

        solution = param.solve(inequalities)

        replicated_graph = make_graph(parametrizations.convert_to_general_sfs(solution, param))
        assert replicated_graph == graph

class TestCouplingInSeries:
    """
    Test coupling 'in series', where c_ij = c_j and c_j < 0.
    For details see Liu, Teunisse, Korovin, Vermaire, Jin, Bense and Van Hecke, PNAS (2024).
    """
    @pytest.mark.parametrize("num_hysts,expected", [
        (1, {'u_1^+', 'u_1^-'}),
        (2, {'u_1^+', 'u_2^+', 'u_1^-', 'u_2^-', 'c_1', 'c_2'}),
        (3, {'u_1^+', 'u_2^+', 'u_3^+', 'u_1^-', 'u_2^-', 'u_3^-',
             'c_1', 'c_2', 'c_3'}),
        (4, {'u_1^+', 'u_2^+', 'u_3^+', 'u_4^+',
            'u_1^-', 'u_2^-', 'u_3^-', 'u_4^-',
             'c_1', 'c_2', 'c_3', 'c_4'})
    ])
    def test_has_correct_variable_names(self, num_hysts:int, expected:set[str]):
        param = parametrizations.PairwiseAdditiveGlobal(num_hysts)

        assert set(param.variables) == expected

    def test_scrambling_is_not_allowed_for_global_coupling(self):
        param = parametrizations.PairwiseAdditiveGlobal(3)
        graph = Graph(3)
        graph.add((0, 0, 0), (2,))
        graph.add((1, 0, 0), (1,))
        switching_field_order = make_design_inequalities(graph)
        inequalities = parametrizations.convert_to_specific_inequalities(switching_field_order, param)

        solvable = parametrizations.check_solvable(inequalities)

        assert solvable == False

    def test_avalanches_must_be_grouped_for_global_coupling(self):
        param = parametrizations.PairwiseAdditiveGlobal(3)
        graph = Graph(3)
        graph.add((0, 0, 1), (1, 2))
        graph.add((1, 0, 1), (1,))
        switching_field_order = make_design_inequalities(graph)
        inequalities = parametrizations.convert_to_specific_inequalities(switching_field_order, param)

        solvable = parametrizations.check_solvable(inequalities)

        assert solvable == False

    def test_grouped_avalanches_are_realizable_for_coupling_in_series(self):
        param = parametrizations.PairwiseAdditiveGlobal(3)
        graph = Graph(3)
        graph.add((0, 0, 1), (1, 2))
        graph.add((1, 0, 1), (1, 2))
        switching_field_order = make_design_inequalities(graph)
        inequalities = parametrizations.convert_to_specific_inequalities(switching_field_order, param)

        solvable = parametrizations.check_solvable(inequalities, param.generate_constraints_antiferro())

        assert solvable == True

class TestConstraints:
    """"
    Test construction and application of constraints.
    """
    def test_inequality_constraint_conflicts_with_inequalities(self):
        variables = ["var_A", "var_B"] # Example variables
        inequalities = {'var_A':[1,], 'var_B':[-1,]} # var_A > var_B
        constraints = parametrizations.Constraints(variables)
        constraints.add_inequality({'var_A':-1, 'var_B':1}, b=1) # var_B - var_A >= 1

        solvable = parametrizations.check_solvable(inequalities, constraints)

        assert solvable == False

    def test_equality_constraint_conflicts_with_inequalities(self):
        variables = ["var_A", "var_B"] # Example variables
        inequalities = {'var_A':[1,], 'var_B':[-1,]} # var_A > var_B
        constraints = parametrizations.Constraints(variables)
        constraints.add_equality({'var_A':-1, 'var_B':1}, b=1) # var_B - var_A = 1

        solvable = parametrizations.check_solvable(inequalities, constraints)

        assert solvable == False

    def test_constraints_conflict_with_one_another(self):
        variables = ["var_A", "var_B"] # Example variables
        inequalities = {'var_A':[], 'var_B':[]} # no inequalities
        constraints = parametrizations.Constraints(variables)
        constraints.add_inequality({'var_A':-1, 'var_B':1}, b=2) # var_B - var_A >= 2
        constraints.add_equality({'var_A':-1, 'var_B':1}, b=1) # var_B - var_A = 1

        solvable = parametrizations.check_solvable(inequalities, constraints)

        assert solvable == False

    def test_constraints_do_not_conflict_with_one_another(self):
        variables = ["var_A", "var_B"] # Example variables
        inequalities = {'var_A':[], 'var_B':[]} # no inequalities
        constraints = parametrizations.Constraints(variables)
        constraints.add_inequality({'var_A':-1, 'var_B':1}, b=-2) # var_A - var_B <= 2
        constraints.add_equality({'var_A':1, 'var_B':-1}, b=1) # var_A - var_B = 1

        solvable = parametrizations.check_solvable(inequalities, constraints)

        assert solvable == True

    def test_constraints_do_not_conflict_with_inequalities(self):
        variables = ["var_A", "var_B"] # Example variables
        inequalities = {'var_A':[1,], 'var_B':[-1,]} # var_A > var_B
        constraints = parametrizations.Constraints(variables)
        constraints.add_inequality({'var_A':-1, 'var_B':1}, b=-2) # var_A - var_B <= 2
        constraints.add_equality({'var_A':1, 'var_B':-1}, b=1) # var_A - var_B = 1

        solvable = parametrizations.check_solvable(inequalities, constraints)

        assert solvable == True