from hysteron_tgraphs.model import *
from hysteron_tgraphs.exceptions import *
import pytest

class TestTransition:
    def test_single_hysteron_can_flip_up(self):
        up_transition = Transition((0,), (0,))

        assert up_transition.final_state == (1,)

    def test_single_hysteron_can_flip_down(self):
        down_transition = Transition((1,), (0,))

        assert down_transition.final_state == (0,)

    def test_hysteron_indices_cannot_exceed_number_of_hysterons(self):
        with pytest.RaisesExc(ValueError):
            Transition((0,), (1,))

    @pytest.mark.parametrize("state, flipped", [
        ((0,), (0, 0)),
        ((0, 0), (1, 0, 1, 0)),
        ((0, 0), (1, 0, 0))
    ])
    def test_recognizes_self_loop(self, state:State, flipped:Tuple[int]):
        assert Transition(state, flipped).is_loop

    @pytest.mark.parametrize("state, flipped", [
        ((0,), (0,)),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 0, 1)),
    ])
    def test_recognizes_non_looping_transition(self, state:State, flipped:Tuple[int]):
        assert not Transition(state, flipped).is_loop

    @pytest.mark.parametrize("state, flipped", [
        ((0,), (0, 0, 0)),
        ((0, 0), (1, 0, 1, 0, 1))
    ])
    def test_cannot_continue_transition_after_looping(self, state:State, flipped:Tuple[int]):
        with pytest.RaisesExc(InvalidTransitionPathException):
            Transition(state, flipped)

    @pytest.mark.parametrize("lamb,expected", [
        (1, (0, 1)),
        (2, (1, 1))
    ])
    def test_can_obtain_intermediate_state(self, lamb:int, expected:State):
        transition = Transition((0, 0), (1, 0, 1))

        intermediate_state: State = transition.intermediate_state(lamb)

        assert intermediate_state == expected

    @pytest.mark.parametrize("lamb", [0, 3, 4])
    def test_cannot_obtain_intermediate_state_at_invalid_index(self, lamb:int):
        transition = Transition((0, 0), (1, 0, 1))

        with pytest.RaisesExc(ValueError):
            transition.intermediate_state(lamb)

    @pytest.mark.parametrize("flipped",[
        (0,),
        (1,),
        (1, 0, 1)
    ])
    def test_can_obtain_critical_hysteron(self, flipped:Tuple[int]):
        transition = Transition((0, 0), flipped)

        critical_hysteron = transition.critical_hysteron

        expected = flipped[0]
        assert critical_hysteron == expected

    @pytest.mark.parametrize("flipped",[
        (0,),
        (1,),
        (1, 0, 1),
        (2, 0),
        (3,)
    ])    
    def test_can_obtain_transition_sign(self, flipped:Tuple[int]):
        """
        Test that transition sign can be obtained;
        this is the initial phase(0/1) of the critical hysteron. 
        """
        state = (0, 0, 1, 1)
        transition = Transition(state, flipped)
        critical_hysteron = flipped[0]

        sign = transition.sign

        assert sign == state[critical_hysteron]

    @pytest.mark.parametrize("flipped,expected",[
        ((0,), 1),
        ((1,), 1),
        ((1, 0, 1),1),
        ((2, 0), -1),
        ((3,), -1)
    ])    
    def test_can_obtain_transition_direction(self, flipped:Tuple[int], expected:int):
        """
        Test that transition direction can be obtained (up = +1, down = -1).
        """
        state = (0, 0, 1, 1)
        transition = Transition(state, flipped)
        critical_hysteron = flipped[0]

        direction = transition.direction

        assert direction == expected

class TestScaffold:
    def test_scaffolds_are_equal(self):
        first_scaffold = Scaffold(2)
        first_scaffold[(0, 0), 1] = 1

        second_scaffold = Scaffold(2)
        second_scaffold[(0, 0), 1] = 1

        assert first_scaffold == second_scaffold

    def test_scaffolds_are_not_equal(self):
        first_scaffold = Scaffold(2)
        first_scaffold[(0, 0), 1] = 0

        second_scaffold = Scaffold(2)
        second_scaffold[(0, 0), 1] = 1

        assert first_scaffold != second_scaffold

    def test_index_must_match_direction(self):
        scaffold = Scaffold(0)

        with pytest.RaisesExc(ValueError):
            scaffold[(0,), -1] = 0

    @pytest.mark.parametrize("state, direction, index, expected", [
        ((0,), 1, 0, (1,)),
        ((1,), -1, 0, (0,)),
        ((0, 0, 1), 1, 1, (0, 1, 1))
    ])
    def test_can_obtain_landing_state(self, state:State, direction:int, index:int, expected:State):
        scaffold = Scaffold(len(state))
        scaffold[state, direction] = index

        landing_state = scaffold.landing_state(state, direction)

        assert landing_state == expected

    def test_can_obtain_avalanche_free_graph_from_scaffold(self):
        scaffold = Scaffold(2)
        scaffold[(0, 0), 1] = 1
        scaffold[(0, 1), 1] = 0
        scaffold[(1, 1), -1] = 1
        scaffold[(1, 0), -1] = 0
        scaffold[(0, 1), -1] = 1
        scaffold[(1, 0), 1] = 1

        graph_from_scaffold = make_graph_from_scaffold(scaffold)

        graph = Graph(2)
        graph.add((0, 0), (1,))
        graph.add((0, 1), (0,))
        graph.add((1, 1), (1,))
        graph.add((1, 0), (0,))
        graph.add((0, 1), (1,))
        graph.add((1, 0), (1,))
        assert graph_from_scaffold == graph

class TestGraph:
    def test_make_single_hysteron_graph(self):
        switching_fields = SwitchingFields(1)
        switching_fields[(0,)] = (2,)
        switching_fields[(1,)] = (1,)
        graph = make_graph(switching_fields)

        check_graph = Graph(1)
        check_graph.add((0,), (0,))
        check_graph.add((1,), (0,))

        assert graph == check_graph

    def test_identifies_graphs_for_different_num_hysterons_as_unequal(self):
        one_hysteron_graph = Graph(1)
        two_hysteron_graph = Graph(2)

        assert one_hysteron_graph != two_hysteron_graph

    def test_identifies_graph_with_different_transitions_as_unequal(self):
        first_graph = Graph(1)
        first_graph.add((0,), (0,))
        second_graph = Graph(1)
        second_graph.add((1,), (0,))

        assert first_graph != second_graph

    def test_self_loop_raises_exception_by_default(self):
        switching_fields = SwitchingFields(1)
        switching_fields[(0,)] = (1,)
        switching_fields[(1,)] = (2,)

        with pytest.RaisesExc(SelfLoopException) as e:
            make_graph(switching_fields)

    def test_self_loop_allowance(self):
        switching_fields = SwitchingFields(1)
        switching_fields[(0,)] = (1,)
        switching_fields[(1,)] = (2,)
        graph = make_graph(switching_fields, allow_self=True)

        check_graph = Graph(1)
        check_graph.add((0,), (0, 0))
        check_graph.add((1,), (0, 0))
    
        assert graph == check_graph

    def test_make_two_hysteron_graph(self):
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

    def test_race_conditions_raise_exception_by_default(self):
        switching_fields = SwitchingFields(2)
        switching_fields[(0, 0)] = (2, 1)
        switching_fields[(0, 1)] = (-1, 2)
        switching_fields[(1, 1)] = (-2, 1)
        switching_fields[(1, 0)] = (0, 3)

        with pytest.RaisesExc(AmbiguousTransitionException) as e:
            make_graph(switching_fields, allow_self=True)

    def test_resolve_race_conds_via_most_unstable_hysteron(self):
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

class TestSwitchingFieldOrder:
    def test_valid_graph(self):
        graph = Graph(2)
        graph.add((0, 0), (1, 0))
        graph.add((0, 1), (0, 1))
        graph.add((0, 1), (1,))
        graph.add((1, 1), (1,))
        graph.add((1, 0), (0,))
        graph.add((1, 0), (1,))

        switching_field_order = make_design_inequalities(graph)
    
        assert switching_field_order.valid

    def test_transitive_closure_for_single_transition(self):
        """
        Test that transitive closure (maximal set of inequalities) can be obtained.
        """
        graph = Graph(2)
        graph.add((0, 0), (1, 0))
        switching_field_order = make_design_inequalities(graph)

        inequalities:Set[Tuple[Tuple[State, int], Tuple[State, int]]] = switching_field_order.get()

        transitive_closure = {(((0, 0), 0), ((0, 0), 1)),
                              (((0, 0), 1), ((0, 1), 0)),
                              (((0, 0), 1), ((0, 1), 1)),
                              (((0, 0), 1), ((1, 1), 0)),
                              (((0, 0), 1), ((1, 1), 1)),
                              (((0, 0), 0), ((0, 1), 0)),
                              (((0, 0), 0), ((0, 1), 1)),
                              (((0, 0), 0), ((1, 1), 0)),
                              (((0, 0), 0), ((1, 1), 1)),}
        assert inequalities == transitive_closure

    def test_transitive_reduction_for_single_transition(self):
        """
        Test that transitive reduction (maximal set of inequalities) can be obtained.
        """
        graph = Graph(2)
        graph.add((0, 0), (1, 0))
        switching_field_order = make_design_inequalities(graph)

        inequalities:Set[Tuple[Tuple[State, int], Tuple[State, int]]] = switching_field_order.get_transitive_reduction()

        transitive_closure = {(((0, 0), 0), ((0, 0), 1)),
                              (((0, 0), 1), ((0, 1), 0)),
                              (((0, 0), 1), ((0, 1), 1)),
                              (((0, 0), 1), ((1, 1), 0)),
                              (((0, 0), 1), ((1, 1), 1))}
        assert inequalities == transitive_closure

    def test_cannot_construct_transitive_reduction_for_invalid_partial_order(self):
        switching_field_order = SwitchingFieldOrder(1)
        switching_field_order.add((0,), 0, (1,), 0)
        switching_field_order.add((1,), 0, (0,), 0)

        with pytest.RaisesExc(InvalidPartialOrderException) as e:
            switching_field_order.get_transitive_reduction()


    def test_valid_self_loop(self):
        """
        Check that the conditions for a self-loop are valid.
        Note that these conditions are well-defined, so no exception is raised.
        """
        graph = Graph(2)
        graph.add((0, 0), (1, 0, 1, 0))

        switching_field_order = make_design_inequalities(graph)

        assert switching_field_order.valid


    def test_invalid_graph_direct_contradiction(self):
        """
        Check that a direct conflict between inequalities is identified correctly.
        See also Teunisse & Van Hecke (2025), Fig. 7a-b. 
        """
        graph = Graph(2)
        graph.add((0, 0), (1,))
        graph.add((0, 1), (0, 1, 0))

        switching_field_order = make_design_inequalities(graph)
        assert switching_field_order.valid == False

    def test_invalid_graph_indirect_contradiction(self):
        """
        Check that an indirect direct conflict between inequalities is identified correctly.
        See also Teunisse & Van Hecke (2025), Fig. 7c-d. 
        """
        graph = Graph(2)
        graph.add((0, 0), (1, 0))
        graph.add((1, 0), (0, 1))

        switching_fields = make_design_inequalities(graph)
        assert switching_fields.valid == False

    def test_irresolvible_race_conditions(self):
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

        assert switching_field_order.valid == False

    @pytest.mark.parametrize("stateA,i,stateB,j", [
        ((0,), 0, (1,), 0),
        ((0, 0), 0, (0, 0), 1)
    ])
    def test_equality_and_inequality_give_contradiction(self, stateA:State, i:int, stateB:State, j:int):
        switching_field_order = SwitchingFieldOrder(len(stateA))
        switching_field_order.add(stateA, i, stateB, j)

        switching_field_order.equate(stateA, i, stateB, j)

        assert switching_field_order.valid == False

    @pytest.mark.parametrize("num_hysts", [1, 2])
    def test_total_linear_extensions_is_factorial(self, num_hysts:int):
        """
        The total amount of linear extensions for a given number of hysterons is the factorial of the number of thresholds n*2^n.
        """
        switching_field_order = SwitchingFieldOrder(num_hysts)

        total_count = count_linear_extensions(switching_field_order)

        # This is just a manual factorial;
        # I'd rather keep my unit tests independent of Numpy.
        num_thresholds = num_hysts*2**num_hysts
        expected = 1
        for factor in range(1, num_thresholds + 1):
            expected*= factor
        assert total_count == expected

    @pytest.mark.parametrize("critical_down_hysteron", [0, 1])
    def test_linear_extensions_for_two_hysteron_scaffold(self, critical_down_hysteron:int):
        """
        Check that linear extensions are counted correctly for scaffold inequalities.
        This can be done analytically, with each scaffold taking up an equal partition of solution space:
        see Teunisse & van Hecke (2025), section 5.2. 
        """
        num_hysts = 2
        scaffold = Scaffold(num_hysts)
        scaffold[((0, 0), 1)] = 1
        scaffold[((0, 1), 1)] = 0
        scaffold[((1, 0), 1)] = 1
        scaffold[((0, 1), -1)] = 1
        scaffold[((1, 0), -1)] = 0
        scaffold[((1, 1), -1)] = critical_down_hysteron
        switching_field_order = make_initial_inequalities(scaffold, stable=False)

        linear_extension_count: int = count_linear_extensions(switching_field_order)

        total_linear_extensions = 40320 # = 8!
        num_scaffolds = 4 # just the 'Preisach' scaffolds for 2 hysterons; more complex for 3+ hysterons
        expected = total_linear_extensions//num_scaffolds # analytical linear extension count
        assert linear_extension_count == expected

    def test_cannot_iadd_switching_field_orders_for_different_hysteron_number(self):
        one_hysteron_order = SwitchingFieldOrder(1)
        two_hysteron_order = SwitchingFieldOrder(2)

        with pytest.RaisesExc(Exception):
            one_hysteron_order += two_hysteron_order

    def test_can_iadd_switching_field_orders_for_same_hysteron_number(self):
        switching_field_order = SwitchingFieldOrder(2)
        switching_field_order.add((0, 0), 1, (0, 1), 1,)
        second_switching_field_order = SwitchingFieldOrder(2)
        second_switching_field_order.add((0, 0), 0, (1, 0), 0,)
        switching_field_order += second_switching_field_order

        combined_inequalities = switching_field_order.get()
        expected = {(((0, 0), 0), ((1, 0), 0)), (((0, 0), 1), ((0, 1), 1))}
        assert combined_inequalities == expected

    @pytest.mark.parametrize("filled,expected", [
        ({}, {(((0,), 0), ((1,), 0)), (((1,), 0), ((0,), 0))}),
        ({(((0,), 0), ((1,), 0))}, set())
    ])
    def test_can_identify_open_entries(self, filled:Set[Tuple[Tuple[State, int]]],
                                       expected:Set[Tuple[Tuple[State, int]]]):
        switching_field_order = SwitchingFieldOrder(1)
        for ((stateA, i), (stateB, j)) in filled:
            switching_field_order.add(stateA, i, stateB, j)

        open_entries = switching_field_order.open_entries

        assert open_entries == expected

    @pytest.mark.parametrize("num_hysts, expected", [
        (1, set()),
        (2, {(((0, 0), 0), ((0, 0), 1))}), 
        (3, {(((0, 0, 0), 0), ((0, 0, 0), 2)), (((0, 0, 0), 1), ((0, 0, 0), 2)),
             (((0, 0, 1), 0), ((0, 0, 1), 1))})
    ])
    def test_symmetry_restriction(self, num_hysts:int, expected:Set[Tuple[Tuple[State, int]]]):
        switching_field_order = SwitchingFieldOrder(num_hysts)

        switching_field_order.enforce_symmetry_restriction()

        inequalities = switching_field_order.get_transitive_reduction()
        assert inequalities == expected

class TestGeneralSolution:
    def test_can_replicate_graph_using_general_solution(self):
        graph = Graph(2)
        graph.add((0, 0), (1, 0))
        graph.add((0, 1), (1,))
        graph.add((0, 1), (0, 1))
        graph.add((1, 1), (1,))
        graph.add((1, 0), (1,))
        graph.add((1, 0), (0,))
        switching_field_order: SwitchingFieldOrder = make_design_inequalities(graph)

        general_solution: SwitchingFields = generate_general_solution(switching_field_order)

        replicated_graph = make_graph(general_solution)
        assert replicated_graph == graph

    def test_cannot_generate_general_solution_for_invalid_order(self):
        switching_field_order = SwitchingFieldOrder(1)
        switching_field_order.add((0,), 0, (1,), 0)
        switching_field_order.add((1,), 0, (0,), 0)

        with pytest.RaisesExc(InvalidPartialOrderException) as e:
            generate_general_solution(switching_field_order)
