from hysteron_tgraphs.model import *
from hysteron_tgraphs.exceptions import InvalidTransitionPathException, SelfLoopException, AmbiguousTransitionException
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
