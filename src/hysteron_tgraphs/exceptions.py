class InvalidTransitionPathException(Exception):
    """
    Is raised when user tries to construct a transition path which is nonsensical.
    In particular, this applies when a transition continues after a self-loop.
    """

class SelfLoopException(Exception):
    """
    Is raised when, for a valid switching threshold order, a graph cannot be constructed due to a self-loop.
    Self-loops are not allowed in graphs by default, but can be marked as allowed upon graph construction.
    See fig.4a in Teunisse & van Hecke, Royal Society Open Science 12(9), 2025.
    """

class AmbiguousTransitionException(Exception):
    """
    Is raised when, for a valid switching field order, a graph cannot be constructed due to race conditions.
    If race conditions only concern switching thresholds of the same direction (up or down),
    these can typically be resolved by flipping the 'critical' hysteron (lowest up or highest down threshold).
    Race conditions which concern both critical hysterons for a given state require more caution.
    See fig.4b in Teunisse & van Hecke, Royal Society Open Science 12(9), 2025.
    """

class InvalidPartialOrderException(Exception):
    """
    Is raised when an operation cannot continue because SwitchingFieldOrder is not a valid partial order.
    This is used purely for mathematical invalidity, as expressed by SwitchingFieldOrder.valid;
    it should not be used for realizability in specific parametrizations.
    """

class InconsistentScaffoldException(Exception):
    """
    Is raised when a consistent scaffold cannot be produced for a graph.
    """

class MarginalSwitchingFieldsException(Exception):
    """
    Is raised when switching fields are too close together for their order to be accurately determined.
    This applies when switching fields are exactly equal, but might also apply when the difference is sufficiently small,
    depending on the system. 
    """