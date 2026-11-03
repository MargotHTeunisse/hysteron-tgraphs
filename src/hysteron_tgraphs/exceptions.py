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