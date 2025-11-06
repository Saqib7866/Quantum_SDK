"""Custom exception types for QX SDK."""

class QXError(Exception):
    """Base class for QX errors."""
    pass


class QXCompileError(ValueError, QXError):
    """Raised for compile / transpiler errors. Inherits from ValueError for backwards compatibility."""
    pass


class QXBackendError(QXError):
    """Raised for backend selection/creation errors."""
    pass


class QXRuntimeError(QXError):
    """Raised for runtime/job-related errors."""
    pass


class QXValidationError(QXError):
    """Raised for IR validation problems."""
    pass


class CircuitError(QXError):
    """Base class for circuit-related errors."""
    pass

class QubitOutOfRangeError(CircuitError, IndexError):
    """Raised when a qubit index is out of the valid range."""
    pass

class MeasurementError(CircuitError):
    """Raised for errors during measurement."""
    pass

class GateError(CircuitError):
    """Raised for errors related to quantum gates."""
    pass
