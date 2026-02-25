"""Custom exceptions for the FLUX versioning system."""


class FluxError(Exception):
    """Base exception for all FLUX errors."""
    pass


class RepositoryNotFoundError(FluxError):
    """Raised when a FLUX repository is not found at the given path."""
    pass


class RepositoryExistsError(FluxError):
    """Raised when trying to initialize a repository that already exists."""
    pass


class VersionNotFoundError(FluxError):
    """Raised when a requested version does not exist."""
    pass


class TagNotFoundError(FluxError):
    """Raised when a requested tag does not exist."""
    pass


class LockError(FluxError):
    """Raised when a lock cannot be acquired within the timeout."""
    pass


class LockTimeoutError(LockError):
    """Raised when lock acquisition times out."""
    pass


class ImportError_(FluxError):
    """Raised when importing a version fails (hash mismatch, already exists, etc.)."""
    pass


class PreprocessingError(FluxError):
    """Raised when a preprocessing step fails."""
    pass


class InvalidConfigError(FluxError):
    """Raised when a preprocessing config is invalid."""
    pass


class DataFormatError(FluxError):
    """Raised when the input data format is invalid (e.g., missing 'text' column)."""
    pass
