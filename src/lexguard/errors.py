class LexguardError(Exception):
    """Base class for every error raised by lexguard."""


class AccessDenied(LexguardError):
    """The policy refused the request."""


class KeyUnavailable(LexguardError):
    """The matter key was never enrolled or has been shredded."""


class IntegrityError(LexguardError):
    """Ciphertext or audit data failed authentication."""


class EgressBlocked(LexguardError):
    """Data was about to cross the trust boundary against policy."""
